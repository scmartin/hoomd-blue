//! This file is only included once in JIT compilation

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/Index1D.h"
#include "hoomd/jit/Evaluator.cuh"
#include "hoomd/jit/EvaluatorUnionGPU.cuh"

#include "GPUHelpers.cuh"
#include "hoomd/WarpTools.cuh"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace hpmc
{
namespace gpu
{

//! Sentinel for the CNF
const unsigned int SAT_sentinel = 0xffffffff;

namespace kernel
{

//! Compute energy in old and new configuration of every particle
/*!
    The energy evaluation function (EVAL_FN) is selected via a preprocessor define
 */
template<unsigned int eval_threads, unsigned int max_threads>
__launch_bounds__(max_threads)
__global__ void hpmc_narrow_phase_patch(const Scalar4 *d_postype,
                           const Scalar4 *d_orientation,
                           const Scalar4 *d_trial_postype,
                           const Scalar4 *d_trial_orientation,
                           const unsigned *d_trial_move_type,
                           const Scalar *d_charge,
                           const Scalar *d_diameter,
                           const unsigned int *d_excell_idx,
                           const unsigned int *d_excell_size,
                           const Index2D excli,
                           const unsigned int *d_update_order_by_ptl,
                           const unsigned int seed,
                           const unsigned int timestep,
                           const unsigned int select,
                           const unsigned int num_types,
                           const BoxDim box,
                           const Scalar3 ghost_width,
                           const uint3 cell_dim,
                           const Index3D ci,
                           const unsigned int N_local,
                           Scalar r_cut_patch,
                           const Scalar *d_additive_cutoff,
                           const unsigned int *d_reject_out_of_cell,
                           const unsigned int max_queue_size,
                           const unsigned int work_offset,
                           const unsigned int nwork,
                           const unsigned int max_extra_bytes,
                           unsigned int *d_n_inequality,
                           unsigned int *d_inequality_literals,
                           const unsigned int maxn_inequality,
                           unsigned int *d_req_n_inequality,
                           double *d_coeff,
                           double *d_rhs,
                           unsigned int *d_n_literals,
                           unsigned int *d_literals,
                           const unsigned int maxn_literals,
                           unsigned int *d_req_n_literals
                           )
    {
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_req_n_literals;
    __shared__ unsigned int s_req_n_inequality;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0) && threadIdx.x == 0;
    unsigned int n_groups = blockDim.y;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    #ifdef UNION_EVAL
    typename jit::union_params_t *s_params = (typename jit::union_params_t *)(&s_data[0]);
    Scalar4 *s_orientation_group_old = (Scalar4*)(s_params + num_types);
    #else
    Scalar4 *s_orientation_group_old = (Scalar4*)(&s_data[0]);
    #endif

    Scalar4 *s_orientation_group_new = (Scalar4*)(s_orientation_group_old + n_groups);
    Scalar3 *s_pos_group_old = (Scalar3*)(s_orientation_group_new + n_groups);
    Scalar3 *s_pos_group_new = (Scalar3*)(s_pos_group_old + n_groups);
    Scalar *s_diameter_group = (Scalar *)(s_pos_group_new + n_groups);
    Scalar *s_charge_group = (Scalar *)(s_diameter_group + n_groups);
    Scalar *s_additive_cutoff = (Scalar *)(s_charge_group + n_groups);
    float *s_rhs_group = (float *)(s_additive_cutoff + num_types);
    unsigned int *s_queue_j =   (unsigned int*)(s_rhs_group + n_groups);
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int *s_idx_group = (unsigned int*)(s_type_group + n_groups);

        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

        #ifdef UNION_EVAL
        unsigned int param_size = num_types*sizeof(typename jit::union_params_t) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)jit::d_union_params)[cur_offset + tidx];
                }
            }
        #endif

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_additive_cutoff[cur_offset + tidx] = d_additive_cutoff[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    #ifdef UNION_EVAL
    // initialize extra shared mem
    char *s_extra = (char *)(s_idx_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();
    #endif

    if (master && group == 0)
        {
        s_queue_size = 0;
        s_still_searching = 1;
        s_req_n_inequality = 0;
        s_req_n_literals = 0;
        }

    bool active = true;
    unsigned int idx = blockIdx.x*n_groups+group;
    if (idx >= nwork)
        active = false;
    idx += work_offset;

    __syncthreads();

    unsigned int my_cell;

    unsigned int update_order_i;
    unsigned int begin_i;
    if (active)
        {
        // load particle i
        Scalar4 postype_i_old(d_postype[idx]);
        Scalar4 postype_i_new(d_trial_postype[idx]);
        unsigned int type_i = __scalar_as_int(postype_i_old.w);

        // find the cell this particle should be in
        vec3<Scalar> pos_i_old(postype_i_old);
        my_cell = computeParticleCell(vec_to_scalar3(pos_i_old), box, ghost_width, cell_dim, ci, false);

        // load order in update sequence
        update_order_i = d_update_order_by_ptl[idx];

        if (master)
            {
            s_pos_group_old[group] = make_scalar3(postype_i_old.x, postype_i_old.y, postype_i_old.z);
            s_pos_group_new[group] = make_scalar3(postype_i_new.x, postype_i_new.y, postype_i_new.z);
            s_type_group[group] = type_i;
            s_orientation_group_old[group] = d_orientation[idx];
            s_orientation_group_new[group] = d_trial_orientation[idx];
            s_diameter_group[group] = d_diameter[idx];
            s_charge_group[group] = d_charge[idx];

            s_idx_group[group] = idx;
            begin_i = d_n_inequality[idx];
            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::HPMCMonoPatch, seed, idx, select, timestep);
            s_rhs_group[group] = log(hoomd::detail::generate_canonical<double>(rng_i));
            }
        }

     // sync so that s_postype_group and s_orientation are available before other threads might process energy evaluations
     __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;

    // true if we are checking against the old configuration
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        }

    // loop while still searching

    // only the first thread of every group of eval_threads threads adds to the queue
    active &= threadIdx.x == 0;

    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if ((k >> 2) < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k >> 2, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && (k >> 2) < excell_size)
                {
                bool old_i = k & 1;
                bool old_j = k & 2;

                // build shape i from shared memory
                vec3<Scalar> pos_i(old_i ? s_pos_group_old[group] : s_pos_group_new[group]);
                unsigned int type_i = s_type_group[group];
                // prefetch next j
                j = next_j;
                k += group_size;
                if ((k >> 2) < excell_size)
                    {
                    next_j = __ldg(&d_excell_idx[excli(k >> 2, my_cell)]);
                    }

                // has j been updated? ghost particles are not updated
                bool j_before_i = j < N_local &&
                    d_update_order_by_ptl[j] < update_order_i &&
                    d_trial_move_type[j] &&
                    !d_reject_out_of_cell[j];

                if (!old_j && !j_before_i)
                    continue; // no dependency

                // check particle circumspheres
                // load particle j (always load ghosts from particle data)
                const Scalar4 postype_j = (old_j || j >= N_local) ? d_postype[j] : d_trial_postype[j];
                unsigned int type_j = __scalar_as_int(postype_j.w);
                vec3<Scalar> pos_j(postype_j);

                // place ourselves into the minimum image
                vec3<Scalar> r_ij = pos_j - pos_i;
                r_ij = box.minImage(r_ij);

                OverlapReal rcut = r_cut_patch + Scalar(0.5)*(s_additive_cutoff[type_i] + s_additive_cutoff[type_j]);
                OverlapReal rsq = dot(r_ij,r_ij);

                if (idx != j && (old_j || j < N_local)
                    && (rsq <= rcut*rcut))
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                    if (insert_point < max_queue_size)
                        {
                        s_queue_gid[insert_point] = (group << 1) | (old_i ? 1 : 0);
                        s_queue_j[insert_point] = (j << 2) | (old_j ? 2 : 0) | (j_before_i ? 1 : 0);
                        }
                    else
                        {
                        // or back up if the queue is already full
                        // we will recheck and insert this on the next time through
                        k -= group_size;
                        }
                    }
                } // end while (s_queue_size < max_queue_size && (k>>2) < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process energy evaluations
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the energy evaluation to perform out of the shared mem queue
            unsigned int check_group_flag = s_queue_gid[tidx_1d];
            unsigned int check_j_flag = s_queue_j[tidx_1d];
            bool check_old_i = check_group_flag & 1;
            bool check_old_j = check_j_flag & 2;
            unsigned int check_group = check_group_flag >> 1;
            unsigned int check_j  = check_j_flag >> 2;
            bool j_before_i = check_j_flag & 1;

            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = check_old_i ? s_pos_group_old[check_group] : s_pos_group_new[check_group];
            unsigned int type_i = s_type_group[check_group];
            Scalar4 orientation_i  = check_old_i ? s_orientation_group_old[check_group] : s_orientation_group_new[check_group];
            Scalar d_i = s_diameter_group[check_group];
            Scalar q_i = s_charge_group[check_group];

            // build shape j from global memory
            Scalar4 postype_j = check_old_j ? d_postype[check_j] : d_trial_postype[check_j];
            Scalar4 orientation_j = check_old_j ? d_orientation[check_j] : d_trial_orientation[check_j];
            Scalar d_j = d_diameter[check_j];
            Scalar q_j = d_charge[check_j];
            unsigned int type_j = __scalar_as_int(postype_j.w);

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            // evaluate energy
            #ifdef UNION_EVAL
            float energy = jit::eval_union(s_params, r_ij, type_i, quat<float>(orientation_i), d_i, q_i, type_j, quat<float>(orientation_j), d_j, q_j);

            // sum up energy from logical warp
            energy = hoomd::detail::WarpReduce<float, eval_threads>().Sum(energy);
            #else
            float energy = 0.0f;
            if (threadIdx.x == 0)
                {
                // no parallelism in user function
                energy = eval(r_ij, type_i, quat<float>(orientation_i), d_i, q_i, type_j, quat<float>(orientation_j), d_j, q_j);
                }
            #endif

            if (threadIdx.x == 0)
                {
                if (!j_before_i)
                    {
                    // add to RHS
                    atomicAdd(&s_rhs_group[check_group], !check_old_i ? energy : -energy);
                    }
                else if (check_j < N_local)
                    {
                    unsigned int i = s_idx_group[check_group];
                    unsigned int n = atomicAdd(&d_n_inequality[i], 1);

                    if (n >= maxn_inequality)
                        atomicMax(&s_req_n_inequality, n + 1);
                    else
                        {
                        if ((energy > 0 && !check_old_i) || (energy < 0 && check_old_i))
                            {
                            d_inequality_literals[i*maxn_inequality+n] = (check_j << 1) | check_old_j;
                            d_coeff[i*maxn_inequality+n] = check_old_i  ? -energy : energy;
                            atomicAdd(&s_rhs_group[check_group], check_old_i ? -energy : energy);
                            }
                        else
                            {
                            d_inequality_literals[i*maxn_inequality+n] = (check_j << 1) | !check_old_j;
                            d_coeff[i*maxn_inequality+n] = check_old_i ? energy : -energy;
                            }
                        }
                    }
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && (k >> 2) < excell_size)
            atomicAdd(&s_still_searching, 1);

        __syncthreads();
        } // end while (s_still_searching)

    if (active && master)
        {
        // complete the inequality for this particle
        unsigned int n = d_n_inequality[idx];

        if (n == begin_i)
            {
            // no dependencies
            if (s_rhs_group[group] > 0)
                {
                // add unit clause for x_i
                unsigned int n = atomicAdd(&d_n_literals[idx], 2);

                if (n + 1 >= maxn_literals)
                    atomicMax(&s_req_n_literals, n + 2);
                else
                    {
                    d_literals[idx*maxn_literals+n] = idx << 1; // x_i
                    d_literals[idx*maxn_literals+n+1] = SAT_sentinel;
                    }
                }
            }
        else
            {
            if (n + 1 >= maxn_inequality)
                atomicMax(&s_req_n_inequality, n + 2);
            else
                {
                if (s_rhs_group[group] > 0)
                    {
                    d_n_inequality[idx] += 2;
                    d_rhs[idx*maxn_inequality + begin_i] = s_rhs_group[group];
                    d_coeff[idx*maxn_inequality+n] = s_rhs_group[group];
                    d_inequality_literals[idx*maxn_inequality+n] = idx << 1; // x_i
                    d_inequality_literals[idx*maxn_inequality+n+1] = SAT_sentinel;
                    }
                else
                    {
                    // tautology, undo terms
                    d_n_inequality[idx] = begin_i;
                    }
                }
            }
        }

    __syncthreads();

    if (group == 0 && master)
        {
        if (s_req_n_inequality > maxn_inequality)
            atomicMax(d_req_n_inequality, s_req_n_inequality);

        if (s_req_n_literals > maxn_literals)
            atomicMax(d_req_n_literals, s_req_n_literals);
        }
    }
} // end namespace kernel

} // end namespace gpu

} // end namespace hpmc

