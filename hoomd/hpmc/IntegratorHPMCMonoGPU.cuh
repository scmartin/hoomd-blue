// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/CachedAllocator.h"

#include "hoomd/hpmc/HPMCCounters.h"

#include "GPUHelpers.cuh"
#include "HPMCMiscFunctions.h"

// base data types
#include "IntegratorHPMCMonoGPUTypes.cuh"

#include <cassert>

namespace hpmc {

namespace gpu {

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 32
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

#ifdef __HIPCC__
namespace kernel
{

//! Check narrow-phase overlaps
template< class Shape, unsigned int max_threads >
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
__global__ void hpmc_narrow_phase(const Scalar4 *d_postype,
                           const Scalar4 *d_orientation,
                           const Scalar4 *d_trial_postype,
                           const Scalar4 *d_trial_orientation,
                           const unsigned int *d_trial_move_type,
                           const unsigned int *d_excell_idx,
                           const unsigned int *d_excell_size,
                           const Index2D excli,
                           hpmc_counters_t *d_counters,
                           const unsigned int num_types,
                           const BoxDim box,
                           const Scalar3 ghost_width,
                           const uint3 cell_dim,
                           const Index3D ci,
                           const unsigned int N_local,
                           const unsigned int *d_check_overlaps,
                           const Index2D overlap_idx,
                           const typename Shape::param_type *d_params,
                           const unsigned int *d_update_order_by_ptl,
                           const unsigned int *d_reject_in,
                           unsigned int *d_reject_out,
                           const unsigned int *d_reject_out_of_cell,
                           const unsigned int max_extra_bytes,
                           const unsigned int max_queue_size,
                           const unsigned int work_offset,
                           const unsigned int nwork,
                           unsigned int *d_num_clauses,
                           const unsigned int max_num_clauses,
                           unsigned int *d_clause,
                           unsigned int *d_n_clause,
                           const unsigned int max_n_clause,
                           unsigned int *d_req_n_literals
                           )
    {
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_max_num_literals;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0) && threadIdx.x == 0;
    unsigned int n_groups = blockDim.y;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int *s_idx_group = (unsigned int *)(s_type_group + n_groups);
    unsigned int *s_reject_group = (unsigned int *)(s_idx_group + n_groups);

        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_reject_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);
    __syncthreads();

    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        s_max_num_literals = 0;
        }

    if (master)
        s_reject_group[group] = 0;

    bool valid_ptl = true;
    unsigned int idx = blockIdx.x*n_groups+group;
    if (idx >= nwork)
        valid_ptl = false;
    idx += work_offset;

    // if this particle is rejected a priori because it has left the cell, don't check overlaps
    // and avoid out of range memory access when computing the cell
    bool active = valid_ptl;
    if (active && d_reject_out_of_cell[idx])
        {
        if (master)
            s_reject_group[group] = 1;

        active = false;
        }

    unsigned int my_cell;

    unsigned int overlap_checks = 0;
    unsigned int overlap_err_count = 0;

    unsigned int update_order_i;
    if (active)
        {
        Scalar4 postype_i(d_trial_postype[idx]);
        vec3<Scalar> pos_i(postype_i);
        unsigned int type_i = __scalar_as_int(postype_i.w);

        // find the cell this particle should be in
        vec3<Scalar> pos_i_old(d_postype[idx]);
        my_cell = computeParticleCell(vec_to_scalar3(pos_i_old), box, ghost_width,
            cell_dim, ci, false);

        // the order of this particle in the chain
        update_order_i = d_update_order_by_ptl[idx];

        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = type_i;
            s_orientation_group[group] = d_trial_orientation[idx];
            s_idx_group[group] = idx;
            }
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;

    // true if we are checking against the old configuration
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += 2*excell_size;
        }

    // loop while still searching

    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active && threadIdx.x == 0 && !s_reject_group[group])
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < 2*excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found

            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && k < 2*excell_size)
                {
                // build some shapes, but we only need them to get diameters, so don't load orientations
                // build shape i from shared memory
                vec3<Scalar> pos_i(s_pos_group[group]);
                Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                bool old = k & 1;

                // prefetch next j
                j = next_j;
                k += group_size;
                if (k < 2*excell_size)
                    {
                    next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                    }

                // has j been updated? ghost particles are not updated
                bool j_before_i = j < N_local &&
                    d_update_order_by_ptl[j] < update_order_i &&
                    d_trial_move_type[j] &&
                    !d_reject_out_of_cell[j];

                if (!old && !j_before_i)
                    continue; // no dependency

                // check particle circumspheres

                // load particle j (always load ghosts from particle data)
                const Scalar4 postype_j = (old || j >= N_local) ? d_postype[j] : d_trial_postype[j];
                unsigned int type_j = __scalar_as_int(postype_j.w);
                vec3<Scalar> pos_j(postype_j);
                Shape shape_j(quat<Scalar>(), s_params[type_j]);

                // place ourselves into the minimum image
                vec3<Scalar> r_ij = pos_j - pos_i;
                r_ij = box.minImage(r_ij);

                if (idx != j && (old || j < N_local)
                    && check_circumsphere_overlap(r_ij, shape_i, shape_j))
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                    if (insert_point < max_queue_size)
                        {
                        s_queue_gid[insert_point] = (group << 1) | j_before_i;
                        s_queue_j[insert_point] = (j << 1) | old;
                        }
                    else
                        {
                        // or back up if the queue is already full
                        // we will recheck and insert this on the next time through
                        k -= group_size;
                        }
                    }
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group_flag = s_queue_gid[tidx_1d];
            unsigned int check_group = check_group_flag >> 1;
            bool j_before_i = check_group_flag & 1;
            unsigned int check_j_flag = s_queue_j[tidx_1d];
            bool check_old = check_j_flag & 1;
            unsigned int check_j  = check_j_flag >> 1;

            Scalar4 postype_j;
            Scalar4 orientation_j;
            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = check_old ? quat<Scalar>(d_orientation[check_j]) : quat<Scalar>(d_trial_orientation[check_j]);

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)]
                && test_overlap(r_ij, shape_i, shape_j, overlap_err_count))
                {
                if (!j_before_i)
                    {
                    // the decision for j does not affect i
                    atomicAdd(&s_reject_group[check_group],1);
                    }
                else
                    {
                    unsigned int i = s_idx_group[check_group];
                    if (i < max_num_clauses && check_j < N_local)
                        {
                        // add the literal x_j to the ith clause, if j is not a ghost
                        unsigned int n = atomicAdd_system(&d_n_clause[i], 1);

                        if (n < max_n_clause)
                            d_clause[i*max_n_clause + n] = (check_j << 1) | !check_old;
                        else
                            atomicMax(&s_max_num_literals, n + 1);
                        }

                    // add a clause with x_i v !x_j (j only if it is not a ghost)
                    unsigned int m = atomicAdd_system(d_num_clauses, 1);
                    if (N_local + m < max_num_clauses)
                        {
                        unsigned int n = atomicAdd(&d_n_clause[N_local+m], 1);
                        if (n < max_n_clause)
                            d_clause[(m+N_local)*max_n_clause+n] = i << 1;

                        if (check_j < N_local)
                            {
                            n = atomicAdd_system(&d_n_clause[N_local+m], 1);
                            if (n < max_n_clause)
                                d_clause[(m+N_local)*max_n_clause+n] = (check_j << 1) | check_old;
                            }

                        if (n >= max_n_clause)
                            atomicMax(&s_max_num_literals, n + 1);
                        }
                    }
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && (threadIdx.x == 0) && !s_reject_group[group] && k < 2*excell_size)
            atomicAdd(&s_still_searching, 1);

        __syncthreads();
        } // end while (s_still_searching)

    __syncthreads();

    if (valid_ptl && master && idx < max_num_clauses)
        {
        // the ith clause contains !x_i (or x_i, if rejected) and all literals that lead to x_i evaluating to true
        unsigned int n = atomicAdd_system(&d_n_clause[idx], 1);
        if (n < max_n_clause)
            d_clause[idx*max_n_clause + n] = (idx << 1) | !s_reject_group[group];
        else
            atomicMax(&s_max_num_literals, n + 1);
        }

    if (master && group == 0)
        {
        unsigned int req_n_literals = s_max_num_literals;

        if (s_max_num_literals > max_n_clause)
            atomicMax(d_req_n_literals, req_n_literals);
        }

    if (master)
        {
        atomicAdd(&s_overlap_checks, overlap_checks);
        atomicAdd(&s_overlap_err_count, overlap_err_count);
        }

    __syncthreads();

    if (master && group == 0)
        {
        // write out counters to global memory
        #if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd_system(&d_counters->overlap_checks, s_overlap_checks);
        #else
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        #endif
        }
    }

//! Launcher for narrow phase kernel with templated launch bounds
template< class Shape, unsigned int cur_launch_bounds >
void narrow_phase_launcher(const hpmc_args_t& args, const typename Shape::param_type *params,
    unsigned int max_threads, detail::int2type<cur_launch_bounds>)
    {
    assert(params);

    if (max_threads == cur_launch_bounds*MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_narrow_phase<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>));
            max_block_size = attr.maxThreadsPerBlock;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int overlap_threads = args.overlap_threads;
        unsigned int tpp = min(args.tpp,run_block_size);

        while (overlap_threads*tpp > run_block_size || run_block_size % (overlap_threads*tpp) != 0)
            {
            tpp--;
            }
        tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

        unsigned int n_groups = run_block_size/(tpp*overlap_threads);
        unsigned int max_queue_size = n_groups*tpp;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type)
            + args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            run_block_size -= args.devprop.warpSize;
            if (run_block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");

            tpp = min(tpp, run_block_size);
            while (overlap_threads*tpp > run_block_size || run_block_size % (overlap_threads*tpp) != 0)
                {
                tpp--;
                }
            tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

            n_groups = run_block_size/(tpp*overlap_threads);
            max_queue_size = n_groups*tpp;

            shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
                + max_queue_size * 2 * sizeof(unsigned int)
                + min_shared_bytes;
            }

        // determine dynamically allocated shared memory size
        unsigned int base_shared_bytes = shared_bytes + attr.sharedSizeBytes;
        unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
        char *ptr = (char *)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        unsigned int extra_bytes = max_extra_bytes - available_bytes;
        shared_bytes += extra_bytes;

        dim3 thread(overlap_threads, n_groups, tpp);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            unsigned int nwork = range.second - range.first;
            const unsigned int num_blocks = nwork/n_groups + 1;

            dim3 grid(num_blocks, 1, 1);

            assert(args.d_postype);
            assert(args.d_orientation);
            assert(args.d_trial_postype);
            assert(args.d_trial_orientation);
            assert(args.d_excell_idx);
            assert(args.d_excell_size);
            assert(args.d_counters);
            assert(args.d_check_overlaps);
            assert(args.d_reject_in);
            assert(args.d_reject_out);
            assert(args.d_reject_out_of_cell);

            hipLaunchKernelGGL((hpmc_narrow_phase<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>),
                grid, thread, shared_bytes, args.streams[idev],
                args.d_postype, args.d_orientation, args.d_trial_postype, args.d_trial_orientation,
                args.d_trial_move_type, args.d_excell_idx, args.d_excell_size, args.excli,
                args.d_counters+idev*args.counters_pitch, args.num_types,
                args.box, args.ghost_width, args.cell_dim, args.ci, args.N, args.d_check_overlaps,
                args.overlap_idx, params, args.d_update_order_by_ptl, args.d_reject_in, args.d_reject_out, args.d_reject_out_of_cell,
                max_extra_bytes, max_queue_size, range.first, nwork,
                args.d_num_clauses, args.max_num_clauses, args.d_clause, args.d_n_clause, args.max_n_clause,
                args.d_req_n_literals);
            }
        }
    else
        {
        narrow_phase_launcher<Shape>(args, params, max_threads, detail::int2type<cur_launch_bounds/2>());
        }
    }
} // end namespace kernel

//! Kernel driver for kernel::hpmc_narrow_phase
template< class Shape >
void hpmc_narrow_phase(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);

    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::narrow_phase_launcher<Shape>(args, params, launch_bounds, detail::int2type<MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
    }
#endif

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE

} // end namespace gpu

} // end namespace hpmc
