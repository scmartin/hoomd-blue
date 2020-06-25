// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUTypes.cuh"
#include "hoomd/GPUPartition.cuh"

namespace hpmc
{
namespace gpu
{
namespace kernel
{

//! Kernel to generate expanded cells
/*! \param d_excell_idx Output array to list the particle indices in the expanded cells
    \param d_excell_size Output array to list the number of particles in each expanded cell
    \param excli Indexer for the expanded cells
    \param d_cell_idx Particle indices in the normal cells
    \param d_cell_size Number of particles in each cell
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer
    \param cli Cell list indexer
    \param cadji Cell adjacency indexer
    \param ngpu Number of active devices

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D ci,
                            const Index2D cli,
                            const Index2D cadji,
                            const unsigned int ngpu)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    my_cell = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];

        // iterate over per-device cell lists
        for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
            {
            unsigned int neigh_cell_size = d_cell_size[neigh_cell+igpu*ci.getNumElements()];

            for (unsigned int k = 0; k < neigh_cell_size; k++)
                {
                // read in the index of the new particle to add to our cell
                unsigned int new_idx = d_cell_idx[cli(k, neigh_cell)+igpu*cli.getNumElements()];
                d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
                my_cell_size++;
                }
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
    }

//! Kernel for grid shift
/*! \param d_postype postype of each particle
    \param d_image Image flags for each particle
    \param N number of particles
    \param box Simulation box
    \param shift Vector by which to translate the particles

    Shift all the particles by a given vector.

    \ingroup hpmc_kernels
*/
__global__ void hpmc_shift(Scalar4 *d_postype,
                          int3 *d_image,
                          const unsigned int N,
                          const BoxDim box,
                          const Scalar3 shift)
    {
    // identify the active cell that this thread handles
    unsigned int my_pidx = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread is inactive if it indexes past the end of the particle list
    if (my_pidx >= N)
        return;

    // pull in the current position
    Scalar4 postype = d_postype[my_pidx];

    // shift the position
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    pos += shift;

    // wrap the particle back into the box
    int3 image = d_image[my_pidx];
    box.wrap(pos, image);

    // write out the new position and orientation
    d_postype[my_pidx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    d_image[my_pidx] = image;
    }

/*! Add implications to the CNF that ensure that a variable is set to zero (accept move),
    if it is not otherwise constrained
 */
__global__ void complete_cnf(
    const unsigned int n_variables,
    unsigned int *d_literals,
    unsigned int *d_n_literals,
    const unsigned int maxn_literals,
    unsigned int *d_req_n_inequality,
    unsigned int *d_inequality_literals,
    unsigned int *d_n_inequality,
    const unsigned int maxn_inequality,
    double *d_coeff,
    double *d_rhs)
    {
    unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (tidx >= n_variables)
        return;

    // we assume that the clauses in row i are all implications for when variable x_i must be true,
    // therefore we can generate the opposite implication for !x_i by negating the left hand side (except
    // the variable itself)

    unsigned int nlit = min(maxn_literals,d_n_literals[tidx]);

    // count the number of new terms
    unsigned int n_new = 0;
    bool have_unit_clause = false;
    bool is_unit_clause = true;
    for (unsigned int i = 0; i < nlit; ++i)
        {
        unsigned int l = d_literals[tidx*maxn_literals+i];

        if (l == SAT_sentinel)
            {
            have_unit_clause |= is_unit_clause;
            is_unit_clause = true;
            continue;
            }

        if (l != (tidx << 1)) // != x_i ?
            {
            n_new++;
            is_unit_clause = false;
            }
        }

    if (have_unit_clause)
        return;

    unsigned int nlit_inequality = d_n_inequality[tidx];

    if (nlit_inequality + n_new + 2 > maxn_inequality)
        atomicMax(d_req_n_inequality, nlit_inequality + n_new + 2);
    else
        {
        // add the new clause as a Pseudo-Boolean constraint
        d_n_inequality[tidx] += n_new + 2; // account for !x_j and marker terms

        unsigned int n = nlit_inequality;
        d_inequality_literals[tidx*maxn_inequality+n] = (tidx << 1) | 1; // !x_i
        d_coeff[tidx*maxn_inequality+n] = 1.0;
        d_rhs[tidx*maxn_inequality+n] = 1.0;
        n++;

        for (unsigned int i = 0; i < nlit; ++i)
            {
            unsigned int l = d_literals[tidx*maxn_literals+i];

            if (l == SAT_sentinel)
                {
                continue;
                }

            // add the negated literal to the clause
            unsigned int v = l >> 1;
            unsigned int b = l & 1;

            if (v != tidx)
                {
                d_inequality_literals[tidx*maxn_inequality+n] = (v << 1) | (b ^ 1);
                d_coeff[tidx*maxn_inequality+n] = 1.0;
                n++;
                }
            }

        d_inequality_literals[tidx*maxn_inequality+n] = SAT_sentinel;
        }
    }

} // end namespace kernel

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_excell));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid(ci.getNumElements() / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_excell, dim3(grid), dim3(threads), 0, 0, d_excell_idx,
                                           d_excell_size,
                                           excli,
                                           d_cell_idx,
                                           d_cell_size,
                                           d_cell_adj,
                                           ci,
                                           cli,
                                           cadji,
                                           ngpu);

    }

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size)
    {
    assert(d_postype);
    assert(d_image);

    // setup the grid to run the kernel
    dim3 threads_shift(block_size, 1, 1);
    dim3 grid_shift(N / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_shift, dim3(grid_shift), dim3(threads_shift), 0, 0, d_postype,
                                                      d_image,
                                                      N,
                                                      box,
                                                      shift);

    // after this kernel we return control of cuda managed memory to the host
    hipDeviceSynchronize();
    }

void complete_cnf(
    const unsigned int n_variables,
    unsigned int *d_literals,
    unsigned int *d_n_literals,
    const unsigned int maxn_literals,
    unsigned int *d_req_n_inequality,
    unsigned int *d_inequality_literals,
    unsigned int *d_n_inequality,
    const unsigned int maxn_inequality,
    double *d_coeff,
    double *d_rhs)
    {
    unsigned int block_size = 256;

    hipLaunchKernelGGL(kernel::complete_cnf, n_variables/block_size + 1, block_size, 0, 0,
        n_variables,
        d_literals,
        d_n_literals,
        maxn_literals,
        d_req_n_inequality,
        d_inequality_literals,
        d_n_inequality,
        maxn_inequality,
        d_coeff,
        d_rhs);
    }

} // end namespace gpu
} // end namespace hpmc

