#include "SATSolver.cuh"

#include <hip/hip_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "hoomd/extern/ECL.cuh"

#ifdef __HIP_PLATFORM_NVCC__
#define check_cusparse(a) \
    {\
    cusparseStatus_t status = (a);\
    if ((int)status != CUSPARSE_STATUS_SUCCESS)\
        {\
        printf("cusparse ERROR %d in file %s line %d\n",status,__FILE__,__LINE__);\
        throw std::runtime_error("Error during clusters update");\
        }\
    }
#endif

#ifdef __HIP_PLATFORM_NVCC__
#include <cusparse.h>
#endif

namespace hpmc {

namespace gpu {

namespace kernel {

__device__ inline bool update_watchlist(
    const unsigned int false_literal,
    const unsigned int maxn_watch,
    unsigned int *d_req_n_watch,
    unsigned int *d_watch,
    unsigned int *d_n_watch,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    const unsigned int *d_assignment,
    bool &reallocate)
    {
    // in the first pass, see if we can successfully swap literals in all clauses watching this literal
    unsigned int n_watch = d_n_watch[false_literal];

    for (unsigned int i = 0; i < n_watch; ++i)
        {
        unsigned int c = d_watch[false_literal*maxn_watch+i];
        bool found_alternative = false;
        unsigned int n_clause = d_n_clause[c];

        for (unsigned int j = 0; j < n_clause; ++j)
            {
            unsigned int alternative = d_clause[c*maxn_clause+j];
            unsigned int v = alternative >> 1;
            unsigned int a = alternative & 1;
            if (d_assignment[v] == 0xffffffff || d_assignment[v] == a ^ 1)
                {
                found_alternative = true;
                break;
                }
            }

        if (! found_alternative)
            return false;
        }

    // when we're here, we know that all clauses can be updated to watch a different literal
    for (unsigned int i = 0; i < n_watch; ++i)
        {
        unsigned int c = d_watch[false_literal*maxn_watch+i];
        unsigned int n_clause = d_n_clause[c];

        for (unsigned int j = 0; j < n_clause; ++j)
            {
            unsigned int alternative = d_clause[c*maxn_clause+j];
            unsigned int v = alternative >> 1;
            unsigned int a = alternative & 1;
            if (d_assignment[v] == 0xffffffff || d_assignment[v] == a ^ 1)
                {
                // add this clause to the watch list for the alternative
                unsigned int n = d_n_watch[alternative]++;
                if (n < maxn_watch)
                    d_watch[alternative*maxn_watch+n] = c;
                else
                    {
                    atomicMax(d_req_n_watch, n + 1);
                    reallocate = true;
                    return false;
                    }

                break;
                }
            }
        }

    // now there cannot be any clauses watching false_literal anymore (as it just was assigned true)
    d_n_watch[false_literal] = 0;

    return true;
    }

__global__ void solve_sat(
    const unsigned int maxn_watch,
    unsigned int *d_watch,
    unsigned int *d_n_watch,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_assignment,
    unsigned int *d_state,
    const unsigned int *d_variables,
    const unsigned int n_components,
    unsigned int *d_unsat,
    unsigned int *d_req_n_watch,
    const unsigned int *d_component_begin,
    const unsigned int *d_component_end)
    {
    unsigned int component = threadIdx.x + blockIdx.x*blockDim.x;

    if (component >= n_components)
        return;

    unsigned int component_start = d_component_begin[component];
    unsigned int component_end = d_component_end[component];

    unsigned int d = component_start; // start with the first variable in this component

    bool reallocate = false;

    while (true)
        {
        if (d == component_end)
            return;

        bool tried_something = false;

        const unsigned int var = d_variables[d];

        for (int a = 1; a >= 0; --a)
            {
            if (((d_state[var] >> a) & 1) == 0)
                {
                tried_something = true;

                // set the bit indicating a has been tried for d
                d_state[var] |= 1 << a;
                d_assignment[var] = a;

                if (!update_watchlist((var << 1) | a,
                                      maxn_watch,
                                      d_req_n_watch,
                                      d_watch,
                                      d_n_watch,
                                      maxn_clause,
                                      d_clause,
                                      d_n_clause,
                                      d_assignment,
                                      reallocate))
                    {
                    d_assignment[var] = 0xffffffff;
                    }
                else
                    {
                    // move on to the next variable
                    d++;
                    break;
                    }

                if (reallocate)
                    return;
                }
            }

        if (!tried_something)
            {
            if (d == component_start)
                {
                // can't backtrack further, no solutions
                atomicAdd(d_unsat, 1);
                return;
                }
            else
                {
                // backtrack
                d_state[var] = 0;
                d_assignment[var] = 0xffffffff;
                d--;
                }
            }
        }
    }

__global__ void setup_watch_list(
    unsigned int n_clauses,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    const unsigned int maxn_watch,
    unsigned int *d_watch,
    unsigned int *d_n_watch)
    {
    unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (tidx >= n_clauses)
        return;

    // ignore empty clauses
    if (d_n_clause[tidx] == 0)
        return;

    unsigned int first_literal = d_clause[tidx*maxn_clause];
    unsigned int n = atomicAdd(&d_n_watch[first_literal],1);
    d_watch[first_literal*maxn_watch+n] = tidx;
    }

__global__ void find_dependencies(
    const unsigned int n_clauses,
    const unsigned int *d_n_clause,
    const unsigned int *d_clause,
    const unsigned int maxn_clause,
    unsigned int *d_n_elem,
    unsigned int *d_rowidx,
    unsigned int *d_colidx,
    const unsigned int max_n_elem)
    {
    const unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    if (tidx >= n_clauses)
        return;

    unsigned int nclause = d_n_clause[tidx];
    for (unsigned int i = 0; i < nclause; ++i)
        {
        unsigned int l = d_clause[tidx*maxn_clause+i];
        unsigned int v = l >> 1;

        for (unsigned int j = 0; j < nclause; ++j)
            {
            if (j == i)
                continue;

            unsigned int m = d_clause[tidx*maxn_clause+j];
            unsigned int w = m >> 1;

            // add dependency
            unsigned int k = atomicAdd(d_n_elem, 1);
            if (k < max_n_elem)
                {
                d_rowidx[k] = v;
                d_colidx[k] = w;
                }
            }
        }
    }

} //end namespace kernel


void find_connected_components(
    const unsigned int n_clauses,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_n_elem,
    const unsigned int max_n_elem,
    unsigned int *d_rowidx,
    unsigned int *d_colidx,
    unsigned int *d_csr_row_ptr,
    const unsigned int n_variables,
    unsigned int *d_phi,
    unsigned int *d_components,
    unsigned int &n_components,
    unsigned int *d_work,
    unsigned int *d_unique_components,
    unsigned int *d_component_begin,
    unsigned int *d_component_end,
    const hipDeviceProp_t devprop,
    const unsigned int block_size,
    CachedAllocator &alloc)
    {
    hipMemsetAsync(d_n_elem, 0, sizeof(unsigned int));

    // fill the connnectivity matrix
    hipLaunchKernelGGL(kernel::find_dependencies, n_clauses/block_size + 1, block_size, 0, 0,
        n_clauses,
        d_n_clause,
        d_clause,
        maxn_clause,
        d_n_elem,
        d_rowidx,
        d_colidx,
        max_n_elem);

    // construct a CSR matrix
    unsigned int nnz;
    hipMemcpy(&nnz, d_n_elem, sizeof(unsigned int), hipMemcpyDeviceToHost);

    if (nnz > max_n_elem)
        return;

    // COO -> CSR
    thrust::device_ptr<unsigned int> rowidx(d_rowidx);
    thrust::device_ptr<unsigned int> colidx(d_colidx);

    // throw out duplicates
    auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(rowidx,colidx));
    thrust::sort(
        zip_it,
        zip_it + nnz);
    auto new_end = thrust::unique(thrust::cuda::par(alloc),
                                  zip_it,
                                  zip_it + nnz);
    nnz = new_end - zip_it;

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    check_cusparse(cusparseXcoo2csr(handle,
        (const int *) d_rowidx,
        nnz,
        n_variables,
        (int *) d_csr_row_ptr,
        CUSPARSE_INDEX_BASE_ZERO));
    cusparseDestroy(handle);

    // find connected components
    ecl_connected_components(
        n_variables,
        nnz,
        (const int *) d_csr_row_ptr,
        (const int *) d_colidx,
        (int *) d_components,
        (int *) d_work,
        devprop);

    // put first member of every component into phi
    thrust::device_ptr<unsigned int> phi(d_phi);
    thrust::device_ptr<unsigned int> components(d_components);
    thrust::counting_iterator<unsigned int> variables_begin(0);
    thrust::copy(variables_begin,
                 variables_begin + n_variables,
                 phi);

    thrust::sort_by_key(
        components,
        components + n_variables,
        phi);

    thrust::device_ptr<unsigned int> unique_components(d_unique_components);
    auto it = thrust::reduce_by_key(
        thrust::cuda::par(alloc),
        components,
        components + n_variables,
        thrust::constant_iterator<unsigned int>(1),
        unique_components,
        thrust::make_discard_iterator());

    n_components = it.first - unique_components;

    // find start and end for every component
    thrust::device_ptr<unsigned int> component_begin(d_component_begin);
    thrust::device_ptr<unsigned int> component_end(d_component_end);
    thrust::lower_bound(components,
        components + n_variables,
        unique_components,
        unique_components + n_components,
        component_begin);
    thrust::upper_bound(components,
        components + n_variables,
        unique_components,
        unique_components + n_components,
        component_end);
    }

// solve the satisfiability problem
void solve_sat(const unsigned int maxn_watch,
    unsigned int *d_watch,
    unsigned int *d_n_watch,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_assignment,
    unsigned int *d_state,
    const unsigned int n_variables,
    const unsigned int n_clauses,
    unsigned int *d_unsat,
    unsigned int *d_req_n_watch,
    const unsigned int *d_phi,
    unsigned int n_components,
    const unsigned int *d_component_begin,
    const unsigned int *d_component_end,
    const unsigned int block_size)
    {
    hipMemsetAsync(d_unsat, 0, sizeof(unsigned int));
    hipMemsetAsync(d_state, 0, sizeof(unsigned int)*n_variables);
    hipMemsetAsync(d_n_watch, 0, sizeof(unsigned int)*2*n_variables);
    hipMemsetAsync(d_assignment, 0xff, sizeof(unsigned int)*n_variables);

    hipLaunchKernelGGL(kernel::setup_watch_list, n_clauses/block_size + 1, block_size, 0, 0,
        n_clauses,
        maxn_clause,
        d_clause,
        d_n_clause,
        maxn_watch,
        d_watch,
        d_n_watch);

    unsigned int sat_block_size = 256;
    hipLaunchKernelGGL(kernel::solve_sat, n_components/sat_block_size + 1, sat_block_size, 0, 0,
        maxn_watch,
        d_watch,
        d_n_watch,
        maxn_clause,
        d_clause,
        d_n_clause,
        d_assignment,
        d_state,
        d_phi,
        n_components,
        d_unsat,
        d_req_n_watch,
        d_component_begin,
        d_component_end);
    }

} //end namespace gpu
} //end namespace hpm

#undef check_cusparse
