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

namespace hpmc {

namespace gpu {

const unsigned int SAT_sentinel = 0xffffffff;

namespace kernel {

__device__ inline bool update_watchlist(
    const unsigned int false_literal,
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    const unsigned int *d_n_clause,
    const unsigned int *d_clause,
    const unsigned int maxn_clause,
    const unsigned int *d_assignment)
    {
    unsigned int c = d_watch[false_literal];

    // false_literal is no longer being watched
    d_watch[false_literal] = SAT_sentinel;

    // update the clauses watching it to a different watched literal
    while (c != SAT_sentinel)
        {
        unsigned int next = d_next_clause[c];
        unsigned int n_clause = d_n_clause[c];
        bool found_alternative = false;
        for (unsigned int j = 0; j < n_clause; ++j)
            {
            unsigned int alternative = d_clause[c*maxn_clause+j];
            unsigned int v = alternative >> 1;
            unsigned int a = alternative & 1;
            if (d_assignment[v] == SAT_sentinel || d_assignment[v] == a ^ 1)
                {
                found_alternative = true;

                // insert clause at begining of alternative literal's watch list
                d_next_clause[c] = d_watch[alternative];
                d_watch[alternative] = c;
                break;
                }
            }

        if (!found_alternative)
            return false; // should never get here

        c = next;
        }

    return true;
    }

// Returns true if literal is being watched by a unit clause
__device__ inline bool is_unit(
    const unsigned int literal,
    const unsigned int *d_watch,
    const unsigned int *d_next_clause,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    const unsigned int *d_assignment)
    {
    unsigned int c = d_watch[literal];

    while (c != SAT_sentinel)
        {
        unsigned int n_clause = d_n_clause[c];

        bool unit_clause = true;
        for (unsigned int j = 0; j < n_clause; ++j)
            {
            unsigned int l = d_clause[c*maxn_clause+j];
            if (l == literal)
                continue;

            unsigned int v = l >> 1;
            unsigned int a = l & 1;

            // if there is a different literal that is either unassigned or true, this clause can not be a unit clause
            if (d_assignment[v] == SAT_sentinel || d_assignment[v] == a ^ 1)
                {
                unit_clause = false;
                break;
                }
            }

        if (unit_clause)
            return true;

        c = d_next_clause[c];
        }

    return false;
    }

__global__ void solve_sat(
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    unsigned int *d_next,
    unsigned int *d_h,
    const unsigned int *d_head,
    const unsigned int *d_tail,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_assignment,
    unsigned int *d_state,
    const unsigned int n_components,
    unsigned int *d_unsat,
    const unsigned int *d_component_begin)
    {
    unsigned int component = threadIdx.x + blockIdx.x*blockDim.x;

    if (component >= n_components)
        return;

    unsigned int component_start = d_component_begin[component];

    unsigned int h = d_head[component];
    unsigned int t = d_tail[component];
    unsigned int d = component_start;

    while (true)
        {
        if (t == SAT_sentinel)
            return; // SAT

        // fetch next variable
        unsigned int k = t;

        bool backtrack = false;
        bool unit = false;
        do
            {
            // look for unit clauses
            h = d_next[k];

            bool is_h_unit = is_unit(h << 1,
                                     d_watch,
                                     d_next_clause,
                                     maxn_clause,
                                     d_clause,
                                     d_n_clause,
                                     d_assignment);
            bool is_neg_h_unit = is_unit((h << 1) | 1,
                                     d_watch,
                                     d_next_clause,
                                     maxn_clause,
                                     d_clause,
                                     d_n_clause,
                                     d_assignment);

            unsigned int f = is_h_unit + (is_neg_h_unit << 1);

            if (f == 1 || f == 2)
                {
                // on of the two literals is true
                d_state[d] = f + 3;
                t = k;
                unit = true;
                break;
                }
            else if (f == 3)
                {
                // conflict
                backtrack = true;
                break;
                }

            k = h;
            }
        while (h != t);

        if (!backtrack && !unit)
            {
            // two way branch
            h = d_next[t];
            d_state[d] = (d_watch[h << 1] == SAT_sentinel) ||
                         (d_watch[(h << 1) | 1] != SAT_sentinel);
            }

        if (!backtrack)
            {
            // move on
            d++;
            d_h[d-1] = k = h;

            if (t == k)
                {
                t = SAT_sentinel;
                }
            else
                {
                // delete k from ring
                d_next[t] = h = d_next[k];
                }
            }
        else
            {
            t = k;

            bool done = d != component_start;
            while (!done && d_state[d-1] >= 2)
                {
                k = d_h[d-1];
                d_assignment[k] = SAT_sentinel;
                if (d_watch[k << 1] != SAT_sentinel || d_watch[(k << 1) | 1] != SAT_sentinel)
                    {
                    d_next[k] = h;
                    h = k;
                    d_next[t] = h;
                    }

                if (d == component_start)
                    done = true;
                else
                    d--;
                }

            if (done)
                {
                // can't backtrack further, no solutions
                atomicAdd(d_unsat, 1);
                return;
                }
            else
                {
                // backtrack
                d_state[d-1] = 3 - d_state[d-1];
                k = d_h[d-1];
                }
            }

        // update watches
        unsigned int b = (d_state[d-1] + 1) & 1;
        d_assignment[k] = b;
        update_watchlist((k << 1) | b,
                         d_watch,
                         d_next_clause,
                         d_n_clause,
                         d_clause,
                         maxn_clause,
                         d_assignment);

        }
    }

__global__ void setup_watch_list(
    unsigned int n_clauses,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_watch,
    unsigned int *d_next_clause)
    {
    unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (tidx >= n_clauses)
        return;

    // ignore empty clauses (Is this really necessary and shouldn't the disjunction be false then?)
    if (d_n_clause[tidx] == 0)
        return;

    unsigned int first_literal = d_clause[tidx*maxn_clause];

    // append to the singly linked list for this literal
    unsigned int p = atomicCAS(&d_watch[first_literal], SAT_sentinel, tidx);
    while (p != SAT_sentinel)
        {
        p = atomicCAS(&d_next_clause[p], SAT_sentinel, tidx);
        }
    }

__global__ void setup_active_ring(
    unsigned int *d_watch,
    unsigned int *d_assignment,
    const unsigned int *d_variables,
    const unsigned int n_components,
    const unsigned int *d_component_begin,
    unsigned int *d_head,
    unsigned int *d_tail,
    unsigned int *d_next)
    {
    unsigned int component = threadIdx.x + blockIdx.x*blockDim.x;

    if (component >= n_components)
        return;

    unsigned int component_start = d_component_begin[component];
    unsigned int component_end = d_component_begin[component+1];

    unsigned int h = SAT_sentinel;
    unsigned int t = SAT_sentinel;

    for (int d = component_end-1; d >= (int) component_start; --d)
        {
        unsigned int var = d_variables[d];
        d_assignment[var] = SAT_sentinel;
        if (d_watch[var << 1] != SAT_sentinel || d_watch[(var << 1) | 1] != SAT_sentinel)
            {
            d_next[var] = h;
            h = var;

            if (t == SAT_sentinel)
                t = var;
            }
        }

    if (t != SAT_sentinel)
        d_next[t] = h;

    d_head[component] = h;
    d_tail[component] = t;
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
    unsigned int *d_component_begin,
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
        thrust::cuda::par(alloc),
        zip_it,
        zip_it + nnz);
    auto new_end = thrust::unique(thrust::cuda::par(alloc),
                                  zip_it,
                                  zip_it + nnz);
    nnz = new_end - zip_it;

    thrust::counting_iterator<unsigned int> rows_begin(0);
    thrust::device_ptr<unsigned int> csr_row_ptr(d_csr_row_ptr);
    thrust::lower_bound(
        thrust::cuda::par(alloc),
        rowidx,
        rowidx + nnz,
        rows_begin,
        rows_begin + n_variables + 1,
        csr_row_ptr);

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
    thrust::sequence(thrust::cuda::par(alloc),
        phi,
        phi + n_variables,
        0);

    thrust::sort_by_key(
        thrust::cuda::par(alloc),
        components,
        components + n_variables,
        phi);

    // find start and end for every component
    thrust::device_ptr<unsigned int> component_begin(d_component_begin);
    auto it = thrust::reduce_by_key(
        thrust::cuda::par(alloc),
        components,
        components + n_variables,
        thrust::counting_iterator<unsigned int>(0),
        thrust::make_discard_iterator(),
        component_begin,
        thrust::equal_to<unsigned int>(),
        thrust::minimum<unsigned int>());

    n_components = it.second - component_begin;

    // set the last element
    thrust::fill(component_begin + n_components,
                 component_begin + n_components + 1,
                 n_variables);
    }

// solve the satisfiability problem
void solve_sat(unsigned int *d_watch,
    unsigned int *d_next_clause,
    unsigned int *d_head,
    unsigned int *d_tail,
    unsigned int *d_next,
    unsigned int *d_h,
    unsigned int *d_state,
    const unsigned int maxn_clause,
    const unsigned int *d_clause,
    const unsigned int *d_n_clause,
    unsigned int *d_assignment,
    const unsigned int n_variables,
    const unsigned int n_clauses,
    unsigned int *d_unsat,
    const unsigned int *d_phi,
    unsigned int n_components,
    const unsigned int *d_component_begin,
    const unsigned int block_size)
    {
    hipMemsetAsync(d_unsat, 0, sizeof(unsigned int));

    // initialize with sentinel values
    hipMemsetAsync(d_watch, 0xff, sizeof(unsigned int)*2*n_variables);
    hipMemsetAsync(d_next_clause, 0xff, sizeof(unsigned int)*n_clauses);

    hipLaunchKernelGGL(kernel::setup_watch_list, n_clauses/block_size + 1, block_size, 0, 0,
        n_clauses,
        maxn_clause,
        d_clause,
        d_n_clause,
        d_watch,
        d_next_clause);

    unsigned int sat_block_size = 256;
    hipLaunchKernelGGL(kernel::setup_active_ring, n_components/sat_block_size + 1, sat_block_size, 0, 0,
        d_watch,
        d_assignment,
        d_phi,
        n_components,
        d_component_begin,
        d_head,
        d_tail,
        d_next);

    hipLaunchKernelGGL(kernel::solve_sat, n_components/sat_block_size + 1, sat_block_size, 0, 0,
        d_watch,
        d_next_clause,
        d_next,
        d_h,
        d_head,
        d_tail,
        maxn_clause,
        d_clause,
        d_n_clause,
        d_assignment,
        d_state,
        n_components,
        d_unsat,
        d_component_begin);
    }

} //end namespace gpu
} //end namespace hpm

#undef check_cusparse
