#include "SATSolver.cuh"

#include <hip/hip_runtime.h>

#include <cub/cub.cuh>
#include "IntegratorHPMCMonoGPUTypes.cuh"

#include "hoomd/extern/ECL.cuh"

namespace hpmc {

namespace gpu {

namespace kernel {

__device__ inline bool update_watchlist(
    const unsigned int false_literal,
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    const unsigned int *d_literals,
    const unsigned int *d_assignment,
    unsigned int *d_next,
    unsigned int &h,
    unsigned int &t)
    {
    unsigned int c = d_watch[false_literal];

    // false_literal is no longer being watched
    d_watch[false_literal] = SAT_sentinel;

    // update the clauses watching it to a different watched literal
    while (c != SAT_sentinel)
        {
        unsigned int next = d_next_clause[c];

        bool found_alternative = false;
        unsigned int j = c;
        while (true)
            {
            unsigned int alternative = d_literals[j++];
            if (alternative == SAT_sentinel)
                break; // end of clause

            unsigned int v = alternative >> 1;
            unsigned int a = alternative & 1;
            if (d_assignment[v] == SAT_sentinel || d_assignment[v] == a ^ 1)
                {
                found_alternative = true;

                // the variable corresponding to 'alternative' might become active at this point,
                // because it might not be watched anywhere else. In such a case, we insert it at the
                // 'beginning' of the active ring (that is, just after t)
                if (d_assignment[v] == SAT_sentinel && d_watch[v << 1] == SAT_sentinel && d_watch[(v << 1) | 1] == SAT_sentinel)
                    {
                    if (t == SAT_sentinel)
                        {
                        t = h = v;
                        d_next[t] = h;
                        }
                    else
                        {
                        d_next[v] = h;
                        h = v;
                        d_next[t] = h;
                        }
                    }

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
    const unsigned int *d_literals,
    const unsigned int *d_assignment)
    {
    unsigned int c = d_watch[literal];

    while (c != SAT_sentinel)
        {
        bool unit_clause = true;
        unsigned int j = c;
        while (true)
            {
            unsigned int l = d_literals[j++];

            if (l == SAT_sentinel)
                break; // end of clause

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
    const unsigned int *d_literals,
    unsigned int *d_assignment,
    unsigned int *d_state,
    const unsigned int *d_representative,
    const unsigned int n_variables,
    unsigned int *d_unsat,
    unsigned int *d_heap)
    {
    unsigned int node_idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (node_idx >= n_variables)
        return;

    // start from the representatives of every component, all other threads just exit
    if (d_representative[node_idx] != node_idx)
        {
        return;
        }

    unsigned int h = d_head[node_idx];

    // chase pointers until we find a tail for the ring buffer
    unsigned int v = h;
    unsigned int n = 0;
    unsigned int t = SAT_sentinel;
    while (v != SAT_sentinel)
        {
        t = v;
        v = d_next[v];
        n++; // the size of the component
        }
    if (t != SAT_sentinel)
        d_next[t] = h;

    // allocate scratch memory for this component
    unsigned int component_start = atomicAdd(d_heap, n);
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
                                     d_literals,
                                     d_assignment);
            bool is_neg_h_unit = is_unit((h << 1) | 1,
                                     d_watch,
                                     d_next_clause,
                                     d_literals,
                                     d_assignment);

            unsigned int f = is_h_unit + (is_neg_h_unit << 1);

            if (f == 1 || f == 2)
                {
                // one of the two literals is true
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
            d_h[d++] = k = h;

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

            while (d > component_start && d_state[d-1] >= 2)
                {
                k = d_h[d-1];
                d_assignment[k] = SAT_sentinel;
                if (d_watch[k << 1] != SAT_sentinel || d_watch[(k << 1) | 1] != SAT_sentinel)
                    {
                    d_next[k] = h;
                    h = k;
                    d_next[t] = h;
                    }

                d--;
                }

            if (d == component_start)
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
                         d_literals,
                         d_assignment,
                         d_next,
                         h,
                         t);
        }
    }

__global__ void setup_watch_list(
    unsigned int n_variables,
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int *d_watch,
    unsigned int *d_next_clause)
    {
    unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (tidx >= n_variables)
        return;

    // go through the literals associated with this variable and pick the first literal of every clause
    unsigned nlit = d_n_literals[tidx];

    bool first = true;
    for (unsigned int i = 0; i < nlit; ++i)
        {
        unsigned int l = d_literals[tidx*maxn_literals+i];

        if (l == SAT_sentinel)
            {
            first = true;
            continue;
            }

        if (first)
            {
            // append the clause to the singly linked list for this literal
            unsigned int c = tidx*maxn_literals+i;
            unsigned int p = atomicCAS(&d_watch[l], SAT_sentinel, c);
            while (p != SAT_sentinel)
                {
                p = atomicCAS(&d_next_clause[p], SAT_sentinel, c);
                }
            }

        first = false;
        }
    }

// Initialize the active list for every component.
__global__ void initialize_components(
    unsigned int *d_watch,
    unsigned int *d_assignment,
    const unsigned int *d_component_ptr,
    const unsigned int n_variables,
    unsigned int *d_representative,
    unsigned int *d_head,
    unsigned int *d_next)
    {
    unsigned int node_idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (node_idx >= n_variables)
        return;

    // jump to the node with the lowest index in this component, which is its label
    unsigned int next, vstat = d_component_ptr[node_idx];
    while (vstat > (next = d_component_ptr[vstat]))
        {
        vstat = next;
        }
    unsigned int component = vstat;

    // store the reprentative for this node's component in global mem
    d_representative[node_idx] = component;

    // assign a sentinel value to the variable for this node
    d_assignment[node_idx] = SAT_sentinel;

    if (d_watch[node_idx << 1] != SAT_sentinel || d_watch[(node_idx << 1) | 1] != SAT_sentinel)
        {
        // append ourselves to the linked list
        unsigned int p = atomicCAS(&d_head[component], SAT_sentinel, node_idx);
        while (p != SAT_sentinel)
            {
            p = atomicCAS(&d_next[p], SAT_sentinel, node_idx);
            }
        }
    }

__global__ void find_dependencies(
    const unsigned int n_variables,
    const unsigned int *d_n_literals,
    const unsigned int *d_literals,
    const unsigned int maxn_literals,
    unsigned int *d_n_columns,
    unsigned int *d_colidx,
    const unsigned int max_n_columns)
    {
    const unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    if (tidx >= n_variables)
        return;

    const unsigned int literal = threadIdx.y + blockIdx.y*blockDim.y;

    unsigned int nlit = d_n_literals[tidx];

    if (literal < nlit)
        {
        // merge all elements in each clause asssociated with this variable, generating 2(n-1) edges per clause
        unsigned int prev = literal > 0 ? d_literals[tidx*maxn_literals + literal - 1] : SAT_sentinel;
        unsigned int l = d_literals[tidx*maxn_literals+literal];

        if (prev != SAT_sentinel && l != SAT_sentinel)
            {
            unsigned int v = l >> 1;
            unsigned int w = prev >> 1;

            // add undirected edge (need to make sure to have sufficient storage)
            unsigned int k = atomicAdd(&d_n_columns[v], 1);
            d_colidx[v*max_n_columns+k] = w;

            k = atomicAdd(&d_n_columns[w], 1);
            d_colidx[w*max_n_columns+k] = v;
            }
        }
    }

__global__ void scatter_rows(
    const unsigned int n_variables,
    const unsigned int maxn_columns,
    const unsigned int *d_n_columns,
    const unsigned int *d_csr_row_ptr,
    const unsigned int *d_colidx_table,
    unsigned int *d_colidx)
    {
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;

    if (tidx >= n_variables)
        return;

    unsigned int v = d_csr_row_ptr[tidx];
    unsigned int start = tidx*maxn_columns;
    unsigned int n = d_n_columns[tidx];

    // this is how we do the segmented scan
    for (unsigned int i = 0; i < n; ++i)
        d_colidx[v+i] = d_colidx_table[start+i];
    }

__global__ void reset_mem(
    const unsigned int n_variables,
    unsigned int *d_head,
    unsigned int *d_next,
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    const unsigned int *d_n_literals,
    const unsigned int maxn_literals,
    unsigned int *d_unsat,
    unsigned int *d_heap)
    {
    unsigned int tidx = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int literal = threadIdx.y+blockDim.y*blockIdx.y;

    if (tidx >= n_variables)
        return;

    if (tidx == 0 && literal == 0)
        {
        *d_unsat = 0;
        *d_heap = 0;
        }

    if (literal == 0)
        {
        d_head[tidx] = SAT_sentinel;
        d_next[tidx] = SAT_sentinel;
        d_watch[tidx << 1] = d_watch[(tidx << 1) | 1] = SAT_sentinel;
        }

    if (literal < d_n_literals[tidx])
        d_next_clause[tidx*maxn_literals+literal] = SAT_sentinel;
    }


} //end namespace kernel

void identify_connected_components(
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int *d_n_columns,
    unsigned int *d_colidx_table,
    unsigned int *d_colidx,
    unsigned int *d_csr_row_ptr,
    const unsigned int n_variables,
    unsigned int *d_component_ptr,
    unsigned int *d_work,
    const hipDeviceProp_t devprop,
    const unsigned int block_size,
    unsigned int literals_per_block,
    CachedAllocator &alloc)
    {
    hipMemsetAsync(d_n_columns, 0, sizeof(unsigned int)*(n_variables+1));

    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::find_dependencies));
        max_block_size = attr.maxThreadsPerBlock;
        }
    unsigned int run_block_size = min(max_block_size, block_size);
    while (run_block_size % literals_per_block)
        literals_per_block--;

    // fill the connnectivity matrix
    unsigned int max_n_columns = 2*maxn_literals;
    dim3 grid(n_variables*literals_per_block/run_block_size + 1, maxn_literals/literals_per_block + 1, 1);
    dim3 block(run_block_size/literals_per_block, literals_per_block, 1);
    hipLaunchKernelGGL(kernel::find_dependencies, grid, block, 0, 0,
        n_variables,
        d_n_literals,
        d_literals,
        maxn_literals,
        d_n_columns,
        d_colidx_table,
        max_n_columns);

    // construct a CSR matrix
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_n_columns,
        d_csr_row_ptr,
        n_variables+1);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_n_columns,
        d_csr_row_ptr,
        n_variables+1);
    alloc.deallocate((char *)d_temp_storage);

    // move the 2d table into a contiguous array
    hipLaunchKernelGGL(kernel::scatter_rows, n_variables/block_size + 1, block_size, 0, 0,
        n_variables,
        max_n_columns,
        d_n_columns,
        d_csr_row_ptr,
        d_colidx_table,
        d_colidx);

    // find connected components
    ecl_connected_components(
        n_variables,
        (const int *) d_csr_row_ptr,
        (const int *) d_colidx,
        (int *) d_component_ptr,
        (int *) d_work,
        devprop,
        false);
    }

// solve the satisfiability problem
void solve_sat(
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    unsigned int *d_head,
    unsigned int *d_next,
    unsigned int *d_h,
    unsigned int *d_state,
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int *d_assignment,
    const unsigned int n_variables,
    unsigned int *d_unsat,
    const unsigned int *d_component_ptr,
    unsigned int *d_representative,
    unsigned int *d_heap,
    const unsigned int block_size,
    unsigned int literals_per_block)
    {
    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::reset_mem));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // initialize memory
    unsigned int run_block_size = min(max_block_size, block_size);
    while (run_block_size % literals_per_block)
        literals_per_block--;
    dim3 grid(n_variables*literals_per_block/run_block_size + 1, maxn_literals/literals_per_block + 1, 1);
    dim3 block(run_block_size/literals_per_block, literals_per_block, 1);
    hipLaunchKernelGGL(kernel::reset_mem, grid, block, 0, 0,
        n_variables,
        d_head,
        d_next,
        d_watch,
        d_next_clause,
        d_n_literals,
        maxn_literals,
        d_unsat,
        d_heap);

    // watch all first literals in each clause
    hipLaunchKernelGGL(kernel::setup_watch_list, n_variables/block_size + 1, block_size, 0, 0,
        n_variables,
        maxn_literals,
        d_literals,
        d_n_literals,
        d_watch,
        d_next_clause);

    // setup active rings for the solver
    unsigned int sat_block_size = 256;
    hipLaunchKernelGGL(kernel::initialize_components, n_variables/sat_block_size + 1, sat_block_size, 0, 0,
        d_watch,
        d_assignment,
        d_component_ptr,
        n_variables,
        d_representative,
        d_head,
        d_next);

    // solve the system of Boolean equations
    hipLaunchKernelGGL(kernel::solve_sat, n_variables/sat_block_size + 1, sat_block_size, 0, 0,
        d_watch,
        d_next_clause,
        d_next,
        d_h,
        d_head,
        d_literals,
        d_assignment,
        d_state,
        d_representative,
        n_variables,
        d_unsat,
        d_heap);
    }

} //end namespace gpu
} //end namespace hpm

#undef check_cusparse
