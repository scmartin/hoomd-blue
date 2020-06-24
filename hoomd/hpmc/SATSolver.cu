#include "SATSolver.cuh"

#include <hip/hip_runtime.h>

#include <cub/cub.cuh>
#include "IntegratorHPMCMonoGPUTypes.cuh"

#include "hoomd/extern/ECL.cuh"

namespace hpmc {

namespace gpu {

namespace kernel {

// insert a newly watched variable into the active ring if necessary
__device__ inline void update_active_ring(
    const unsigned int v,
    const unsigned int *d_assignment,
    const unsigned int *d_watch,
    const unsigned int *d_watch_inequality,
    unsigned int *d_next,
    unsigned int &t,
    unsigned int &h)
    {
    if (d_assignment[v] == SAT_sentinel &&
        d_watch[v << 1] == SAT_sentinel &&
        d_watch[(v << 1) | 1] == SAT_sentinel &&
        d_watch_inequality[v << 1] == SAT_sentinel &&
        d_watch_inequality[(v << 1) | 1] == SAT_sentinel)
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
    }

/*! Update the watch sum for the (linear) inequalities

    Input: set of inequalities with literals sorted by their non-negative
           coefficients from large to small, and a watch list
    Output: the updated watch list such that the invariant

        S_W >= b

    is maintained, where S_W is the sum of coefficients of watched literals.

    This implements a variant of the lazy data structures for Pseudo-Boolean constraints
    discussed in Chai and Kuehlmann, "A Fast Pseudo-Boolean Constraint Solver",
    http://dx.doi.org/10.1145/775832.776041,
    and Sheini and Sakallah, "Pueblo: A Hybrid Pseudo-Boolean SAT Solver",
    http://dx.doi.org/10.3233/SAT190020

    \param false_literal The literal that is set to false
    \param d_watch_inequality Points to the first inequality watching a literal
    \param d_next_inequality The next pointer in a literal's watch list
    \param d_inequality_literals Stores the clauses (literals) of the inequalities
    \param d_inequality_begin Points to the inequality's first literal for every literal
    \param d_coeff Linear coefficients of the inequality
    \param d_rhs Right hand sides, one per inequality
    \param d_is_watching A boolean field indicating for every literal if it is being watched
    \param d_watch_sum The sum of coefficients of non-false literals being watched
    \param d_assignment Partial assignment of variables to truth values
    \param d_next Points to the next member of the active ring
    \param h Current head of active ring
    \param t Current tail of active ring
 */
__device__ inline bool update_watch_sum(
    const unsigned int false_literal,
    unsigned int *d_watch_inequality,
    const unsigned int *d_watch,
    unsigned int *d_next_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_inequality_begin,
    const double *d_coeff,
    const double *d_rhs,
    unsigned int *d_is_watching,
    double *d_watch_sum,
    const unsigned int *d_assignment,
    unsigned int *d_next,
    unsigned int &h,
    unsigned int &t)
    {
    // traverse the watch list for the false literal
    unsigned int q = d_watch_inequality[false_literal];

    // false_literal is no longer being watched
    d_watch_inequality[false_literal] = SAT_sentinel;

    while (q != SAT_sentinel)
        {
        // for every literal in an inequality, a pointer to the start of that inequality
        unsigned int c = d_inequality_begin[q];

        double rhs = d_rhs[c];

        // update the watch sum
        double watch_sum = d_watch_sum[c];

        // unwatch the literal inside the clause
        d_is_watching[q] = 0;
        watch_sum -= d_coeff[q];

        unsigned int j = c;
        bool found_alternative = false;
        while (watch_sum < rhs)
            {
            unsigned int l = d_inequality_literals[j++];

            if (l == SAT_sentinel)
                break; // end of inequality

            unsigned int v = l >> 1;
            unsigned int a = l & 1;

            if ((d_assignment[v] == SAT_sentinel || d_assignment[v] == a ^ 1) && !d_is_watching[j])
                {
                watch_sum += d_coeff[j];

                // add this inequality to the watch list for l
                found_alternative = true;
                d_is_watching[j] = 1;

                // add variable to the active ring if it was not being watched before
                update_active_ring(v, d_assignment, d_watch, d_watch_inequality, d_next, t, h);

                // add ours to the list of inequalites watching the literal
                d_next_inequality[j] = d_watch_inequality[l];
                d_watch_inequality[l] = j;
                }
            }

        if (!found_alternative)
            return false; // should never get here

        d_watch_sum[c] = watch_sum;

        // follow the watch list to the next inequality
        q = d_next_inequality[q];
        }

    return true;
    }

/*! 1. Clear the watch list for a literal
    2. Make the clause watching it watch an alternative literal
    3. If a new literal is put on a watch list that was not being watched before,
       add it to the active ring

    See also D. Knuth, The Art of Computer Programming, vol4 fascicle 6 (Satisfiability),
        exercise 130

    \param false_literal The literal being updated
    \param d_watch Points to the first clause in a literal's watch list
    \param d_next_clause Links clauses in the watch list
    \param d_literals Stores the literals of the CNF
    \param d_assignment Partial assignment of variables
    \param d_next Pointer to the next variable in the active ring
    \param h Head of active ring (the list unassigned variables being watched)
    \param t Tail of active ring
 */
__device__ inline bool update_watchlist(
    const unsigned int false_literal,
    unsigned int *d_watch,
    const unsigned int *d_watch_inequality,
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
                update_active_ring(v, d_assignment, d_watch, d_watch_inequality, d_next, t, h);

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

// Returns true if the watched literal is forced true by an inequality and a partial assignment
__device__ inline bool is_unit_inequality(
    const unsigned int literal,
    const unsigned int *d_watch_inequality,
    const unsigned int *d_inequality_begin,
    const unsigned int *d_next_inequality,
    const double *d_rhs,
    const double *d_coeff,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_assignment
    )
    {
    // traverse the watch list for the literal
    unsigned int q = d_watch_inequality[literal];

    while (q != SAT_sentinel)
        {
        unsigned int c = d_inequality_begin[q];

        // compute the slack of this inequality
        double s = -d_rhs[c];

        unsigned int j = c;
        while (true)
            {
            unsigned int l = d_inequality_literals[j++];

            if (l == SAT_sentinel)
                break; // end of inequality

            unsigned int v = l >> 1;
            unsigned int a = l & 1;

            if (d_assignment[v] == SAT_sentinel || d_assignment[v] == a ^ 1)
                s += d_coeff[j];
            }

        if (s - d_coeff[q] < 0)
            {
            // this literal is implied true
            return true;
            }

        q = d_next_inequality[q];
        }

    return false;
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

            // if there is a different literal that is either unassigned or true,
            // this clause can not be a unit clause
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

/* A DPLL SAT Solver with lazy data structures,
   implementing Algorithm D of Knuth, TACOP v4f6

   with generalization to Pseudo-Boolean constraints
 */
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
    const unsigned int *d_component_size,
    const unsigned int n_variables,
    unsigned int *d_heap,
    unsigned int *d_watch_inequality,
    unsigned int *d_next_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_inequality_begin,
    unsigned int *d_is_watching,
    double *d_watch_sum,
    const double *d_coeff,
    const double *d_rhs)
    {
    unsigned int node_idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (node_idx >= n_variables)
        return;

    // start from the representatives of every component, all other threads just exit
    if (d_representative[node_idx] != node_idx)
        return;

    unsigned int component = d_representative[node_idx];
    unsigned int h = d_head[component];

    // chase pointers until we find a tail for the ring buffer
    unsigned int v = h;
    unsigned int t = SAT_sentinel;
    while (v != SAT_sentinel)
        {
        t = v;
        v = d_next[v];
        }
    if (node_idx == component && t != SAT_sentinel)
        d_next[t] = h;

    // allocate scratch memory for this component
    unsigned int n = d_component_size[node_idx];
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
            is_h_unit |= is_unit_inequality(h << 1,
                                            d_watch_inequality,
                                            d_inequality_begin,
                                            d_next_inequality,
                                            d_rhs,
                                            d_coeff,
                                            d_inequality_literals,
                                            d_assignment);

            bool is_neg_h_unit = is_unit((h << 1) | 1,
                                     d_watch,
                                     d_next_clause,
                                     d_literals,
                                     d_assignment);
            is_neg_h_unit |= is_unit_inequality((h << 1) | 1,
                                            d_watch_inequality,
                                            d_inequality_begin,
                                            d_next_inequality,
                                            d_rhs,
                                            d_coeff,
                                            d_inequality_literals,
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
                if (d_watch[k << 1] != SAT_sentinel || d_watch[(k << 1) | 1] != SAT_sentinel ||
                    d_watch_inequality[k << 1] != SAT_sentinel || d_watch_inequality[(k << 1) | 1] != SAT_sentinel)
                    {
                    d_next[k] = h;
                    h = k;
                    d_next[t] = h;
                    }

                d--;
                }

            if (d == component_start)
                {
                // can't backtrack further, no solution (should not get here)
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
        bool b = (d_state[d-1] + 1) & 1;
        d_assignment[k] = b;

        // Boolean formula
        update_watchlist((k << 1) | b,
                         d_watch,
                         d_watch_inequality,
                         d_next_clause,
                         d_literals,
                         d_assignment,
                         d_next,
                         h,
                         t);

        // inequalities
        update_watch_sum((k << 1) | b,
            d_watch_inequality,
            d_watch,
            d_next_inequality,
            d_inequality_literals,
            d_inequality_begin,
            d_coeff,
            d_rhs,
            d_is_watching,
            d_watch_sum,
            d_assignment,
            d_next,
            h,
            t);
        }
    }

//! Sort values by key using bitonic sort
__device__ inline void bitonic_sort_descending(
    double *keys,
    unsigned int *values,
    unsigned int n)
    {
    unsigned int i,j,k;

    // next power of two
    unsigned int n2 = 1;
    while (n2 < n) n2 >>=1;

    for (k=2; k <= n2; k=2*k)
        {
        for (j = k >> 1; j > 0; j = j >> 1)
            {
            for (i = 0; i < n2; i++)
                {
                unsigned int ij = i ^ j;
                if (ij > i && i < n && ij < n)
                    {
                    if ((i & k) == 0 && keys[i] < keys[ij] ||
                        (i & k) != 0 && keys[i] > keys[ij])
                        {
                        // swap
                        double t = keys[i];
                        keys[i] = keys[ij];
                        keys[ij] = t;

                        unsigned int v = values[i];
                        values[i] = values[ij];
                        values[ij] = v;
                        }
                    }
                }
            }
        }
    }

/* Preprocess the list of inequalities.

   Input: A list of inequalities (literals and coefficients), with arbitrary real coefficients
   Output: Transformed inequalities, s.t. all coefficients are non-negative and sorted from large to small

   Uses a bitonic sort algorithm to sort the literals by their coefficients
*/
__global__ void preprocess_inequalities(
    unsigned int n_variables,
    const unsigned int maxn_inequality,
    unsigned int *d_inequality_literals,
    const unsigned int *d_n_inequality,
    double *d_coeff,
    double *d_rhs)
    {
    unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    if (tidx >= n_variables)
        return;

    unsigned int nlit = d_n_inequality[tidx];
    unsigned int first_idx = tidx*maxn_inequality;

    // make all coefficients nonnegative
    for (unsigned int i = 0; i < nlit; ++i)
        {
        unsigned int q = tidx*maxn_inequality+i;
        unsigned int l = d_inequality_literals[q];

        if (l == SAT_sentinel)
            {
            first_idx = q + 1;
            continue;
            }

        if (d_coeff[q] < 0)
            {
            unsigned int v = l >> 1;
            unsigned int a = l & 1;
            d_inequality_literals[q] = (v << 1) | a ^ 1;
            d_rhs[first_idx] += fabs(d_coeff[q]);
            }
        }

    // sort literals in every inequality by their coefficients
    for (unsigned int i = 0; i < nlit; ++i)
        {
        unsigned int q = tidx*maxn_inequality+i;
        unsigned int l = d_inequality_literals[q];

        if (l == SAT_sentinel)
            {
            // end of inequality, sort the preceding sequence
            bitonic_sort_descending(d_coeff + first_idx,
                         d_inequality_literals + first_idx,
                         q - first_idx);

            first_idx = q+1;
            continue;
            }
        }
    }

/* Input:
      1. CNF for Boolean variables
      2. Pseudo-Boolean constraints with literals sorted by their coefficients from large to small

         Literals must have non-negative real coefficients

   Output:
     initialized watch lists for Boolean and inequality constraints
 */
__global__ void setup_watch_list(
    unsigned int n_variables,
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    const unsigned int maxn_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_n_inequality,
    const double *d_coeff,
    const double *d_rhs,
    unsigned int *d_inequality_begin,
    unsigned int *d_is_watching,
    unsigned int *d_watch_inequality,
    unsigned int *d_next_inequality,
    double *d_watch_sum
    )
    {
    unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (tidx >= n_variables)
        return;

    // go through the literals associated with this variable and pick the first literal of every clause
    unsigned int nlit = d_n_literals[tidx];

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

    // add all variables in the inequality associated with this variable to the watch list
    nlit = d_n_inequality[tidx];

    bool adding_watches = true;
    unsigned int first_idx = tidx*maxn_inequality;
    double watch_sum = 0.0;
    double rhs = d_rhs[first_idx];

    for (unsigned int i = 0; i < nlit; ++i)
        {
        unsigned int q = tidx*maxn_inequality+i;
        unsigned int l = d_inequality_literals[q];

        if (l == SAT_sentinel)
            {
            adding_watches = true;
            first_idx = q+1;
            watch_sum = 0.0;
            continue;
            }

        // initialize pointer
        d_inequality_begin[q] = first_idx;

        d_is_watching[q] = adding_watches;

        if (adding_watches)
            {
            // add to the sum of watched literals
            watch_sum += d_coeff[q];

            // append the literal in the inequality to the singly linked list for this literal
            unsigned int p = atomicCAS(&d_watch_inequality[l], SAT_sentinel, q);
            while (p != SAT_sentinel)
                {
                p = atomicCAS(&d_next_inequality[p], SAT_sentinel, q);
                }

            if (watch_sum >= rhs)
                {
                adding_watches = false;
                d_watch_sum[first_idx] = watch_sum;
                }
            }
        }
    }

// Initialize the active list for every component.
__global__ void initialize_components(
    const unsigned int *d_watch,
    const unsigned int *d_watch_inequality,
    unsigned int *d_assignment,
    const unsigned int *d_component_ptr,
    const unsigned int n_variables,
    unsigned int *d_representative,
    unsigned int *d_component_size,
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

    // count the number of members of this component
    atomicAdd(&d_component_size[component], 1);

    // assign a sentinel value to the variable for this node
    d_assignment[node_idx] = SAT_sentinel;

    if (d_watch[node_idx << 1] != SAT_sentinel || d_watch[(node_idx << 1) | 1] != SAT_sentinel ||
        d_watch_inequality[node_idx << 1] != SAT_sentinel || d_watch_inequality[(node_idx << 1) | 1] != SAT_sentinel)
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
    const unsigned int *d_n_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int maxn_inequality,
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
        // merge all literals in each clause asssociated with this variable,
        // generating 2(n-1) edges per clause
        unsigned int prev = literal > 0 ? d_literals[tidx*maxn_literals + literal - 1] : SAT_sentinel;
        unsigned int l = d_literals[tidx*maxn_literals+literal];

        if (prev != SAT_sentinel && l != SAT_sentinel)
            {
            unsigned int v = l >> 1;
            unsigned int w = prev >> 1;

            // add undirected edge (need to allocate sufficient storage before launching this kernel)
            unsigned int k = atomicAdd(&d_n_columns[v], 1);
            d_colidx[v*max_n_columns+k] = w;

            k = atomicAdd(&d_n_columns[w], 1);
            d_colidx[w*max_n_columns+k] = v;
            }
        }

    // inequalities (maximum one per variable)
    nlit = d_n_inequality[tidx];

    if (literal < nlit)
        {
        // merge all literals in each linear inequality asssociated with this variable
        unsigned int prev = literal > 0 ?
            d_inequality_literals[tidx*maxn_inequality + literal - 1] : SAT_sentinel;
        unsigned int l = d_inequality_literals[tidx*maxn_inequality+literal];

        if (prev != SAT_sentinel && l != SAT_sentinel)
            {
            unsigned int v = l >> 1;
            unsigned int w = prev >> 1;

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
    unsigned int *d_watch_inequality,
    unsigned int *d_next_inequality,
    const unsigned int *d_n_literals,
    const unsigned int maxn_literals,
    const unsigned int *d_n_inequality,
    const unsigned int maxn_inequality,
    unsigned int *d_heap,
    unsigned int *d_component_size)
    {
    unsigned int tidx = threadIdx.x+blockDim.x*blockIdx.x;
    unsigned int literal = threadIdx.y+blockDim.y*blockIdx.y;

    if (tidx >= n_variables)
        return;

    if (tidx == 0 && literal == 0)
        {
        *d_heap = 0;
        }

    if (literal == 0)
        {
        d_head[tidx] = SAT_sentinel;
        d_next[tidx] = SAT_sentinel;
        d_watch[tidx << 1] = d_watch[(tidx << 1) | 1] = SAT_sentinel;
        d_watch_inequality[tidx << 1] = d_watch_inequality[(tidx << 1) | 1] = SAT_sentinel;
        d_component_size[tidx] = 0;
        }

    if (literal < d_n_literals[tidx])
        d_next_clause[tidx*maxn_literals+literal] = SAT_sentinel;

    if (literal < d_n_inequality[tidx])
        d_next_inequality[tidx*maxn_inequality+literal] = SAT_sentinel;
    }

} //end namespace kernel

void identify_connected_components(
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    const unsigned int *d_n_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int maxn_inequality,
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
    unsigned int max_n_columns = 2*maxn_literals + 2*maxn_inequality;
    unsigned int maxn = max(maxn_literals, maxn_inequality);
    dim3 grid(n_variables*literals_per_block/run_block_size + 1, maxn/literals_per_block + 1, 1);
    dim3 block(run_block_size/literals_per_block, literals_per_block, 1);
    hipLaunchKernelGGL(kernel::find_dependencies, grid, block, 0, 0,
        n_variables,
        d_n_literals,
        d_literals,
        maxn_literals,
        d_n_inequality,
        d_inequality_literals,
        maxn_inequality,
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

void preprocess_inequalities(
    unsigned int n_variables,
    const unsigned int maxn_inequality,
    unsigned int *d_inequality_literals,
    const unsigned int *d_n_inequality,
    double *d_coeff,
    double *d_rhs,
    unsigned int block_size)
    {
    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::preprocess_inequalities));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(max_block_size, block_size);
    dim3 grid(n_variables/run_block_size + 1, 1, 1);
    dim3 block(run_block_size, 1, 1);
    hipLaunchKernelGGL(kernel::preprocess_inequalities, grid, block, 0, 0,
        n_variables,
        maxn_inequality,
        d_inequality_literals,
        d_n_inequality,
        d_coeff,
        d_rhs);
    }

// initialie the memory for SAT
void initialize_sat_mem(
    unsigned int *d_watch,
    unsigned int *d_next_clause,
    unsigned int *d_head,
    unsigned int *d_next,
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int *d_assignment,
    const unsigned int n_variables,
    const unsigned int *d_component_ptr,
    unsigned int *d_representative,
    unsigned int *d_component_size,
    unsigned int *d_heap,
    const unsigned int maxn_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_n_inequality,
    const double *d_coeff,
    const double *d_rhs,
    unsigned int *d_inequality_begin,
    unsigned int *d_is_watching,
    unsigned int *d_watch_inequality,
    unsigned int *d_next_inequality,
    double *d_watch_sum,
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
    unsigned int maxn = max(maxn_literals, maxn_inequality);
    dim3 grid(n_variables*literals_per_block/run_block_size + 1, maxn/literals_per_block + 1, 1);
    dim3 block(run_block_size/literals_per_block, literals_per_block, 1);
    hipLaunchKernelGGL(kernel::reset_mem, grid, block, 0, 0,
        n_variables,
        d_head,
        d_next,
        d_watch,
        d_next_clause,
        d_watch_inequality,
        d_next_inequality,
        d_n_literals,
        maxn_literals,
        d_n_inequality,
        maxn_inequality,
        d_heap,
        d_component_size);

    // watch all first literals in each clause, and the minimum
    // number of literals in each inequality satisfying the invariant
    hipLaunchKernelGGL(kernel::setup_watch_list, n_variables/block_size + 1, block_size, 0, 0,
        n_variables,
        maxn_literals,
        d_literals,
        d_n_literals,
        d_watch,
        d_next_clause,
        maxn_inequality,
        d_inequality_literals,
        d_n_inequality,
        d_coeff,
        d_rhs,
        d_inequality_begin,
        d_is_watching,
        d_watch_inequality,
        d_next_inequality,
        d_watch_sum);

    // setup active rings for the solver
    unsigned int sat_block_size = 256;
    hipLaunchKernelGGL(kernel::initialize_components, n_variables/sat_block_size + 1, sat_block_size, 0, 0,
        d_watch,
        d_watch_inequality,
        d_assignment,
        d_component_ptr,
        n_variables,
        d_representative,
        d_component_size,
        d_head,
        d_next);
    }

// solve the Boolean formula plus inequalities
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
    const unsigned int *d_component_ptr,
    const unsigned int *d_representative,
    const unsigned int *d_component_size,
    unsigned int *d_heap,
    unsigned int *d_watch_inequality,
    unsigned int *d_next_inequality,
    const unsigned int *d_inequality_literals,
    const unsigned int *d_inequality_begin,
    unsigned int *d_is_watching,
    double *d_watch_sum,
    const double *d_coeff,
    const double *d_rhs,
    const unsigned int block_size)
    {
    hipLaunchKernelGGL(kernel::solve_sat, n_variables/block_size + 1, block_size, 0, 0,
        d_watch,
        d_next_clause,
        d_next,
        d_h,
        d_head,
        d_literals,
        d_assignment,
        d_state,
        d_representative,
        d_component_size,
        n_variables,
        d_heap,
        d_watch_inequality,
        d_next_inequality,
        d_inequality_literals,
        d_inequality_begin,
        d_is_watching,
        d_watch_sum,
        d_coeff,
        d_rhs);
    }

} //end namespace gpu
} //end namespace hpm
