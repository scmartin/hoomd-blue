#pragma once

#include "hoomd/CachedAllocator.h"

namespace hpmc {

namespace gpu {

// Label connected components
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
    CachedAllocator& alloc);


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
    const unsigned int n_components,
    const unsigned int *d_component_begin,
    const unsigned int *d_component_end,
    unsigned int block_size);

} //end namespace gpu
} //end namespace hpm
