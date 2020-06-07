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
    unsigned int *d_component_begin,
    const hipDeviceProp_t devprop,
    const unsigned int block_size,
    CachedAllocator& alloc);

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
    const unsigned int block_size);

} //end namespace gpu
} //end namespace hpm
