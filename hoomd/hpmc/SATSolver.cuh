#pragma once

#include "hoomd/CachedAllocator.h"

namespace hpmc {

namespace gpu {

void identify_connected_components(
    const unsigned int maxn_literals,
    const unsigned int *d_literals,
    const unsigned int *d_n_literals,
    unsigned int &req_n_columns,
    unsigned int *d_req_n_columns,
    const unsigned int max_n_columns,
    unsigned int *d_n_columns,
    unsigned int *d_colidx_table,
    unsigned int *d_compact_indices,
    unsigned int *d_colidx,
    unsigned int *d_csr_row_ptr,
    const unsigned int n_variables,
    unsigned int *d_component_ptr,
    unsigned int *d_work,
    const hipDeviceProp_t devprop,
    const unsigned int block_size,
    CachedAllocator &alloc);

void solve_sat(unsigned int *d_watch,
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
    const unsigned int block_size);

} //end namespace gpu
} //end namespace hpm
