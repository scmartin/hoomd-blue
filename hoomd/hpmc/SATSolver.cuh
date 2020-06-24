#pragma once

#include "hoomd/CachedAllocator.h"

namespace hpmc {

namespace gpu {

void preprocess_inequalities(
    unsigned int n_variables,
    const unsigned int maxn_inequality,
    unsigned int *d_inequality_literals,
    const unsigned int *d_n_inequality,
    double *d_coeff,
    double *d_rhs,
    unsigned int block_size);

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
    CachedAllocator &alloc);

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
    unsigned int literals_per_block);

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
    const unsigned int block_size);

} //end namespace gpu
} //end namespace hpm
