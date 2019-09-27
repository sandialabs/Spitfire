/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

#ifndef GRIFFON_BTDDOD_MATRIX_KERNELS_H
#define GRIFFON_BTDDOD_MATRIX_KERNELS_H

namespace griffon {
namespace btddod {
void btddod_full_factorize(const double *matrix_values, const int num_blocks, const int block_size, double *out_l_values, int *out_d_pivots,
    double *out_d_factors);

void btddod_full_solve(const double *matrix_values, const double *l_values, const int *d_pivots, const double *d_factors, const double *rhs,
    const int num_blocks, const int block_size, double *out_solution);

void btddod_full_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size, double *out_matvec);

void btddod_blockdiag_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
    double *out_matvec);

void btddod_offdiag_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size, double *out_matvec);

void btddod_lowerfulltriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
    double *out_matvec);

void btddod_upperfulltriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
    double *out_matvec);

void btddod_lowerofftriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
    double *out_matvec);

void btddod_upperofftriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
    double *out_matvec);

void btddod_blockdiag_factorize(const double *matrix_values, const int num_blocks, const int block_size, int *out_pivots,
    double *out_factors);

void btddod_blockdiag_solve(const int *pivots, const double *factors, const double *rhs, const int num_blocks, const int block_size,
    double *out_solution);

void btddod_lowerfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values, const double *rhs,
    const int num_blocks, const int block_size, double *out_solution);

void btddod_upperfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values, const double *rhs,
    const int num_blocks, const int block_size, double *out_solution);

// A <- matrix_scale * A + diag_scale * block_diag
void btddod_scale_and_add_scaled_block_diagonal(double *in_out_matrix_values, const double matrix_scale, const double *block_diag,
    const double diag_scale, const int num_blocks, const int block_size);

// A <- matrix_scale * A + diag_scale * diagonal
void btddod_scale_and_add_diagonal(double *in_out_matrix_values, const double matrix_scale, const double *diagonal, const double diag_scale,
    const int num_blocks, const int block_size);

// TODO: add block row and column scaling
}
}

#endif //GRIFFON_BTDDOD_MATRIX_KERNELS_H
