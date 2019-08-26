/*
*  Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
*  You may use, distribute and modify this code under the terms of the MIT license.
*
*  You should have received a copy of the MIT license with this file.
*  If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
*/

#include "btddod_matrix_kernels.h"
#include "blas_lapack_kernels.h"

namespace griffon {
  namespace btddod {

    void btddod_full_factorize( const double *matrix_values,
                                const int num_blocks,
                                const int block_size,
                                double *out_l_values,
                                int *out_d_pivots,
                                double *out_d_factors ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;

      double d_local_tmp[nelem_block];

      double identity_matrix[nelem_block];
      for ( int i = 0; i < block_size * block_size; ++i ) {
        identity_matrix[i] = 0.;
      }
      for ( int i = 0; i < block_size; ++i ) {
        identity_matrix[i * (block_size + 1)] = 1.;
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      griffon::lapack::lu_factorize( block_size, matrix_values, out_d_pivots, out_d_factors );

      for ( int i = 1; i < num_blocks; ++i ) {
        griffon::lapack::lu_solve_on_matrix( block_size,
                                             &out_d_factors[(i - 1) * nelem_block],
                                             &out_d_pivots[(i - 1) * block_size],
                                             identity_matrix,
                                             &out_l_values[i * nelem_block] );
        const int i_base = i * block_size;
        const int im1_base = (i - 1) * block_size;
        for ( int j = 0; j < block_size; ++j ) {
          for ( int k = 0; k < block_size; ++k ) {
            out_l_values[i * nelem_block + j * block_size + k] *= sub[im1_base + k];
          }
        }

        griffon::blas::copy_vector( nelem_block, d_local_tmp, &matrix_values[i * nelem_block] );
        for ( int j = 0; j < block_size; ++j ) {
          griffon::blas::vector_plus_scaled_vector( block_size, &d_local_tmp[j * block_size], -sup[im1_base + j], &out_l_values[i * nelem_block + j * block_size] );
        }

        griffon::lapack::lu_factorize( block_size,
                                       d_local_tmp,
                                       &out_d_pivots[i * block_size],
                                       &out_d_factors[i * nelem_block] );
      }
    }

    void btddod_full_solve( const double *matrix_values,
                            const double *l_values,
                            const int *d_pivots,
                            const double *d_factors,
                            const double *rhs,
                            const int num_blocks,
                            const int block_size,
                            double *out_solution ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      double y[nelem_vector];
      double local_tmp[block_size];

      griffon::blas::copy_vector( block_size, y, rhs );
      for ( int i = 1; i < num_blocks; ++i ) {
        griffon::blas::copy_vector( block_size, &y[i * block_size], &rhs[i * block_size] );
        griffon::blas::matrix_vector_multiply( block_size, &y[i * block_size], -1., &l_values[i * nelem_block], &y[(i - 1) * block_size], 1. );
      }

      const int i = num_blocks - 1;
      griffon::lapack::lu_solve( block_size, &d_factors[i * nelem_block], &d_pivots[i * block_size], &y[i * block_size], &out_solution[i * block_size] );
      for ( int i = num_blocks - 2; i >= 0; --i ) {
        const int i_base = i * block_size;
        const int ip1_base = (i + 1) * block_size;
        for ( int j = 0; j < block_size; ++j ) {
          local_tmp[j] = -sup[i_base + j] * out_solution[ip1_base + j];
        }
        griffon::blas::vector_plus_scaled_vector( block_size, local_tmp, 1., &y[i * block_size] );
        griffon::lapack::lu_solve( block_size, &d_factors[i * nelem_block], &d_pivots[i * block_size], local_tmp, &out_solution[i * block_size] );
      }
    }

    void btddod_full_matvec( const double *matrix_values,
                             const double *vec,
                             const int num_blocks,
                             const int block_size,
                             double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      // block diagonal contribution
      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::blas::matrix_vector_multiply( block_size,
                                               &out_matvec[i * block_size],
                                               1.,
                                               &matrix_values[i * nelem_block],
                                               &vec[i * block_size],
                                               0. );
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq] + sub[im1_base + iq] * vec[im1_base + iq];
        }
      }

      const int iz1 = num_blocks - 1;
      const int i1_base = iz1 * block_size;
      const int im1_base = (iz1 - 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
      }

      const int iz2 = 0;
      const int i2_base = iz2 * block_size;
      const int ip1_base = (iz2 + 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
      }
    }

    void btddod_blockdiag_matvec( const double *matrix_values,
                                  const double *vec,
                                  const int num_blocks,
                                  const int block_size,
                                  double *out_matvec ) {
      const int nelem_block = block_size * block_size;

      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::blas::matrix_vector_multiply( block_size,
                                               &out_matvec[i * block_size],
                                               1.,
                                               &matrix_values[i * nelem_block],
                                               &vec[i * block_size],
                                               0. );
      }
    }

    void btddod_offdiag_matvec( const double *matrix_values,
                                const double *vec,
                                const int num_blocks,
                                const int block_size,
                                double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      for ( int i = 0; i < nelem_vector; ++i ) {
        out_matvec[i] = 0.;
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq] + sub[im1_base + iq] * vec[im1_base + iq];
        }
      }

      const int iz1 = num_blocks - 1;
      const int i1_base = iz1 * block_size;
      const int im1_base = (iz1 - 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
      }

      const int iz2 = 0;
      const int i2_base = iz2 * block_size;
      const int ip1_base = (iz2 + 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
      }
    }

    void btddod_lowerfulltriangle_matvec( const double *matrix_values,
                                          const double *vec,
                                          const int num_blocks,
                                          const int block_size,
                                          double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      // block diagonal contribution
      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::blas::matrix_vector_multiply( block_size,
                                               &out_matvec[i * block_size],
                                               1.,
                                               &matrix_values[i * nelem_block],
                                               &vec[i * block_size],
                                               0. );
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
        }
      }

      const int iz1 = num_blocks - 1;
      const int i1_base = iz1 * block_size;
      const int im1_base = (iz1 - 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
      }
    }

    void btddod_upperfulltriangle_matvec( const double *matrix_values,
                                          const double *vec,
                                          const int num_blocks,
                                          const int block_size,
                                          double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      // block diagonal contribution
      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::blas::matrix_vector_multiply( block_size,
                                               &out_matvec[i * block_size],
                                               1.,
                                               &matrix_values[i * nelem_block],
                                               &vec[i * block_size],
                                               0. );
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq];
        }
      }

      const int iz2 = 0;
      const int i2_base = iz2 * block_size;
      const int ip1_base = (iz2 + 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
      }
    }

    void btddod_lowerofftriangle_matvec( const double *matrix_values,
                                         const double *vec,
                                         const int num_blocks,
                                         const int block_size,
                                         double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      for ( int i = 0; i < nelem_vector; ++i ) {
        out_matvec[i] = 0.;
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
        }
      }

      const int iz1 = num_blocks - 1;
      const int i1_base = iz1 * block_size;
      const int im1_base = (iz1 - 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
      }
    }

    void btddod_upperofftriangle_matvec( const double *matrix_values,
                                         const double *vec,
                                         const int num_blocks,
                                         const int block_size,
                                         double *out_matvec ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      const int nelem_vector = num_blocks * block_size;

      for ( int i = 0; i < nelem_vector; ++i ) {
        out_matvec[i] = 0.;
      }

      const double *sub = &matrix_values[nelem_blockdiagonals];
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

      // off diagonal contributions
      for ( int iz = 1; iz < num_blocks - 1; ++iz ) {
        const int i_base = iz * block_size;
        const int im1_base = (iz - 1) * block_size;
        const int ip1_base = (iz + 1) * block_size;
        for ( int iq = 0; iq < block_size; ++iq ) {
          out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq];
        }
      }

      const int iz2 = 0;
      const int i2_base = iz2 * block_size;
      const int ip1_base = (iz2 + 1) * block_size;
      for ( int iq = 0; iq < block_size; ++iq ) {
        out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
      }
    }

    void btddod_blockdiag_factorize( const double *matrix_values,
                                     const int num_blocks,
                                     const int block_size,
                                     int *out_pivots,
                                     double *out_factors ) {
      const int nelem_block = block_size * block_size;
      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::lapack::lu_factorize( block_size,
                                       &matrix_values[i * nelem_block],
                                       &out_pivots[i * block_size],
                                       &out_factors[i * nelem_block] );
      }
    }

    void btddod_blockdiag_solve( const int *pivots,
                                 const double *factors,
                                 const double *rhs,
                                 const int num_blocks,
                                 const int block_size,
                                 double *out_solution ) {
      const int nelem_block = block_size * block_size;
      for ( int i = 0; i < num_blocks; ++i ) {
        griffon::lapack::lu_solve( block_size,
                                   &factors[i * nelem_block],
                                   &pivots[i * block_size],
                                   &rhs[i * block_size],
                                   &out_solution[i * block_size] );
      }
    }

    void btddod_lowerfulltriangle_solve( const int *pivots,
                                         const double *factors,
                                         const double *matrix_values,
                                         const double *rhs,
                                         const int num_blocks,
                                         const int block_size,
                                         double *out_solution ) {
      const int nelem_block = block_size * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      double rhsTemp[block_size];

      griffon::lapack::lu_solve( block_size,
                                 factors,
                                 pivots,
                                 rhs,
                                 out_solution );
      const double *sub = &matrix_values[nelem_blockdiagonals];
      for ( int i = 1; i < num_blocks; ++i ) {
        for ( int iq = 0; iq < block_size; ++iq ) {
          rhsTemp[iq] = rhs[i * block_size + iq] - sub[(i - 1) * block_size + iq] * out_solution[(i - 1) * block_size + iq];
        }
        griffon::lapack::lu_solve( block_size,
                                   &factors[i * nelem_block],
                                   &pivots[i * block_size],
                                   rhsTemp,
                                   &out_solution[i * block_size] );
      }
    }

    void btddod_upperfulltriangle_solve( const int *pivots,
                                         const double *factors,
                                         const double *matrix_values,
                                         const double *rhs,
                                         const int num_blocks,
                                         const int block_size,
                                         double *out_solution ) {
      const int nelem_block = block_size * block_size;
      const int nelem_offdiagonal = (num_blocks - 1) * block_size;
      const int nelem_blockdiagonals = num_blocks * nelem_block;
      double rhsTemp[block_size];

      const int i = num_blocks - 1;
      griffon::lapack::lu_solve( block_size,
                                 &factors[i * nelem_block],
                                 &pivots[i * block_size],
                                 &rhs[i * block_size],
                                 &out_solution[i * block_size] );
      const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];
      for ( int i = num_blocks - 2; i >= 0; --i ) {
        for ( int iq = 0; iq < block_size; ++iq ) {
          rhsTemp[iq] = rhs[i * block_size + iq] - sup[i * block_size + iq] * out_solution[(i + 1) * block_size + iq];
        }
        griffon::lapack::lu_solve( block_size,
                                   &factors[i * nelem_block],
                                   &pivots[i * block_size],
                                   rhsTemp,
                                   &out_solution[i * block_size] );
      }
    }

    // A <- matrix_scale * A + diag_scale * block_diag
    void btddod_scale_and_add_scaled_block_diagonal( double *in_out_matrix_values,
                                                     const double matrix_scale,
                                                     const double *block_diag,
                                                     const double diag_scale,
                                                     const int num_blocks,
                                                     const int block_size ) {
      const int nelem_block = block_size * block_size;
      const int nelem_matrix = block_size * (num_blocks * block_size + 2 * (num_blocks - 1));
      const int nelem_blockdiag = nelem_block * num_blocks;
      griffon::blas::scale_vector( nelem_matrix, in_out_matrix_values, matrix_scale );
      griffon::blas::vector_plus_scaled_vector( nelem_blockdiag, in_out_matrix_values, diag_scale, block_diag );
    }

    // A <- matrix_scale * A + diag_scale * diagonal
    void btddod_scale_and_add_diagonal( double *in_out_matrix_values,
                                        const double matrix_scale,
                                        const double *diagonal,
                                        const double diag_scale,
                                        const int num_blocks,
                                        const int block_size ) {
      const int nelem_block = block_size * block_size;
      const int nelem_matrix = block_size * (num_blocks * block_size + 2 * (num_blocks - 1));
      griffon::blas::scale_vector( nelem_matrix, in_out_matrix_values, matrix_scale );

      for ( int iz = 0; iz < num_blocks; ++iz ) {
        for ( int iq = 0; iq < block_size; ++iq ) {
          in_out_matrix_values[iz * nelem_block + iq * (block_size + 1)] += diag_scale * diagonal[iz * block_size + iq];
        }
      }
    }

  }
}
