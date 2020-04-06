/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

#include "btddod_matrix_kernels.h"
#include "blas_lapack_kernels.h"

namespace griffon
{
namespace btddod
{

void btddod_full_factorize(double *out_d_factors, const int num_blocks, const int block_size, double *out_l_values,
                           int *out_d_pivots)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;

  double d_local_tmp[nelem_block];

  double identity_matrix[nelem_block];
  for (int i = 0; i < nelem_block; ++i)
  {
    identity_matrix[i] = 0.;
  }
  for (int i = 0; i < block_size; ++i)
  {
    identity_matrix[i * (block_size + 1)] = 1.;
  }

  const double *sub = &out_d_factors[nelem_blockdiagonals];
  const double *sup = &out_d_factors[nelem_blockdiagonals + nelem_offdiagonal];

  int info;
  const int one = 1;
  char trans = 'N';
  griffon::lapack::dgetrf_(&block_size, &block_size, out_d_factors, &block_size, out_d_pivots, &info);

  for (int i = 1; i < num_blocks; ++i)
  {
    const int o1 = i * nelem_block;
    const int om = o1 - nelem_block;

    for (int l = 0; l < nelem_block; ++l)
    {
      out_l_values[o1 + l] = identity_matrix[l];
    }
    griffon::lapack::dgetrs_(&trans, &block_size, &block_size, &out_d_factors[om], &block_size,
                             &out_d_pivots[(i - 1) * block_size], &out_l_values[o1], &block_size, &info);

    const int i_base = i * block_size;
    const int im1_base = (i - 1) * block_size;
    for (int j = 0; j < block_size; ++j)
    {
      const int o2 = o1 + j * block_size;
      for (int k = 0; k < block_size; ++k)
      {
        out_l_values[o2 + k] *= sub[im1_base + k];
      }
    }

    for (int j = 0; j < block_size; ++j)
    {
      const int o2 = j * block_size;
      const int o3 = o1 + o2;
      const double fac = -sup[im1_base + j];

      for (int k = 0; k < block_size; ++k)
      {
        out_d_factors[o3 + k] += fac * out_l_values[o3 + k];
      }
    }

    griffon::lapack::dgetrf_(&block_size, &block_size, &out_d_factors[o1], &block_size,
                             &out_d_pivots[i * block_size], &info);
  }
}

void btddod_full_solve(const double *d_factors, const double *l_values, const int *d_pivots, const double *rhs,
                       const int num_blocks, const int block_size, double *out_solution)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  const double *sup = &d_factors[nelem_blockdiagonals + nelem_offdiagonal];

  double y[nelem_vector];
  double local_tmp[block_size];

  for (int i = 0; i < block_size; ++i)
    y[i] = rhs[i];
  for (int i = 1; i < num_blocks; ++i)
  {
    for (int l = 0; l < block_size; ++l)
      y[i * block_size + l] = rhs[i * block_size + l];
    griffon::blas::matrix_vector_multiply(block_size, &y[i * block_size], -1., &l_values[i * nelem_block],
                                          &y[(i - 1) * block_size], 1.);
  }

  const int i = num_blocks - 1;
  griffon::lapack::lu_solve_with_copy(block_size, &d_factors[i * nelem_block], &d_pivots[i * block_size],
                                      &y[i * block_size], &out_solution[i * block_size]);
  for (int i = num_blocks - 2; i >= 0; --i)
  {
    const int i_base = i * block_size;
    const int ip1_base = (i + 1) * block_size;
    for (int j = 0; j < block_size; ++j)
    {
      local_tmp[j] = y[i * block_size + j] - sup[i_base + j] * out_solution[ip1_base + j];
    }
    griffon::lapack::lu_solve_with_copy(block_size, &d_factors[i * nelem_block], &d_pivots[i * block_size],
                                        local_tmp, &out_solution[i * block_size]);
  }
}

void btddod_full_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
                        double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  // block diagonal contribution
  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::blas::matrix_vector_multiply(block_size, &out_matvec[i * block_size], 1.,
                                          &matrix_values[i * nelem_block], &vec[i * block_size], 0.);
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq] + sub[im1_base + iq] * vec[im1_base + iq];
    }
  }

  const int iz1 = num_blocks - 1;
  const int i1_base = iz1 * block_size;
  const int im1_base = (iz1 - 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
  }

  const int iz2 = 0;
  const int i2_base = iz2 * block_size;
  const int ip1_base = (iz2 + 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
  }
}

void btddod_blockdiag_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
                             double *out_matvec)
{
  const int nelem_block = block_size * block_size;

  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::blas::matrix_vector_multiply(block_size, &out_matvec[i * block_size], 1.,
                                          &matrix_values[i * nelem_block], &vec[i * block_size], 0.);
  }
}

void btddod_offdiag_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size,
                           double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  for (int i = 0; i < nelem_vector; ++i)
  {
    out_matvec[i] = 0.;
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq] + sub[im1_base + iq] * vec[im1_base + iq];
    }
  }

  const int iz1 = num_blocks - 1;
  const int i1_base = iz1 * block_size;
  const int im1_base = (iz1 - 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
  }

  const int iz2 = 0;
  const int i2_base = iz2 * block_size;
  const int ip1_base = (iz2 + 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
  }
}

void btddod_lowerfulltriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks,
                                     const int block_size, double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  // block diagonal contribution
  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::blas::matrix_vector_multiply(block_size, &out_matvec[i * block_size], 1.,
                                          &matrix_values[i * nelem_block], &vec[i * block_size], 0.);
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
    }
  }

  const int iz1 = num_blocks - 1;
  const int i1_base = iz1 * block_size;
  const int im1_base = (iz1 - 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
  }
}

void btddod_upperfulltriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks,
                                     const int block_size, double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  // block diagonal contribution
  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::blas::matrix_vector_multiply(block_size, &out_matvec[i * block_size], 1.,
                                          &matrix_values[i * nelem_block], &vec[i * block_size], 0.);
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq];
    }
  }

  const int iz2 = 0;
  const int i2_base = iz2 * block_size;
  const int ip1_base = (iz2 + 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
  }
}

void btddod_lowerofftriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks,
                                    const int block_size, double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  for (int i = 0; i < nelem_vector; ++i)
  {
    out_matvec[i] = 0.;
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
    }
  }

  const int iz1 = num_blocks - 1;
  const int i1_base = iz1 * block_size;
  const int im1_base = (iz1 - 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i1_base + iq] += sub[im1_base + iq] * vec[im1_base + iq];
  }
}

void btddod_upperofftriangle_matvec(const double *matrix_values, const double *vec, const int num_blocks,
                                    const int block_size, double *out_matvec)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  const int nelem_vector = num_blocks * block_size;

  for (int i = 0; i < nelem_vector; ++i)
  {
    out_matvec[i] = 0.;
  }

  const double *sub = &matrix_values[nelem_blockdiagonals];
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];

  // off diagonal contributions
  for (int iz = 1; iz < num_blocks - 1; ++iz)
  {
    const int i_base = iz * block_size;
    const int im1_base = (iz - 1) * block_size;
    const int ip1_base = (iz + 1) * block_size;
    for (int iq = 0; iq < block_size; ++iq)
    {
      out_matvec[i_base + iq] += sup[i_base + iq] * vec[ip1_base + iq];
    }
  }

  const int iz2 = 0;
  const int i2_base = iz2 * block_size;
  const int ip1_base = (iz2 + 1) * block_size;
  for (int iq = 0; iq < block_size; ++iq)
  {
    out_matvec[i2_base + iq] += sup[i2_base + iq] * vec[ip1_base + iq];
  }
}

void btddod_blockdiag_factorize(const double *matrix_values, const int num_blocks, const int block_size, int *out_pivots,
                                double *out_factors)
{
  const int nelem_block = block_size * block_size;
  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::lapack::lu_factorize_with_copy(block_size, &matrix_values[i * nelem_block],
                                            &out_pivots[i * block_size], &out_factors[i * nelem_block]);
  }
}

void btddod_blockdiag_solve(const int *pivots, const double *factors, const double *rhs, const int num_blocks,
                            const int block_size, double *out_solution)
{
  const int nelem_block = block_size * block_size;
  for (int i = 0; i < num_blocks; ++i)
  {
    griffon::lapack::lu_solve_with_copy(block_size, &factors[i * nelem_block], &pivots[i * block_size],
                                        &rhs[i * block_size], &out_solution[i * block_size]);
  }
}

void btddod_lowerfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values,
                                    const double *rhs, const int num_blocks, const int block_size, double *out_solution)
{
  const int nelem_block = block_size * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  double rhsTemp[block_size];

  griffon::lapack::lu_solve_with_copy(block_size, factors, pivots, rhs, out_solution);
  const double *sub = &matrix_values[nelem_blockdiagonals];
  for (int i = 1; i < num_blocks; ++i)
  {
    for (int iq = 0; iq < block_size; ++iq)
    {
      rhsTemp[iq] = rhs[i * block_size + iq] - sub[(i - 1) * block_size + iq] * out_solution[(i - 1) * block_size + iq];
    }
    griffon::lapack::lu_solve_with_copy(block_size, &factors[i * nelem_block], &pivots[i * block_size], rhsTemp,
                                        &out_solution[i * block_size]);
  }
}

void btddod_upperfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values,
                                    const double *rhs, const int num_blocks, const int block_size, double *out_solution)
{
  const int nelem_block = block_size * block_size;
  const int nelem_offdiagonal = (num_blocks - 1) * block_size;
  const int nelem_blockdiagonals = num_blocks * nelem_block;
  double rhsTemp[block_size];

  const int i = num_blocks - 1;
  griffon::lapack::lu_solve_with_copy(block_size, &factors[i * nelem_block], &pivots[i * block_size],
                                      &rhs[i * block_size], &out_solution[i * block_size]);
  const double *sup = &matrix_values[nelem_blockdiagonals + nelem_offdiagonal];
  for (int i = num_blocks - 2; i >= 0; --i)
  {
    for (int iq = 0; iq < block_size; ++iq)
    {
      rhsTemp[iq] = rhs[i * block_size + iq] - sup[i * block_size + iq] * out_solution[(i + 1) * block_size + iq];
    }
    griffon::lapack::lu_solve_with_copy(block_size, &factors[i * nelem_block], &pivots[i * block_size], rhsTemp,
                                        &out_solution[i * block_size]);
  }
}

// A <- matrix_scale * A + diag_scale * block_diag
void btddod_scale_and_add_scaled_block_diagonal(double *in_out_matrix_values, const double matrix_scale,
                                                const double *block_diag, const double diag_scale, const int num_blocks,
                                                const int block_size)
{
  const int nelem_block = block_size * block_size;
  const int nelem_matrix = block_size * (num_blocks * block_size + 2 * (num_blocks - 1));
  const int nelem_blockdiag = nelem_block * num_blocks;
  for (int i = 0; i < nelem_matrix; ++i)
  {
    in_out_matrix_values[i] *= matrix_scale;
  }
  for (int i = 0; i < nelem_blockdiag; ++i)
  {
    in_out_matrix_values[i] += block_diag[i] * diag_scale;
  }
}

// A <- matrix_scale * A + diag_scale * diagonal
void btddod_scale_and_add_diagonal(double *in_out_matrix_values, const double matrix_scale, const double *diagonal,
                                   const double diag_scale, const int num_blocks, const int block_size)
{
  const int nelem_block = block_size * block_size;
  const int nelem_matrix = block_size * (num_blocks * block_size + 2 * (num_blocks - 1));
  for (int i = 0; i < nelem_matrix; ++i)
  {
    in_out_matrix_values[i] *= matrix_scale;
  }

  for (int iz = 0; iz < num_blocks; ++iz)
  {
    for (int iq = 0; iq < block_size; ++iq)
    {
      in_out_matrix_values[iz * nelem_block + iq * (block_size + 1)] += diag_scale * diagonal[iz * block_size + iq];
    }
  }
}

} // namespace btddod
} // namespace griffon
