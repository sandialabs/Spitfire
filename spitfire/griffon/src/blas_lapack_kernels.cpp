/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

#include "blas_lapack_kernels.h"

namespace griffon {
namespace blas {

double inner_product(const int n, const double *x, const double *y) {
  const int one = 1;
  return ddot_(&n, x, &one, y, &one);
}

void copy_vector(const int n, double *y, const double *x) {
  const int one = 1;
  dcopy_(&n, x, &one, y, &one);
}

void scale_vector(const int n, double *x, const double alpha) {
  const int one = 1;
  dscal_(&n, &alpha, x, &one);
}

void vector_plus_scaled_vector(const int n, double *y, const double alpha, const double *x) {
  const int one = 1;
  daxpy_(&n, &alpha, x, &one, y, &one);
}

void matrix_vector_multiply(const int n, double *y, const double alpha, const double *mat, const double *x, const double beta) {
  const int one = 1;
  char trans = 'N';
  dgemv_(&trans, &n, &n, &alpha, mat, &n, x, &one, &beta, y, &one);
}

}

namespace lapack {

void lu_factorize(const int nrows, const double *matrix, int *ipiv, double *factor) {
  griffon::blas::copy_vector(nrows * nrows, factor, matrix);
  int info;
  dgetrf_(&nrows, &nrows, factor, &nrows, ipiv, &info);
}

void lu_factorize_blocks(const double *blocks, const int numBlocks, const int blockSize, double *out_factors, int *out_pivots) {
  const int blockElements = blockSize * blockSize;

  const int numElements = numBlocks * blockElements;
  griffon::blas::copy_vector(numElements, out_factors, blocks);

  for (int iblock = 0; iblock < numBlocks; ++iblock) {
    const int blockIdxBaseMatrix = iblock * blockElements;
    const int blockIdxBasePivots = iblock * blockSize;
    griffon::lapack::lu_factorize(blockSize, &blocks[blockIdxBaseMatrix], &out_pivots[blockIdxBasePivots],
        &out_factors[blockIdxBaseMatrix]);
  }
}

void lu_solve(const int nrows, const double *factor, const int *ipiv, const double *rhs, double *solution) {
  const int one = 1;
  griffon::blas::copy_vector(nrows, solution, rhs);
  int info;
  char trans = 'N';
  dgetrs_(&trans, &nrows, &one, factor, &nrows, ipiv, solution, &nrows, &info);
}

void lu_solve_on_matrix(const int nrows, const double *factor, const int *ipiv, const double *rhsmatrix, double *solutionmatrix) {
  const int one = 1;
  griffon::blas::copy_vector(nrows * nrows, solutionmatrix, rhsmatrix);
  int info;
  char trans = 'N';
  dgetrs_(&trans, &nrows, &nrows, factor, &nrows, ipiv, solutionmatrix, &nrows, &info);
}

void eigenvalues(const int nrows, const double *matrix, double *realparts, double *imagparts) {
  double matrixcopy[nrows * nrows];
  griffon::blas::copy_vector(nrows * nrows, matrixcopy, matrix);

  const char leftchar = 'N';
  const char rightchar = 'N';

  double null[nrows * nrows];

  int lwork, info;
  double wkopt;

  // first query dgeev for the optimal workspace size
  lwork = -1;
  dgeev_(&leftchar, &rightchar, &nrows, matrixcopy, &nrows, realparts, imagparts, null, &nrows, null, &nrows, &wkopt, &lwork, &info);

  // allocate the workspace and compute the decomposition
  lwork = static_cast<std::size_t>(wkopt);
  double* work = new double[lwork];
  dgeev_(&leftchar, &rightchar, &nrows, matrixcopy, &nrows, realparts, imagparts, null, &nrows, null, &nrows, work, &lwork, &info);
  delete[] work;
}

}

}

