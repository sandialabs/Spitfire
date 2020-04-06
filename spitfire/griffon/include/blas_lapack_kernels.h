/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

/*
 * This file includes a few wrappers around LAPACK and BLAS routines,
 * some of which are written natively here to unleash optimizing compilers
 * that seem to be better than BLAS/LAPACK for certain operations.
 *
 * Note that the factorization and solution routines below will make copies of
 * input matrices or solution vectors/matrices that LAPACK otherwise modifies
 * in place. This is for convenience when copying is required or acceptable.
 * If copies are not acceptable, the methods should make it simple enough to
 * see how to directly use the underlying LAPACK routines.
 */

#ifndef GRIFFON_LAPACK_WRAPPER_H
#define GRIFFON_LAPACK_WRAPPER_H

#include <cstddef>

namespace griffon
{

namespace blas
{

/*
     * @brief compute an inner product of two arrays
     *
     * @param n the number of elements
     * @param x the first vector
     * @param y the second vector
     */
inline double
inner_product(const int n, const double *x, const double *y)
{
  double d = 0.;
  for (int i = 0; i < n; ++i)
    d += y[i] * x[i];
  return d;
}

/*
     * @brief add a matvec to a vector, y = beta * y + alpha * mat * x
     *
     * @param n the number of elements
     * @param y the output vector
     * @param alpha the scale on the matvec product
     * @param mat the column-major matrix
     * @param x the vector being multiplied by the matrix
     * @param beta the scale on the y vector
     */
inline void
matrix_vector_multiply(const int n, double *y, const double alpha, const double *mat, const double *x,
                       const double beta)
{
  for (int j = 0; j < n; ++j)
  {
    y[j] *= beta;
  }
  for (int i = 0; i < n; ++i)
  {
    const double axi = alpha * x[i];
    const int offset = i * n;
    for (int j = 0; j < n; ++j)
    {
      y[j] = y[j] + mat[offset + j] * axi;
    }
  }
}

} // namespace blas

namespace lapack
{

extern "C" void
dgetrf_(const int *nrows, const int *ncols, double *a, const int *lda, int *ipiv, int *info);
/*
     * @brief factorize a linear system of equations for later solution
     *
     * @param n number of rows in the matrix
     * @param matrix the left-hand side matrix
     * @param ipiv the pivots from dgetrf (out argument)
     * @param factor the factored matrix (out argument)
     */
inline void
lu_factorize_with_copy(const int n, const double *matrix, int *ipiv, double *factor)
{
  for (int i = 0; i < n * n; ++i)
    factor[i] = matrix[i];
  int info;
  dgetrf_(&n, &n, factor, &n, ipiv, &info);
}

extern "C" void
dgetrs_(const char *transpose, const int *nrows, const int *nrhs, const double *plu, const int *lda,
        const int *ipiv, double *rhs, const int *ldb, int *info);
/*
     * @brief compute the solution to a prefactored linear system of equations
     *
     * @param n number of rows in the matrix
     * @param factor the factored matrix from dgetrf
     * @param ipiv the pivots from dgetrf
     * @param rhs the right-hand side of the linear system
     * @param solution the solution (out argument)
     */
inline void
lu_solve_with_copy(const int n, const double *factor, const int *ipiv, const double *rhs, double *solution)
{
  const int one = 1;
  char trans = 'N';
  int info;
  for (int i = 0; i < n; ++i)
    solution[i] = rhs[i];
  dgetrs_(&trans, &n, &one, factor, &n, ipiv, solution, &n, &info);
}
/*
     * @brief compute the solution to a set of prefactored linear system of equations with the same left-hand side matrix
     *
     * @param n number of rows in the matrix
     * @param factor the factored matrix from dgetrf
     * @param ipiv the pivots from dgetrf
     * @param rhsmatrix the right-hand side matrix
     * @param solutionmatrix the solution (out argument)
     */
inline void
lu_solve_on_matrix_with_copy(const int n, const double *factor, const int *ipiv, const double *rhsmatrix,
                             double *solutionmatrix)
{
  const int one = 1;
  char trans = 'N';
  int info;
  for (int i = 0; i < n * n; ++i)
    solutionmatrix[i] = rhsmatrix[i];
  dgetrs_(&trans, &n, &n, factor, &n, ipiv, solutionmatrix, &n, &info);
}

extern "C" void
dgeev_(const char *doleft, const char *doright, const int *n, double *a, const int *lda, double *real, double *imag,
       double *leftmat, const int *ldleft, double *rightmat, const int *ldright, double *work, int *lwork,
       int *info);
/*
     * @brief compute the eigenvalues of a matrix
     *
     * @param n number of rows in the matrix
     * @param matrix the values of the matrix in column-major form
     * @param realparts the real parts of the eigenvalues (out argument)
     * @param imagparts the imaginary parts of the eigenvalues (out argument)
     */
inline void
eigenvalues(const int n, const double *matrix, double *realparts, double *imagparts)
{
  const char rightchar = 'N';
  const char leftchar = 'N';
  double null[1];
  int lwork, info;
  double wkopt;

  double matrixcopy[n * n];
  for (int i = 0; i < n * n; ++i)
    matrixcopy[i] = matrix[i];

  // first query dgeev for the optimal workspace size
  lwork = -1;
  dgeev_(&leftchar, &rightchar, &n, matrixcopy, &n, realparts, imagparts, null, &n, null, &n, &wkopt, &lwork,
         &info);

  // allocate the workspace and compute the decomposition
  lwork = static_cast<std::size_t>(wkopt);
  double *work = new double[lwork];
  dgeev_(&leftchar, &rightchar, &n, matrixcopy, &n, realparts, imagparts, null, &n, null, &n, work, &lwork, &info);
  delete[] work;
}

} // namespace lapack

} // namespace griffon

#endif //GRIFFON_LAPACK_WRAPPER_H
