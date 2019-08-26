/*
 *  Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
 *  You may use, distribute and modify this code under the terms of the MIT license.
 *
 *  You should have received a copy of the MIT license with this file.
 *  If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
 */

#ifndef GRIFFON_LAPACK_WRAPPER_H
#define GRIFFON_LAPACK_WRAPPER_H

#include <cstddef>

namespace griffon {

namespace blas {

extern "C" double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
double inner_product(const int n, const double *x, const double *y);

extern "C" void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy);
void copy_vector(const int n, double *y, const double *x);

extern "C" void dscal_(const int *n, const double *alpha, double *x, const int *incx);
void scale_vector(const int n, double *x, const double alpha);

extern "C" void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
void vector_plus_scaled_vector(const int n, double *y, const double alpha, const double *x);

extern "C" void dgemv_(const char *transpose, const int *m, const int *n, const double *alpha, const double *mat, const int *lda,
    const double *x, const int *incx, const double *beta, double *y, const int *incy);
void matrix_vector_multiply(const int n, double *y, const double alpha, const double *mat, const double *x, const double beta);

}

namespace lapack {

/*
 * @brief LU factorization of a dense matrix
 *
 * @param nrows number of rows in A, integer reference
 * @param ncols number of columns in A, integer reference
 * @param a a matrix in column-major form, double array
 * @param lda leading dimension of A, integer reference
 * @param ipiv pivot indices, integer array of size nrows
 * @param info info structure, integer reference
 */
extern "C" void dgetrf_(const int *nrows, const int *ncols, double *a, const int *lda, int *ipiv, int *info);

void lu_factorize(const int nrows, const double *matrix, int *ipiv, double *factor);
void lu_factorize_blocks(const double *blocks, const int numBlocks, const int blockSize, double *out_factors, int *out_pivots);

/*
 * @brief solution to a dense system of equations with LU factorization pre-computed
 *
 * @param nrows number of rows in A, integer reference
 * @param ncols number of columns in A, integer reference
 * @param a a matrix in column-major form, double array
 * @param lda leading dimension of A, integer reference
 * @param ipiv pivot indices, integer array of size nrows
 * @param info info structure, integer reference
 */
extern "C" void dgetrs_(const char *transpose, const int *nrows, const int *nrhs, const double *plu, const int *lda, const int *ipiv,
    double *rhs, const int *ldb, int *info);

void lu_solve(const int nrows, const double *factor, const int *ipiv, const double *rhs, double *solution);
void lu_solve_on_matrix(const int nrows, const double *factor, const int *ipiv, const double *rhsmatrix, double *solutionmatrix);

/*
 * @brief full spectral decomposition of a general, double-precision matrix
 *
 * @param doleft 'N' for no left eigenmatrix and 'V' for computing it, char ref.
 * @param doright 'N' for no right eigenmatrix and 'V' for computing it, char ref.
 * @param n number of rows in A, integer reference
 * @param a a matrix in column-major form, double array
 * @param lda leading dimension of A, integer reference
 * @param real eigenvalue real parts, double array
 * @param imag eigenvalue imaginary parts, double array
 * @param leftmat left eigenmatrix, double array
 * @param ldleft leading dim. of left eig matrix, integer reference
 * @param rightmat right eigenmatrix, double array
 * @param ldright leading dim. of right eig matrix, integer reference
 * @param work work array size, double reference
 * @param lwork work array, double array
 * @param info info structure, integer reference
 */
extern "C" void dgeev_(const char *doleft, const char *doright, const int *n, double *a, const int *lda, double *real, double *imag,
    double *leftmat, const int *ldleft, double *rightmat, const int *ldright, double *work, int *lwork, int *info);

void eigenvalues(const int nrows, const double *matrix, double *realparts, double *imagparts);

}

}

#endif //GRIFFON_LAPACK_WRAPPER_H
