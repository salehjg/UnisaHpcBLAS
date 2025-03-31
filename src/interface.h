//
// Created by saleh on 3/29/25.
//

#pragma once

#include <complex>
#include <memory>

template <typename Impl>
class interface {
public:
    using cfloat_t = std::complex<float>;
    using cdouble_t = std::complex<double>;
    enum order_t { CblasRowMajor = 101, CblasColMajor = 102 };
    enum transpose_t { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114 };
    enum uplo_t { CblasUpper = 121, CblasLower = 122 };
    enum diag_t { CblasNonUnit = 131, CblasUnit = 132 };
    enum side_t { CblasLeft = 141, CblasRight = 142 };

public:
    interface();

public:
    float sdsdot(int32_t n, float alpha, const float* __restrict x, int32_t incx, const float* __restrict y,
                 int32_t incy);
    double dsdot(int32_t n, const float* __restrict x, int32_t incx, const float* __restrict y, int32_t incy);
    float sdot(int32_t n, const float* __restrict x, int32_t incx, const float* __restrict y, int32_t incy);
    double ddot(int32_t n, const double* __restrict x, int32_t incx, const double* __restrict y, int32_t incy);

    cfloat_t cdotu(int32_t n, const cfloat_t* __restrict x, int32_t incx, const cfloat_t* __restrict y, int32_t incy);
    cfloat_t cdotc(int32_t n, const cfloat_t* __restrict x, int32_t incx, const cfloat_t* __restrict y, int32_t incy);
    cdouble_t zdotu(int32_t n, const cdouble_t* __restrict x, int32_t incx, const cdouble_t* __restrict y,
                    int32_t incy);
    cdouble_t zdotc(int32_t n, const cdouble_t* __restrict x, int32_t incx, const cdouble_t* __restrict y,
                    int32_t incy);

    void cdotu_sub(int32_t n, const cfloat_t* __restrict x, int32_t incx, const cfloat_t* __restrict y, int32_t incy,
                   cfloat_t* __restrict ret);
    void cdotc_sub(int32_t n, const cfloat_t* __restrict x, int32_t incx, const cfloat_t* __restrict y, int32_t incy,
                   cfloat_t* __restrict ret);
    void zdotu_sub(int32_t n, const cdouble_t* __restrict x, int32_t incx, const cdouble_t* __restrict y, int32_t incy,
                   cdouble_t* __restrict ret);
    void zdotc_sub(int32_t n, const cdouble_t* __restrict x, int32_t incx, const cdouble_t* __restrict y, int32_t incy,
                   cdouble_t* __restrict ret);

    float sasum(int32_t n, const float* __restrict x, int32_t incx);
    double dasum(int32_t n, const double* __restrict x, int32_t incx);
    float scasum(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    double dzasum(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    float ssum(int32_t n, const float* __restrict x, int32_t incx);
    double dsum(int32_t n, const double* __restrict x, int32_t incx);
    float scsum(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    double dzsum(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    float snrm2(int32_t N, const float* __restrict X, int32_t incX);
    double dnrm2(int32_t N, const double* __restrict X, int32_t incX);
    float scnrm2(int32_t N, const cfloat_t* __restrict X, int32_t incX);
    double dznrm2(int32_t N, const cdouble_t* __restrict X, int32_t incX);

    size_t isamax(int32_t n, const float* __restrict x, int32_t incx);
    size_t idamax(int32_t n, const double* __restrict x, int32_t incx);
    size_t icamax(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    size_t izamax(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    size_t isamin(int32_t n, const float* __restrict x, int32_t incx);
    size_t idamin(int32_t n, const double* __restrict x, int32_t incx);
    size_t icamin(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    size_t izamin(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    float samax(int32_t n, const float* __restrict x, int32_t incx);
    double damax(int32_t n, const double* __restrict x, int32_t incx);
    float scamax(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    double dzamax(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    float samin(int32_t n, const float* __restrict x, int32_t incx);
    double damin(int32_t n, const double* __restrict x, int32_t incx);
    float scamin(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    double dzamin(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    size_t ismax(int32_t n, const float* __restrict x, int32_t incx);
    size_t idmax(int32_t n, const double* __restrict x, int32_t incx);
    size_t icmax(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    size_t izmax(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    size_t ismin(int32_t n, const float* __restrict x, int32_t incx);
    size_t idmin(int32_t n, const double* __restrict x, int32_t incx);
    size_t icmin(int32_t n, const cfloat_t* __restrict x, int32_t incx);
    size_t izmin(int32_t n, const cdouble_t* __restrict x, int32_t incx);

    void saxpy(int32_t n, float alpha, const float* __restrict x, int32_t incx, float* __restrict y, int32_t incy);
    void daxpy(int32_t n, double alpha, const double* __restrict x, int32_t incx, double* __restrict y, int32_t incy);
    void caxpy(int32_t n, const cfloat_t* __restrict alpha, const cfloat_t* __restrict x, int32_t incx,
               cfloat_t* __restrict y, int32_t incy);
    void zaxpy(int32_t n, const cdouble_t* __restrict alpha, const cdouble_t* __restrict x, int32_t incx,
               cdouble_t* __restrict y, int32_t incy);

    void caxpyc(int32_t n, const cfloat_t* __restrict alpha, const cfloat_t* __restrict x, int32_t incx,
                cfloat_t* __restrict y, int32_t incy);
    void zaxpyc(int32_t n, const cdouble_t* __restrict alpha, const cdouble_t* __restrict x, int32_t incx,
                cdouble_t* __restrict y, int32_t incy);

    void scopy(int32_t n, const float* __restrict x, int32_t incx, float* __restrict y, int32_t incy);
    void dcopy(int32_t n, const double* __restrict x, int32_t incx, double* __restrict y, int32_t incy);
    void ccopy(int32_t n, const cfloat_t* __restrict x, int32_t incx, cfloat_t* __restrict y, int32_t incy);
    void zcopy(int32_t n, const cdouble_t* __restrict x, int32_t incx, cdouble_t* __restrict y, int32_t incy);

    void sswap(int32_t n, float* __restrict x, int32_t incx, float* __restrict y, int32_t incy);
    void dswap(int32_t n, double* __restrict x, int32_t incx, double* __restrict y, int32_t incy);
    void cswap(int32_t n, cfloat_t* __restrict x, int32_t incx, cfloat_t* __restrict y, int32_t incy);
    void zswap(int32_t n, cdouble_t* __restrict x, int32_t incx, cdouble_t* __restrict y, int32_t incy);

    void srot(int32_t N, float* __restrict X, int32_t incX, float* __restrict Y, int32_t incY, float c, float s);
    void drot(int32_t N, double* __restrict X, int32_t incX, double* __restrict Y, int32_t incY, double c, double s);
    void csrot(int32_t n, const cfloat_t* __restrict x, int32_t incx, cfloat_t* __restrict y, int32_t incY, float c,
               float s);
    void zdrot(int32_t n, const cdouble_t* __restrict x, int32_t incx, cdouble_t* __restrict y, int32_t incY, double c,
               double s);

    void srotg(float* __restrict a, float* __restrict b, float* __restrict c, float* __restrict s);
    void drotg(double* __restrict a, double* __restrict b, double* __restrict c, double* __restrict s);
    void crotg(cfloat_t* __restrict a, cfloat_t* __restrict b, float* __restrict c, cfloat_t* __restrict s);
    void zrotg(cdouble_t* __restrict a, cdouble_t* __restrict b, double* __restrict c, cdouble_t* __restrict s);

    void srotm(int32_t N, float* __restrict X, int32_t incX, float* __restrict Y, int32_t incY,
               const float* __restrict P);
    void drotm(int32_t N, double* __restrict X, int32_t incX, double* __restrict Y, int32_t incY,
               const double* __restrict P);

    void srotmg(float* __restrict d1, float* __restrict d2, float* __restrict b1, float b2, float* __restrict P);
    void drotmg(double* __restrict d1, double* __restrict d2, double* __restrict b1, double b2, double* __restrict P);

    void sscal(int32_t N, float alpha, float* __restrict X, int32_t incX);
    void dscal(int32_t N, double alpha, double* __restrict X, int32_t incX);
    void cscal(int32_t N, const cfloat_t* __restrict alpha, cfloat_t* __restrict X, int32_t incX);
    void zscal(int32_t N, const cdouble_t* __restrict alpha, cdouble_t* __restrict X, int32_t incX);
    void csscal(int32_t N, float alpha, cfloat_t* __restrict X, int32_t incX);
    void zdscal(int32_t N, double alpha, cdouble_t* __restrict X, int32_t incX);

    void sgemv(enum order_t order, enum transpose_t trans, int32_t m, int32_t n, float alpha, const float* __restrict a,
               int32_t lda, const float* __restrict x, int32_t incx, float beta, float* __restrict y, int32_t incy);
    void dgemv(enum order_t order, enum transpose_t trans, int32_t m, int32_t n, double alpha,
               const double* __restrict a, int32_t lda, const double* __restrict x, int32_t incx, double beta,
               double* __restrict y, int32_t incy);
    void cgemv(enum order_t order, enum transpose_t trans, int32_t m, int32_t n, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict a, int32_t lda, const cfloat_t* __restrict x, int32_t incx,
               const cfloat_t* __restrict beta, cfloat_t* __restrict y, int32_t incy);
    void zgemv(enum order_t order, enum transpose_t trans, int32_t m, int32_t n, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict a, int32_t lda, const cdouble_t* __restrict x, int32_t incx,
               const cdouble_t* __restrict beta, cdouble_t* __restrict y, int32_t incy);

    void sger(enum order_t order, int32_t M, int32_t N, float alpha, const float* __restrict X, int32_t incX,
              const float* __restrict Y, int32_t incY, float* __restrict A, int32_t lda);
    void dger(enum order_t order, int32_t M, int32_t N, double alpha, const double* __restrict X, int32_t incX,
              const double* __restrict Y, int32_t incY, double* __restrict A, int32_t lda);
    void cgeru(enum order_t order, int32_t M, int32_t N, const cfloat_t* __restrict alpha, const cfloat_t* __restrict X,
               int32_t incX, const cfloat_t* __restrict Y, int32_t incY, cfloat_t* __restrict A, int32_t lda);
    void cgerc(enum order_t order, int32_t M, int32_t N, const cfloat_t* __restrict alpha, const cfloat_t* __restrict X,
               int32_t incX, const cfloat_t* __restrict Y, int32_t incY, cfloat_t* __restrict A, int32_t lda);
    void zgeru(enum order_t order, int32_t M, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict X, int32_t incX, const cdouble_t* __restrict Y, int32_t incY,
               cdouble_t* __restrict A, int32_t lda);
    void zgerc(enum order_t order, int32_t M, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict X, int32_t incX, const cdouble_t* __restrict Y, int32_t incY,
               cdouble_t* __restrict A, int32_t lda);

    void strsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const float* __restrict A, int32_t lda, float* __restrict X, int32_t incX);
    void dtrsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const double* __restrict A, int32_t lda, double* __restrict X, int32_t incX);
    void ctrsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cfloat_t* __restrict A, int32_t lda, cfloat_t* __restrict X, int32_t incX);
    void ztrsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cdouble_t* __restrict A, int32_t lda, cdouble_t* __restrict X, int32_t incX);

    void strmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const float* __restrict A, int32_t lda, float* __restrict X, int32_t incX);
    void dtrmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const double* __restrict A, int32_t lda, double* __restrict X, int32_t incX);
    void ctrmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cfloat_t* __restrict A, int32_t lda, cfloat_t* __restrict X, int32_t incX);
    void ztrmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cdouble_t* __restrict A, int32_t lda, cdouble_t* __restrict X, int32_t incX);

    void ssyr(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict X, int32_t incX,
              float* __restrict A, int32_t lda);
    void dsyr(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict X, int32_t incX,
              double* __restrict A, int32_t lda);
    void cher(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const cfloat_t* __restrict X, int32_t incX,
              cfloat_t* __restrict A, int32_t lda);
    void zher(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const cdouble_t* __restrict X,
              int32_t incX, cdouble_t* __restrict A, int32_t lda);


    void ssyr2(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict X, int32_t incX,
               const float* __restrict Y, int32_t incY, float* __restrict A, int32_t lda);
    void dsyr2(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict X, int32_t incX,
               const double* __restrict Y, int32_t incY, double* __restrict A, int32_t lda);
    void cher2(enum order_t order, enum uplo_t Uplo, int32_t N, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict X, int32_t incX, const cfloat_t* __restrict Y, int32_t incY,
               cfloat_t* __restrict A, int32_t lda);
    void zher2(enum order_t order, enum uplo_t Uplo, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict X, int32_t incX, const cdouble_t* __restrict Y, int32_t incY,
               cdouble_t* __restrict A, int32_t lda);

    void sgbmv(enum order_t order, enum transpose_t TransA, int32_t M, int32_t N, int32_t KL, int32_t KU, float alpha,
               const float* __restrict A, int32_t lda, const float* __restrict X, int32_t incX, float beta,
               float* __restrict Y, int32_t incY);
    void dgbmv(enum order_t order, enum transpose_t TransA, int32_t M, int32_t N, int32_t KL, int32_t KU, double alpha,
               const double* __restrict A, int32_t lda, const double* __restrict X, int32_t incX, double beta,
               double* __restrict Y, int32_t incY);
    void cgbmv(enum order_t order, enum transpose_t TransA, int32_t M, int32_t N, int32_t KL, int32_t KU,
               const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               const cfloat_t* __restrict X, int32_t incX, const cfloat_t* __restrict beta, cfloat_t* __restrict Y,
               int32_t incY);
    void zgbmv(enum order_t order, enum transpose_t TransA, int32_t M, int32_t N, int32_t KL, int32_t KU,
               const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               const cdouble_t* __restrict X, int32_t incX, const cdouble_t* __restrict beta, cdouble_t* __restrict Y,
               int32_t incY);

    void ssbmv(enum order_t order, enum uplo_t Uplo, int32_t N, int32_t K, float alpha, const float* __restrict A,
               int32_t lda, const float* __restrict X, int32_t incX, float beta, float* __restrict Y, int32_t incY);
    void dsbmv(enum order_t order, enum uplo_t Uplo, int32_t N, int32_t K, double alpha, const double* __restrict A,
               int32_t lda, const double* __restrict X, int32_t incX, double beta, double* __restrict Y, int32_t incY);

    void stbmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const float* __restrict A, int32_t lda, float* __restrict X, int32_t incX);
    void dtbmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const double* __restrict A, int32_t lda, double* __restrict X, int32_t incX);
    void ctbmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const cfloat_t* __restrict A, int32_t lda, cfloat_t* __restrict X, int32_t incX);
    void ztbmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const cdouble_t* __restrict A, int32_t lda, cdouble_t* __restrict X, int32_t incX);

    void stbsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const float* __restrict A, int32_t lda, float* __restrict X, int32_t incX);
    void dtbsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const double* __restrict A, int32_t lda, double* __restrict X, int32_t incX);
    void ctbsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const cfloat_t* __restrict A, int32_t lda, cfloat_t* __restrict X, int32_t incX);
    void ztbsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N, int32_t K,
               const cdouble_t* __restrict A, int32_t lda, cdouble_t* __restrict X, int32_t incX);

    void stpmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const float* __restrict Ap, float* __restrict X, int32_t incX);
    void dtpmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const double* __restrict Ap, double* __restrict X, int32_t incX);
    void ctpmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cfloat_t* __restrict Ap, cfloat_t* __restrict X, int32_t incX);
    void ztpmv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cdouble_t* __restrict Ap, cdouble_t* __restrict X, int32_t incX);

    void stpsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const float* __restrict Ap, float* __restrict X, int32_t incX);
    void dtpsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const double* __restrict Ap, double* __restrict X, int32_t incX);
    void ctpsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cfloat_t* __restrict Ap, cfloat_t* __restrict X, int32_t incX);
    void ztpsv(enum order_t order, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag, int32_t N,
               const cdouble_t* __restrict Ap, cdouble_t* __restrict X, int32_t incX);

    void ssymv(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict A, int32_t lda,
               const float* __restrict X, int32_t incX, float beta, float* __restrict Y, int32_t incY);
    void dsymv(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict A, int32_t lda,
               const double* __restrict X, int32_t incX, double beta, double* __restrict Y, int32_t incY);
    void chemv(enum order_t order, enum uplo_t Uplo, int32_t N, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict A, int32_t lda, const cfloat_t* __restrict X, int32_t incX,
               const cfloat_t* __restrict beta, cfloat_t* __restrict Y, int32_t incY);
    void zhemv(enum order_t order, enum uplo_t Uplo, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict A, int32_t lda, const cdouble_t* __restrict X, int32_t incX,
               const cdouble_t* __restrict beta, cdouble_t* __restrict Y, int32_t incY);

    void sspmv(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict Ap,
               const float* __restrict X, int32_t incX, float beta, float* __restrict Y, int32_t incY);
    void dspmv(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict Ap,
               const double* __restrict X, int32_t incX, double beta, double* __restrict Y, int32_t incY);

    void sspr(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict X, int32_t incX,
              float* __restrict Ap);
    void dspr(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict X, int32_t incX,
              double* __restrict Ap);

    void chpr(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const cfloat_t* __restrict X, int32_t incX,
              cfloat_t* __restrict A);
    void zhpr(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const cdouble_t* __restrict X,
              int32_t incX, cdouble_t* __restrict A);

    void sspr2(enum order_t order, enum uplo_t Uplo, int32_t N, float alpha, const float* __restrict X, int32_t incX,
               const float* __restrict Y, int32_t incY, float* __restrict A);
    void dspr2(enum order_t order, enum uplo_t Uplo, int32_t N, double alpha, const double* __restrict X, int32_t incX,
               const double* __restrict Y, int32_t incY, double* __restrict A);
    void chpr2(enum order_t order, enum uplo_t Uplo, int32_t N, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict X, int32_t incX, const cfloat_t* __restrict Y, int32_t incY,
               cfloat_t* __restrict Ap);
    void zhpr2(enum order_t order, enum uplo_t Uplo, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict X, int32_t incX, const cdouble_t* __restrict Y, int32_t incY,
               cdouble_t* __restrict Ap);

    void chbmv(enum order_t order, enum uplo_t Uplo, int32_t N, int32_t K, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict A, int32_t lda, const cfloat_t* __restrict X, int32_t incX,
               const cfloat_t* __restrict beta, cfloat_t* __restrict Y, int32_t incY);
    void zhbmv(enum order_t order, enum uplo_t Uplo, int32_t N, int32_t K, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict A, int32_t lda, const cdouble_t* __restrict X, int32_t incX,
               const cdouble_t* __restrict beta, cdouble_t* __restrict Y, int32_t incY);

    void chpmv(enum order_t order, enum uplo_t Uplo, int32_t N, const cfloat_t* __restrict alpha,
               const cfloat_t* __restrict Ap, const cfloat_t* __restrict X, int32_t incX,
               const cfloat_t* __restrict beta, cfloat_t* __restrict Y, int32_t incY);
    void zhpmv(enum order_t order, enum uplo_t Uplo, int32_t N, const cdouble_t* __restrict alpha,
               const cdouble_t* __restrict Ap, const cdouble_t* __restrict X, int32_t incX,
               const cdouble_t* __restrict beta, cdouble_t* __restrict Y, int32_t incY);

    void sgemm(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
               float alpha, const float* __restrict A, int32_t lda, const float* __restrict B, int32_t ldb, float beta,
               float* __restrict C, int32_t ldc);
    void dgemm(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
               double alpha, const double* __restrict A, int32_t lda, const double* __restrict B, int32_t ldb,
               double beta, double* __restrict C, int32_t ldc);
    void cgemm(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
               const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
               int32_t ldc);
    void cgemm3m(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
                 const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
                 const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
                 int32_t ldc);
    void zgemm(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
               const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
               int32_t ldc);
    void zgemm3m(enum order_t Order, enum transpose_t TransA, enum transpose_t TransB, int32_t M, int32_t N, int32_t K,
                 const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
                 const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
                 int32_t ldc);

    void sgemmt(enum order_t Order, enum uplo_t Uplo, enum transpose_t TransA, enum transpose_t TransB, int32_t M,
                int32_t K, float alpha, const float* __restrict A, int32_t lda, const float* __restrict B, int32_t ldb,
                float beta, float* __restrict C, int32_t ldc);
    void dgemmt(enum order_t Order, enum uplo_t Uplo, enum transpose_t TransA, enum transpose_t TransB, int32_t M,
                int32_t K, double alpha, const double* __restrict A, int32_t lda, const double* __restrict B,
                int32_t ldb, double beta, double* __restrict C, int32_t ldc);
    void cgemmt(enum order_t Order, enum uplo_t Uplo, enum transpose_t TransA, enum transpose_t TransB, int32_t M,
                int32_t K, const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
                const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
                int32_t ldc);
    void zgemmt(enum order_t Order, enum uplo_t Uplo, enum transpose_t TransA, enum transpose_t TransB, int32_t M,
                int32_t K, const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
                const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
                int32_t ldc);

    void ssymm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N, float alpha,
               const float* __restrict A, int32_t lda, const float* __restrict B, int32_t ldb, float beta,
               float* __restrict C, int32_t ldc);
    void dsymm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N, double alpha,
               const double* __restrict A, int32_t lda, const double* __restrict B, int32_t ldb, double beta,
               double* __restrict C, int32_t ldc);
    void csymm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N,
               const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
               int32_t ldc);
    void zsymm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N,
               const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
               int32_t ldc);

    void ssyrk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, float alpha,
               const float* __restrict A, int32_t lda, float beta, float* __restrict C, int32_t ldc);
    void dsyrk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, double alpha,
               const double* __restrict A, int32_t lda, double beta, double* __restrict C, int32_t ldc);
    void csyrk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
               const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               const cfloat_t* __restrict beta, cfloat_t* __restrict C, int32_t ldc);
    void zsyrk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
               const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               const cdouble_t* __restrict beta, cdouble_t* __restrict C, int32_t ldc);

    void ssyr2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, float alpha,
                const float* __restrict A, int32_t lda, const float* __restrict B, int32_t ldb, float beta,
                float* __restrict C, int32_t ldc);
    void dsyr2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, double alpha,
                const double* __restrict A, int32_t lda, const double* __restrict B, int32_t ldb, double beta,
                double* __restrict C, int32_t ldc);
    void csyr2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
                const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
                const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
                int32_t ldc);
    void zsyr2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
                const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
                const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
                int32_t ldc);

    void strmm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, float alpha, const float* __restrict A, int32_t lda, float* __restrict B,
               int32_t ldb);
    void dtrmm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, double alpha, const double* __restrict A, int32_t lda, double* __restrict B,
               int32_t ldb);
    void ctrmm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               cfloat_t* __restrict B, int32_t ldb);
    void ztrmm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               cdouble_t* __restrict B, int32_t ldb);

    void strsm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, float alpha, const float* __restrict A, int32_t lda, float* __restrict B,
               int32_t ldb);
    void dtrsm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, double alpha, const double* __restrict A, int32_t lda, double* __restrict B,
               int32_t ldb);
    void ctrsm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               cfloat_t* __restrict B, int32_t ldb);
    void ztrsm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, enum transpose_t TransA, enum diag_t Diag,
               int32_t M, int32_t N, const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               cdouble_t* __restrict B, int32_t ldb);

    void chemm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N,
               const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
               const cfloat_t* __restrict B, int32_t ldb, const cfloat_t* __restrict beta, cfloat_t* __restrict C,
               int32_t ldc);
    void zhemm(enum order_t Order, enum side_t Side, enum uplo_t Uplo, int32_t M, int32_t N,
               const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
               const cdouble_t* __restrict B, int32_t ldb, const cdouble_t* __restrict beta, cdouble_t* __restrict C,
               int32_t ldc);

    void cherk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, float alpha,
               const cfloat_t* __restrict A, int32_t lda, float beta, cfloat_t* __restrict C, int32_t ldc);
    void zherk(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K, double alpha,
               const cdouble_t* __restrict A, int32_t lda, double beta, cdouble_t* __restrict C, int32_t ldc);

    void cher2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
                const cfloat_t* __restrict alpha, const cfloat_t* __restrict A, int32_t lda,
                const cfloat_t* __restrict B, int32_t ldb, float beta, cfloat_t* __restrict C, int32_t ldc);
    void zher2k(enum order_t Order, enum uplo_t Uplo, enum transpose_t Trans, int32_t N, int32_t K,
                const cdouble_t* __restrict alpha, const cdouble_t* __restrict A, int32_t lda,
                const cdouble_t* __restrict B, int32_t ldb, double beta, cdouble_t* __restrict C, int32_t ldc);

    void xerbla(int32_t p, const char* rout, const char* form, ...);

    void saxpby(int32_t n, float alpha, const float* __restrict x, int32_t incx, float beta, float* __restrict y,
                int32_t incy);
    void daxpby(int32_t n, double alpha, const double* __restrict x, int32_t incx, double beta, double* __restrict y,
                int32_t incy);
    void caxpby(int32_t n, const cfloat_t* __restrict alpha, const cfloat_t* __restrict x, int32_t incx,
                const cfloat_t* __restrict beta, cfloat_t* __restrict y, int32_t incy);
    void zaxpby(int32_t n, const cdouble_t* __restrict alpha, const cdouble_t* __restrict x, int32_t incx,
                const cdouble_t* __restrict beta, cdouble_t* __restrict y, int32_t incy);

    void somatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols, float calpha,
                   const float* __restrict a, int32_t clda, float* __restrict b, int32_t cldb);
    void domatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols, double calpha,
                   const double* __restrict a, int32_t clda, double* __restrict b, int32_t cldb);
    void comatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols,
                   const cfloat_t* __restrict calpha, const cfloat_t* __restrict a, int32_t clda,
                   cfloat_t* __restrict b, int32_t cldb);
    void zomatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols,
                   const cdouble_t* __restrict calpha, const cdouble_t* __restrict a, int32_t clda,
                   cdouble_t* __restrict b, int32_t cldb);

    void simatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols, float calpha,
                   float* __restrict a, int32_t clda, int32_t cldb);
    void dimatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols, double calpha,
                   double* __restrict a, int32_t clda, int32_t cldb);
    void cimatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols,
                   const cfloat_t* __restrict calpha, cfloat_t* __restrict a, int32_t clda, int32_t cldb);
    void zimatcopy(enum order_t CORDER, enum transpose_t CTRANS, int32_t crows, int32_t ccols,
                   const cdouble_t* __restrict calpha, cdouble_t* __restrict a, int32_t clda, int32_t cldb);

    void sgeadd(enum order_t CORDER, int32_t crows, int32_t ccols, float calpha, const float* __restrict a,
                int32_t clda, float cbeta, float* __restrict c, int32_t cldc);
    void dgeadd(enum order_t CORDER, int32_t crows, int32_t ccols, double calpha, const double* __restrict a,
                int32_t clda, double cbeta, double* __restrict c, int32_t cldc);
    void cgeadd(enum order_t CORDER, int32_t crows, int32_t ccols, const cfloat_t* __restrict calpha,
                const cfloat_t* __restrict a, int32_t clda, const cfloat_t* __restrict cbeta, cfloat_t* __restrict c,
                int32_t cldc);
    void zgeadd(enum order_t CORDER, int32_t crows, int32_t ccols, const cdouble_t* __restrict calpha,
                const cdouble_t* __restrict a, int32_t clda, const cdouble_t* __restrict cbeta, cdouble_t* __restrict c,
                int32_t cldc);

    void sgemm_batch(enum order_t Order, const enum transpose_t* __restrict TransA_array,
                     const enum transpose_t* __restrict TransB_array, const int32_t* __restrict M_array,
                     const int32_t* __restrict N_array, const int32_t* __restrict K_array,
                     const float* __restrict alpha_array, const float* __restrict* __restrict A_array,
                     const int32_t* __restrict lda_array, const float* __restrict* __restrict B_array,
                     const int32_t* __restrict ldb_array, const float* __restrict beta_array,
                     float* __restrict* __restrict C_array, const int32_t* __restrict ldc_array, int32_t group_count,
                     const int32_t* __restrict group_size);
    void dgemm_batch(enum order_t Order, const enum transpose_t* __restrict TransA_array,
                     const enum transpose_t* __restrict TransB_array, const int32_t* __restrict M_array,
                     const int32_t* __restrict N_array, const int32_t* __restrict K_array,
                     const double* __restrict alpha_array, const double* __restrict* __restrict A_array,
                     const int32_t* __restrict lda_array, const double* __restrict* __restrict B_array,
                     const int32_t* __restrict ldb_array, const double* __restrict beta_array,
                     double* __restrict* __restrict C_array, const int32_t* __restrict ldc_array, int32_t group_count,
                     const int32_t* __restrict group_size);
    void cgemm_batch(enum order_t Order, const enum transpose_t* __restrict TransA_array,
                     const enum transpose_t* __restrict TransB_array, const int32_t* __restrict M_array,
                     const int32_t* __restrict N_array, const int32_t* __restrict K_array,
                     const cfloat_t* __restrict alpha_array, const cfloat_t* __restrict* __restrict A_array,
                     const int32_t* __restrict lda_array, const cfloat_t* __restrict* __restrict B_array,
                     const int32_t* __restrict ldb_array, const cfloat_t* __restrict beta_array,
                     cfloat_t* __restrict* __restrict C_array, const int32_t* __restrict ldc_array, int32_t group_count,
                     const int32_t* __restrict group_size);

    void zgemm_batch(enum order_t Order, const enum transpose_t* __restrict TransA_array,
                     const enum transpose_t* __restrict TransB_array, const int32_t* __restrict M_array,
                     const int32_t* __restrict N_array, const int32_t* __restrict K_array,
                     const cdouble_t* __restrict alpha_array, const cdouble_t* __restrict* __restrict A_array,
                     const int32_t* __restrict lda_array, const cdouble_t* __restrict* __restrict B_array,
                     const int32_t* __restrict ldb_array, const cdouble_t* __restrict beta_array,
                     cdouble_t* __restrict* __restrict C_array, const int32_t* __restrict ldc_array,
                     int32_t group_count, const int32_t* __restrict group_size);
    /*
    // BFLOAT16 and INT8 extensions
    // convert float array to BFLOAT16 array by rounding
    void sbstobf16(const int32_t n, const float* in, const int32_t incin, bfloat16* out, const int32_t incout);
    // convert double array to BFLOAT16 array by rounding
    void sbdtobf16(const int32_t n, const double* in, const int32_t incin, bfloat16* out, const int32_t incout);
    // convert BFLOAT16 array to float array
    void sbf16tos(const int32_t n, const bfloat16* in, const int32_t incin, float* out, const int32_t incout);
    // convert BFLOAT16 array to double array
    void dbf16tod(const int32_t n, const bfloat16* in, const int32_t incin, double* out, const int32_t incout);
    // dot production of BFLOAT16 input arrays, and output as float
    float sbdot(const int32_t n, const bfloat16* x, const int32_t incx, const bfloat16* y, const int32_t incy);
    void sbgemv(const enum order_t order, const enum transpose_t trans, const int32_t m, const int32_t n,
                      const float alpha, const bfloat16* a, const int32_t lda, const bfloat16* x, const int32_t incx,
                      const float beta, float* y, const int32_t incy);

    void sbgemm(const enum order_t Order, const enum transpose_t TransA,
                      const enum transpose_t TransB, const int32_t M, const int32_t N, const int32_t K,
                      const float alpha, const bfloat16* A, const int32_t lda, const bfloat16* B, const int32_t ldb,
                      const float beta, float* C, const int32_t ldc);
    void sbgemm_batch(const enum order_t Order, const enum transpose_t* TransA_array,
                            const enum transpose_t* TransB_array, const int32_t* M_array, const int32_t* N_array,
                            const int32_t* K_array,
                            const float* alpha_array, const bfloat16** A_array, const int32_t* lda_array,
                            const bfloat16** B_array, const int32_t* ldb_array, const float* beta_array,
                            float** C_array, const int32_t* ldc_array, const int32_t group_count,
                            const int32_t* group_size);
    */

private:
    std::unique_ptr<Impl> impl;
};


#include "impl_scalar.h"
template class interface<impl_scalar>;
