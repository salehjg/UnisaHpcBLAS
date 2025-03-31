//
// Created by saleh on 3/30/25.
//

#include <iostream>
#include "timer_scope.h"
#include "timer_stats.h"
#include "aligned_tensor.h"

#include "interface.h"
#include "impl_scalar.h"

#include <cblas.h>

int main() {
    interface<impl_scalar> scalar_blas;
    constexpr int REPS = 100;
    constexpr int N = 1000;
    aligned_tensor<float> a(N, 64);
    aligned_tensor<float> b(N, 64);
    a.initialize(2.0f);
    b.initialize(3.0f);

    timer_stats ts("scalar_blas", {{"N", N}, {"REPS", REPS}});
    float uut1 = 0.0f;
    for (volatile size_t i = 0; i < REPS; ++i) {
        timer_scope tsc(ts);
        uut1 = scalar_blas.sdot(N, a.data_t(), 1, b.data_t(), 1);
    }

    int n = a.sizet();
    int incx = 1;         // Stride for x vector
    int incy = 1;         // Stride for y vector
    float cblas_result = 0.0f;
    timer_stats ts2("cblas", {{"N", N}, {"REPS", REPS}});
    for (volatile size_t i = 0; i < REPS; ++i) {
        timer_scope tsc(ts2);
        cblas_result = cblas_sdot(n, a.data_t(), incx, b.data_t(), incy);
    }


    if (uut1 != cblas_result) {
        std::cerr << "Unit under test does not match cblas result!" << std::endl;
        std::cerr << "uut1: " << uut1 << std::endl;
        std::cerr << "cblas_result: " << cblas_result << std::endl;
        return -1;
    }

}