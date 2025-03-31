//
// Created by saleh on 3/29/25.
//

#include "impl_scalar.h"

float impl_scalar::sdot(int32_t n, const float* x, int32_t incx, const float* y, int32_t incy) {
    float dot_product = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        dot_product += x[i * incx] * y[i * incy];
    }
    return dot_product;
}
