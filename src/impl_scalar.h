//
// Created by saleh on 3/29/25.
//

#pragma once

#include <memory>
#include <complex>

class impl_scalar {
public:
    float sdot(int32_t n, const float* __restrict x, int32_t incx, const float* __restrict y, int32_t incy);
};

