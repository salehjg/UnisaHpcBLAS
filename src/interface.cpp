//
// Created by saleh on 3/29/25.
//

#include "interface.h"

template <typename Impl>
interface<Impl>::interface() : impl(std::make_unique<Impl>()) {}


template <typename Impl>
float interface<Impl>::sdot(int32_t n, const float* x, int32_t incx, const float* y, int32_t incy) {
    return impl->sdot(n, x, incx, y, incy);
}

