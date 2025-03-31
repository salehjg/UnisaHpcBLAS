/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh on 11/5/24.
//

// From `RISCV-NN` repo.

#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <exception>
#include <random>

template <typename T>
T* aligned_alloc_array(size_t size, size_t alignment) {
    // Calculate the total size in bytes, ensuring it’s a multiple of alignment
    size_t total_size = size * sizeof(T);
    if (total_size % alignment != 0) {
        total_size += alignment - (total_size % alignment); // Round up to next multiple
    }
    // Allocate aligned memory
    void* ptr = std::aligned_alloc(alignment, total_size);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Return the pointer as type T*
    return static_cast<T*>(ptr);
}
