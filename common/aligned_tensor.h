//
// Created by saleh on 3/30/25.
//

#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

template <typename T>
class aligned_tensor {
    // Type punning is a undefined behavior in C++ when its done using unions.
    // The only exception is to use char as the dest type.
    // C++20 introduced std::bit_cast to do type punning in a safe way when possible.
    // For older versions of C++, we can use `std::memcpy` to do type punning.
    // Avoiding type punning is the best practice.

    size_t _size, _size_bytes;
    const int _alignment;
    std::vector<size_t> _shape;
    T* _data = nullptr;

protected:
    void infer_size_and_alloc() {
        if (_shape.empty())
            throw std::runtime_error("Shape not set");
        _size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
        _size_bytes = _size * sizeof(T);
        if (_size_bytes % _alignment != 0) {
            _size_bytes += _alignment - (_size_bytes % _alignment); // Round up to next multiple
        }
        void* ptr = std::aligned_alloc(_alignment, _size_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }

        if (_data) {
            std::free(_data);
        }
        _data = static_cast<T*>(ptr);
    }

public:
    enum class init_type {
        random,
        twos
    };

    ~aligned_tensor() {
        if (_data) {
            std::free(_data);
        }
    }

    aligned_tensor(size_t words, int alignment) : _alignment(alignment) {
        _shape.push_back(words);
        infer_size_and_alloc();
    }

    aligned_tensor(const std::vector<size_t>& shape, int alignment) : _alignment(alignment) {
        _shape = shape;
        infer_size_and_alloc();
    }

    /**
     * @return Number of elements of type T in the tensor
     */
    size_t sizet() {
        return _size;
    }

    /**
     * @return Number of elements of type uint16_t in the tensor
     */
    size_t sizeu16() {
        return _size * 2;
    }

    /**
     * @return Number of elements of type uint8_t in the tensor
     */
    size_t sizeu8() {
        return _size * 4;
    }

    /**
     * @return The size of the aligned buffer in bytes with padding.
     */
    size_t size_bytes() {
        return _size_bytes;
    }

    std::vector<size_t> shape() {
        return _shape;
    }

    /**
     * @return Pointer to the aligned buffer of type T.
     */
    T* data_t() {
        return _data;
    }

    /**
     * @return Pointer to the aligned buffer of type uint16_t.
     */
    uint16_t* data_u16() {
        throw std::runtime_error("Not implemented");
    }

    /**
     * @return Pointer to the aligned buffer of type uint8_t.
     */
    uint8_t* data_u8() {
        throw std::runtime_error("Not implemented");
    }

    void wipe() {
        for (size_t i = 0; i < _size; i++) {
            _data[i] = 0;
        }
    }

    void initialize(T val) {
        for (size_t i = 0; i < _size; i++) {
            _data[i] = val;
        }
    }

    void initialize(T* vals, size_t len) {
        if (len != _size) {
            throw std::runtime_error("Invalid length");
        }
        for (size_t i = 0; i < _size; i++) {
            _data[i] = vals[i];
        }
    }

    void initialize(std::vector<T>& vals) {
        if (vals.size() != _size) {
            throw std::runtime_error("Invalid length");
        }
        for (size_t i = 0; i < _size; i++) {
            _data[i] = vals[i];
        }
    }

    void initialize(init_type type, T lower_bound, T upper_bound) {
        if (type == init_type::random) {
            // Seed the random number generator with the current time
            std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
            if constexpr (std::is_floating_point<T>::value) {
                std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
                for (size_t i = 0; i < _size; ++i) {
                    _data[i] = dist(rng);
                }
            } else if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(lower_bound, upper_bound);
                for (size_t i = 0; i < _size; ++i) {
                    _data[i] = dist(rng);
                }
            } else {
                throw std::runtime_error("Unsupported type");
            }
            return;
        }
        if (type == init_type::twos) {
          std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
            std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
            for (size_t i = 0; i < _size; ++i) {
                _data[i] = std::pow(2, std::floor(dist(rng)));
            }
            return;
        }
        throw std::runtime_error("Unsupported init type.");
    }

protected:
    void compute_indices(size_t flattened_idx, std::vector<size_t>& indices, const std::vector<size_t>& shape) {
        for (int axis = shape.size() - 1; axis >= 0; --axis) {
            indices[axis] = flattened_idx % shape[axis];
            flattened_idx /= shape[axis];
        }
    }

public:
    void compare(aligned_tensor<T>& other) {
        size_t total_size = 1;
        for (const auto& dim : _shape) {
            total_size *= dim;
        }

        auto *p2 = other.data_t();

        std::vector<size_t> indices(_shape.size());
        for (size_t flattened_idx = 0; flattened_idx < total_size; ++flattened_idx) {
            compute_indices(flattened_idx, indices, _shape);

            if (_data[flattened_idx] != p2[flattened_idx]) {
                std::cout << "Results mismatch at index ";
                for (size_t i = 0; i < indices.size(); ++i) {
                    std::cout << indices[i] << (i + 1 == indices.size() ? "" : ", ");
                }
                std::cout << std::endl;
                std::cout << "_data1 = " << _data[flattened_idx] << ", _data2 = " << p2[flattened_idx] << std::endl;
                std::cout << "Difference = " << _data[flattened_idx] - p2[flattened_idx] << std::endl;
                return;
            }
        }
        std::cout << "Results match!" << std::endl;
    }
};
