//
// Created by saleh on 3/27/25.
//

#include <iostream>
#include <vector>
#include <random>
#include <cblas.h>
#include <chrono>

#include "common.h"

// Function to generate random matrix
std::vector<double> generate_random_matrix(int rows, int cols) {
    // Use current time as seed for random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // Normal distribution with mean 0 and standard deviation 1
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Create and fill matrix
    std::vector<double> matrix(rows * cols);
    for (auto& elem : matrix) {
        elem = distribution(generator);
    }

    return matrix;
}

int main() {
    // Large matrix size (NxN)
    const int N = 2048;

    // Generate random matrices
    auto A = generate_random_matrix(N, N);
    auto B = generate_random_matrix(N, N);

    // Output matrix (initialized to zero)
    std::vector<double> C(N * N, 0.0);

    // GEMM parameters
    const double alpha = 1.0;  // Scaling factor for A * B
    const double beta = 0.0;   // Scaling factor for C

    // Time the matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication C = alpha * A * B + beta * C
    cblas_dgemm(
        CblasRowMajor,   // Matrix storage order
        CblasNoTrans,    // Transpose A?
        CblasNoTrans,    // Transpose B?
        N,               // Number of rows in A and C
        N,               // Number of columns in B and C
        N,               // Number of columns in A / rows in B
        alpha,           // Scalar alpha
        A.data(),        // Matrix A
        N,               // Leading dimension of A
        B.data(),        // Matrix B
        N,               // Leading dimension of B
        beta,            // Scalar beta
        C.data(),        // Matrix C
        N                // Leading dimension of C
    );

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;

    // Print timing information
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Computation time: " << diff.count() << " seconds" << std::endl;

    // Optionally print a few elements to verify computation
    std::cout << "Sample results (first 3x3 elements):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}