/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file cmatinv_batched_test.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_solver.h"


#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

void printTensor(const std::complex<float> *tensorData, int64_t batchSize, int64_t rows, int64_t cols)
{
    for(int64_t b = 0; b < batchSize; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                std::cout << tensorData[b * rows * cols + i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

bool allclose(const std::complex<float>* a, const std::complex<float>* b, int64_t size, float rtol, float atol)
{
    for (int64_t i = 0; i < size; i++) {
        float diffReal = std::abs(a[i].real() - b[i].real());
        float diffImag = std::abs(a[i].imag() - b[i].imag());
        float absAReal = std::abs(a[i].real());
        float absAImag = std::abs(a[i].imag());
        if (diffReal > atol + rtol * absAReal || diffImag > atol + rtol * absAImag) {
            std::cout << "[Failed] Case accuracy is verification failed!"
                      << std::endl;
            return false;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return true;
}

const float EPSILON = 1e-10f;

bool matrix_inverse(const std::complex<float>* input, std::complex<float>* output, int B, int N) {
    // Create augmented matrix [A | I] of size N x 2N
    // We'll use row-major layout: row i, column j is at aug[i * (2*N) + j]
    const int cols = 2 * N;
    std::complex<float>* aug = new std::complex<float>[N * cols];

    bool all_invertible = true;

    for (int b = 0; b < B; b++) {
        const std::complex<float>* A = input + b * N * N;
        std::complex<float>* inv = output + b * N * N;

        // Initialize augmented matrix [A | I]
        for (int i = 0; i < N; i++) {
            // Left side: copy A
            for (int j = 0; j < N; j++) {
                aug[i * cols + j] = A[i * N + j];
            }
            // Right side: identity
            for (int j = 0; j < N; j++) {
                aug[i * cols + (N + j)] = (i == j) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
            }
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < N; col++) {
            // Find pivot (row with maximum absolute value in current column)
            int pivot_row = col;
            float max_val = std::norm(aug[col * cols + col]);

            for (int row = col + 1; row < N; row++) {
                float val = std::norm(aug[row * cols + col]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = row;
                }
            }

            // Check for singularity
            if (max_val < EPSILON) {
                std::memset(inv, 0, N * N * sizeof(std::complex<float>));
                all_invertible = false;
                goto next_matrix;
            }

            // Swap rows if necessary
            if (pivot_row != col) {
                for (int j = 0; j < cols; j++) {
                    std::swap(aug[col * cols + j], aug[pivot_row * cols + j]);
                }
            }

            // Scale pivot row to make pivot element 1
            std::complex<float> pivot_val = aug[col * cols + col];
            for (int j = 0; j < cols; j++) {
                aug[col * cols + j] /= pivot_val;
            }

            // Eliminate column in all other rows
            for (int row = 0; row < N; row++) {
                if (row != col) {
                    std::complex<float> factor = aug[row * cols + col];
                    for (int j = 0; j < cols; j++) {
                        aug[row * cols + j] -= factor * aug[col * cols + j];
                    }
                }
            }
        }

        // Extract inverse from right half of augmented matrix
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                inv[i * N + j] = aug[i * cols + (N + j)];
            }
        }

next_matrix:
        continue;
    }

    delete[] aug;
    return all_invertible;
}


int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    int64_t batchSize = 2;
    int64_t n = 3;

    int64_t tensorASize = batchSize * n * n;
    std::vector<int64_t> shape = {batchSize, n, n };
    std::vector<std::complex<float>> tensorInAData;
    std::vector<std::complex<float>> tensorInAinvData;
    std::vector<int32_t> tensorInInfoData;
    tensorInAData.resize(tensorASize);
    tensorInAinvData.resize(tensorASize);
    tensorInInfoData.resize(batchSize);

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                if (i == j) {
                    tensorInAData[n * n * batchIdx + n * i + j] = {2.0f + batchIdx, -2.0f - batchIdx};
                } else {
                    tensorInAData[n * n * batchIdx + n * i + j] = {1.0f, -1.0f};
                }
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                tensorInAinvData[n * n * batchIdx + n * i + j] = {-1.0f, -1.0f};
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        tensorInInfoData[batchIdx] = 0;
    }

    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), batchSize, n, n);
    std::cout << "------- input TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    auto ret = aclsolverCmatinvBatched(n, tensorInAData.data(), n, tensorInAinvData.data(), n,tensorInInfoData.data(), batchSize, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCmatinvBatched failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    std::vector<std::complex<float>> goldenData(tensorASize);
    matrix_inverse(tensorInAData.data(), goldenData.data(), batchSize, n);
    std::cout << "------- golden TensorInAinv -------" << std::endl;
    printTensor(goldenData.data(), batchSize, n, n);

    if (!allclose(goldenData.data(), tensorInAinvData.data(), tensorASize, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        printf("judge not equal\n");
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 1;
}