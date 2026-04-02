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
 * \file utils.h
 * \brief
 */

#pragma once

#include <algorithm>
#include <cmath>

typedef unsigned short ushort;//占用2个字节
typedef unsigned int uint;    //占用4个字节

inline uint as_uint(const float x) {
    return *(uint*)&x;
}

inline float as_float(const uint x) {
    return *(float*)&x;
}

inline float half_to_float(const ushort x) { 
    // IEEE-754 16-bit floating-point format (without infinity): 
    // 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x & 0x7C00) >> 10; // exponent
    const uint m = (x & 0x03FF) << 13; // mantissa
    const uint v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) * 
                    ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}

inline ushort float_to_half(const float x) { 
    // IEEE-754 16-bit floating-point format (without infinity): 
    // 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint e = (b & 0x7F800000) >> 23; // exponent
    const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) | ((e < 113) & (e > 101)) * 
            ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

template<typename T>
inline void PrintMatrix(uint8_t* matrix_raw, int rows, int cols) {
    T* mat = reinterpret_cast<T*>(matrix_raw);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float value;
            if constexpr(sizeof(T) == 2) {
                value = half_to_float(mat[i * cols + j]);
            }
            else {
                value = *reinterpret_cast<float*>(&mat[i * cols + j]);
            }
            printf("%d ", static_cast<int>(value));
        }
        printf("\n");
    }
}

template<typename T>
inline void PrintPartOfMatrix(uint8_t* matrix_raw, int rows, int cols, int m, int n) {
    T* mat = reinterpret_cast<T*>(matrix_raw);
    for (int i = 0; i < std::min(rows, m); ++i) {
        for (int j = 0; j < std::min(cols, n); ++j) {
            float value;
            if constexpr(sizeof(T) == 2) {
                value = half_to_float(mat[i * cols + j]);
            }
            else {
                value = *reinterpret_cast<float*>(&mat[i * cols + j]);
            }
            printf("%f ", static_cast<float>(value));
        }
        printf("\n");
    }
}

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

inline void printTensor(const std::complex<float> *tensorData, int64_t batchSize, int64_t rows, int64_t cols)
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

inline bool allclose(const std::complex<float>* a, const std::complex<float>* b, int64_t size, float rtol, float atol)
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