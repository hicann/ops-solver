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
 * \file eye_matrix.h
 * \brief eye (identity) matrix initialization shared by solver host implementations.
 */

#ifndef ACLSOLVER_UTILS_EYE_MATRIX_H
#define ACLSOLVER_UTILS_EYE_MATRIX_H

#include <securec.h>

#include <cstdint>

// eye 矩阵每个元素包含的 float 分量数：实数矩阵为 1，复数矩阵为 2（实部/虚部）
static constexpr size_t EYE_FLOATS_PER_REAL_ELEMENT = 1;
static constexpr size_t EYE_FLOATS_PER_COMPLEX_ELEMENT = 2;

// 初始化 eye 缓冲区为单位阵（对角 1，其余 0），floatsPerElement 为每元素的 float 分量数，
// 初始化失败（memset_s 非 EOK）返回 false
// 初始化 eye 缓冲区为单位阵（对角 1，其余 0），floatsPerElement 为每元素的 float 分量数。
// paddedRows 为缓冲的完整行数（行对齐 + padding 行）：清零范围必须覆盖整个缓冲，
// 否则 padding 行携带宿主未初始化内存被整体 H2D 上设备（issue #136）。
// 初始化失败（memset_s 非 EOK）返回 false
inline bool GenerateEyeMatrix(int64_t N, int64_t strideN, int64_t paddedRows, uint8_t *eyeBuf, size_t floatsPerElement)
{
    auto buf = reinterpret_cast<float *>(eyeBuf);
    const int64_t zeroRows = paddedRows > ((N + 15) / 16 * 16) ? paddedRows : (N + 15) / 16 * 16;
    const size_t eyeSize =
        static_cast<size_t>(zeroRows) * static_cast<size_t>(strideN) * sizeof(float) * floatsPerElement;
    if (memset_s(buf, eyeSize, 0, eyeSize) != EOK)
    {
        return false;
    }
    for (int64_t i = 0; i < N; ++i)
    {
        buf[i * (strideN + 1)] = 1;
    }
    return true;
}

#endif  // ACLSOLVER_UTILS_EYE_MATRIX_H
