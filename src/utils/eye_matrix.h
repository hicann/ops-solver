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

// 初始化 eye 缓冲区首个"单 float 平面"为单位阵（对角 1，其余 0）。
//
// floatsPerElement 仅用于计算清零字节数（本函数写入的平面所在的整体缓冲按
// fpe 放大，fpe=1 实数 / fpe=2 复数平面分离布局），**不参与对角偏移**：
// 本函数产出的平面是单 float 布局（行宽 strideN 个 float），与 kernel 消费
// 口径一致——trsm/restore 均按"实部平面 + 偏移 M*strideN 的虚部平面"两个
// 独立单 float 平面访问（trsm_custom.hpp: lGlobalImag = lGlobalReal[M*strideN]），
// 不存在交错复数布局的调用方；若未来引入交错布局，须另行实现对角偏移乘
// fpe 的变体，勿直接复用本函数（issue #156）。
// paddedRows 为缓冲的完整行数（行对齐 + padding 行）：清零范围覆盖 zeroRows
// 行 × fpe，与调用方传入的缓冲大小一致。
// 初始化失败（memset_s 非 EOK）返回 false。
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
