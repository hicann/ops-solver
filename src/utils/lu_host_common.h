/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSOLVER_UTILS_LU_HOST_COMMON_H
#define ACLSOLVER_UTILS_LU_HOST_COMMON_H

#include <cstdint>

#include "eye_matrix.h"

// ================= LU 系 host 共享常量（issue #143） =================
// 口径说明：列方向按 COL_ALIGNED(128) 对齐；行方向按 ROW_ALIGNED(16) 对齐后再追加
// EYE_ROW_PADDING(128) 行 padding（供 kernel 的 tile 换块过渡区使用，须初始化清零，issue #136）。

constexpr int64_t LU_BLOCK_N = 16;          // kernel 消元子块列宽
constexpr int64_t LU_TILE_M = 512;          // kernel tile 行高（gather1/gather2 重排表按 tileM*blockN 布局）
constexpr int64_t LU_ROW_ALIGNED = 16;      // 行对齐粒度
constexpr int64_t LU_COL_ALIGNED = 128;     // strideN 列对齐粒度
constexpr int64_t EYE_ROW_PADDING = 128;    // eye/工作区行 padding 高度（行对齐之上追加）
constexpr uint32_t LU_MAX_NUM_BLOCKS = 20;  // kernel 按固定 8+12 分块设计，block 数上限 20

// gather3（实虚平面合并重排表）的工作 tile 逻辑长度：表项按 (src, src+4096) 偏移对组织
constexpr int64_t GATHER3_TILE_LENGTH = 4096;
// kernel MergeRealImag 按 8 元素对齐消费 gather3（alignedN = ceil8(n)，issue #137）
constexpr int64_t GATHER3_ALIGN = 8;

// 行/列向上对齐（对齐粒度必须为正）
inline int64_t LuAlignUp(int64_t v, int64_t align) { return (v + align - 1) / align * align; }

// ================= gather 重排表共享生成（issue #144 收敛五处复制） =================
//
// gather1/gather2：tileM*blockN 块内的列/行主序倒序重排偏移（单位：字节），sgetrf/cgetrf/
// sgetri/cgetri 四算子共用，语义与各文件原逐字复制的实现完全一致。
// gather3：复数实/虚平面合并偏移表。kernel 侧按 alignedN = ceil8(n) 的行主序消费
// linesPerIter*alignedN 对偏移（src/utils/kernel/c64/pad.hpp 的 SetMatrix/LoadGather），
// host 生成必须按同一对齐口径展开，否则 n 非 8 倍数时 kernel 越界读取未写入表项（issue #137）。
//
// 返回 false 仅在 eye 初始化失败（memset_s 非 EOK）时发生；偏移表生成为纯整数运算，无失败路径。

inline bool GenerateGatherTables(int64_t tileM, int64_t blockN, int64_t n, int64_t strideN, int64_t paddedRows,
                                 uint8_t *gatherBuf1, uint8_t *gatherBuf2, uint8_t *gatherBuf3, uint8_t *eyeBuf,
                                 size_t floatsPerElement)
{
    {
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf1);
        int idx = 0;
        for (int64_t j = 0; j < blockN; ++j)
            for (int64_t i = tileM - 1; i >= 0; --i) buf[idx++] = static_cast<uint32_t>(i * blockN + j) * sizeof(float);
    }
    {
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf2);
        int idx = 0;
        for (int64_t i = tileM - 1; i >= 0; --i)
            for (int64_t j = 0; j < blockN; ++j) buf[idx++] = static_cast<uint32_t>(j * tileM + i) * sizeof(float);
    }
    if (gatherBuf3 != nullptr)
    {
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf3);
        int idx = 0;
        if (n > 0)
        {
            // 与 kernel 消费口径一致：按 alignedN = ceil8(n) 展开偏移对；k 超出 n 的表项
            // 落在列 padding 区，指向的有效数据由 kernel 的列裁剪保证不被消费为输出（issue #137）
            const int64_t alignedN = (n + GATHER3_ALIGN - 1) / GATHER3_ALIGN * GATHER3_ALIGN;
            const int64_t effN = alignedN < GATHER3_TILE_LENGTH ? alignedN : GATHER3_TILE_LENGTH;
            const int64_t m = GATHER3_TILE_LENGTH / effN;
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < effN; ++j)
                {
                    buf[idx++] = static_cast<uint32_t>(i * effN + j) * sizeof(float);
                    buf[idx++] = static_cast<uint32_t>(i * effN + j + GATHER3_TILE_LENGTH) * sizeof(float);
                }
        }
    }
    // 无 eye 缓冲的调用方（sgetrf/cgetrf 的 gather-only 场景）：跳过 eye 初始化
    if (eyeBuf == nullptr)
    {
        return true;
    }
    return GenerateEyeMatrix(n, strideN, paddedRows, eyeBuf, floatsPerElement);
}

#endif  // ACLSOLVER_UTILS_LU_HOST_COMMON_H
