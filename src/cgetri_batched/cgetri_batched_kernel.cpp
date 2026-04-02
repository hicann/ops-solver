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
 * \file cgetri_batched_kernel.cpp
 * \brief
 */

#ifndef CGETRI_BATCHED_MIX_H
#define CGETRI_BATCHED_MIX_H

#include "../utils/kernel/c64/getri_custom.hpp"

static constexpr uint32_t OFFSET_TWO = 2;
static constexpr uint32_t OFFSET_THREE = 3;
static constexpr uint32_t OFFSET_FOUR = 4;
static constexpr uint32_t OFFSET_FIVE = 5;

static constexpr int64_t EYE_MATRIX_NUM = 3;
static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t TILE_ELENUM = 512;
static constexpr int64_t COMPLEX_ELENUM = 2;

__global__ __aicore__ void cgetri_batched_kernel(
    GM_ADDR sync, GM_ADDR A_org, GM_ADDR W, GM_ADDR gather1_gm,
    GM_ADDR gather2_gm, GM_ADDR gather3_gm, GM_ADDR eye_gm, GM_ADDR A_work,
    GM_ADDR work_gm, GM_ADDR A_inv, GM_ADDR workspace, GM_ADDR tiling_gm) {

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr((unsigned long)sync);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);
    int64_t dtype = static_cast<int64_t>(*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf)); // 0-c32; 1-c64
    int64_t n = static_cast<int64_t>(*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + sizeof(uint32_t)));

    int64_t batchSize = static_cast<int64_t>(*(__gm__ uint32_t *)
        ((__gm__ uint8_t *)tiling_buf + OFFSET_TWO * sizeof(uint32_t)));
    int64_t blockM = static_cast<int64_t>(*(__gm__ uint32_t *)
        ((__gm__ uint8_t *)tiling_buf + OFFSET_THREE * sizeof(uint32_t)));
    int64_t blockN = static_cast<int64_t>(*(__gm__ uint32_t *)
        ((__gm__ uint8_t *)tiling_buf + OFFSET_FOUR * sizeof(uint32_t)));
    int64_t tileM = static_cast<int64_t>(*(__gm__ uint32_t *)
        ((__gm__ uint8_t *)tiling_buf + OFFSET_FIVE * sizeof(uint32_t)));

    int64_t orgM = n;
    int64_t orgN = n;

    int64_t minMN = orgM > orgN ? orgN : orgM;

    int64_t wEleNum = (minMN + blockN - 1) / blockN * blockN;
    int64_t eyeMatEleNum = ((orgN + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM *
        COL_ALIGNED_ELENUM) * ((orgM + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM *
        BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;

    int64_t dtypeSize = dtype > 0 ? sizeof(float) : sizeof(half);

    for (int64_t bid = 0; bid < batchSize; bid++)
    {
        GM_ADDR cur_W = W + bid * wEleNum * sizeof(uint32_t);
        GM_ADDR cur_eye_gm = eye_gm + bid * eyeMatEleNum * sizeof(float);
        GM_ADDR cur_A_org = A_org + bid * orgM * orgN * COMPLEX_ELENUM * dtypeSize;
        GM_ADDR cur_A_inv = A_inv + bid * orgM * orgN * COMPLEX_ELENUM * dtypeSize;

        custom_getri(dtype, orgM, orgN, blockM, blockN, tileM, cur_A_org, cur_A_inv,
            A_work, cur_W, work_gm, gather1_gm, gather2_gm, gather3_gm, cur_eye_gm, workspace);
    }
}

void cgetri_batched_kernel_do(GM_ADDR sync, GM_ADDR A_org, GM_ADDR W,
                              GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                              GM_ADDR gather3_gm, GM_ADDR eye_gm,
                              GM_ADDR A_work, GM_ADDR work_gm, GM_ADDR A_inv,
                              GM_ADDR workspace, GM_ADDR tiling_gm,
                              uint32_t numBlocks, void *stream) {
    cgetri_batched_kernel<<<numBlocks, nullptr, stream>>>(
        sync, A_org, W, gather1_gm, gather2_gm, gather3_gm, eye_gm, A_work,
        work_gm, A_inv, workspace, tiling_gm);
}
#endif  // CGETRI_BATCHED_MIX_H