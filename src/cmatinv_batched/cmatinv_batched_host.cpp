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
 * \file cmatinv_batched_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <iterator>
#include <securec.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "cann_ops_solver.h"
#include "../utils/assert.h"


#define GM_ADDR uint8_t*

extern void cmatinv_batched_kernel_do(GM_ADDR dA, GM_ADDR dUniReal,
                                      GM_ADDR dUniImag, GM_ADDR dOffset,
                                      GM_ADDR dAinv, GM_ADDR workSpace,
                                      GM_ADDR tilingGm, uint32_t numBlocks,
                                      void *stream);

static constexpr uint32_t COMPLEX_ELENUM = 2;
static constexpr uint32_t MAX_CORE_CNT = 40;
static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_COMPLEX32 = 0;
static constexpr int64_t DTYPE_COMPLEX64 = 1;

static constexpr int ELEMENT_NUM_PER_BLOCK = 8;
static constexpr int ELEMENT_NUM_PER_REPEAT = 64;
static constexpr uint32_t BASE_COMPLEX64_NUM = 8192; // 64kb / 8b
static constexpr uint32_t MAX_COMPUTE_NUM = 128; // max compute num per repeat

struct CmatinvBatchedTilingData {
    uint32_t dtype;
    uint32_t n;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

CmatinvBatchedTilingData CalTilingData(uint32_t vecCoreNum, uint32_t dtype, uint32_t n, uint32_t batchSize)
{
    CmatinvBatchedTilingData tilingData;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;
    vecCoreNum = (vecCoreNum == 0) ? 1 : vecCoreNum;

    // init tiling data
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t matPerCore = batchSize / vecCoreNum;
    uint32_t remainMatNum = batchSize % vecCoreNum;
    if (matPerCore == 0) {
        for (uint32_t i = 0; i < remainMatNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i * n * n * COMPLEX_ELENUM; // complex element num
        }
    } else {
        uint32_t currComputeNum;
        uint32_t currOffset = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainMatNum) {
                currComputeNum = matPerCore + 1;
            } else {
                currComputeNum = matPerCore;
            }
            tilingData.calNum[i] = currComputeNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currComputeNum * n * n * COMPLEX_ELENUM; // complex element num
        }
    }
    tilingData.dtype = dtype;
    tilingData.n = n;

    return tilingData;
}

aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A,
                                 const int64_t lda, std::complex<float> *Ainv,
                                 const int64_t lda_inv, int32_t *info,
                                 int64_t batchSize) {
    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        aclsolverGetStream(handle, &stream);
    }

    if (n > MATRIX_SHAPE_LIMIT) {
        LOG_PRINT("CmatinvBatched only supports n ≤ 32. For n > 32, use CgetriBatched instead.\n");
        return aclsolverCgetriBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    }
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr) {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    if (lda <= 0 || lda_inv <= 0) {
        LOG_PRINT("CmatinvBatched get lda <= 0 or lda_inv <= 0.\n");
    }
    SOLVER_ECHECK(n > 0 && batchSize > 0, "CmatinvBatched get n <= 0 || batchSize <= 0.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "CmatinvBatched get n <= 256 || batchSize <= 3000.",
                  ACL_ERROR_INVALID_PARAM);

    uint32_t N = static_cast<uint32_t>(n);
    uint32_t alignedN = (N + (ELEMENT_NUM_PER_BLOCK - 1)) / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    uint32_t computeNumPerRepeat = BASE_COMPLEX64_NUM / (N * alignedN);
    computeNumPerRepeat = std::min(computeNumPerRepeat, MAX_COMPUTE_NUM);
    computeNumPerRepeat = computeNumPerRepeat / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;
    std::vector<float> uniBatchRealData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatRealData(N * alignedN, 0.0f);
    std::vector<float> uniBatchImagData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatImagData(N * alignedN, 0.0f);
    for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
        for (uint32_t colIdx = 0; colIdx < alignedN; colIdx++) {
            if (rowIdx == colIdx) {
                uniMatRealData[alignedN * rowIdx + colIdx] = 1.0f;
            } else {
                uniMatRealData[alignedN * rowIdx + colIdx] = 0.0f;
            }
            uniMatImagData[alignedN * rowIdx + colIdx] = 0.0f;
        }
    }
    errno_t rc;
    for (uint32_t i = 0; i < computeNumPerRepeat; i++) {
        rc = memcpy_s(static_cast<float *>(uniBatchRealData.data() + N * alignedN * i),
            N * alignedN * sizeof(float), uniMatRealData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatRealData by memcpy_s failed.", ACL_ERROR_INTERNAL_ERROR);
        rc = memcpy_s(static_cast<float *>(uniBatchImagData.data() + N * alignedN * i),
            N * alignedN * sizeof(float), uniMatImagData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatImagData by memcpy_s failed.",ACL_ERROR_INTERNAL_ERROR);
    }

    uint32_t offsetSize = N * alignedN * computeNumPerRepeat * 2;
    offsetSize = (offsetSize + (ELEMENT_NUM_PER_REPEAT - 1)) / ELEMENT_NUM_PER_REPEAT * ELEMENT_NUM_PER_REPEAT;
    std::vector<uint32_t> offsetData(offsetSize, 0);
    uint32_t k = 0;
    uint32_t realBase = 0;
    uint32_t imagBase = 32 * 1024;
    for (uint32_t batchIdx = 0; batchIdx < computeNumPerRepeat; batchIdx++) {
        for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
            for (uint32_t colIdx = 0; colIdx < N; colIdx++) {
                offsetData[k++] = realBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
                offsetData[k++] = imagBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
            }
        }
    }

    uint8_t *hostA = reinterpret_cast<uint8_t *>(A);
    uint8_t *hostAinv = reinterpret_cast<uint8_t *>(Ainv);
    uint8_t *hostinfo = reinterpret_cast<uint8_t *>(info);
    uint8_t *hostUniReal = reinterpret_cast<uint8_t *>(uniBatchRealData.data());
    uint8_t *hostUniImag = reinterpret_cast<uint8_t *>(uniBatchImagData.data());
    uint8_t *hostOffset = reinterpret_cast<uint8_t *>(offsetData.data());
    auto sizeA = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeAinv = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeinfo = batchSize * sizeof(int32_t);
    auto sizeUniReal = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeUniImag = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeOffset = offsetSize * sizeof(uint32_t);

    uint8_t *d_A = nullptr;
    uint8_t *d_Ainv = nullptr;
    uint8_t *d_info = nullptr;
    uint8_t *d_UniReal = nullptr;
    uint8_t *d_UniImag = nullptr;
    uint8_t *d_Offset = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_Ainv, sizeAinv, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_info, sizeinfo, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniReal, sizeUniReal, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniImag, sizeUniImag, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_Offset, sizeOffset, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(d_A, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_Ainv, sizeAinv, hostAinv, sizeAinv, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_info, sizeinfo, hostinfo, sizeinfo, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_UniReal, sizeUniReal, hostUniReal, sizeUniReal, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_UniImag, sizeUniImag, hostUniImag, sizeUniImag, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_Offset, sizeOffset, hostOffset, sizeOffset, ACL_MEMCPY_HOST_TO_DEVICE));

    CmatinvBatchedTilingData tiling = CalTilingData(numBlocks, DTYPE_COMPLEX64, n, batchSize);
    uint8_t *tilingDevice = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&tilingDevice, sizeof(CmatinvBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CmatinvBatchedTilingData),
                            &tiling, sizeof(CmatinvBatchedTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *workSpace = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&workSpace, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));

    cmatinv_batched_kernel_do(d_A, d_UniReal, d_UniImag, d_Offset, d_Ainv, workSpace, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    CHECK_ACLRT(aclrtMemcpy(hostAinv, sizeAinv, d_Ainv, sizeAinv, ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(d_A);
    aclrtFree(d_UniReal);
    aclrtFree(d_UniImag);
    aclrtFree(d_Offset);
    aclrtFree(d_Ainv);
    aclrtFree(workSpace);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
