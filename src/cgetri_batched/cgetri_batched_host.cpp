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
 * \file cgetri_batched_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <securec.h>
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "../utils/assert.h"

#include <complex>
#include "tiling/platform/platform_ascendc.h"


#define GM_ADDR uint8_t*

extern void cgetri_batched_kernel_do(GM_ADDR sync, GM_ADDR A_org, GM_ADDR W,
                                     GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                                     GM_ADDR gather3_gm, GM_ADDR eye_gm,
                                     GM_ADDR A_work, GM_ADDR work_gm,
                                     GM_ADDR A_inv, GM_ADDR workspace,
                                     GM_ADDR tiling_gm, uint32_t numBlocks,
                                     void *stream);

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_COMPLEX32 = 0;
static constexpr int64_t DTYPE_COMPLEX64 = 1;


static constexpr int64_t EYE_MATRIX_NUM = 3;
static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t TILE_ELENUM = 512;
static constexpr int64_t TILE_LENGTH = 4096;
static constexpr int64_t COMPLEX_ELENUM = 2;

static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CgetriBatchedTilingData {
    uint32_t dtype;
    uint32_t n;
    uint32_t batchSize;
    uint32_t blockM;
    uint32_t blockN;
    uint32_t tileM;
};

aclError aclsolverCgetriBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A,
                                        const int64_t lda,
                                        std::complex<float> *Ainv,
                                        const int64_t lda_inv, int32_t *info,
                                        int64_t batchSize) {
    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        aclsolverGetStream(handle, &stream);
    }
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr) {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    if (lda <= 0 || lda_inv <= 0) {
        LOG_PRINT("CgetriBatched get lda <= 0 or lda_inv <= 0.\n");
    }
    SOLVER_ECHECK(n > 0 && batchSize > 0,
                  "CgetriBatched get n <= 0 || batchSize <= 0.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "CgetriBatched get n <= 256 || batchSize <= 3000.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n > MATRIX_SHAPE_LIMIT,
                  "CgetriBatched only supports n > 32. For n ≤ 32, use "
                  "CmatinvBatched instead.",
                  ACL_ERROR_INVALID_PARAM);

    int64_t M = n;
    int64_t N = n;
    int64_t batchNum = batchSize;
    int64_t tileM = TILE_ELENUM;
    int64_t blockM = tileM;
    int64_t blockN = BASE_BLOCK_ELENUM;

    // 1、 gen batch w data
    int64_t t = (std::min(M, N) + blockN - 1) / blockN * blockN;
    int64_t wEleNum = batchNum * t;
    std::vector<uint32_t> wBatchData(wEleNum, 0);
    memset_s(wBatchData.data(), wEleNum * sizeof(uint32_t), -1, wEleNum * sizeof(uint32_t));

    // 2、gen work data
    int64_t aMatWorkEleNum = ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM *
        COL_ALIGNED_ELENUM) * ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM *
        BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;
    int64_t workM = (M + tileM - 1) / tileM * tileM;
    int64_t workEleNum = workM * blockN * COMPLEX_ELENUM;
    std::vector<float> aWorkData(aMatWorkEleNum, 0.0f);
    std::vector<float> workGmData(workEleNum, 0.0f);

    // 3、generate offset and eye mat data
    int64_t gatherEleNum = tileM * blockN;
    int64_t eyeMatEleNum = ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM) *
        ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;

    std::vector<uint32_t> gatherOffset1(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset2(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset3(gatherEleNum, 0);
    std::vector<float> eyeBatchMatData(batchNum * eyeMatEleNum, 0.0f);

    int64_t strideN = (N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM;

    int64_t idx1 = 0;
    for (int64_t j = 0; j < blockN; ++j) {
        for (int64_t i = blockM - 1; i >= 0; --i) {
            gatherOffset1[idx1++] = static_cast<uint32_t>((i * blockN + j) * sizeof(float));
        }
    }
    int64_t idx2 = 0;
    for (int64_t i = blockM - 1; i >= 0; --i) {
        for (int64_t j = 0; j < blockN; ++j) {
            gatherOffset2[idx2++] = static_cast<uint32_t>((j * blockM + i) * sizeof(float));
        }
    }
    int64_t idx3 = 0;
    int64_t nn = std::min(N, TILE_LENGTH);
    int64_t mm = TILE_LENGTH / nn;
    for (int64_t i = 0; i < mm; ++i) {
        for (int64_t j = 0; j < nn; ++j) {
            gatherOffset3[idx3++] = static_cast<uint32_t>((i * nn + j) * sizeof(float));
            gatherOffset3[idx3++] = static_cast<uint32_t>((i * nn + j + TILE_LENGTH) * sizeof(float));
        }
    }
    for (int64_t b = 0; b < batchNum; ++b) {
        for (int64_t i = 0; i < N; ++i) {
            eyeBatchMatData[b * eyeMatEleNum + i * (strideN + 1)] = 1.0f;
        }
    }

    uint8_t* host_wBatch = reinterpret_cast<uint8_t*>(wBatchData.data());
    uint8_t* host_aWork = reinterpret_cast<uint8_t*>(aWorkData.data());
    uint8_t* host_workGm = reinterpret_cast<uint8_t*>(workGmData.data());
    uint8_t* host_gather1Offset = reinterpret_cast<uint8_t*>(gatherOffset1.data());
    uint8_t* host_gather2Offset = reinterpret_cast<uint8_t*>(gatherOffset2.data());
    uint8_t* host_gather3Offset = reinterpret_cast<uint8_t*>(gatherOffset3.data());
    uint8_t* host_eyeBatchMat = reinterpret_cast<uint8_t*>(eyeBatchMatData.data());

    size_t size_wBatch = wEleNum * sizeof(uint32_t);
    size_t size_aWork = aMatWorkEleNum * sizeof(float);
    size_t size_workGm = workEleNum * sizeof(float);
    size_t size_gather1Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_gather2Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_gather3Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_eyeBatchMat = batchNum * eyeMatEleNum * sizeof(float);

    uint8_t* d_wBatch = nullptr;
    uint8_t* d_aWork = nullptr;
    uint8_t* d_workGm = nullptr;
    uint8_t* d_gather1Offset = nullptr;
    uint8_t* d_gather2Offset = nullptr;
    uint8_t* d_gather3Offset = nullptr;
    uint8_t* d_eyeBatchMat = nullptr;

    CHECK_ACLRT(aclrtMalloc((void**)&d_wBatch, size_wBatch, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_aWork, size_aWork, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_workGm, size_workGm, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_gather1Offset, size_gather1Offset, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_gather2Offset, size_gather2Offset, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_gather3Offset, size_gather3Offset, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&d_eyeBatchMat, size_eyeBatchMat, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACLRT(aclrtMemcpy(d_wBatch, size_wBatch, host_wBatch, size_wBatch, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_aWork, size_aWork, host_aWork, size_aWork, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_workGm, size_workGm, host_workGm, size_workGm, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(
        d_gather1Offset, size_gather1Offset, host_gather1Offset, size_gather1Offset, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(
        d_gather2Offset, size_gather2Offset, host_gather2Offset, size_gather2Offset, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(
        d_gather3Offset, size_gather3Offset, host_gather3Offset, size_gather3Offset, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(
        d_eyeBatchMat, size_eyeBatchMat, host_eyeBatchMat, size_eyeBatchMat, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *hostA = reinterpret_cast<uint8_t *>(A);
    uint8_t *hostAinv = reinterpret_cast<uint8_t *>(Ainv);
    auto sizeA = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeAinv = batchSize * n * n * sizeof(std::complex<float>);
    uint8_t *d_A= nullptr;
    uint8_t *d_Ainv = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_Ainv, sizeAinv, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(d_A, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_Ainv, sizeAinv, hostAinv, sizeAinv, ACL_MEMCPY_HOST_TO_DEVICE));

    CgetriBatchedTilingData tilingData;
    tilingData.dtype = DTYPE_COMPLEX64;
    tilingData.n = n;
    tilingData.batchSize = batchSize;
    tilingData.blockM = BASE_BLOCK_ELENUM;
    tilingData.blockN = BASE_BLOCK_ELENUM;
    tilingData.tileM = TILE_ELENUM;
    uint8_t *tilingDevice = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&tilingDevice, sizeof(CgetriBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CgetriBatchedTilingData),
                            &tilingData, sizeof(CgetriBatchedTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *workSpace = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&workSpace, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *sync = nullptr;
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync));

    cgetri_batched_kernel_do(sync, d_A, d_wBatch, d_gather1Offset,
                             d_gather2Offset, d_gather3Offset, d_eyeBatchMat,
                             d_aWork, d_workGm, d_Ainv, workSpace, tilingDevice,
                             numBlocks, stream);
    aclrtSynchronizeStream(stream);

    CHECK_ACLRT(aclrtMemcpy(hostAinv, sizeAinv, d_Ainv, sizeAinv, ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(d_A);
    aclrtFree(d_Ainv);
    aclrtFree(d_wBatch);
    aclrtFree(d_gather1Offset);
    aclrtFree(d_gather2Offset);
    aclrtFree(d_gather3Offset);
    aclrtFree(d_eyeBatchMat);
    aclrtFree(d_aWork);
    aclrtFree(d_workGm);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}