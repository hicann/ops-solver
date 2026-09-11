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
 * \file sgetri_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/assert.h"
#include "../utils/eye_matrix.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "tiling/platform/platform_ascendc.h"

#define GM_ADDR uint8_t *

extern void sgetri_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                             GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                             GM_ADDR eye_gm, uint32_t numBlocks, void *stream);
bool GenerateTiling(int blockM, int blockN, int64_t N, int64_t strideN, uint8_t *gatherBuf1, uint8_t *gatherBuf2,
                    uint8_t *eyeBuf)
{
    try
    {
        {
            auto buf = reinterpret_cast<uint32_t *>(gatherBuf1);
            int idx = 0;
            for (int j = 0; j < blockN; ++j)
                for (int i = blockM - 1; i >= 0; --i) buf[idx++] = (i * blockN + j) * sizeof(float);
        }
        {
            auto buf = reinterpret_cast<uint32_t *>(gatherBuf2);
            int idx = 0;
            for (int i = blockM - 1; i >= 0; --i)
                for (int j = 0; j < blockN; ++j) buf[idx++] = (j * blockM + i) * sizeof(float);
        }
        if (!GenerateEyeMatrix(N, strideN, eyeBuf, EYE_FLOATS_PER_REAL_ELEMENT))
        {
            return false;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}

aclError aclsolverSgetri(aclsolverHandle_t handle, const int64_t n, float *A, const int64_t lda, int32_t *info)
{
    SOLVER_ECHECK(n > 0 && lda > 0 && A != nullptr && info != nullptr,
                  "aclsolverSgetri invalid param: n, lda <= 0, or A, info is nullptr.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= INT32_MAX, "aclsolverSgetri invalid param: n exceeds int32 range.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n * n <= INT32_MAX, "aclsolverSgetri invalid param: n * n exceeds INT32_MAX elements.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(
        lda == n,
        "aclsolverSgetri only supports lda == n in current version, matrix A must be stored contiguously as n * n.",
        ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

    int32_t deviceId = 0;
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wHost = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *eyeMatrixHost = nullptr;
    uint8_t *eyeMatrixDevice = nullptr;
    uint8_t *gatherHost1 = nullptr, *gatherHost2 = nullptr;
    uint8_t *gatherDevice1 = nullptr, *gatherDevice2 = nullptr;
    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (aMatrixDeviceWork) aclrtFree(aMatrixDeviceWork);
        if (eyeMatrixDevice) aclrtFree(eyeMatrixDevice);
        if (eyeMatrixHost) aclrtFreeHost(eyeMatrixHost);
        if (wDevice) aclrtFree(wDevice);
        if (wHost) aclrtFreeHost(wHost);
        if (workDevice) aclrtFree(workDevice);
        if (gatherDevice1) aclrtFree(gatherDevice1);
        if (gatherDevice2) aclrtFree(gatherDevice2);
        if (gatherHost1) aclrtFreeHost(gatherHost1);
        if (gatherHost2) aclrtFreeHost(gatherHost2);
    };

    CHECK_ACLRT(aclrtGetDevice(&deviceId), cleanup());
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
    {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    if (numBlocks > 20)
    {
        numBlocks = 20;
    }
    // 平台信息查询失败时 GetCoreNumAic 可能返回 0，按 1 处理，避免以 0 block 启动内核
    if (numBlocks == 0)
    {
        numBlocks = 1;
    }

    int M, N, blockDim, blockN, blockM, tileM;
    M = n;
    N = n;
    blockDim = numBlocks;

    blockN = 16;
    blockM = 16;
    tileM = 512;
    // 尺寸推导统一在 int64_t 域计算，避免 N 较大时 int 乘加溢出导致缓冲尺寸回绕
    int64_t t = (static_cast<int64_t>(std::min(M, N)) + blockN - 1) / blockN * blockN;
    int64_t alignedM = (static_cast<int64_t>(M) + 15) / 16 * 16;
    int64_t strideN = (static_cast<int64_t>(N) + 127) / 128 * 128;

    size_t aMatrixFileSize = static_cast<int64_t>(M) * N * sizeof(float);
    size_t wFileSize = t * sizeof(int);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    size_t aMatrixWorkSize = static_cast<size_t>(strideN) * static_cast<size_t>(alignedM + 128) * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&wHost), wFileSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemset(wDevice, wFileSize, -1, wFileSize), cleanup());

    int64_t workM = (static_cast<int64_t>(M) + tileM - 1) / tileM * tileM;
    size_t workSize = static_cast<size_t>(workM) * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&eyeMatrixHost), aMatrixWorkSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&eyeMatrixDevice, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    size_t gatherSize = tileM * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost1), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost2), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    if (!GenerateTiling(tileM, blockN, N, strideN, gatherHost1, gatherHost2, eyeMatrixHost))
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice1, gatherSize, gatherHost1, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice2, gatherSize, gatherHost2, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(
        aclrtMemcpy(eyeMatrixDevice, aMatrixWorkSize, eyeMatrixHost, aMatrixWorkSize, ACL_MEMCPY_HOST_TO_DEVICE),
        cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    sgetri_kernel_do(sync, M, N, blockM, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice,
                     gatherDevice1, gatherDevice2, eyeMatrixDevice, blockDim, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    cleanup();

    return ACL_SUCCESS;
}
