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
 * \file sgetrf_host.cpp
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

#include "tiling/platform/platform_ascendc.h"


#define GM_ADDR uint8_t*

extern void sgetrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockN, int tileM,
                             GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm,
                             GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                             uint32_t numBlocks, void *stream);

void GenerateGather(int blockM, int blockN, int M, int N, uint8_t *gatherBuf1, uint8_t *gatherBuf2) {
    try {
        {
            auto buf = reinterpret_cast<uint32_t *>(gatherBuf1);
            int idx = 0;
            for (int j = 0; j < blockN; ++j)
                for (int i = blockM - 1; i >= 0; --i)
                    buf[idx++] = (i * blockN + j) * sizeof(float);
        }
        {
            auto buf = reinterpret_cast<uint32_t *>(gatherBuf2);
            int idx = 0;
            for (int i = blockM - 1; i >= 0; --i)
                for (int j = 0; j < blockN; ++j)
                    buf[idx++] = (j * blockM + i) * sizeof(float);
        }
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
}

aclError aclsolverSgetrf(const int64_t m, const int64_t n, float *A, const int64_t lda,
                         int32_t *ipiv, int32_t *info, void *stream) {
    // Get current device ID
    int32_t deviceId = 0;
    CHECK_ACLRT(aclrtGetDevice(&deviceId));
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr) {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    int M, N, blockDim, blockN, blockM, tileM;
    M = m;
    N = n;
    blockDim = numBlocks;

    blockN = 16;
    blockM = 16;
    tileM = 512;
    int t = (std::min(M, N) + blockN - 1) / blockN * blockN;

    size_t aMatrixFileSize = M * N * sizeof(float);
    size_t wFileSize = t * sizeof(int);

    uint8_t* aMatrixHost = reinterpret_cast<uint8_t*>(A);
    uint8_t* aMatrixDevice;
    CHECK_ACLRT(aclrtMalloc((void**)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t aMatrixWorkSize = ((N + 127) / 128 * 128) * ((M + 15) / 16 * 16) * sizeof(float);
    uint8_t* aMatrixDeviceWork;
    CHECK_ACLRT(aclrtMalloc((void**)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* wHost;
    uint8_t* wDevice;
    CHECK_ACLRT(aclrtMallocHost((void**)(&wHost), wFileSize));
    CHECK_ACLRT(aclrtMalloc((void**)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtMemset(wDevice, wFileSize, -1, wFileSize);

    int workM = (M + tileM - 1) / tileM * tileM;
    size_t workSize = workM * blockN * sizeof(float);
    uint8_t* workDevice;
    CHECK_ACLRT(aclrtMalloc((void**)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t gatherSize = tileM * blockN * sizeof(float);
    uint8_t *gatherHost1, *gatherHost2;
    uint8_t *gatherDevice1, *gatherDevice2;
    CHECK_ACLRT(aclrtMallocHost((void**)(&gatherHost1), gatherSize));
    CHECK_ACLRT(aclrtMallocHost((void**)(&gatherHost2), gatherSize));
    CHECK_ACLRT(aclrtMalloc((void**)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void**)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST));

    GenerateGather(tileM, blockN, (M + 15) / 16 * 16, (N + 15) / 16 * 16, gatherHost1, gatherHost2);
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice1, gatherSize, gatherHost1, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice2, gatherSize, gatherHost2, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream));

    uint8_t *sync = nullptr;
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync));

    sgetrf_kernel_do(sync, M, N, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice,
                     gatherDevice1, gatherDevice2, blockDim, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream));

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Copy pivot information back to ipiv
    int32_t *wHostInt = reinterpret_cast<int32_t*>(wHost);
    CHECK_ACLRT(aclrtMemcpy(wHost, wFileSize, wDevice, wFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < std::min(M, N); ++i) {
        ipiv[i] = wHostInt[i];  // Convert to 1-based indexing
    }

    CHECK_ACLRT(aclrtFree(aMatrixDevice));
    CHECK_ACLRT(aclrtFree(aMatrixDeviceWork));
    CHECK_ACLRT(aclrtFree(wDevice));
    CHECK_ACLRT(aclrtFreeHost(wHost));
    CHECK_ACLRT(aclrtFree(workDevice));
    CHECK_ACLRT(aclrtFree(gatherDevice1));
    CHECK_ACLRT(aclrtFree(gatherDevice2));
    CHECK_ACLRT(aclrtFreeHost(gatherHost1));
    CHECK_ACLRT(aclrtFreeHost(gatherHost2));

    return ACL_SUCCESS;
}
