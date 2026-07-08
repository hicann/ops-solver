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
 * \file cgetri_batched_test.cpp
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
#include "../utils/test_utils.h"

int32_t main(int32_t argc, char *argv[])
{
    int deviceId, batchSize, n;
    deviceId = (argc>1) ? std::atoi(argv[1]) : 0;
    batchSize = (argc>2) ? std::atoi(argv[2]) : 2;
    n = (argc>3) ? std::atoi(argv[3]) : 48;

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    std::complex<float>* A = nullptr;

    auto cleanup = [&]() {
        if (A) aclrtFreeHost(A);
        if (handle) aclsolverDestroy(handle);
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) { aclFinalize(); return ret; }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) { cleanup(); return ret; }
    ret = aclsolverCreate(&handle);
    if (ret != ACL_SUCCESS) { cleanup(); return ret; }
    ret = aclsolverSetStream(handle, stream);
    if (ret != ACL_SUCCESS) { cleanup(); return ret; }

    size_t aMatrixFileSize = batchSize * n * n * sizeof(std::complex<float>);
    ret = aclrtMallocHost((void**)(&A), aMatrixFileSize);
    if (ret != ACL_SUCCESS) { cleanup(); return ret; }

    ReadFile("./test/cgetri_batched/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize);
    std::vector<std::complex<float>> Ainv(batchSize * n * n, {-1.0f, -1.0f});
    std::vector<int32_t> info(batchSize, 0);

    ret = aclsolverCgetriBatched(handle, n, A, n, Ainv.data(), n, info.data(), batchSize);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclsolverCgetriBatched failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }

    WriteFile("./test/cgetri_batched/data/output/Ainv_gm.bin", Ainv.data(), aMatrixFileSize);

    cleanup();
    return 0;
}
