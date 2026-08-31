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
 * \file cmatinv_batched_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int32_t main(int32_t argc, char* argv[])
{
    int deviceId, batchSize, n;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    batchSize = (argc > 2) ? std::atoi(argv[2]) : 2;
    n = (argc > 3) ? std::atoi(argv[3]) : 3;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = batchSize * n * n * sizeof(std::complex<float>);
    std::complex<float>* A = nullptr;
    auto cleanup = [&]() -> aclError
    {
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    CHECK_RET(ReadFile("./test/cmatinv_batched/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);
    std::vector<std::complex<float>> Ainv(batchSize * n * n, {-1.0f, -1.0f});
    std::vector<int32_t> info(batchSize, 0);

    std::cout << "[Input] A:" << std::endl;
    printTensor(A, batchSize, n, n);
    std::cout << "[Input] Ainv:" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    auto ret = aclsolverCmatinvBatched(handle, n, A, n, Ainv.data(), n, info.data(), batchSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCmatinvBatched failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] Ainv:" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    WriteFile("./test/cmatinv_batched/data/output/Ainv_gm.bin", Ainv.data(), aMatrixFileSize);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
