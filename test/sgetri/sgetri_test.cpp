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
 * \file sgetri_test.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "../utils/test_utils.h"

int main(int argc, char **argv) {
    int deviceId, M, N;
    deviceId = (argc>1) ? std::atoi(argv[1]) : 0;
    N = (argc>2) ? std::atoi(argv[2]) : 32;
    M = N;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    int t = (std::min(M, N) + 15) / 16 * 16;
    size_t aMatrixFileSize = M * N * sizeof(float);

    float* A;
    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    ReadFile("./test/sgetri/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    int32_t infoVal = 0;
    int32_t *info = &infoVal;

    auto ret = aclsolverSgetri(handle, N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetri failed. ERROR: %d\n", ret); return ret);

    std::cout << "[Output] A (inverse):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    WriteFile("./test/sgetri/data/output/A_gm.bin", A, aMatrixFileSize);

    CHECK_ACL(aclrtFreeHost(A));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclsolverDestroy(handle));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}