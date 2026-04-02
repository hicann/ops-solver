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
 * \file sgetrf_test.cpp
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
#include "../utils/data_utils.h"
#include "../utils/utils.h"

int main(int argc, char **argv) {
    int deviceId, M, N;
    deviceId = (argc>1) ? std::atoi(argv[1]) : 0;
    M = (argc>2) ? std::atoi(argv[2]) : 32;
    N = (argc>3) ? std::atoi(argv[3]) : 32;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    size_t aMatrixFileSize = M * N * sizeof(float);

    float* A;
    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    ReadFile("./test/sgetrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize);

    int32_t *ipiv = new int32_t[std::min(M, N)];
    int32_t *info;

    auto ret = aclsolverSgetrf(M, N, A, N, ipiv, info, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetrf failed. ERROR: %d\n", ret); return ret);

    WriteFile("./test/sgetrf/data/output/A_gm.bin", A, aMatrixFileSize);

    // Write pivot information
    std::vector<int32_t> ipivData(ipiv, ipiv + std::min(M, N));
    WriteFile("./test/sgetrf/data/output/W_gm.bin", ipivData.data(), ipivData.size() * sizeof(int32_t));

    delete[] ipiv;
    CHECK_ACL(aclrtFreeHost(A));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}