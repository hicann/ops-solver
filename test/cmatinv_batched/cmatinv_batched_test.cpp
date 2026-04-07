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
#include "../utils/data_utils.h"
#include "../utils/utils.h"

int32_t main(int32_t argc, char *argv[])
{
    int deviceId, batchSize, n;
    deviceId = (argc>1) ? std::atoi(argv[1]) : 0;
    batchSize = (argc>2) ? std::atoi(argv[2]) : 2;
    n = (argc>3) ? std::atoi(argv[3]) : 3;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    
    size_t aMatrixFileSize = batchSize * n * n * sizeof(std::complex<float>);

    std::complex<float>* A;
    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    ReadFile("./test/cmatinv_batched/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize);

    std::vector<std::complex<float>> Ainv(batchSize * n * n, {-1.0f, -1.0f});
    std::vector<int32_t> info(batchSize, 0);

    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(A, batchSize, n, n);
    std::cout << "------- input TensorInAinv -------" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    auto ret = aclsolverCmatinvBatched(n, A, n, Ainv.data(), n, info.data(), batchSize, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCmatinvBatched failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInAinv -------" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    WriteFile("./test/cmatinv_batched/data/output/Ainv_gm.bin", Ainv.data(), aMatrixFileSize);

    // 读取 golden 数据进行对比
    std::vector<std::complex<float>> goldenAinv(batchSize * n * n);
    ReadFile("./test/cmatinv_batched/data/golden/Ainv_gm.bin", aMatrixFileSize, goldenAinv.data(), aMatrixFileSize);

    bool passed = allclose(goldenAinv.data(), Ainv.data(), batchSize * n * n, 1e-4f, 1e-4f);
    if (!passed) {
        printf("judge not equal\n");
    }

    CHECK_ACL(aclrtFreeHost(A));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return passed ? 0 : 1;
}