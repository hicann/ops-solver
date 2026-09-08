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
 * \file cheevj_benchmark.cpp
 * \brief In-process public API wall-time benchmark for aclsolverCheevj.
 */

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

namespace
{
constexpr const char *INPUT_A_PATH = "./test/cheevj/data/input/A_gm.bin";

struct BenchmarkConfig
{
    int32_t deviceId = 0;
    int64_t n = 0;
    int64_t warmup = 0;
    int64_t repeat = 0;
    char jobz = 'V';
    char uplo = 'L';
    size_t matrixElements = 0;
    size_t matrixBytes = 0;
};

char NormalizeOption(const char *arg, char defaultValue)
{
    if (arg == nullptr || arg[0] == '\0')
    {
        return defaultValue;
    }
    return static_cast<char>(std::toupper(static_cast<unsigned char>(arg[0])));
}

bool ParseNonNegativeInt(const char *arg, int64_t *value)
{
    if (arg == nullptr || value == nullptr || arg[0] == '\0')
    {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || parsed < 0)
    {
        return false;
    }
    *value = static_cast<int64_t>(parsed);
    return true;
}

aclsolverEigMode_t ParseJobz(char jobz)
{
    return jobz == 'N' ? ACLSOLVER_EIG_MODE_NOVECTOR : ACLSOLVER_EIG_MODE_VECTOR;
}

aclsolverFillMode_t ParseUplo(char uplo) { return uplo == 'U' ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER; }

double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() % 2) != 0)
    {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

void PrintUsage(const char *program)
{
    std::cerr << "Usage: " << program << " <device> <n> <jobz:N|V> <uplo:L|U> <warmup>=0 <repeat>>0\n";
}

bool ParseArguments(int32_t argc, char *argv[], BenchmarkConfig *config)
{
    int64_t deviceValue = 0;
    if (argc != 7 || !ParseNonNegativeInt(argv[1], &deviceValue) || !ParseNonNegativeInt(argv[2], &config->n) ||
        !ParseNonNegativeInt(argv[5], &config->warmup) || !ParseNonNegativeInt(argv[6], &config->repeat) ||
        deviceValue > std::numeric_limits<int32_t>::max() || config->n == 0 || config->repeat == 0 ||
        config->warmup > std::numeric_limits<int64_t>::max() - config->repeat)
    {
        return false;
    }
    config->deviceId = static_cast<int32_t>(deviceValue);
    config->jobz = NormalizeOption(argv[3], 'V');
    config->uplo = NormalizeOption(argv[4], 'L');
    if ((config->jobz != 'V' && config->jobz != 'N') || (config->uplo != 'L' && config->uplo != 'U'))
    {
        return false;
    }
    const uint64_t n = static_cast<uint64_t>(config->n);
    if (n > std::numeric_limits<size_t>::max() / n / sizeof(std::complex<float>))
    {
        std::cerr << "Matrix byte size overflows size_t.\n";
        return false;
    }
    config->matrixElements = static_cast<size_t>(config->n) * static_cast<size_t>(config->n);
    config->matrixBytes = config->matrixElements * sizeof(std::complex<float>);
    return true;
}

int32_t RunTimedCalls(aclsolverHandle_t handle, const BenchmarkConfig &config, std::complex<float> *matrix,
                      const std::vector<std::complex<float>> &originalMatrix, std::vector<float> *eigenvalues,
                      std::vector<double> *samples)
{
    for (int64_t call = 0; call < config.warmup + config.repeat; ++call)
    {
        if (config.jobz == 'V' || call == 0)
        {
            std::copy_n(originalMatrix.data(), originalMatrix.size(), matrix);
        }
        int32_t info = 0;
        const auto started = std::chrono::steady_clock::now();
        const aclError status = aclsolverCheevj(handle, ParseJobz(config.jobz), ParseUplo(config.uplo), config.n,
                                                matrix, config.n, eigenvalues->data(), &info);
        const auto finished = std::chrono::steady_clock::now();
        if (status != ACL_SUCCESS || info != 0)
        {
            std::cerr << "aclsolverCheevj failed: call=" << call << " acl_error=" << status << " info=" << info
                      << "\n";
            return status != ACL_SUCCESS ? static_cast<int32_t>(status) : info;
        }
        const double elapsedMs = std::chrono::duration<double, std::milli>(finished - started).count();
        if (call < config.warmup)
        {
            std::cout << "[WARMUP] iteration=" << (call + 1) << " wall_ms=" << std::fixed << std::setprecision(6)
                      << elapsedMs << std::endl;
            continue;
        }
        samples->push_back(elapsedMs);
        std::cout << "[ITER] iteration=" << (call - config.warmup + 1) << " wall_ms=" << std::fixed
                  << std::setprecision(6) << elapsedMs << std::endl;
    }
    return 0;
}
}  // namespace

int32_t main(int32_t argc, char *argv[])
{
    BenchmarkConfig config;
    if (!ParseArguments(argc, argv, &config))
    {
        PrintUsage(argv[0]);
        return 2;
    }

    // Runtime setup, input file I/O, and all host allocations are deliberately
    // outside the timed region. The public API currently synchronizes its
    // stream internally, so the interval below covers one complete API call.
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(config.deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    std::complex<float> *matrix = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&matrix), config.matrixBytes));
    size_t readSize = config.matrixBytes;
    CHECK_RET(ReadFile(INPUT_A_PATH, readSize, matrix, config.matrixBytes) && readSize == config.matrixBytes,
              LOG_PRINT("Read input matrix failed: %s\n", INPUT_A_PATH);
              return 1);

    const std::vector<std::complex<float>> originalMatrix(matrix, matrix + config.matrixElements);
    std::vector<float> eigenvalues(static_cast<size_t>(config.n), 0.0f);
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(config.repeat));
    const int32_t runStatus = RunTimedCalls(handle, config, matrix, originalMatrix, &eigenvalues, &samples);
    CHECK_RET(runStatus == 0, return runStatus > 0 ? runStatus : 1);

    const double medianMs = Median(samples);
    const double minMs = *std::min_element(samples.begin(), samples.end());
    std::cout << "[SUMMARY] n=" << config.n << " jobz=" << config.jobz << " uplo=" << config.uplo
              << " warmup=" << config.warmup << " repeat=" << config.repeat << " median_ms=" << std::fixed
              << std::setprecision(6) << medianMs << " min_ms=" << minMs << std::endl;

    CHECK_ACL(aclrtFreeHost(matrix));
    CHECK_ACL(aclsolverDestroy(handle));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(config.deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
