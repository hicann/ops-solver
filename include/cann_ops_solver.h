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
 * \file cann_ops_solver.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <complex>

aclError aclsolverCmatinvBatched(const int64_t n, std::complex<float> *A, const int64_t lda, std::complex<float> *Ainv,
                                 const int64_t lda_inv, int32_t *info, int64_t batchSize, void *stream);

aclError aclsolverCgetriBatched(const int64_t n, std::complex<float> *A, const int64_t lda, std::complex<float> *Ainv,
                                const int64_t lda_inv, int32_t *info, int64_t batchSize, void *stream);

aclError aclsolverCgetri(const int64_t n, std::complex<float> *A, const int64_t lda, int32_t *info, void *stream);

aclError aclsolverSgetri(const int64_t n, float *A, const int64_t lda, int32_t *info, void *stream);

aclError aclsolverCgetrf(const int64_t m, const int64_t n, std::complex<float> *A, const int64_t lda, int32_t *ipiv,
                         int32_t *info, void *stream);

aclError aclsolverSgetrf(const int64_t m, const int64_t n, float *A, const int64_t lda, int32_t *ipiv, int32_t *info,
                         void *stream);
