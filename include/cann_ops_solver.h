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
#include <cstddef>
#include <acl/acl.h>
#include "cann_ops_solver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create an aclsolver handle
 *
 * Allocates an aclsolver handle on the heap and outputs it via @p handle. The internal
 * structure is opaque to users; the returned void* is used with other APIs.
 *
 * @param handle Output parameter to receive the created handle pointer (void**). *handle
 *               must be nullptr before the call, otherwise it is treated as already existing
 *               to prevent memory leaks.
 * @return ACLSOLVER_STATUS_SUCCESS success
 *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle pointer is null
 *         ACLSOLVER_STATUS_INVALID_VALUE *handle is not null (prevent repeated creation causing memory leak)
 *         ACLSOLVER_STATUS_NOT_INITIALIZED CANN context not initialized
 *         ACLSOLVER_STATUS_ALLOC_FAILED memory allocation failed
 */
aclsolverStatus_t aclsolverCreate(aclsolverHandle_t *handle);

/**
 * @brief Destroy an aclsolver handle
 *
 * Releases all resources occupied by the handle created by aclsolverCreate.
 *
 * @param handle The handle to destroy (void*)
 * @return ACLSOLVER_STATUS_SUCCESS success
 *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
 */
aclsolverStatus_t aclsolverDestroy(aclsolverHandle_t handle);

/**
 * @brief Set the stream for a handle
 * @param handle aclsolver handle (void*)
 * @param stream The stream to set, nullptr means default stream
 * @return ACLSOLVER_STATUS_SUCCESS success
 *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
 */
aclsolverStatus_t aclsolverSetStream(aclsolverHandle_t handle, aclrtStream stream);

/**
 * @brief Get the stream of a handle
 * @param handle aclsolver handle (void*)
 * @param stream Output parameter, returns the current stream
 * @return ACLSOLVER_STATUS_SUCCESS success
 *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
 *         ACLSOLVER_STATUS_INVALID_VALUE stream output parameter is null
 */
aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream *stream);

#ifdef __cplusplus
}
#endif

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
