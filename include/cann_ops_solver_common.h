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
 * \file cann_ops_solver_common.h
 * \brief aclsolver common types and status codes.
 */

#pragma once
              
#include <cstddef>
#include <acl/acl.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief aclsolver status codes definition */
typedef enum aclsolverStatus
{
    ACLSOLVER_STATUS_SUCCESS = 0,              /**< Function succeeds */
    ACLSOLVER_STATUS_NOT_INITIALIZED = 1,      /**< aclsolver library not initialized */
    ACLSOLVER_STATUS_ALLOC_FAILED = 2,         /**< resource allocation failed */
    ACLSOLVER_STATUS_INVALID_VALUE = 3,        /**< unsupported numerical value was passed to function */
    ACLSOLVER_STATUS_MAPPING_ERROR = 4,        /**< access to memory space failed */
    ACLSOLVER_STATUS_EXECUTION_FAILED = 5,     /**< program failed to execute */
    ACLSOLVER_STATUS_INTERNAL_ERROR = 6,       /**< an internal aclsolver operation failed */
    ACLSOLVER_STATUS_NOT_SUPPORTED = 7,        /**< function not implemented */
    ACLSOLVER_STATUS_ARCH_MISMATCH = 8,        /**< architecture mismatch */
    ACLSOLVER_STATUS_HANDLE_IS_NULLPTR = 9,    /**< aclsolver handle is null pointer */
    ACLSOLVER_STATUS_INVALID_ENUM = 10,        /**< unsupported enum value was passed to function */
    ACLSOLVER_STATUS_UNKNOWN = 11              /**< back-end returned an unsupported status code */
} aclsolverStatus_t;

typedef void* aclsolverHandle_t;

#ifdef __cplusplus
}
#endif
