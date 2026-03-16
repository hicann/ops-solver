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
 * \file assert.h
 * \brief
 */

#ifndef ASSERT_H
#define ASSERT_H

#define SOLVER_ECHECK(condition, logExpr, handleExpr)                          \
  do {                                                                         \
    if (!(condition)) {                                                        \
      flockfile(stderr);                                                       \
      fprintf(stderr, "[ERROR] %s | Return value: %s\n", (logExpr),            \
              #handleExpr);                                                    \
      fflush(stderr);                                                          \
      funlockfile(stderr);                                                     \
      return handleExpr;                                                       \
    }                                                                          \
  } while (0)

#define SOLVER_CHECK(condition, logExpr, handleExpr)                           \
  do {                                                                         \
    if (!(condition)) {                                                        \
      flockfile(stderr);                                                       \
      fprintf(stderr, "[ERROR] %s\n", (logExpr));                              \
      fflush(stderr);                                                          \
      funlockfile(stderr);                                                     \
      handleExpr;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

#define CHECK_ACLRT(func)                                                      \
  {                                                                            \
    aclError status = (func);                                                  \
    if (status != ACL_SUCCESS) {                                               \
      std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__      \
                << " (error code: " << status << ")" << std::endl;             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                       \
  do {                                                                         \
    int32_t error = status;                                                    \
    if (error != 0) {                                                          \
      std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error         \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

#endif // ASSERT_H