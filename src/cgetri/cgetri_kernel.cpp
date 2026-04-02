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
 * \file cgetri_kernel.cpp
 * \brief
 */

#ifndef CGETRI_MIX_H
#define CGETRI_MIX_H

#include <cstdint>
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>
#include "../utils/kernel/c64/getri_custom.hpp"

using namespace AscendC;
using namespace matmul;

static constexpr int64_t DTYPE_COMPLEX64 = 1;

__global__ __aicore__ void cgetri_kernel(GM_ADDR sync, int orgM, int orgN,
                                         int blockM, int blockN, int tileM,
                                         GM_ADDR A_org, GM_ADDR A_work,
                                         GM_ADDR W, GM_ADDR work_gm,
                                         GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                                         GM_ADDR gather3_gm, GM_ADDR eye_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  GM_ADDR workspace;
  custom_getri(DTYPE_COMPLEX64, orgM, orgN, blockM, blockN, tileM, A_org, A_org,
            A_work, W, work_gm, gather1_gm, gather2_gm, gather3_gm, eye_gm, workspace);
}

void cgetri_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN,
                      int tileM, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W,
                      GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                      GM_ADDR gather3_gm, GM_ADDR eye_gm, uint32_t numBlocks,
                      void *stream) {
  cgetri_kernel<<<numBlocks, nullptr, stream>>>(
      sync, orgM, orgN, blockM, blockN, tileM, A_org, A_work, W, work_gm, gather1_gm, gather2_gm, gather3_gm, eye_gm);
}
#endif  // CGETRI_MIX_H