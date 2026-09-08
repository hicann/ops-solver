/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHEEVJ_ASCENDC_SYMBOLS_HPP
#define CHEEVJ_ASCENDC_SYMBOLS_HPP

#include "kernel_operator.h"

namespace Cheevj
{
using AscendC::Add;
using AscendC::CacheLine;
using AscendC::CreateVecIndex;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DcciDst;
using AscendC::Duplicate;
using AscendC::Gather;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::ReduceMax;
using AscendC::ShiftLeft;
using AscendC::ShiftRight;
using AscendC::Sub;
using AscendC::TBuf;
using AscendC::TBufPool;
using AscendC::TPipe;
using AscendC::TPosition;
}  // namespace Cheevj

#endif  // CHEEVJ_ASCENDC_SYMBOLS_HPP
