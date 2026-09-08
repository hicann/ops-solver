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
 * \file complex_vec.hpp
 * \brief AscendC vector helpers for planar complex64 local tensors.
 */

#ifndef CHEEVJ_C64_COMPLEX_VEC_HPP
#define CHEEVJ_C64_COMPLEX_VEC_HPP

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace Cheevj
{

__aicore__ inline void ComplexAbs2(LocalTensor<float> dst, LocalTensor<float> real, LocalTensor<float> imag,
                                   LocalTensor<float> scratch, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    Mul(scratch, imag, imag, count);
    Mul(dst, real, real, count);
    PipeBarrier<PIPE_V>();
    Add(dst, dst, scratch, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexAbs(LocalTensor<float> dst, LocalTensor<float> real, LocalTensor<float> imag,
                                  LocalTensor<float> scratch, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    ComplexAbs2(dst, real, imag, scratch, count);
    Sqrt(dst, dst, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexMulProducts(LocalTensor<float> realProduct, LocalTensor<float> imagProduct,
                                          LocalTensor<float> aR, LocalTensor<float> aI, LocalTensor<float> bR,
                                          LocalTensor<float> bI, uint32_t count)
{
    Mul(realProduct, aR, bR, count);
    Mul(imagProduct, aI, bI, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexMul(LocalTensor<float> outR, LocalTensor<float> outI, LocalTensor<float> aR,
                                  LocalTensor<float> aI, LocalTensor<float> bR, LocalTensor<float> bI,
                                  LocalTensor<float> tmp0, LocalTensor<float> tmp1, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    ComplexMulProducts(tmp0, tmp1, aR, aI, bR, bI, count);
    Sub(tmp0, tmp0, tmp1, count);
    PipeBarrier<PIPE_V>();

    Mul(tmp1, aR, bI, count);
    Mul(outI, aI, bR, count);
    PipeBarrier<PIPE_V>();
    Add(outI, outI, tmp1, count);
    PipeBarrier<PIPE_V>();

    Muls(outR, tmp0, 1.0f, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexMulConjB(LocalTensor<float> outR, LocalTensor<float> outI, LocalTensor<float> aR,
                                       LocalTensor<float> aI, LocalTensor<float> bR, LocalTensor<float> bI,
                                       LocalTensor<float> tmp0, LocalTensor<float> tmp1, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    ComplexMulProducts(tmp0, tmp1, aR, aI, bR, bI, count);
    Add(tmp0, tmp0, tmp1, count);
    PipeBarrier<PIPE_V>();

    Mul(tmp1, aR, bI, count);
    Mul(outI, aI, bR, count);
    PipeBarrier<PIPE_V>();
    Sub(outI, outI, tmp1, count);
    PipeBarrier<PIPE_V>();

    Muls(outR, tmp0, 1.0f, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexRot2(LocalTensor<float> outPR, LocalTensor<float> outPI, LocalTensor<float> outQR,
                                   LocalTensor<float> outQI, LocalTensor<float> pR, LocalTensor<float> pI,
                                   LocalTensor<float> qR, LocalTensor<float> qI, float c, float sR, float sI,
                                   LocalTensor<float> tmp0, LocalTensor<float> tmp1, LocalTensor<float> tmp2,
                                   LocalTensor<float> tmp3, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    (void)tmp1;
    (void)tmp2;
    (void)tmp3;

    Muls(outPR, pR, c, count);
    Muls(tmp0, qR, sR, count);
    PipeBarrier<PIPE_V>();
    Sub(outPR, outPR, tmp0, count);
    PipeBarrier<PIPE_V>();
    Muls(tmp0, qI, sI, count);
    PipeBarrier<PIPE_V>();
    Sub(outPR, outPR, tmp0, count);
    PipeBarrier<PIPE_V>();

    Muls(outPI, pI, c, count);
    Muls(tmp0, qI, sR, count);
    PipeBarrier<PIPE_V>();
    Sub(outPI, outPI, tmp0, count);
    PipeBarrier<PIPE_V>();
    Muls(tmp0, qR, sI, count);
    PipeBarrier<PIPE_V>();
    Add(outPI, outPI, tmp0, count);
    PipeBarrier<PIPE_V>();

    Muls(outQR, pR, sR, count);
    Muls(tmp0, pI, sI, count);
    PipeBarrier<PIPE_V>();
    Sub(outQR, outQR, tmp0, count);
    PipeBarrier<PIPE_V>();
    Muls(tmp0, qR, c, count);
    PipeBarrier<PIPE_V>();
    Add(outQR, outQR, tmp0, count);
    PipeBarrier<PIPE_V>();

    Muls(outQI, pI, sR, count);
    Muls(tmp0, pR, sI, count);
    PipeBarrier<PIPE_V>();
    Add(outQI, outQI, tmp0, count);
    PipeBarrier<PIPE_V>();
    Muls(tmp0, qI, c, count);
    PipeBarrier<PIPE_V>();
    Add(outQI, outQI, tmp0, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComplexCopy(LocalTensor<float> dstR, LocalTensor<float> dstI, LocalTensor<float> srcR,
                                   LocalTensor<float> srcI, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    Muls(dstR, srcR, 1.0f, count);
    Muls(dstI, srcI, 1.0f, count);
    PipeBarrier<PIPE_V>();
}

}  // namespace Cheevj

#endif  // CHEEVJ_C64_COMPLEX_VEC_HPP
