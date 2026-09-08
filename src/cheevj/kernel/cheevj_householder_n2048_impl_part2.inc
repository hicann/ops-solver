/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


    __aicore__ inline void MultiplyInPlace(const AscendC::LocalTensor<float> &real,
                                           const AscendC::LocalTensor<float> &imag, float scaleReal, float scaleImag,
                                           const AscendC::LocalTensor<float> &tmp0,
                                           const AscendC::LocalTensor<float> &tmp1)
    {
        AscendC::Muls(tmp0, real, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(tmp1, imag, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp1, imag, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(imag, real, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(imag, tmp1, imag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(real, tmp0, 1.0f, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DotConjugate(const AscendC::LocalTensor<float> &leftReal,
                                        const AscendC::LocalTensor<float> &leftImag,
                                        const AscendC::LocalTensor<float> &rightReal,
                                        const AscendC::LocalTensor<float> &rightImag,
                                        const AscendC::LocalTensor<float> &tmp0,
                                        const AscendC::LocalTensor<float> &tmp1,
                                        const AscendC::LocalTensor<float> &result, float &dotReal, float &dotImag)
    {
        AscendC::Mul(tmp0, leftReal, rightReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Mul(tmp1, leftImag, rightImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        dotReal = CheevjN2048Reduce(tmp0, tmp1, result);

        AscendC::Mul(tmp0, leftReal, rightImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Mul(tmp1, leftImag, rightReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        dotImag = CheevjN2048Reduce(tmp0, tmp1, result);
    }

    __aicore__ inline void BuildReflector(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_N2048_STREAMED_N - begin;
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto tmp0 = accumulatorRealBuf.Get<float>();
        auto tmp1 = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();

        ClearDense(vectorReal);
        ClearDense(vectorImag);
        // The lower column is the conjugate of the contiguous upper row,
        // avoiding a strided matrix transpose/read.  DataCopyPad accepts the
        // unaligned GM row offset while every following SIMD operation starts
        // at an aligned UB base.
        const int rowOffset = step * CHEEVJ_HH_N2048_STREAMED_N + begin;
        LoadContiguous(vectorReal, matrixRealGlobal, rowOffset, active);
        LoadContiguous(vectorImag, matrixImagGlobal, rowOffset, active);
        ClearActivePadding(vectorReal, active);
        ClearActivePadding(vectorImag, active);
        AscendC::Muls(vectorImag, vectorImag, -1.0f, CHEEVJ_HH_N2048_STREAMED_N);
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 2
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION

        const int columnOffset = step * CHEEVJ_HH_N2048_STREAMED_N;
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 3
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 13
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#define CHEEVJ_HH_VARIANT_N CHEEVJ_HH_N2048_STREAMED_N
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 19
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#define CHEEVJ_HH_VARIANT_REDUCE CheevjN2048Reduce
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 15
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_VARIANT_REDUCE
#undef CHEEVJ_HH_VARIANT_N

    __aicore__ inline void ComputeRowGemv(const AscendC::LocalTensor<float> &matrixReal,
                                          const AscendC::LocalTensor<float> &matrixImag,
                                          const AscendC::LocalTensor<float> &result,
                                          const AscendC::LocalTensor<float> &resultImag, int resultLaneBegin)
    {
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto vectorImagNeg = vectorImagNegBuf.Get<float>();
        auto partials = accumulatorRealBuf.Get<float>();
        auto reduction = accumulatorImagBuf.Get<float>();
        // The controller-only correction buffers are independent per AIV, so
        // row workers may reuse their local instances as full-width products.
        auto product0 = control0Buf.Get<float>();
        auto product1 = control1Buf.Get<float>();

        // A 2048-float row stride is 256 data blocks, one beyond the C220
        // vector repeat-stride range.  Process the four resident rows with
        // full-width contiguous SIMD operations instead of encoding that
        // stride in BinaryRepeatParams.  No matrix element enters the scalar
        // unit; only the two final reduction controls do.
        for (int localRow = 0; localRow < CHEEVJ_HH_N2048_STREAMED_ROWS; ++localRow)
        {
            const int rowOffset = localRow * CHEEVJ_HH_N2048_STREAMED_N;
            AscendC::Mul(product0, matrixReal[rowOffset], vectorReal, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::Mul(product1, matrixImag[rowOffset], vectorImagNeg, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(product0, product0, product1, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::PipeBarrier<PIPE_V>();
            const float dotReal = CheevjN2048Reduce(product0, partials, reduction);

            AscendC::Mul(product0, matrixReal[rowOffset], vectorImag, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::Mul(product1, matrixImag[rowOffset], vectorReal, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(product0, product0, product1, CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::PipeBarrier<PIPE_V>();
            const float dotImag = CheevjN2048Reduce(product0, partials, reduction);

            uint64_t laneMask[2] = {1ULL << static_cast<uint32_t>(resultLaneBegin + localRow), 0ULL};
            AscendC::Duplicate(result, dotReal, laneMask, static_cast<uint8_t>(1), 1, 8);
            AscendC::Duplicate(resultImag, dotImag, laneMask, static_cast<uint8_t>(1), 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadReflectorForGemv(int step)
    {
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto vectorImagNeg = vectorImagNegBuf.Get<float>();
        const int columnOffset = step * CHEEVJ_HH_N2048_STREAMED_N;
        LoadContiguous(vectorReal, internalVRealGlobal, columnOffset, CHEEVJ_HH_N2048_STREAMED_N);
        LoadContiguous(vectorImag, internalVImagGlobal, columnOffset, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(vectorImagNeg, vectorImag, -1.0f, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CorrectAndFormW(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_N2048_STREAMED_N - begin;
        const int columnOffset = step * CHEEVJ_HH_N2048_STREAMED_N;
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 4
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
        AscendC::Muls(tmp0, yImag, -1.0f, CHEEVJ_HH_N2048_STREAMED_N);
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 5
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
#define CHEEVJ_HH_VARIANT_N CHEEVJ_HH_N2048_STREAMED_N
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 16
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_VARIANT_N
