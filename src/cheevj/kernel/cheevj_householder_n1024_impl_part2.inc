/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


    __aicore__ inline void BuildReflector(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_N1024_STREAMED_N - begin;
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto tmp0 = accumulatorRealBuf.Get<float>();
        auto tmp1 = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();

        if (active <= 0)
        {
            const float diagonal = LoadScalar(matrixRealGlobal, step * CHEEVJ_HH_N1024_STREAMED_N + step, tmp0);
            StoreScalar(diagonalGlobal, step, tmp0, diagonal);
            StoreScalar(offDiagonalGlobal, step, tmp0, 0.0f);
            StoreScalar(tauRealGlobal, step, tmp0, 0.0f);
            StoreScalar(tauImagGlobal, step, tmp0, 0.0f);
            return;
        }

        ClearDense(vectorReal, active);
        ClearDense(vectorImag, active);
        // Read the conjugate lower column from its contiguous upper row; SIMD data remains UB-aligned.
        const int rowOffset = step * CHEEVJ_HH_N1024_STREAMED_N + begin;
        LoadContiguous(vectorReal, matrixRealGlobal, rowOffset, active);
        LoadContiguous(vectorImag, matrixImagGlobal, rowOffset, active);
        ClearActivePadding(vectorReal, active);
        ClearActivePadding(vectorImag, active);
        AscendC::Muls(vectorImag, vectorImag, -1.0f, CHEEVJ_HH_N1024_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();

        const float diagonal = ApplyPriorReflectors(step, begin, active, vectorReal, vectorImag, tmp0, tmp1);
        float beta = 0.0f;
        float tauR = 0.0f;
        float tauI = 0.0f;
        FormReflector(active, vectorReal, vectorImag, tmp0, tmp1, result, beta, tauR, tauI);

        uint64_t firstMask[2] = {1ULL, 0ULL};
        AscendC::Duplicate(vectorReal, 1.0f, firstMask, static_cast<uint8_t>(1), 1, 8);
        AscendC::Duplicate(vectorImag, 0.0f, firstMask, static_cast<uint8_t>(1), 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        const int columnOffset = step * CHEEVJ_HH_N1024_STREAMED_N;
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 3
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
    __aicore__ inline float ApplyPriorReflectors(int step, int begin, int active,
                                                 const AscendC::LocalTensor<float> &vectorReal,
                                                 const AscendC::LocalTensor<float> &vectorImag,
                                                 const AscendC::LocalTensor<float> &tmp0,
                                                 const AscendC::LocalTensor<float> &tmp1)
    {
        auto previousReal = control2Buf.Get<float>();
        auto previousImag = control3Buf.Get<float>();
        float diagonal = LoadScalar(matrixRealGlobal, step * CHEEVJ_HH_N1024_STREAMED_N + step, tmp0);
        for (int previous = 0; previous < step; ++previous)
        {
            const int previousOffset = previous * CHEEVJ_HH_N1024_STREAMED_N;
            const float vjr = LoadScalar(internalVRealGlobal, previousOffset + step, tmp0);
            const float vji = LoadScalar(internalVImagGlobal, previousOffset + step, tmp0);
            const float wjr = LoadScalar(internalWRealGlobal, previousOffset + step, tmp0);
            const float wji = LoadScalar(internalWImagGlobal, previousOffset + step, tmp0);
            diagonal -= 2.0f * (vjr * wjr + vji * wji);

            ClearDense(previousReal, active);
            ClearDense(previousImag, active);
            LoadContiguous(previousReal, internalVRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalVImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(vectorReal, vectorImag, previousReal, previousImag, wjr, -wji, tmp0, tmp1, active);

            ClearDense(previousReal, active);
            ClearDense(previousImag, active);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(vectorReal, vectorImag, previousReal, previousImag, vjr, -vji, tmp0, tmp1, active);
        }
        return diagonal;
    }

    __aicore__ inline void FormReflector(int active, const AscendC::LocalTensor<float> &vectorReal,
                                         const AscendC::LocalTensor<float> &vectorImag,
                                         const AscendC::LocalTensor<float> &tmp0,
                                         const AscendC::LocalTensor<float> &tmp1,
                                         const AscendC::LocalTensor<float> &result, float &beta, float &tauR,
                                         float &tauI)
    {
        const float alphaReal = ReadLocalControlLane(vectorReal, 0, result);
        const float alphaImag = ReadLocalControlLane(vectorImag, 0, result);
        AscendC::Mul(tmp0, vectorReal, vectorReal, CHEEVJ_HH_N1024_STREAMED_N);
        AscendC::Mul(tmp1, vectorImag, vectorImag, CHEEVJ_HH_N1024_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_N1024_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        const float totalNorm2 = CheevjN1024ReduceActive(tmp0, tmp1, result, active);
        const float alphaNorm2 = alphaReal * alphaReal + alphaImag * alphaImag;
        const float rawTailNorm2 = totalNorm2 - alphaNorm2;
        const float tailNorm2 = rawTailNorm2 > 0.0f ? rawTailNorm2 : 0.0f;

        beta = alphaReal;
        tauR = 0.0f;
        tauI = 0.0f;
        if (tailNorm2 == 0.0f && alphaImag == 0.0f)
        {
            ClearDense(vectorReal, active);
            ClearDense(vectorImag, active);
        }
        else
        {
            ScaleReflector(alphaReal, alphaImag, alphaNorm2, tailNorm2, vectorReal, vectorImag, tmp0, tmp1, result,
                           beta, tauR, tauI);
        }
    }

#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 13
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#define CHEEVJ_HH_VARIANT_N CHEEVJ_HH_N1024_STREAMED_N
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 19
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_VARIANT_N

    __aicore__ inline void ComputeAndStoreRowGemv(const AscendC::LocalTensor<float> &matrixReal,
                                                  const AscendC::LocalTensor<float> &matrixImag, int rowBegin)
    {
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto vectorImagNeg = vectorImagNegBuf.Get<float>();
        auto accumulatorReal = accumulatorRealBuf.Get<float>();
        auto accumulatorImag = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();
        auto resultImag = result[CHEEVJ_HH_N1024_STREAMED_ROWS];

        AccumulateGemvSegments(matrixReal, matrixImag, vectorReal, vectorImag, vectorImagNeg, accumulatorReal,
                               accumulatorImag);
        AscendC::WholeReduceSum<float, true>(result, accumulatorReal, CHEEVJ_HH_N1024_STREAMED_SEGMENT,
                                             CHEEVJ_HH_N1024_STREAMED_ROWS, 1, 1, CHEEVJ_HH_N1024_STREAMED_SEGMENT / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float, true>(resultImag, accumulatorImag, CHEEVJ_HH_N1024_STREAMED_SEGMENT,
                                             CHEEVJ_HH_N1024_STREAMED_ROWS, 1, 1, CHEEVJ_HH_N1024_STREAMED_SEGMENT / 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreContiguous(productRealGlobal, rowBegin, result, CHEEVJ_HH_N1024_STREAMED_ROWS);
        StoreContiguous(productImagGlobal, rowBegin, resultImag, CHEEVJ_HH_N1024_STREAMED_ROWS);
    }

#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 8
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
        params.dstRepStride = CHEEVJ_HH_N1024_STREAMED_SEGMENT / 8;
        params.src0RepStride = CHEEVJ_HH_N1024_STREAMED_N / 8;
        params.src1RepStride = 0;
        constexpr uint8_t repeats = CHEEVJ_HH_N1024_STREAMED_ROWS;

        for (int segment = 0; segment < CHEEVJ_HH_N1024_STREAMED_SEGMENTS; ++segment)
        {
            const int offset = segment * CHEEVJ_HH_N1024_STREAMED_SEGMENT;
            if (segment == 0)
            {
                AscendC::Mul(accumulatorReal, matrixReal[offset], vectorReal[offset],
                             static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(accumulatorImag, matrixReal[offset], vectorImag[offset],
                             static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
            else
            {
                AscendC::MulAddDst(accumulatorReal, matrixReal[offset], vectorReal[offset],
                                   static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(accumulatorImag, matrixReal[offset], vectorImag[offset],
                                   static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::MulAddDst(accumulatorReal, matrixImag[offset], vectorImagNeg[offset],
                               static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(accumulatorImag, matrixImag[offset], vectorReal[offset],
                               static_cast<uint64_t>(CHEEVJ_HH_N1024_STREAMED_SEGMENT), repeats, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CorrectAndFormW(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_N1024_STREAMED_N - begin;
        const int columnOffset = step * CHEEVJ_HH_N1024_STREAMED_N;
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto yReal = control0Buf.Get<float>();
        auto yImag = control1Buf.Get<float>();
        auto tmp0 = accumulatorRealBuf.Get<float>();
        auto tmp1 = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();

        if (active <= 0)
        {
            return;
        }
        ClearDense(vectorReal, active);
        ClearDense(vectorImag, active);
        LoadContiguous(vectorReal, internalVRealGlobal, columnOffset + begin, active);
        LoadContiguous(vectorImag, internalVImagGlobal, columnOffset + begin, active);
        ClearActivePadding(vectorReal, active);
        ClearActivePadding(vectorImag, active);
        ClearDense(yReal, active);
        ClearDense(yImag, active);
        LoadContiguous(yReal, productRealGlobal, begin, active);
        LoadContiguous(yImag, productImagGlobal, begin, active);
        ClearActivePadding(yReal, active);
        ClearActivePadding(yImag, active);
        ApplyPreviousWCorrectionsN1024(step, begin, active, vectorReal, vectorImag, yReal, yImag, tmp0, tmp1, result);
        const float tauR = LoadScalar(tauRealGlobal, step, tmp0);
        const float tauI = LoadScalar(tauImagGlobal, step, tmp0);
        MultiplyInPlace(yReal, yImag, tauR, tauI, tmp0, tmp1, active);

        float dotReal = 0.0f;
        float dotImag = 0.0f;
        DotConjugate(yReal, yImag, vectorReal, vectorImag, tmp0, tmp1, result, dotReal, dotImag, active);
        const float gammaReal = -0.5f * (tauR * dotReal - tauI * dotImag);
        const float gammaImag = -0.5f * (tauR * dotImag + tauI * dotReal);
        AddScaled(yReal, yImag, vectorReal, vectorImag, gammaReal, gammaImag, tmp0, tmp1, active);

        StoreContiguous(internalWRealGlobal, columnOffset + begin, yReal, active);
        StoreContiguous(internalWImagGlobal, columnOffset + begin, yImag, active);
        StoreContiguous(wHRealGlobal, columnOffset + begin, yReal, active);
        AscendC::Muls(tmp0, yImag, -1.0f, CHEEVJ_HH_N1024_STREAMED_N);
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 5
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 18
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
            const int previousOffset = previous * CHEEVJ_HH_N1024_STREAMED_N;
            ClearDense(previousReal, active);
            ClearDense(previousImag, active);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            float dotWReal = 0.0f;
            float dotWImag = 0.0f;
            DotConjugate(previousReal, previousImag, vectorReal, vectorImag, tmp0, tmp1, result, dotWReal, dotWImag,
                         active);

            ClearDense(previousReal, active);
            ClearDense(previousImag, active);
            LoadContiguous(previousReal, internalVRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalVImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            float dotVReal = 0.0f;
            float dotVImag = 0.0f;
            DotConjugate(previousReal, previousImag, vectorReal, vectorImag, tmp0, tmp1, result, dotVReal, dotVImag,
                         active);
            SubtractScaled(yReal, yImag, previousReal, previousImag, dotWReal, dotWImag, tmp0, tmp1, active);

            ClearDense(previousReal, active);
            ClearDense(previousImag, active);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(yReal, yImag, previousReal, previousImag, dotVReal, dotVImag, tmp0, tmp1, active);
        }
    }
