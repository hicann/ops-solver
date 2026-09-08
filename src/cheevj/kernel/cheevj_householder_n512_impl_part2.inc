/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


    __aicore__ inline void DotConjugate(const AscendC::LocalTensor<float> &leftReal,
                                        const AscendC::LocalTensor<float> &leftImag,
                                        const AscendC::LocalTensor<float> &rightReal,
                                        const AscendC::LocalTensor<float> &rightImag,
                                        const AscendC::LocalTensor<float> &tmp0,
                                        const AscendC::LocalTensor<float> &tmp1,
                                        const AscendC::LocalTensor<float> &result, float &dotReal, float &dotImag)
    {
        AscendC::Mul(tmp0, leftReal, rightReal, CHEEVJ_HH_PERSISTENT_N);
        AscendC::Mul(tmp1, leftImag, rightImag, CHEEVJ_HH_PERSISTENT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_PERSISTENT_N);
        AscendC::PipeBarrier<PIPE_V>();
        dotReal = CheevjPersistentReduce512(tmp0, tmp1, result);

        AscendC::Mul(tmp0, leftReal, rightImag, CHEEVJ_HH_PERSISTENT_N);
        AscendC::Mul(tmp1, leftImag, rightReal, CHEEVJ_HH_PERSISTENT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp0, tmp1, CHEEVJ_HH_PERSISTENT_N);
        AscendC::PipeBarrier<PIPE_V>();
        dotImag = CheevjPersistentReduce512(tmp0, tmp1, result);
    }

    __aicore__ inline void BuildReflector(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_PERSISTENT_N - begin;
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto tmp0 = accumulatorRealBuf.Get<float>();
        auto tmp1 = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();

        ClearDense(vectorReal);
        ClearDense(vectorImag);
        const int rowOffset = step * CHEEVJ_HH_PERSISTENT_N + begin;
        LoadContiguous(vectorReal, matrixRealGlobal, rowOffset, active);
        LoadContiguous(vectorImag, matrixImagGlobal, rowOffset, active);
        ClearActivePadding(vectorReal, active);
        ClearActivePadding(vectorImag, active);
        AscendC::Muls(vectorImag, vectorImag, -1.0f, CHEEVJ_HH_PERSISTENT_N);
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 2
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION

        const int columnOffset = step * CHEEVJ_HH_PERSISTENT_N;
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 3
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 13
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#define CHEEVJ_HH_VARIANT_N CHEEVJ_HH_PERSISTENT_N
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 19
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#define CHEEVJ_HH_VARIANT_REDUCE CheevjPersistentReduce512
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 15
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_VARIANT_REDUCE
#undef CHEEVJ_HH_VARIANT_N

    template <int VectorN>
    __aicore__ inline void ComputeAndStoreRowGemv(int step, const AscendC::LocalTensor<float> &matrixReal,
                                                  const AscendC::LocalTensor<float> &matrixImag, int rowBegin)
    {
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto vectorImagNeg = vectorImagNegBuf.Get<float>();
        auto accumulatorReal = accumulatorRealBuf.Get<float>();
        auto accumulatorImag = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();
        auto resultImag = result[CHEEVJ_HH_PERSISTENT_ROWS];

        const int columnOffset = step * CHEEVJ_HH_PERSISTENT_N;
        LoadContiguous(vectorReal, internalVRealGlobal, columnOffset, CHEEVJ_HH_PERSISTENT_N);
        LoadContiguous(vectorImag, internalVImagGlobal, columnOffset, CHEEVJ_HH_PERSISTENT_N);
        AscendC::Muls(vectorImagNeg, vectorImag, -1.0f, CHEEVJ_HH_PERSISTENT_N);
        AscendC::PipeBarrier<PIPE_V>();

        AccumulateGemvSegments<VectorN>(matrixReal, matrixImag, vectorReal, vectorImag, vectorImagNeg, accumulatorReal,
                                        accumulatorImag);
        AscendC::WholeReduceSum<float, true>(result, accumulatorReal, CHEEVJ_HH_PERSISTENT_SEGMENT,
                                             CHEEVJ_HH_PERSISTENT_ROWS, 1, 1, CHEEVJ_HH_PERSISTENT_SEGMENT / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float, true>(resultImag, accumulatorImag, CHEEVJ_HH_PERSISTENT_SEGMENT,
                                             CHEEVJ_HH_PERSISTENT_ROWS, 1, 1, CHEEVJ_HH_PERSISTENT_SEGMENT / 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreContiguous(productRealGlobal, rowBegin, result, CHEEVJ_HH_PERSISTENT_ROWS);
        StoreContiguous(productImagGlobal, rowBegin, resultImag, CHEEVJ_HH_PERSISTENT_ROWS);
    }

    template <int VectorN>
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 8
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
        params.dstRepStride = CHEEVJ_HH_PERSISTENT_SEGMENT / 8;
        params.src0RepStride = CHEEVJ_HH_PERSISTENT_N / 8;
        params.src1RepStride = 0;
        constexpr uint8_t repeats = CHEEVJ_HH_PERSISTENT_ROWS;

        constexpr int segments = VectorN / CHEEVJ_HH_PERSISTENT_SEGMENT;
        for (int segment = 0; segment < segments; ++segment)
        {
            const int offset = segment * CHEEVJ_HH_PERSISTENT_SEGMENT;
            if (segment == 0)
            {
                AscendC::Mul(accumulatorReal, matrixReal[offset], vectorReal[offset],
                             static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(accumulatorImag, matrixReal[offset], vectorImag[offset],
                             static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
            else
            {
                AscendC::MulAddDst(accumulatorReal, matrixReal[offset], vectorReal[offset],
                                   static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(accumulatorImag, matrixReal[offset], vectorImag[offset],
                                   static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::MulAddDst(accumulatorReal, matrixImag[offset], vectorImagNeg[offset],
                               static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(accumulatorImag, matrixImag[offset], vectorReal[offset],
                               static_cast<uint64_t>(CHEEVJ_HH_PERSISTENT_SEGMENT), repeats, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CorrectAndFormW(int step)
    {
        const int begin = step + 1;
        const int active = CHEEVJ_HH_PERSISTENT_N - begin;
        const int columnOffset = step * CHEEVJ_HH_PERSISTENT_N;
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 4
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
        AscendC::Muls(tmp0, yImag, -1.0f, CHEEVJ_HH_PERSISTENT_N);
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 5
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    }
#define CHEEVJ_HH_VARIANT_N CHEEVJ_HH_PERSISTENT_N
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 16
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_VARIANT_N
