/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 11
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
            ProcessWorkerWaves(block, step, matrixReal, matrixImag);
        }
        PanelBarrier(barrierLocal);
        if (block == 0)
        {
            CorrectAndFormW(step);
        }
    }

    __aicore__ inline void ProcessWorkerWaves(int block, int step, const AscendC::LocalTensor<float> &matrixReal,
                                             const AscendC::LocalTensor<float> &matrixImag)
    {
        LoadReflectorForGemv(step);
        auto pairResult = resultBuf.Get<float>();
        auto pairResultImag = pairResult[8];
        for (int waveGroup = 0; waveGroup < CHEEVJ_HH_N2048_STREAMED_WAVES / 2; ++waveGroup)
        {
            const int rowBegin = (waveGroup * CHEEVJ_HH_N2048_STREAMED_ROW_WORKERS + block - 1) *
                                 (2 * CHEEVJ_HH_N2048_STREAMED_ROWS);
            if (rowBegin >= activeN)
            {
                continue;
            }
            AscendC::Duplicate(pairResult, 0.0f, 16);
            AscendC::PipeBarrier<PIPE_V>();
            LoadSlab(matrixReal, matrixRealGlobal, rowBegin * CHEEVJ_HH_N2048_STREAMED_N);
            LoadSlab(matrixImag, matrixImagGlobal, rowBegin * CHEEVJ_HH_N2048_STREAMED_N);
            AscendC::PipeBarrier<PIPE_ALL>();
            ComputeRowGemv(matrixReal, matrixImag, pairResult, pairResultImag, 0);

            const int secondRowBegin = rowBegin + CHEEVJ_HH_N2048_STREAMED_ROWS;
            if (secondRowBegin < activeN)
            {
                LoadSlab(matrixReal, matrixRealGlobal, secondRowBegin * CHEEVJ_HH_N2048_STREAMED_N);
                LoadSlab(matrixImag, matrixImagGlobal, secondRowBegin * CHEEVJ_HH_N2048_STREAMED_N);
                AscendC::PipeBarrier<PIPE_ALL>();
                ComputeRowGemv(matrixReal, matrixImag, pairResult, pairResultImag, CHEEVJ_HH_N2048_STREAMED_ROWS);
            }
            StoreContiguous(productRealGlobal, rowBegin, pairResult, 2 * CHEEVJ_HH_N2048_STREAMED_ROWS);
            StoreContiguous(productImagGlobal, rowBegin, pairResultImag, 2 * CHEEVJ_HH_N2048_STREAMED_ROWS);
        }
    }

#define CHEEVJ_HH_COMMON_WS_V_REAL CHEEVJ_HH_N2048_STREAMED_WS_V_REAL
#define CHEEVJ_HH_COMMON_WS_V_IMAG CHEEVJ_HH_N2048_STREAMED_WS_V_IMAG
#define CHEEVJ_HH_COMMON_WS_W_REAL CHEEVJ_HH_N2048_STREAMED_WS_W_REAL
#define CHEEVJ_HH_COMMON_WS_W_IMAG CHEEVJ_HH_N2048_STREAMED_WS_W_IMAG
#define CHEEVJ_HH_COMMON_WS_Y_REAL CHEEVJ_HH_N2048_STREAMED_WS_Y_REAL
#define CHEEVJ_HH_COMMON_WS_Y_IMAG CHEEVJ_HH_N2048_STREAMED_WS_Y_IMAG
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 1
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HH_COMMON_WS_V_REAL
#undef CHEEVJ_HH_COMMON_WS_V_IMAG
#undef CHEEVJ_HH_COMMON_WS_W_REAL
#undef CHEEVJ_HH_COMMON_WS_W_IMAG
#undef CHEEVJ_HH_COMMON_WS_Y_REAL
#undef CHEEVJ_HH_COMMON_WS_Y_IMAG
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION

    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        constexpr int vectorBytes = CHEEVJ_HH_N2048_STREAMED_N * sizeof(float);
        constexpr int slabBytes = CHEEVJ_HH_N2048_STREAMED_SLAB * sizeof(float);
        constexpr int accumulatorBytes = CHEEVJ_HH_N2048_STREAMED_ACCUM * sizeof(float);
        constexpr int scatterBytes = CHEEVJ_HH_N2048_STREAMED_SCATTER * sizeof(float);
        pipe->InitBuffer(matrixRealBuf, slabBytes);
        pipe->InitBuffer(matrixImagBuf, slabBytes);
        pipe->InitBuffer(vectorRealBuf, vectorBytes);
        pipe->InitBuffer(vectorImagBuf, vectorBytes);
        pipe->InitBuffer(vectorImagNegBuf, vectorBytes);
        pipe->InitBuffer(accumulatorRealBuf, accumulatorBytes);
        pipe->InitBuffer(accumulatorImagBuf, accumulatorBytes);
        pipe->InitBuffer(resultBuf, CHEEVJ_HH_N2048_STREAMED_SEGMENT * sizeof(float));
        pipe->InitBuffer(control0Buf, vectorBytes);
        pipe->InitBuffer(control1Buf, vectorBytes);
        pipe->InitBuffer(control2Buf, vectorBytes);
        pipe->InitBuffer(control3Buf, vectorBytes);
        pipe->InitBuffer(scatterBuf, scatterBytes);
        pipe->InitBuffer(scatterIndexBuf, scatterBytes);
        pipe->InitBuffer(barrierBuf, CHEEVJ_HH_N2048_STREAMED_BARRIER_INTS * sizeof(int32_t));
    }

#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 10
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
    int activeN = CHEEVJ_HH_N2048_STREAMED_N;

    __aicore__ inline void PanelBarrier(const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>(barrierGlobal, barrierLocal, CHEEVJ_HH_N2048_STREAMED_PARTICIPANTS);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ClearDense(const AscendC::LocalTensor<float> &tensor)
    {
        AscendC::Duplicate(tensor, 0.0f, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClearActivePadding(const AscendC::LocalTensor<float> &tensor, int active)
    {
        // DataCopyPad may expose adjacent GM words in the final aligned block
        // when pad=false.  All nb=4 active lengths are 2044..2047, so mask the
        // inactive lanes in the last 64-lane repeat before a fixed-2048 op.
        constexpr int repeatBase = CHEEVJ_HH_N2048_STREAMED_N - 64;
        const int firstPaddingLane = active - repeatBase;
        uint64_t mask[2] = {~0ULL << firstPaddingLane, 0ULL};
        AscendC::Duplicate(tensor[repeatBase], 0.0f, mask, static_cast<uint8_t>(1), 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadSlab(const AscendC::LocalTensor<float> &destination,
                                    const AscendC::GlobalTensor<float> &source, int offset)
    {
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(CHEEVJ_HH_N2048_STREAMED_ROWS),
                                        static_cast<uint32_t>(CHEEVJ_HH_N2048_STREAMED_N * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(destination, source[offset], copy, pad);
    }

#define CHEEVJ_HH_SCATTER_INDEX_COUNT CHEEVJ_HH_N2048_STREAMED_SCATTER
#define CHEEVJ_HOUSEHOLDER_COMMON_SECTION 9
#include "cheevj_householder_common.inc"
#undef CHEEVJ_HOUSEHOLDER_COMMON_SECTION
#undef CHEEVJ_HH_SCATTER_INDEX_COUNT

    __aicore__ inline void ScatterPanelColumn(AscendC::GlobalTensor<float> destination, int step, int begin,
                                              const AscendC::LocalTensor<float> &source)
    {
        const int count = CHEEVJ_HH_N2048_STREAMED_N - begin;
        auto staged = scatterBuf.Get<float>();
        auto index = scatterIndexBuf.Get<uint32_t>();
        for (int denseBase = 0; denseBase < count; denseBase += CHEEVJ_HH_N2048_STREAMED_SCATTER_VALUES)
        {
            const int remaining = count - denseBase;
            const int valid = remaining < CHEEVJ_HH_N2048_STREAMED_SCATTER_VALUES
                                  ? remaining
                                  : CHEEVJ_HH_N2048_STREAMED_SCATTER_VALUES;
            AscendC::Duplicate(staged, 0.0f, CHEEVJ_HH_N2048_STREAMED_SCATTER);
            AscendC::PipeBarrier<PIPE_V>();

            uint64_t sparseMask[2] = {0x0101010101010101ULL, 0ULL};
            const int fullRepeats = valid / 8;
            if (fullRepeats > 0)
            {
                AscendC::Gather(staged, source[denseBase], index, 0, sparseMask,
                                static_cast<uint8_t>(fullRepeats), 8);
            }
            const int tail = valid - fullRepeats * 8;
            if (tail > 0)
            {
                const uint64_t tailBits = 0x0101010101010101ULL >> ((8 - tail) * 8);
                uint64_t tailMask[2] = {tailBits, 0ULL};
                const int sparseOffset = fullRepeats * 64;
                AscendC::Gather(staged[sparseOffset], source[denseBase], index[sparseOffset], 0, tailMask,
                                static_cast<uint8_t>(1), 8);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(valid), static_cast<uint32_t>(sizeof(float)), 0,
                                            static_cast<uint32_t>((CHEEVJ_HH_N2048_STREAMED_LD - 1) * sizeof(float)),
                                            0};
            const int outputOffset = (begin + denseBase) * CHEEVJ_HH_N2048_STREAMED_LD + step;
            AscendC::DataCopyPad(destination[outputOffset], staged, copy);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void SubtractScaled(const AscendC::LocalTensor<float> &destinationReal,
                                          const AscendC::LocalTensor<float> &destinationImag,
                                          const AscendC::LocalTensor<float> &sourceReal,
                                          const AscendC::LocalTensor<float> &sourceImag, float scaleReal,
                                          float scaleImag, const AscendC::LocalTensor<float> &tmp0,
                                          const AscendC::LocalTensor<float> &tmp1)
    {
        AscendC::Muls(tmp0, sourceReal, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(tmp1, sourceImag, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(destinationReal, destinationReal, tmp0, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(tmp0, sourceImag, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(tmp1, sourceReal, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(destinationImag, destinationImag, tmp0, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddScaled(const AscendC::LocalTensor<float> &destinationReal,
                                     const AscendC::LocalTensor<float> &destinationImag,
                                     const AscendC::LocalTensor<float> &sourceReal,
                                     const AscendC::LocalTensor<float> &sourceImag, float scaleReal, float scaleImag,
                                     const AscendC::LocalTensor<float> &tmp0, const AscendC::LocalTensor<float> &tmp1)
    {
        AscendC::Muls(tmp0, sourceReal, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(tmp1, sourceImag, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(destinationReal, destinationReal, tmp0, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(tmp0, sourceImag, scaleReal, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::Muls(tmp1, sourceReal, scaleImag, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(destinationImag, destinationImag, tmp0, CHEEVJ_HH_N2048_STREAMED_N);
        AscendC::PipeBarrier<PIPE_V>();
    }
