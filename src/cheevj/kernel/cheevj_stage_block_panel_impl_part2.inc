/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE

    __aicore__ inline void ProcessSinglePairRightPanelAivSplit(
        const CheevjStageBlockPair16 &pair, int pairIndex, int localDim, int slotIndex, int lane,
        GlobalTensor<float> matrixReal, GlobalTensor<float> matrixImag, GlobalTensor<float> outputReal,
        GlobalTensor<float> outputImag, GlobalTensor<float> outputImagNeg, bool updateImagNeg, bool packUnitary)
    {
        (void)pairIndex;
        (void)packUnitary;
        const int rowsPerLane = (localDim + 1) / 2;
        const int rowBegin = lane * rowsPerLane;
        const int rowEnd = MinInt(localDim, rowBegin + rowsPerLane);
        PackPairRightTransposeRows(pair, localDim, slotIndex, rowBegin, rowEnd, matrixReal, matrixImag);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CHEEVJ_STAGE_PANEL_PACK_READY_FLAG);
        CrossCoreWaitFlag(CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG);
        ScatterPairRightTransposeRows(pair, localDim, slotIndex, rowBegin, rowEnd, outputReal, outputImag,
                                      outputImagNeg, updateImagNeg);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PackPairLeftRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                            int localRowBegin, int localRowEnd)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PreparePanelGatherIndex(gatherLocal, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE);

        GlobalTensor<float> packedReal = PanelSlotInputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotInputImag(slotIndex, localDim);
        GlobalTensor<float> packedImagNeg = LeftPanelImagNeg(slotIndex, localDim);
#define CHEEVJ_PANEL_COMMON_SECTION 2
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
                Muls(denseImagNeg, denseImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                DataCopyExtParams packedParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int packedOffset = localRow * planar.layout.n + colOffset;
                DataCopyPad(packedReal[packedOffset], denseReal, packedParams);
                DataCopyPad(packedImag[packedOffset], denseImag, packedParams);
                DataCopyPad(packedImagNeg[packedOffset], denseImagNeg, packedParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void PackPairLeftRowsBuffered(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                    int bufferIndex, int localRowBegin, int localRowEnd)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PreparePanelGatherIndex(gatherLocal, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE);

        GlobalTensor<float> packedReal = RightPanelBufferReal(slotIndex, localDim, bufferIndex);
        GlobalTensor<float> packedImag = RightPanelBufferImag(slotIndex, localDim, bufferIndex);
#define CHEEVJ_PANEL_COMMON_SECTION 2
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
                DataCopyExtParams packedParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int packedOffset = localRow * planar.layout.n + colOffset;
                DataCopyPad(packedReal[packedOffset], denseReal, packedParams);
                DataCopyPad(packedImag[packedOffset], denseImag, packedParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void PackPairUnitaryHermitianRowMajor(int pairIndex, int localDim)
    {
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagConj = tmp0Buf.Get<float>();
        GlobalTensor<float> storedReal = StoredUnitaryReal(pairIndex);
        GlobalTensor<float> storedImag = StoredUnitaryImag(pairIndex);
        GlobalTensor<float> packedReal = planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 0)];
        GlobalTensor<float> packedImag = planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 1)];
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(localDim * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int dstRow = 0; dstRow < localDim; ++dstRow)
        {
            const int storedOffset = dstRow * UnitaryStorageDim();
            DataCopyPad(denseReal, storedReal[storedOffset], copyParams, padParams);
            DataCopyPad(denseImag, storedImag[storedOffset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Muls(denseImagConj, denseImag, -1.0f, localDim);
            PipeBarrier<PIPE_V>();
            const int packedOffset = dstRow * localDim;
            DataCopyPad(packedReal[packedOffset], denseReal, copyParams);
            DataCopyPad(packedImag[packedOffset], denseImagConj, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ScatterPairLeftRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                               int localRowBegin, int localRowEnd)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> stagedImagNeg = rowStageImagNegBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        GlobalTensor<float> packedReal = PanelSlotOutputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotOutputImag(slotIndex, localDim);
#define CHEEVJ_PANEL_COMMON_SECTION 3
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION

    __aicore__ inline void ScatterPairLeftRowsBuffered(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                       int bufferIndex, int localRowBegin, int localRowEnd)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> stagedImagNeg = rowStageImagNegBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        GlobalTensor<float> packedReal = RightPanelBufferReal(slotIndex, localDim, bufferIndex);
        GlobalTensor<float> packedImag = RightPanelBufferImag(slotIndex, localDim, bufferIndex);
#define CHEEVJ_PANEL_COMMON_SECTION 3
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION

    __aicore__ inline bool ProcessLeftPanelStageAivSplit(int stage, int slotIndex, int lane)
    {
        const int slotCount = PanelServiceSlotCount();
        if (slotIndex < 0 || slotIndex >= slotCount || lane < 0 || lane > 1)
        {
            return false;
        }

        const int pairCount = schedule.PairCountPerStage();
        for (int groupPair = slotIndex; groupPair < pairCount; groupPair += 2 * slotCount)
        {
            PackLeftPanelBuffers(stage, groupPair, pairCount, slotCount, slotIndex, lane);
            ScatterLeftPanelBuffers(stage, groupPair, pairCount, slotCount, slotIndex, lane);
        }
        return true;
    }

    __aicore__ inline void PackLeftPanelBuffers(int stage, int groupPair, int pairCount, int slotCount, int slotIndex,
                                               int lane)
    {
        PackPanelBuffers<true>(stage, groupPair, pairCount, slotCount, slotIndex, lane, planar.aReal, planar.aImag);
    }

    __aicore__ inline void ScatterLeftPanelBuffers(int stage, int groupPair, int pairCount, int slotCount,
                                                  int slotIndex, int lane)
    {
        ScatterPanelBuffers<true>(stage, groupPair, pairCount, slotCount, slotIndex, lane, planar.aReal, planar.aImag,
                                  planar.aImagNeg, true);
    }

    template <bool LeftUpdate>
    __aicore__ inline void PackPanelBuffers(int stage, int groupPair, int pairCount, int slotCount, int slotIndex,
                                           int lane, GlobalTensor<float> matrixReal, GlobalTensor<float> matrixImag)
#define CHEEVJ_PANEL_COMMON_SECTION 5
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
            const int rowsPerLane = (localDim + 1) / 2;
            const int rowBegin = lane * rowsPerLane;
            const int rowEnd = MinInt(localDim, rowBegin + rowsPerLane);
            if constexpr (LeftUpdate)
            {
                PackPairLeftRowsBuffered(pair, localDim, slotIndex, bufferIndex, rowBegin, rowEnd);
            }
            else
            {
                PackPairRightBufferedRows(pair, localDim, slotIndex, bufferIndex, rowBegin, rowEnd, 0,
                                          planar.layout.n, matrixReal, matrixImag);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(bufferIndex == 0 ? CHEEVJ_STAGE_PANEL_PACK_READY_FLAG
                                                              : CHEEVJ_STAGE_PANEL_PACK_READY_FLAG_PONG);
        }
    }

    template <bool LeftUpdate>
    __aicore__ inline void ScatterPanelBuffers(int stage, int groupPair, int pairCount, int slotCount, int slotIndex,
                                              int lane, GlobalTensor<float> outputReal,
                                              GlobalTensor<float> outputImag, GlobalTensor<float> outputImagNeg,
                                              bool updateImagNeg)
#define CHEEVJ_PANEL_COMMON_SECTION 5
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
            CrossCoreWaitFlag(RightPanelGemmDoneFlag(bufferIndex));
            const int rowsPerLane = (localDim + 1) / 2;
            const int rowBegin = lane * rowsPerLane;
            const int rowEnd = MinInt(localDim, rowBegin + rowsPerLane);
            if constexpr (LeftUpdate)
            {
                ScatterPairLeftRowsBuffered(pair, localDim, slotIndex, bufferIndex, rowBegin, rowEnd);
            }
            else
            {
                ScatterPairRightBufferedRows(pair, localDim, slotIndex, bufferIndex, rowBegin, rowEnd, 0,
                                             planar.layout.n, outputReal, outputImag, outputImagNeg, updateImagNeg);
            }
        }
    }

    __aicore__ inline void ProcessSinglePairLeftPanelAivSplit(const CheevjStageBlockPair16 &pair, int pairIndex,
                                                              int localDim, int slotIndex, int lane)
    {
        const int rowsPerLane = (localDim + 1) / 2;
        const int localRowBegin = lane * rowsPerLane;
        const int localRowEnd = MinInt(localDim, localRowBegin + rowsPerLane);
        PackPairLeftRows(pair, localDim, slotIndex, localRowBegin, localRowEnd);
        if (lane == 0)
        {
            PackPairUnitaryHermitianRowMajor(pairIndex, localDim);
        }
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CHEEVJ_STAGE_PANEL_PACK_READY_FLAG);
        CrossCoreWaitFlag(CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG);
        ScatterPairLeftRows(pair, localDim, slotIndex, localRowBegin, localRowEnd);
        PipeBarrier<PIPE_ALL>();
    }
#endif
