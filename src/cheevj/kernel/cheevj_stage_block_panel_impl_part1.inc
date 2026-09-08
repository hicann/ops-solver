/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
    __aicore__ inline bool CanUseSinglePairRightPanel(const CheevjStageBlockPair16 &pair, int localDim) const
    {
        if (planar.layout.n < CHEEVJ_STAGE_PANEL_MIN_N || planar.layout.n % CHEEVJ_STAGE_PANEL_N_ALIGN != 0 ||
            !pair.valid || localDim <= 1 || localDim > CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM ||
            planar.layout.scratchPlanes < CHEEVJ_STAGE_BLOCK_JACOBI16_SCRATCH_PLANES)
        {
            return false;
        }
        const int slotCount = ComputePanelServiceSlotCount(localDim);
        if (slotCount <= 0)
        {
            return false;
        }
        const int maxPairIndex = schedule.PairCountPerStage() - 1;
        const int packedUnitaryEnd = PackedUnitaryStorageOffset(maxPairIndex, 2) + UnitaryStorageElems();
        return packedUnitaryEnd <= planar.layout.planeElems;
    }

    __aicore__ inline void ConfigureSinglePairRightPanelUpdate(CheevjPanelUpdate *panelUpdate,
                                                               const CheevjStageBlockPair16 &pair, int pairIndex,
                                                               int slotIndex, int localDim, int packReadyFlag,
                                                               int gemmDoneFlag)
    {
        panelUpdate->SetPairShape(planar.layout.n, planar.layout.strideN, pair.left.begin, pair.left.size,
                                  pair.right.begin, pair.right.size);
        panelUpdate->SetFlagIds(packReadyFlag, gemmDoneFlag);
        panelUpdate->SetMatrix(planar.aReal, planar.aImag);
        panelUpdate->SetColumnMajorUnitary(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                                           UnitaryStorageDim());
        panelUpdate->SetUnitary(planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 0)],
                                planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 1)],
                                planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 2)]);
        panelUpdate->SetPacked(PanelSlotInputReal(slotIndex, localDim), PanelSlotInputImag(slotIndex, localDim),
                               PanelSlotOutputReal(slotIndex, localDim), PanelSlotOutputImag(slotIndex, localDim));
        panelUpdate->SetOutput(planar.aReal, planar.aImag, planar.aImagNeg);
    }

    __aicore__ inline void PreparePanelGatherIndex(LocalTensor<uint32_t> gatherLocal, int elems)
    {
        LocalTensor<int32_t> signedIndex = gatherLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherLocal, gatherLocal, static_cast<uint32_t>(5), elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackPairPanelRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                             int rowBegin, int rowEnd, GlobalTensor<float> matrixReal,
                                             GlobalTensor<float> matrixImag)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PreparePanelGatherIndex(gatherLocal, localDim);

        GlobalTensor<float> packedReal = PanelSlotInputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotInputImag(slotIndex, localDim);
        const uint32_t srcStride = static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float));
        DataCopyExtParams leftParams{static_cast<uint16_t>(pair.left.size), static_cast<uint32_t>(sizeof(float)),
                                     srcStride, 0, 0};
        DataCopyExtParams rightParams{static_cast<uint16_t>(pair.right.size), static_cast<uint32_t>(sizeof(float)),
                                      srcStride, 0, 0};
        DataCopyExtParams packedParams{1, static_cast<uint32_t>(localDim * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int row = rowBegin; row < rowEnd; ++row)
        {
            Duplicate(stagedReal, 0.0f, localDim * 8);
            Duplicate(stagedImag, 0.0f, localDim * 8);
            PipeBarrier<PIPE_V>();
            DataCopyPad(stagedReal, matrixReal[ColumnOffset(pair.left.begin, row)], leftParams, padParams);
            DataCopyPad(stagedImag, matrixImag[ColumnOffset(pair.left.begin, row)], leftParams, padParams);
            DataCopyPad(stagedReal[pair.left.size * 8], matrixReal[ColumnOffset(pair.right.begin, row)], rightParams,
                        padParams);
            DataCopyPad(stagedImag[pair.left.size * 8], matrixImag[ColumnOffset(pair.right.begin, row)], rightParams,
                        padParams);
            PipeBarrier<PIPE_ALL>();
            Gather(denseReal, stagedReal, gatherLocal, 0, localDim);
            Gather(denseImag, stagedImag, gatherLocal, 0, localDim);
            PipeBarrier<PIPE_ALL>();
            const int packedOffset = row * localDim;
            DataCopyPad(packedReal[packedOffset], denseReal, packedParams);
            DataCopyPad(packedImag[packedOffset], denseImag, packedParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void PackPairRightTransposeRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                      int localRowBegin, int localRowEnd,
                                                      GlobalTensor<float> matrixReal, GlobalTensor<float> matrixImag)
    {
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        GlobalTensor<float> packedReal = PanelSlotInputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotInputImag(slotIndex, localDim);
        GlobalTensor<float> packedImagNeg = LeftPanelImagNeg(slotIndex, localDim);
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localRow = localRowBegin; localRow < localRowEnd; ++localRow)
        {
            const int globalCol = GlobalIndexForLocal(pair, localRow);
            for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int matrixOffset = ColumnOffset(globalCol, rowOffset);
                DataCopyPad(denseReal, matrixReal[matrixOffset], copyParams, padParams);
                DataCopyPad(denseImag, matrixImag[matrixOffset], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();
                Muls(denseImagNeg, denseImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                const int packedOffset = localRow * planar.layout.n + rowOffset;
                DataCopyPad(packedReal[packedOffset], denseReal, copyParams);
                DataCopyPad(packedImag[packedOffset], denseImag, copyParams);
                DataCopyPad(packedImagNeg[packedOffset], denseImagNeg, copyParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ScatterPairRightTransposeRows(const CheevjStageBlockPair16 &pair, int localDim,
                                                         int slotIndex, int localRowBegin, int localRowEnd,
                                                         GlobalTensor<float> outputReal, GlobalTensor<float> outputImag,
                                                         GlobalTensor<float> outputImagNeg, bool updateImagNeg)
    {
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        GlobalTensor<float> packedReal = PanelSlotOutputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotOutputImag(slotIndex, localDim);
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localRow = localRowBegin; localRow < localRowEnd; ++localRow)
        {
            const int globalCol = GlobalIndexForLocal(pair, localRow);
            for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
#define CHEEVJ_PANEL_COMMON_SECTION 4
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION

    __aicore__ inline void PackPairRightBufferedRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                     int bufferIndex, int localRowBegin, int localRowEnd,
                                                     int sliceBegin, int sliceEnd, GlobalTensor<float> matrixReal,
                                                     GlobalTensor<float> matrixImag)
    {
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
#define CHEEVJ_PANEL_COMMON_SECTION 1
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int matrixOffset = ColumnOffset(globalCol, rowOffset);
                DataCopyPad(denseReal, matrixReal[matrixOffset], copyParams, padParams);
                DataCopyPad(denseImag, matrixImag[matrixOffset], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();
                const int packedOffset = localRow * planar.layout.n + rowOffset;
                DataCopyPad(packedReal[packedOffset], denseReal, copyParams);
                DataCopyPad(packedImag[packedOffset], denseImag, copyParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ScatterPairRightBufferedRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                        int bufferIndex, int localRowBegin, int localRowEnd,
                                                        int sliceBegin, int sliceEnd, GlobalTensor<float> outputReal,
                                                        GlobalTensor<float> outputImag,
                                                        GlobalTensor<float> outputImagNeg, bool updateImagNeg)
    {
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
#define CHEEVJ_PANEL_COMMON_SECTION 1
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION
#define CHEEVJ_PANEL_COMMON_SECTION 4
#include "cheevj_panel_common.inc"
#undef CHEEVJ_PANEL_COMMON_SECTION

    __aicore__ inline bool ProcessRightPanelStageAivSplit(int stage, int slotIndex, int lane,
                                                          GlobalTensor<float> matrixReal,
                                                          GlobalTensor<float> matrixImag,
                                                          GlobalTensor<float> outputReal,
                                                          GlobalTensor<float> outputImag,
                                                          GlobalTensor<float> outputImagNeg, bool updateImagNeg)
    {
        const int slotCount = PanelServiceSlotCount();
        if (slotIndex < 0 || slotIndex >= slotCount || lane < 0 || lane > 1)
        {
            return false;
        }

        const int pairCount = schedule.PairCountPerStage();
        for (int groupPair = slotIndex; groupPair < pairCount; groupPair += 2 * slotCount)
        {
            PackRightPanelBuffers(stage, groupPair, pairCount, slotCount, slotIndex, lane, matrixReal, matrixImag);
            ScatterRightPanelBuffers(stage, groupPair, pairCount, slotCount, slotIndex, lane, outputReal, outputImag,
                                     outputImagNeg, updateImagNeg);
        }
        return true;
    }

    __aicore__ inline void PackRightPanelBuffers(int stage, int groupPair, int pairCount, int slotCount, int slotIndex,
                                                int lane, GlobalTensor<float> matrixReal,
                                                GlobalTensor<float> matrixImag)
    {
        PackPanelBuffers<false>(stage, groupPair, pairCount, slotCount, slotIndex, lane, matrixReal, matrixImag);
    }

    __aicore__ inline void ScatterRightPanelBuffers(int stage, int groupPair, int pairCount, int slotCount,
                                                   int slotIndex, int lane, GlobalTensor<float> outputReal,
                                                   GlobalTensor<float> outputImag, GlobalTensor<float> outputImagNeg,
                                                   bool updateImagNeg)
    {
        ScatterPanelBuffers<false>(stage, groupPair, pairCount, slotCount, slotIndex, lane, outputReal, outputImag,
                                   outputImagNeg, updateImagNeg);
    }

    __aicore__ inline void PackPairUnitaryRowMajor(int pairIndex, int localDim)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PreparePanelGatherIndex(gatherLocal, localDim);

        GlobalTensor<float> packedReal = planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 0)];
        GlobalTensor<float> packedImag = planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 1)];
        GlobalTensor<float> packedImagNeg = planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 2)];
        GlobalTensor<float> storedReal = StoredUnitaryReal(pairIndex);
        GlobalTensor<float> storedImag = StoredUnitaryImag(pairIndex);
        const uint32_t srcStride = static_cast<uint32_t>((UnitaryStorageDim() - 1) * sizeof(float));
        DataCopyExtParams rowParams{static_cast<uint16_t>(localDim), static_cast<uint32_t>(sizeof(float)), srcStride, 0,
                                    0};
        DataCopyExtParams packedParams{1, static_cast<uint32_t>(localDim * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int row = 0; row < localDim; ++row)
        {
            Duplicate(stagedReal, 0.0f, localDim * 8);
            Duplicate(stagedImag, 0.0f, localDim * 8);
            PipeBarrier<PIPE_V>();
            DataCopyPad(stagedReal, storedReal[row], rowParams, padParams);
            DataCopyPad(stagedImag, storedImag[row], rowParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Gather(denseReal, stagedReal, gatherLocal, 0, localDim);
            Gather(denseImag, stagedImag, gatherLocal, 0, localDim);
            PipeBarrier<PIPE_ALL>();
            Muls(denseImagNeg, denseImag, -1.0f, localDim);
            PipeBarrier<PIPE_V>();
            const int packedOffset = row * localDim;
            DataCopyPad(packedReal[packedOffset], denseReal, packedParams);
            DataCopyPad(packedImag[packedOffset], denseImag, packedParams);
            DataCopyPad(packedImagNeg[packedOffset], denseImagNeg, packedParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ScatterPairPanelRows(const CheevjStageBlockPair16 &pair, int localDim, int slotIndex,
                                                int rowBegin, int rowEnd, GlobalTensor<float> outputReal,
                                                GlobalTensor<float> outputImag, GlobalTensor<float> outputImagNeg,
                                                bool updateImagNeg)
    {
        LocalTensor<float> stagedReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> stagedImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> denseReal = srcRealBuf.Get<float>();
        LocalTensor<float> denseImag = srcImagBuf.Get<float>();
        LocalTensor<float> denseImagNeg = tmp0Buf.Get<float>();
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PreparePanelGatherIndex(gatherLocal, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE);

        GlobalTensor<float> packedReal = PanelSlotOutputReal(slotIndex, localDim);
        GlobalTensor<float> packedImag = PanelSlotOutputImag(slotIndex, localDim);
        const uint32_t srcStride = static_cast<uint32_t>((localDim - 1) * sizeof(float));
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalIndexForLocal(pair, localCol);
            for (int rowOffset = rowBegin; rowOffset < rowEnd; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, rowEnd - rowOffset);
                Duplicate(stagedReal, 0.0f, elems * 8);
                Duplicate(stagedImag, 0.0f, elems * 8);
                PipeBarrier<PIPE_V>();
                DataCopyExtParams gatherParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)),
                                               srcStride, 0, 0};
                const int packedOffset = rowOffset * localDim + localCol;
                DataCopyPad(stagedReal, packedReal[packedOffset], gatherParams, padParams);
                DataCopyPad(stagedImag, packedImag[packedOffset], gatherParams, padParams);
                PipeBarrier<PIPE_ALL>();
                Gather(denseReal, stagedReal, gatherLocal, 0, elems);
                Gather(denseImag, stagedImag, gatherLocal, 0, elems);
                PipeBarrier<PIPE_ALL>();
                if (updateImagNeg)
                {
                    Muls(denseImagNeg, denseImag, -1.0f, elems);
                    PipeBarrier<PIPE_V>();
                }
                DataCopyExtParams outputParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int outputOffset = ColumnOffset(globalCol, rowOffset);
                DataCopyPad(outputReal[outputOffset], denseReal, outputParams);
                DataCopyPad(outputImag[outputOffset], denseImag, outputParams);
                if (updateImagNeg)
                {
                    DataCopyPad(outputImagNeg[outputOffset], denseImagNeg, outputParams);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
#endif
