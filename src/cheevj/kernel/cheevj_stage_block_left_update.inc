/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void ApplyLeftUnitaryRows(const CheevjStageBlockPair16 &pair, int localDim,
                                                LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
        for (int colOffset = 0; colOffset < planar.layout.n; colOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - colOffset);
            ApplyLeftUnitaryRowsTile(pair, localDim, unitaryReal, unitaryImag, colOffset, elems);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ApplyLeftUnitaryRowsDistributed(int stage, int waveStart, int wavePairCount,
                                                           LocalTensor<float> unitaryReal,
                                                           LocalTensor<float> unitaryImag)
    {
        const int colTiles =
            (planar.layout.n + CHEEVJ_STAGE_BLOCK_JACOBI16_TILE - 1) / CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
        const int workerCount = CheevjWorkspaceWorkerCount() > 0 ? CheevjWorkspaceWorkerCount() : 1;
        const int totalTasks = wavePairCount * colTiles;
        for (int task = GetBlockIdx(); task < totalTasks; task += workerCount)
        {
            const int localPair = task / colTiles;
            const int tileIndex = task - localPair * colTiles;
            const int pairIndex = waveStart + localPair;
            const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
            const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
            if (CanUseSinglePairRightPanel(pair, localDim))
            {
                continue;
            }
#endif
            if (!pair.valid || localDim <= 1 || localDim > CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM)
            {
                continue;
            }
            LoadUnitaryForPair(pairIndex, unitaryReal, unitaryImag);
            const int colOffset = tileIndex * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - colOffset);
            ApplyLeftUnitaryRowsTile(pair, localDim, unitaryReal, unitaryImag, colOffset, elems);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ApplyLeftUnitaryRowsTile(const CheevjStageBlockPair16 &pair, int localDim,
                                                    LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                    int colOffset, int elems)
    {
        LocalTensor<float> leftRowsReal = leftRowsRealBuf.Get<float>();
        LocalTensor<float> leftRowsImag = leftRowsImagBuf.Get<float>();
        LocalTensor<float> accReal = accRealBuf.Get<float>();
        LocalTensor<float> accImag = accImagBuf.Get<float>();

        LoadLeftSourceRows(pair, localDim, colOffset, elems, leftRowsReal, leftRowsImag);

        for (int dstLocalRow = 0; dstLocalRow < localDim; ++dstLocalRow)
        {
            Duplicate(accReal, 0.0f, elems);
            Duplicate(accImag, 0.0f, elems);
            PipeBarrier<PIPE_V>();
            for (int srcLocalRow = 0; srcLocalRow < localDim; ++srcLocalRow)
            {
                AccumulateScaledComplex(accReal, accImag, leftRowsReal[LeftRowsOffset(srcLocalRow, 0)],
                                        leftRowsImag[LeftRowsOffset(srcLocalRow, 0)],
                                        unitaryReal.GetValue(LocalOffset(srcLocalRow, dstLocalRow)),
                                        -unitaryImag.GetValue(LocalOffset(srcLocalRow, dstLocalRow)), elems);
            }
            WriteMatrixRowTile(GlobalIndexForLocal(pair, dstLocalRow), colOffset, elems, accReal, accImag);
        }
    }

    __aicore__ inline void LoadLeftSourceRows(const CheevjStageBlockPair16 &pair, int localDim, int colOffset,
                                              int elems, LocalTensor<float> leftRowsReal,
                                              LocalTensor<float> leftRowsImag)
    {
        for (int srcLocalRow = 0; srcLocalRow < localDim; ++srcLocalRow)
        {
            const int globalRow = GlobalIndexForLocal(pair, srcLocalRow);
            LoadMatrixRowTile(planar.aReal, globalRow, colOffset, elems, leftRowsReal[LeftRowsOffset(srcLocalRow, 0)],
                              rowStageRealBuf.Get<float>());
            LoadMatrixRowTile(planar.aImag, globalRow, colOffset, elems, leftRowsImag[LeftRowsOffset(srcLocalRow, 0)],
                              rowStageImagBuf.Get<float>());
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadMatrixRowTile(GlobalTensor<float> plane, int globalRow, int colOffset, int elems,
                                             LocalTensor<float> dense, LocalTensor<float> staged)
    {
        const int stageElems = elems * 8;
        Duplicate(staged, 0.0f, stageElems);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams rowCopyParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)),
                                        static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(staged, plane[RowOffset(globalRow, colOffset)], rowCopyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        CompactStagedTileToDense(dense, staged, elems);
    }

    __aicore__ inline void CompactStagedTileToDense(LocalTensor<float> dense, LocalTensor<float> staged, int elems)
    {
        LocalTensor<uint32_t> gatherLocal = tmp1Buf.Get<uint32_t>();
        PrepareStridedFloatGather(gatherLocal, elems);
        Gather(dense, staged, gatherLocal, 0, elems);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareStridedFloatGather(LocalTensor<uint32_t> gatherLocal, int elems)
    {
        LocalTensor<int32_t> signedIndex = gatherLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherLocal, gatherLocal, static_cast<uint32_t>(5), elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteMatrixRowTile(int globalRow, int colOffset, int elems, LocalTensor<float> realDense,
                                              LocalTensor<float> imagDense)
    {
        LocalTensor<float> rowStageReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> rowStageImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> rowStageImagNeg = rowStageImagNegBuf.Get<float>();
        LocalTensor<float> imagNegDense = tmp0Buf.Get<float>();

        StageDenseTileForStridedRow(rowStageReal, realDense, elems);
        StageDenseTileForStridedRow(rowStageImag, imagDense, elems);
        Muls(imagNegDense, imagDense, -1.0f, elems);
        PipeBarrier<PIPE_V>();
        StageDenseTileForStridedRow(rowStageImagNeg, imagNegDense, elems);

        DataCopyExtParams rowCopyParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)), 0,
                                        static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float)), 0};
        const int rowOffset = RowOffset(globalRow, colOffset);
        DataCopyPad(planar.aReal[rowOffset], rowStageReal, rowCopyParams);
        DataCopyPad(planar.aImag[rowOffset], rowStageImag, rowCopyParams);
        DataCopyPad(planar.aImagNeg[rowOffset], rowStageImagNeg, rowCopyParams);
    }

    __aicore__ inline void StageDenseTileForStridedRow(LocalTensor<float> staged, LocalTensor<float> dense, int elems)
    {
        const int stageElems = elems * 8;
        Duplicate(staged, 0.0f, stageElems);
        PipeBarrier<PIPE_V>();

        constexpr int indexElems = CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * 8;
        LocalTensor<int32_t> signedIndex = tmp1Buf.Get<int32_t>();
        CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(indexElems));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> gatherIndex = signedIndex.template ReinterpretCast<uint32_t>();
        ShiftRight(gatherIndex, gatherIndex, static_cast<uint32_t>(3), indexElems);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, static_cast<uint32_t>(2), indexElems);
        PipeBarrier<PIPE_V>();

        uint64_t sparseMask[2] = {0x0101010101010101ULL, 0ULL};
        const int fullRepeats = elems / 8;
        if (fullRepeats > 0)
        {
            Gather(staged, dense, gatherIndex, 0, sparseMask, static_cast<uint8_t>(fullRepeats), 8);
        }
        const int tail = elems - fullRepeats * 8;
        if (tail > 0)
        {
            uint64_t tailBits = 0;
            for (int idx = 0; idx < tail; ++idx)
            {
                tailBits |= 1ULL << (idx * 8);
            }
            uint64_t tailMask[2] = {tailBits, 0ULL};
            const int sparseOffset = fullRepeats * 64;
            Gather(staged[sparseOffset], dense, gatherIndex[sparseOffset], 0, tailMask, static_cast<uint8_t>(1), 8);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ScatterPrincipalBlock(const CheevjStageBlockPair16 &pair, int localDim,
                                                 LocalTensor<float> blockReal, LocalTensor<float> blockImag)
    {
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalIndexForLocal(pair, localCol);
#define CHEEVJ_JACOBI_COMMON_SECTION 5
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        PipeBarrier<PIPE_ALL>();
#define CHEEVJ_JACOBI_COMMON_SECTION 3
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
