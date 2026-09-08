/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void BuildRightScratch(const CheevjStageBlockPair16 &pair, int localDim,
                                             LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
            BuildRightScratchTile(pair, localDim, unitaryReal, unitaryImag, rowOffset, elems);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ApplyRightUnitaryToMatrixColumnsFused(const CheevjStageBlockPair16 &pair, int localDim,
                                                                 LocalTensor<float> unitaryReal,
                                                                 LocalTensor<float> unitaryImag)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
            BuildRightScratchTile(pair, localDim, unitaryReal, unitaryImag, rowOffset, elems);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ApplyRightUnitaryToMatrixColumnsDistributed(int stage, int waveStart, int wavePairCount,
                                                                       LocalTensor<float> unitaryReal,
                                                                       LocalTensor<float> unitaryImag)
    {
        ApplyRightUnitaryToPlaneColumnsDistributed(stage, waveStart, wavePairCount, unitaryReal, unitaryImag,
                                                   planar.aReal, planar.aImag, true);
    }

    __aicore__ inline void BuildRightScratchTile(const CheevjStageBlockPair16 &pair, int localDim,
                                                 LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                 int rowOffset, int elems)
    {
        BuildRightScratchPlaneTile(pair, localDim, unitaryReal, unitaryImag, planar.aReal, planar.aImag, true,
                                   rowOffset, elems);
    }

    __aicore__ inline void BuildRightScratchPlaneTile(const CheevjStageBlockPair16 &pair, int localDim,
                                                      LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                      GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                      bool updateImagNeg, int rowOffset, int elems)
    {
        LocalTensor<float> leftRowsReal = leftRowsRealBuf.Get<float>();
        LocalTensor<float> leftRowsImag = leftRowsImagBuf.Get<float>();
        LocalTensor<float> accReal = accRealBuf.Get<float>();
        LocalTensor<float> accImag = accImagBuf.Get<float>();
        LocalTensor<float> imagNeg = tmp0Buf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int srcLocalCol = 0; srcLocalCol < localDim; ++srcLocalCol)
        {
            const int srcGlobalCol = GlobalIndexForLocal(pair, srcLocalCol);
            DataCopyPad(leftRowsReal[LeftRowsOffset(srcLocalCol, 0)], realPlane[ColumnOffset(srcGlobalCol, rowOffset)],
                        copyParams, padParams);
            DataCopyPad(leftRowsImag[LeftRowsOffset(srcLocalCol, 0)], imagPlane[ColumnOffset(srcGlobalCol, rowOffset)],
                        copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
        }

        for (int dstLocalCol = 0; dstLocalCol < localDim; ++dstLocalCol)
        {
            Duplicate(accReal, 0.0f, elems);
            Duplicate(accImag, 0.0f, elems);
            PipeBarrier<PIPE_V>();
            for (int srcLocalCol = 0; srcLocalCol < localDim; ++srcLocalCol)
            {
                AccumulateScaledComplex(accReal, accImag, leftRowsReal[LeftRowsOffset(srcLocalCol, 0)],
                                        leftRowsImag[LeftRowsOffset(srcLocalCol, 0)],
                                        unitaryReal.GetValue(LocalOffset(srcLocalCol, dstLocalCol)),
                                        unitaryImag.GetValue(LocalOffset(srcLocalCol, dstLocalCol)), elems);
            }

            const int dstGlobalCol = GlobalIndexForLocal(pair, dstLocalCol);
            DataCopyPad(realPlane[ColumnOffset(dstGlobalCol, rowOffset)], accReal, copyParams);
            DataCopyPad(imagPlane[ColumnOffset(dstGlobalCol, rowOffset)], accImag, copyParams);
            if (updateImagNeg)
            {
                Muls(imagNeg, accImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                DataCopyPad(planar.aImagNeg[ColumnOffset(dstGlobalCol, rowOffset)], imagNeg, copyParams);
            }
        }
    }

    __aicore__ inline void AccumulateScaledComplex(LocalTensor<float> accReal, LocalTensor<float> accImag,
                                                   LocalTensor<float> srcReal, LocalTensor<float> srcImag,
                                                   float scaleReal, float scaleImag, int elems)
    {
        Axpy(accReal, srcReal, scaleReal, elems);
        Axpy(accReal, srcImag, -scaleImag, elems);
        Axpy(accImag, srcReal, scaleImag, elems);
        Axpy(accImag, srcImag, scaleReal, elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyRightScratchBack(const CheevjStageBlockPair16 &pair, int localDim)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
            CopyRightScratchBackTile(pair, localDim, rowOffset, elems);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyRightScratchBackTile(const CheevjStageBlockPair16 &pair, int localDim, int rowOffset,
                                                    int elems)
    {
        CopyRightScratchPlaneBackTile(pair, localDim, planar.aReal, planar.aImag, planar.aImagNeg, false, rowOffset,
                                      elems);
    }

    __aicore__ inline void CopyRightScratchPlaneBackTile(const CheevjStageBlockPair16 &pair, int localDim,
                                                         GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                         GlobalTensor<float> imagNegPlane, bool updateImagNeg,
                                                         int rowOffset, int elems)
    {
        LocalTensor<float> srcReal = srcRealBuf.Get<float>();
        LocalTensor<float> srcImag = srcImagBuf.Get<float>();
        LocalTensor<float> imagNeg = tmp0Buf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalIndexForLocal(pair, localCol);
            DataCopyPad(srcReal, planar.scratch0[ColumnOffset(globalCol, rowOffset)], copyParams, padParams);
            DataCopyPad(srcImag, planar.scratch1[ColumnOffset(globalCol, rowOffset)], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(realPlane[ColumnOffset(globalCol, rowOffset)], srcReal, copyParams);
            DataCopyPad(imagPlane[ColumnOffset(globalCol, rowOffset)], srcImag, copyParams);
            if (updateImagNeg)
            {
                Muls(imagNeg, srcImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                DataCopyPad(imagNegPlane[ColumnOffset(globalCol, rowOffset)], imagNeg, copyParams);
            }
        }
    }

    __aicore__ inline void ApplyRightUnitaryToVectorColumns(const CheevjStageBlockPair16 &pair, int localDim,
                                                            LocalTensor<float> unitaryReal,
                                                            LocalTensor<float> unitaryImag)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
            BuildRightScratchPlaneTile(pair, localDim, unitaryReal, unitaryImag, planar.vReal, planar.vImag, false,
                                       rowOffset, elems);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ApplyRightUnitaryToVectorColumnsDistributed(int stage, int waveStart, int wavePairCount,
                                                                       LocalTensor<float> unitaryReal,
                                                                       LocalTensor<float> unitaryImag)
    {
        ApplyRightUnitaryToPlaneColumnsDistributed(stage, waveStart, wavePairCount, unitaryReal, unitaryImag,
                                                   planar.vReal, planar.vImag, false);
    }

    __aicore__ inline void ApplyRightUnitaryToPlaneColumnsDistributed(
        int stage, int waveStart, int wavePairCount, LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
        GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane, bool updateImagNeg)
    {
        const int rowTiles =
            (planar.layout.n + CHEEVJ_STAGE_BLOCK_JACOBI16_TILE - 1) / CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
        const int workerCount = CheevjWorkspaceWorkerCount() > 0 ? CheevjWorkspaceWorkerCount() : 1;
        const int totalTasks = wavePairCount * rowTiles;
        for (int task = GetBlockIdx(); task < totalTasks; task += workerCount)
        {
            const int localPair = task / rowTiles;
            const int tileIndex = task - localPair * rowTiles;
            const int pairIndex = waveStart + localPair;
            const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
            const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;

#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
            if (CanUseSinglePairRightPanel(pair, localDim))
            {
                continue;
            }
#endif

            if (!pair.valid || localDim <= 1 ||
                localDim > CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM)
            {
                continue;
            }
            LoadUnitaryForPair(pairIndex, unitaryReal, unitaryImag);
            const int rowOffset = tileIndex * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
            const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowOffset);
            BuildRightScratchPlaneTile(pair, localDim, unitaryReal, unitaryImag, realPlane, imagPlane, updateImagNeg,
                                       rowOffset, elems);
            PipeBarrier<PIPE_ALL>();
        }
    }
