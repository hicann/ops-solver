/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(CHEEVJ_PANEL_COMMON_SECTION) && CHEEVJ_PANEL_COMMON_SECTION == 1
        GlobalTensor<float> packedReal = RightPanelBufferReal(slotIndex, localDim, bufferIndex);
        GlobalTensor<float> packedImag = RightPanelBufferImag(slotIndex, localDim, bufferIndex);
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localRow = localRowBegin; localRow < localRowEnd; ++localRow)
        {
            const int globalCol = GlobalIndexForLocal(pair, localRow);
            for (int rowOffset = sliceBegin; rowOffset < sliceEnd; rowOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, sliceEnd - rowOffset);
#elif defined(CHEEVJ_PANEL_COMMON_SECTION) && CHEEVJ_PANEL_COMMON_SECTION == 2
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localRow = localRowBegin; localRow < localRowEnd; ++localRow)
        {
            const int globalRow = GlobalIndexForLocal(pair, localRow);
            for (int colOffset = 0; colOffset < planar.layout.n; colOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - colOffset);
                Duplicate(stagedReal, 0.0f, elems * 8);
                Duplicate(stagedImag, 0.0f, elems * 8);
                PipeBarrier<PIPE_V>();
                DataCopyExtParams rowParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)),
                                            static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float)), 0, 0};
                DataCopyPad(stagedReal, planar.aReal[RowOffset(globalRow, colOffset)], rowParams, padParams);
                DataCopyPad(stagedImag, planar.aImag[RowOffset(globalRow, colOffset)], rowParams, padParams);
                PipeBarrier<PIPE_ALL>();
                Gather(denseReal, stagedReal, gatherLocal, 0, elems);
                Gather(denseImag, stagedImag, gatherLocal, 0, elems);
                PipeBarrier<PIPE_ALL>();
#elif defined(CHEEVJ_PANEL_COMMON_SECTION) && CHEEVJ_PANEL_COMMON_SECTION == 3
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int localRow = localRowBegin; localRow < localRowEnd; ++localRow)
        {
            const int globalRow = GlobalIndexForLocal(pair, localRow);
            for (int colOffset = 0; colOffset < planar.layout.n; colOffset += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
            {
                const int elems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - colOffset);
                DataCopyExtParams packedParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int packedOffset = localRow * planar.layout.n + colOffset;
                DataCopyPad(denseReal, packedReal[packedOffset], packedParams, padParams);
                DataCopyPad(denseImag, packedImag[packedOffset], packedParams, padParams);
                PipeBarrier<PIPE_ALL>();
                Muls(denseImagNeg, denseImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                StageDenseTileForStridedRow(stagedReal, denseReal, elems);
                StageDenseTileForStridedRow(stagedImag, denseImag, elems);
                StageDenseTileForStridedRow(stagedImagNeg, denseImagNeg, elems);
                DataCopyExtParams rowParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)), 0,
                                            static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float)), 0};
                const int outputOffset = RowOffset(globalRow, colOffset);
                DataCopyPad(planar.aReal[outputOffset], stagedReal, rowParams);
                DataCopyPad(planar.aImag[outputOffset], stagedImag, rowParams);
                DataCopyPad(planar.aImagNeg[outputOffset], stagedImagNeg, rowParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
#elif defined(CHEEVJ_PANEL_COMMON_SECTION) && CHEEVJ_PANEL_COMMON_SECTION == 4
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
                const int packedOffset = localRow * planar.layout.n + rowOffset;
                DataCopyPad(denseReal, packedReal[packedOffset], copyParams, padParams);
                DataCopyPad(denseImag, packedImag[packedOffset], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();
                if (updateImagNeg)
                {
                    Muls(denseImagNeg, denseImag, -1.0f, elems);
                    PipeBarrier<PIPE_V>();
                }
                const int outputOffset = ColumnOffset(globalCol, rowOffset);
                DataCopyPad(outputReal[outputOffset], denseReal, copyParams);
                DataCopyPad(outputImag[outputOffset], denseImag, copyParams);
                if (updateImagNeg)
                {
                    DataCopyPad(outputImagNeg[outputOffset], denseImagNeg, copyParams);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
#elif defined(CHEEVJ_PANEL_COMMON_SECTION) && CHEEVJ_PANEL_COMMON_SECTION == 5
    {
        for (int bufferIndex = 0; bufferIndex < 2; ++bufferIndex)
        {
            const int pairIndex = groupPair + bufferIndex * slotCount;
            if (pairIndex >= pairCount)
            {
                continue;
            }
            const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
            const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
            if (!CanUseSinglePairRightPanel(pair, localDim))
            {
                continue;
            }
#endif
