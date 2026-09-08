/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline bool ProcessPair(const CheevjBlockPair& pair)
    {
        const int localDim = pair.left.size + pair.right.size;
        if (localDim <= 1)
        {
            return true;
        }

        LocalTensor<float> blockReal = blockRealBuf.Get<float>();
        LocalTensor<float> blockImag = blockImagBuf.Get<float>();
        LocalTensor<float> unitaryReal = unitaryRealBuf.Get<float>();
        LocalTensor<float> unitaryImag = unitaryImagBuf.Get<float>();

        ExtractPrincipalBlock(pair, blockReal, blockImag);
        InitLocalIdentity(unitaryReal, unitaryImag, localDim);
        const bool localConverged = RunLocalJacobi(blockReal, blockImag, unitaryReal, unitaryImag, localDim);

        ApplyUnitaryToMatrixColumns(pair, localDim, unitaryReal, unitaryImag, blockReal, blockImag);
        if (computeVectors)
        {
            ApplyUnitaryToVectorColumns(pair, localDim, unitaryReal, unitaryImag);
        }
        if (GetBlockIdx() == 0)
        {
            ScatterPrincipalBlock(pair, localDim, blockReal, blockImag);
        }
        CheevjWorkspaceSync();
        return localConverged;
    }

    __aicore__ inline int LocalOffset(int row, int col) const { return row + col * CHEEVJ_BLOCK_JACOBI_LOCAL_DIM; }

    __aicore__ inline int ColumnOffset(int col, int rowOffset) const { return col * planar.layout.strideN + rowOffset; }

    __aicore__ inline int GlobalColumnForLocal(const CheevjBlockPair& pair, int localCol) const
    {
        if (localCol < pair.left.size)
        {
            return pair.left.begin + localCol;
        }
        return pair.right.begin + (localCol - pair.left.size);
    }

    __aicore__ inline int LocalIndexForGlobalInPair(const CheevjBlockPair& pair, int globalIndex) const
    {
        if (globalIndex >= pair.left.begin && globalIndex < pair.left.begin + pair.left.size)
        {
            return globalIndex - pair.left.begin;
        }
        if (globalIndex >= pair.right.begin && globalIndex < pair.right.begin + pair.right.size)
        {
            return pair.left.size + globalIndex - pair.right.begin;
        }
        return -1;
    }

    __aicore__ inline void ExtractPrincipalBlock(const CheevjBlockPair& pair, LocalTensor<float> blockReal,
                                                 LocalTensor<float> blockImag)
    {
        Duplicate(blockReal, 0.0f, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS);
        Duplicate(blockImag, 0.0f, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();

        const int localDim = pair.left.size + pair.right.size;
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalColumnForLocal(pair, localCol);
#define CHEEVJ_JACOBI_COMMON_SECTION 2
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        Duplicate(realLocal, 0.0f, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS);
        Duplicate(imagLocal, 0.0f, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();
        for (int idx = 0; idx < localDim; ++idx)
        {
            realLocal.SetValue(LocalOffset(idx, idx), 1.0f);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline CheevjComplex32 GetLocalComplex(LocalTensor<float> realLocal, LocalTensor<float> imagLocal,
                                                      int row, int col) const
    {
        const int offset = LocalOffset(row, col);
        return BlockMakeComplex(realLocal.GetValue(offset), imagLocal.GetValue(offset));
    }

    __aicore__ inline void SetLocalComplex(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int row, int col,
                                           CheevjComplex32 value)
    {
        const int offset = LocalOffset(row, col);
        realLocal.SetValue(offset, value.real);
        imagLocal.SetValue(offset, value.imag);
    }

    __aicore__ inline float EstimateLocalNorm(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int localDim)
    {
        float norm = 1.0f;
        for (int row = 0; row < localDim; ++row)
        {
            float rowSum = 0.0f;
            for (int col = 0; col < localDim; ++col)
            {
                rowSum += ComplexAbs(GetLocalComplex(realLocal, imagLocal, row, col));
            }
            norm = BlockMaxFloat(norm, rowSum);
        }
        return norm;
    }

    __aicore__ inline bool RunLocalJacobi(LocalTensor<float> blockReal, LocalTensor<float> blockImag,
                                          LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag, int localDim)
    {
        const float tolerance = CHEEVJ_BLOCK_JACOBI_REL_TOL * EstimateLocalNorm(blockReal, blockImag, localDim);
        for (int sweep = 0; sweep < maxLocalSweeps; ++sweep)
        {
            float maxOffDiagonal = 0.0f;
            for (int p = 0; p < localDim - 1; ++p)
            {
                for (int q = p + 1; q < localDim; ++q)
                {
#define CHEEVJ_JACOBI_COMMON_SECTION 7
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
                }
            }
#define CHEEVJ_JACOBI_COMMON_SECTION 1
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        }
        return false;
    }

    __aicore__ inline void ApplyLocalMatrixRotation(LocalTensor<float> blockReal, LocalTensor<float> blockImag,
                                                    int localDim, int p, int q, float absPq, float t, float c,
                                                    CheevjComplex32 s)
    {
        const float app = blockReal.GetValue(LocalOffset(p, p));
        const float aqq = blockReal.GetValue(LocalOffset(q, q));

        RotateLocalColumns(blockReal, blockImag, localDim, p, q, c, s.real, s.imag);

        for (int k = 0; k < localDim; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const int kp = LocalOffset(k, p);
            const int kq = LocalOffset(k, q);
            const int pk = LocalOffset(p, k);
            const int qk = LocalOffset(q, k);
            blockReal.SetValue(pk, blockReal.GetValue(kp));
            blockImag.SetValue(pk, -blockImag.GetValue(kp));
            blockReal.SetValue(qk, blockReal.GetValue(kq));
            blockImag.SetValue(qk, -blockImag.GetValue(kq));
        }

        blockReal.SetValue(LocalOffset(p, p), app - t * absPq);
        blockImag.SetValue(LocalOffset(p, p), 0.0f);
        blockReal.SetValue(LocalOffset(q, q), aqq + t * absPq);
        blockImag.SetValue(LocalOffset(q, q), 0.0f);
        SetLocalComplex(blockReal, blockImag, p, q, BlockMakeComplex(0.0f, 0.0f));
        SetLocalComplex(blockReal, blockImag, q, p, BlockMakeComplex(0.0f, 0.0f));
    }

    __aicore__ inline void RotateLocalColumns(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int localDim,
                                              int p, int q, float c, float sReal, float sImag)
    {
        LocalTensor<float> outPReal = outRealBuf.Get<float>();
        LocalTensor<float> outPImag = outImagBuf.Get<float>();
#define CHEEVJ_JACOBI_COMMON_SECTION 4
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION

    __aicore__ inline void ApplyUnitaryToMatrixColumns(const CheevjBlockPair& pair, int localDim,
                                                       LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                       LocalTensor<float> blockReal, LocalTensor<float> blockImag)
    {
        ApplyUnitaryToPlaneColumns(pair, localDim, unitaryReal, unitaryImag, planar.aReal, planar.aImag,
                                   planar.aImagNeg, true);
        RepairUpdatedMatrixRowsFromScratch(pair, localDim, blockReal, blockImag);
    }

    __aicore__ inline void ApplyUnitaryToVectorColumns(const CheevjBlockPair& pair, int localDim,
                                                       LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
        ApplyUnitaryToPlaneColumns(pair, localDim, unitaryReal, unitaryImag, planar.vReal, planar.vImag, planar.vImag,
                                   false);
    }

    __aicore__ inline void ApplyUnitaryToPlaneColumns(const CheevjBlockPair& pair, int localDim,
                                                      LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                      GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                      GlobalTensor<float> imagNegPlane, bool updateImagNeg)
    {
        const int workerCount = CheevjWorkspaceWorkerCount();
        const int rowsPerWorker = RowsPerWorker(workerCount);
        for (int rowOffset = GetBlockIdx() * rowsPerWorker; rowOffset < planar.layout.n;
             rowOffset += workerCount * rowsPerWorker)
        {
            const int elems = MinInt(rowsPerWorker, planar.layout.n - rowOffset);
            BuildScratchColumnsForRowTile(pair, localDim, unitaryReal, unitaryImag, realPlane, imagPlane, rowOffset,
                                          elems);
        }
        CheevjWorkspaceSync();

        for (int rowOffset = GetBlockIdx() * rowsPerWorker; rowOffset < planar.layout.n;
             rowOffset += workerCount * rowsPerWorker)
        {
            const int elems = MinInt(rowsPerWorker, planar.layout.n - rowOffset);
            CopyScratchColumnsBack(pair, localDim, realPlane, imagPlane, imagNegPlane, rowOffset, elems, updateImagNeg);
        }
        CheevjWorkspaceSync();
    }

    __aicore__ inline int RowsPerWorker(int workerCount) const
    {
        const int activeWorkerCount = workerCount > 0 ? workerCount : 1;
        const int rows = AlignUp((planar.layout.n + activeWorkerCount - 1) / activeWorkerCount, 16);
        return MinInt(CHEEVJ_BLOCK_JACOBI_ROW_TILE, rows > 0 ? rows : 16);
    }

    __aicore__ inline void BuildScratchColumnsForRowTile(const CheevjBlockPair& pair, int localDim,
                                                         LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag,
                                                         GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                         int rowOffset, int elems)
    {
        LocalTensor<float> srcReal = srcRealBuf.Get<float>();
        LocalTensor<float> srcImag = srcImagBuf.Get<float>();
        LocalTensor<float> accReal = accRealBuf.Get<float>();
        LocalTensor<float> accImag = accImagBuf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int dstLocalCol = 0; dstLocalCol < localDim; ++dstLocalCol)
        {
            Duplicate(accReal, 0.0f, elems);
            Duplicate(accImag, 0.0f, elems);
            PipeBarrier<PIPE_V>();
            for (int srcLocalCol = 0; srcLocalCol < localDim; ++srcLocalCol)
            {
                const int srcGlobalCol = GlobalColumnForLocal(pair, srcLocalCol);
                DataCopyPad(srcReal, realPlane[ColumnOffset(srcGlobalCol, rowOffset)], copyParams, padParams);
                DataCopyPad(srcImag, imagPlane[ColumnOffset(srcGlobalCol, rowOffset)], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();
                AccumulateScaledComplex(accReal, accImag, srcReal, srcImag,
                                        unitaryReal.GetValue(LocalOffset(srcLocalCol, dstLocalCol)),
                                        unitaryImag.GetValue(LocalOffset(srcLocalCol, dstLocalCol)), elems);
            }

            const int dstGlobalCol = GlobalColumnForLocal(pair, dstLocalCol);
            DataCopyPad(planar.scratch0[ColumnOffset(dstGlobalCol, rowOffset)], accReal, copyParams);
            DataCopyPad(planar.scratch1[ColumnOffset(dstGlobalCol, rowOffset)], accImag, copyParams);
        }
    }

    __aicore__ inline void AccumulateScaledComplex(LocalTensor<float> accReal, LocalTensor<float> accImag,
                                                   LocalTensor<float> srcReal, LocalTensor<float> srcImag,
                                                   float scaleReal, float scaleImag, int elems)
    {
        LocalTensor<float> tmp0 = tmp0Buf.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();

        Muls(tmp0, srcReal, scaleReal, elems);
        Muls(tmp1, srcImag, scaleImag, elems);
        PipeBarrier<PIPE_V>();
        Sub(tmp0, tmp0, tmp1, elems);
        PipeBarrier<PIPE_V>();
        Add(accReal, accReal, tmp0, elems);

        Muls(tmp0, srcReal, scaleImag, elems);
        Muls(tmp1, srcImag, scaleReal, elems);
        PipeBarrier<PIPE_V>();
        Add(tmp0, tmp0, tmp1, elems);
        PipeBarrier<PIPE_V>();
        Add(accImag, accImag, tmp0, elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyScratchColumnsBack(const CheevjBlockPair& pair, int localDim,
                                                  GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                  GlobalTensor<float> imagNegPlane, int rowOffset, int elems,
                                                  bool updateImagNeg)
    {
        LocalTensor<float> srcReal = srcRealBuf.Get<float>();
        LocalTensor<float> srcImag = srcImagBuf.Get<float>();
        LocalTensor<float> tmp0 = tmp0Buf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalColumnForLocal(pair, localCol);
            DataCopyPad(srcReal, planar.scratch0[ColumnOffset(globalCol, rowOffset)], copyParams, padParams);
            DataCopyPad(srcImag, planar.scratch1[ColumnOffset(globalCol, rowOffset)], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(realPlane[ColumnOffset(globalCol, rowOffset)], srcReal, copyParams);
            DataCopyPad(imagPlane[ColumnOffset(globalCol, rowOffset)], srcImag, copyParams);
            if (updateImagNeg)
            {
                Muls(tmp0, srcImag, -1.0f, elems);
                PipeBarrier<PIPE_V>();
                DataCopyPad(imagNegPlane[ColumnOffset(globalCol, rowOffset)], tmp0, copyParams);
            }
        }
    }

    __aicore__ inline void RepairUpdatedMatrixRowsFromScratch(const CheevjBlockPair& pair, int localDim,
                                                              LocalTensor<float> blockReal,
                                                              LocalTensor<float> blockImag)
    {
        const int workerCount = CheevjWorkspaceWorkerCount();
        const int rowsPerWorker = RowsPerWorker(workerCount);
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalColumnForLocal(pair, localCol);
            for (int offset = GetBlockIdx() * rowsPerWorker; offset < planar.layout.n;
                 offset += workerCount * rowsPerWorker)
            {
                const int elems = MinInt(rowsPerWorker, planar.layout.n - offset);
                RepairMatrixRowTileFromScratch(pair, localDim, localCol, globalCol, offset, elems, blockReal,
                                               blockImag);
            }
        }
        CheevjWorkspaceSync();
    }

    __aicore__ inline void RepairMatrixRowTileFromScratch(const CheevjBlockPair& pair, int localDim, int localCol,
                                                          int globalCol, int offset, int elems,
                                                          LocalTensor<float> blockReal, LocalTensor<float> blockImag)
    {
        LocalTensor<float> realLocal = srcRealBuf.Get<float>();
        LocalTensor<float> imagLocal = srcImagBuf.Get<float>();
        LocalTensor<float> imagNegLocal = tmp0Buf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams scratchCopyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(realLocal, planar.scratch0[ColumnOffset(globalCol, offset)], scratchCopyParams, padParams);
        DataCopyPad(imagLocal, planar.scratch1[ColumnOffset(globalCol, offset)], scratchCopyParams, padParams);
        PipeBarrier<PIPE_ALL>();

        // Scratch holds the freshly right-updated global columns. For rows inside the
        // active 64x64 principal block, the correct two-sided values live in blockReal/blockImag.
        OverlayPrincipalRowsFromLocalBlock(pair, localDim, localCol, offset, elems, realLocal, imagLocal, blockReal,
                                           blockImag);

#define CHEEVJ_JACOBI_COMMON_SECTION 6
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        const int rowOffset = offset * planar.layout.strideN + globalCol;
        DataCopyPad(planar.aReal[rowOffset], rowStageReal, rowCopyParams);
        DataCopyPad(planar.aImag[rowOffset], rowStageImag, rowCopyParams);
        DataCopyPad(planar.aImagNeg[rowOffset], rowStageImagNeg, rowCopyParams);
    }

    __aicore__ inline void OverlayPrincipalRowsFromLocalBlock(const CheevjBlockPair& pair, int localDim, int localCol,
                                                              int offset, int elems, LocalTensor<float> realLocal,
                                                              LocalTensor<float> imagLocal,
                                                              LocalTensor<float> blockReal,
                                                              LocalTensor<float> blockImag)
    {
        for (int idx = 0; idx < elems; ++idx)
        {
            const int localRow = LocalIndexForGlobalInPair(pair, offset + idx);
            if (localRow >= 0 && localRow < localDim)
            {
                realLocal.SetValue(idx, blockReal.GetValue(LocalOffset(localRow, localCol)));
                imagLocal.SetValue(idx, blockImag.GetValue(LocalOffset(localRow, localCol)));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ScatterPrincipalBlock(const CheevjBlockPair& pair, int localDim,
                                                 LocalTensor<float> blockReal, LocalTensor<float> blockImag)
    {
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalColumnForLocal(pair, localCol);
#define CHEEVJ_JACOBI_COMMON_SECTION 5
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
#define CHEEVJ_JACOBI_COMMON_SECTION 3
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
