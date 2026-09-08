/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void StoreUnitaryForPair(int pairIndex, LocalTensor<float> unitaryReal,
                                               LocalTensor<float> unitaryImag)
    {
        LocalTensor<float> unitaryImagNeg = tmp0Buf.Get<float>();
        const int storageDim = UnitaryStorageDim();
        const uint32_t bytes = static_cast<uint32_t>(storageDim * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        for (int col = 0; col < storageDim; ++col)
        {
            const int storedOffset = col * storageDim;
            const int localOffset = LocalOffset(0, col);
            Muls(unitaryImagNeg, unitaryImag[localOffset], -1.0f, storageDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(StoredUnitaryReal(pairIndex)[storedOffset], unitaryReal[localOffset], copyParams);
            DataCopyPad(StoredUnitaryImag(pairIndex)[storedOffset], unitaryImag[localOffset], copyParams);
            DataCopyPad(StoredUnitaryImagNeg(pairIndex)[storedOffset], unitaryImagNeg, copyParams);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadUnitaryForPair(int pairIndex, LocalTensor<float> unitaryReal,
                                              LocalTensor<float> unitaryImag)
    {
        const int storageDim = UnitaryStorageDim();
        const uint32_t bytes = static_cast<uint32_t>(storageDim * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        for (int col = 0; col < storageDim; ++col)
        {
            const int storedOffset = col * storageDim;
            const int localOffset = LocalOffset(0, col);
            DataCopyPad(unitaryReal[localOffset], StoredUnitaryReal(pairIndex)[storedOffset], copyParams, padParams);
            DataCopyPad(unitaryImag[localOffset], StoredUnitaryImag(pairIndex)[storedOffset], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ExtractPrincipalBlock(const CheevjStageBlockPair16 &pair, LocalTensor<float> blockReal,
                                                 LocalTensor<float> blockImag)
    {
        Duplicate(blockReal, 0.0f, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        Duplicate(blockImag, 0.0f, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();

        const int localDim = pair.left.size + pair.right.size;
        for (int localCol = 0; localCol < localDim; ++localCol)
        {
            const int globalCol = GlobalIndexForLocal(pair, localCol);
#define CHEEVJ_JACOBI_COMMON_SECTION 2
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        Duplicate(realLocal, 0.0f, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        Duplicate(imagLocal, 0.0f, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();
        // Use the proven C220 masked-vector form for one diagonal position in
        // each contiguous local column. This removes scalar SetValue while
        // avoiding a large repeat stride that is not portable across firmware.
        for (int idx = 0; idx < localDim; ++idx)
        {
            uint64_t diagonalMask[2] = {1ULL << idx, 0ULL};
            Duplicate(realLocal[LocalOffset(0, idx)], 1.0f, diagonalMask, static_cast<uint8_t>(1), 1, 8);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline CheevjStageComplex32 GetLocalComplex(LocalTensor<float> realLocal, LocalTensor<float> imagLocal,
                                                           int row, int col) const
    {
        const int offset = LocalOffset(row, col);
        return MakeStageComplex(realLocal.GetValue(offset), imagLocal.GetValue(offset));
    }

    __aicore__ inline float EstimateLocalNorm(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int localDim)
    {
        LocalTensor<float> rowSum = accRealBuf.Get<float>();
        LocalTensor<float> columnAbs = srcRealBuf.Get<float>();
        LocalTensor<float> absScratch = srcImagBuf.Get<float>();
        LocalTensor<float> reduceScratch = tmp0Buf.Get<float>();

        Duplicate(rowSum, 0.0f, localDim);
        PipeBarrier<PIPE_V>();
        for (int col = 0; col < localDim; ++col)
        {
            ComplexAbs(columnAbs, realLocal[LocalOffset(0, col)], imagLocal[LocalOffset(0, col)], absScratch,
                       static_cast<uint32_t>(localDim));
            Add(rowSum, rowSum, columnAbs, localDim);
            PipeBarrier<PIPE_V>();
        }
        ReduceMax(rowSum, rowSum, reduceScratch, localDim, true);
        PipeBarrier<PIPE_ALL>();
        return StageMaxFloat(1.0f, rowSum.GetValue(0));
    }

    __aicore__ inline bool RunLocalJacobi(LocalTensor<float> blockReal, LocalTensor<float> blockImag,
                                          LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag, int localDim)
    {
        const float tolerance = CHEEVJ_STAGE_BLOCK_JACOBI16_REL_TOL * EstimateLocalNorm(blockReal, blockImag, localDim);
        PrepareLocalRowMirrorGatherIndex();
        for (int sweep = 0; sweep < maxLocalSweeps; ++sweep)
        {
            float maxOffDiagonal = 0.0f;
            for (int p = 0; p < localDim - 1; ++p)
            {
                for (int q = p + 1; q < localDim; ++q)
                {
#define CHEEVJ_JACOBI_COMMON_SECTION 8
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
                }
            }
#define CHEEVJ_JACOBI_COMMON_SECTION (1)
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        }
        return false;
    }

    __aicore__ inline void ApplyLocalMatrixRotation(LocalTensor<float> blockReal, LocalTensor<float> blockImag,
                                                    int localDim, int p, int q, float app, float aqq, float absPq,
                                                    float t, float c, CheevjStageComplex32 s)
    {
        RotateLocalColumns(blockReal, blockImag, localDim, p, q, c, s.real, s.imag);

        // Snapshot the updated columns before scattering their conjugates into the
        // corresponding rows. Source/destination overlap across Gather repeats is
        // not allowed, and the snapshots also let us patch the pivot entries once.
        LocalTensor<float> rowReal = srcRealBuf.Get<float>();
        LocalTensor<float> rowImag = srcImagBuf.Get<float>();
        Muls(rowReal, blockReal[LocalOffset(0, p)], 1.0f, localDim);
        Muls(rowReal[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], blockReal[LocalOffset(0, q)], 1.0f, localDim);
        Muls(rowImag, blockImag[LocalOffset(0, p)], -1.0f, localDim);
        Muls(rowImag[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], blockImag[LocalOffset(0, q)], -1.0f, localDim);
        PipeBarrier<PIPE_V>();

        uint64_t pMask[2] = {1ULL << p, 0ULL};
        uint64_t qMask[2] = {1ULL << q, 0ULL};
        uint64_t pivotMask[2] = {(1ULL << p) | (1ULL << q), 0ULL};
        const float appNew = app - t * absPq;
        const float aqqNew = aqq + t * absPq;
        Duplicate(rowReal, appNew, pMask, static_cast<uint8_t>(1), 1, 8);
        Duplicate(rowReal, 0.0f, qMask, static_cast<uint8_t>(1), 1, 8);
        Duplicate(rowReal[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], 0.0f, pMask, static_cast<uint8_t>(1), 1, 8);
        Duplicate(rowReal[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], aqqNew, qMask, static_cast<uint8_t>(1), 1, 8);
        Duplicate(rowImag, 0.0f, pivotMask, static_cast<uint8_t>(1), 1, 8);
        Duplicate(rowImag[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], 0.0f, pivotMask, static_cast<uint8_t>(1), 1, 8);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> gatherIndex = leftRowsRealBuf.Get<uint32_t>();
        const uint8_t repeatTimes = static_cast<uint8_t>(localDim);
        Gather(blockReal, rowReal, gatherIndex, 0, pMask, repeatTimes, 8);
        Gather(blockImag, rowImag, gatherIndex, 0, pMask, repeatTimes, 8);
        Gather(blockReal, rowReal[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], gatherIndex, 0, qMask, repeatTimes, 8);
        Gather(blockImag, rowImag[CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM], gatherIndex, 0, qMask, repeatTimes, 8);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareLocalRowMirrorGatherIndex()
    {
        LocalTensor<uint32_t> gatherIndex = leftRowsRealBuf.Get<uint32_t>();
        LocalTensor<int32_t> signedIndex = gatherIndex.template ReinterpretCast<int32_t>();
        CreateVecIndex(signedIndex, static_cast<int32_t>(0),
                       static_cast<uint32_t>(CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS));
        PipeBarrier<PIPE_V>();
        ShiftRight(gatherIndex, gatherIndex, static_cast<uint32_t>(6), CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, static_cast<uint32_t>(2), CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RotateLocalColumns(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int localDim,
                                              int p, int q, float c, float sReal, float sImag)
    {
        LocalTensor<float> outPReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> outPImag = rowStageImagBuf.Get<float>();
#define CHEEVJ_JACOBI_COMMON_SECTION 4
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
