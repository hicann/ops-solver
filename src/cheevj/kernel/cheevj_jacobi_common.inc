/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 1
            PipeBarrier<PIPE_ALL>();
            if (maxOffDiagonal <= tolerance)
            {
                return true;
            }
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 2
            CopyGlobalRowsToLocalColumn(planar.aReal, blockReal, globalCol, pair.left.begin, 0, pair.left.size,
                                        localCol);
            CopyGlobalRowsToLocalColumn(planar.aImag, blockImag, globalCol, pair.left.begin, 0, pair.left.size,
                                        localCol);
            CopyGlobalRowsToLocalColumn(planar.aReal, blockReal, globalCol, pair.right.begin, pair.left.size,
                                        pair.right.size, localCol);
            CopyGlobalRowsToLocalColumn(planar.aImag, blockImag, globalCol, pair.right.begin, pair.left.size,
                                        pair.right.size, localCol);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyGlobalRowsToLocalColumn(GlobalTensor<float> globalPlane, LocalTensor<float> localPlane,
                                                       int globalCol, int globalRowBegin, int localRowBegin,
                                                       int rowCount, int localCol)
    {
        if (rowCount <= 0)
        {
            return;
        }
        const uint32_t bytes = static_cast<uint32_t>(rowCount * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(localPlane[LocalOffset(localRowBegin, localCol)],
                    globalPlane[ColumnOffset(globalCol, globalRowBegin)], copyParams, padParams);
    }

    __aicore__ inline void InitLocalIdentity(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int localDim)
    {
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 3
    }

    __aicore__ inline void CopyLocalRowsToGlobalColumn(GlobalTensor<float> globalPlane, LocalTensor<float> localPlane,
                                                       int globalCol, int globalRowBegin, int localRowBegin,
                                                       int rowCount, int localCol)
    {
        if (rowCount <= 0)
        {
            return;
        }
        const uint32_t bytes = static_cast<uint32_t>(rowCount * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(globalPlane[ColumnOffset(globalCol, globalRowBegin)],
                    localPlane[LocalOffset(localRowBegin, localCol)], copyParams);
    }

    __aicore__ inline void CopyLocalImagNegRowsToGlobalColumn(int globalCol, int globalRowBegin, int localRowBegin,
                                                              int rowCount, int localCol, LocalTensor<float> blockImag)
    {
        if (rowCount <= 0)
        {
            return;
        }
        LocalTensor<float> tmp0 = tmp0Buf.Get<float>();
        Muls(tmp0, blockImag[LocalOffset(localRowBegin, localCol)], -1.0f, rowCount);
        PipeBarrier<PIPE_V>();
        const uint32_t bytes = static_cast<uint32_t>(rowCount * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(planar.aImagNeg[ColumnOffset(globalCol, globalRowBegin)], tmp0, copyParams);
    }
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 4
        LocalTensor<float> outQReal = accRealBuf.Get<float>();
        LocalTensor<float> outQImag = accImagBuf.Get<float>();
        LocalTensor<float> tmp0 = tmp0Buf.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();

        ComplexRot2(outPReal, outPImag, outQReal, outQImag, realLocal[LocalOffset(0, p)], imagLocal[LocalOffset(0, p)],
                    realLocal[LocalOffset(0, q)], imagLocal[LocalOffset(0, q)], c, sReal, sImag, tmp0, tmp1, tmp0, tmp1,
                    localDim);

        Muls(realLocal[LocalOffset(0, p)], outPReal, 1.0f, localDim);
        Muls(imagLocal[LocalOffset(0, p)], outPImag, 1.0f, localDim);
        Muls(realLocal[LocalOffset(0, q)], outQReal, 1.0f, localDim);
        Muls(imagLocal[LocalOffset(0, q)], outQImag, 1.0f, localDim);
        PipeBarrier<PIPE_V>();
    }
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 5
            CopyLocalRowsToGlobalColumn(planar.aReal, blockReal, globalCol, pair.left.begin, 0, pair.left.size,
                                        localCol);
            CopyLocalRowsToGlobalColumn(planar.aImag, blockImag, globalCol, pair.left.begin, 0, pair.left.size,
                                        localCol);
            CopyLocalImagNegRowsToGlobalColumn(globalCol, pair.left.begin, 0, pair.left.size, localCol, blockImag);

            CopyLocalRowsToGlobalColumn(planar.aReal, blockReal, globalCol, pair.right.begin, pair.left.size,
                                        pair.right.size, localCol);
            CopyLocalRowsToGlobalColumn(planar.aImag, blockImag, globalCol, pair.right.begin, pair.left.size,
                                        pair.right.size, localCol);
            CopyLocalImagNegRowsToGlobalColumn(globalCol, pair.right.begin, pair.left.size, pair.right.size, localCol,
                                               blockImag);
        }
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 6
        Muls(imagNegLocal, imagLocal, -1.0f, elems);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> rowStageReal = rowStageRealBuf.Get<float>();
        LocalTensor<float> rowStageImag = rowStageImagBuf.Get<float>();
        LocalTensor<float> rowStageImagNeg = rowStageImagNegBuf.Get<float>();
        StageDenseTileForStridedRow(rowStageReal, realLocal, elems);
        StageDenseTileForStridedRow(rowStageImag, imagNegLocal, elems);
        StageDenseTileForStridedRow(rowStageImagNeg, imagLocal, elems);

        DataCopyExtParams rowCopyParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)), 0,
                                        static_cast<uint32_t>((planar.layout.strideN - 1) * sizeof(float)), 0};
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 7
                    const CheevjComplex32 apq = GetLocalComplex(blockReal, blockImag, p, q);
                    const float absPq = ComplexAbs(apq);
                    maxOffDiagonal = BlockMaxFloat(maxOffDiagonal, absPq);
                    if (absPq <= tolerance || absPq <= CHEEVJ_BLOCK_JACOBI_MIN_PIVOT)
                    {
                        continue;
                    }

                    const float app = blockReal.GetValue(LocalOffset(p, p));
                    const float aqq = blockReal.GetValue(LocalOffset(q, q));
                    const float tau = (aqq - app) / (2.0f * absPq);
                    const float signTau = tau >= 0.0f ? 1.0f : -1.0f;
                    const float absTau = BlockAbsFloat(tau);
                    const float t = signTau / (absTau + BlockSqrtApprox(1.0f + tau * tau));
                    const float c = 1.0f / BlockSqrtApprox(1.0f + t * t);
                    const float scale = absPq > 0.0f ? t * c / absPq : 0.0f;
                    const CheevjComplex32 s = BlockMakeComplex(apq.real * scale, apq.imag * scale);

                    ApplyLocalMatrixRotation(blockReal, blockImag, localDim, p, q, absPq, t, c, s);
                    RotateLocalColumns(unitaryReal, unitaryImag, localDim, p, q, c, s.real, s.imag);
#elif defined(CHEEVJ_JACOBI_COMMON_SECTION) && CHEEVJ_JACOBI_COMMON_SECTION == 8
                    const CheevjStageComplex32 apq = GetLocalComplex(blockReal, blockImag, p, q);
                    const float absPq = StageComplexAbs(apq);
                    maxOffDiagonal = StageMaxFloat(maxOffDiagonal, absPq);
                    if (absPq <= tolerance || absPq <= CHEEVJ_STAGE_BLOCK_JACOBI16_MIN_PIVOT)
                    {
                        continue;
                    }

                    const float app = blockReal.GetValue(LocalOffset(p, p));
                    const float aqq = blockReal.GetValue(LocalOffset(q, q));
                    const float tau = (aqq - app) / (2.0f * absPq);
                    const float signTau = tau >= 0.0f ? 1.0f : -1.0f;
                    const float absTau = StageAbsFloat(tau);
                    const float t = signTau / (absTau + StageSqrtApprox(1.0f + tau * tau));
                    const float c = 1.0f / StageSqrtApprox(1.0f + t * t);
                    const float scale = absPq > 0.0f ? t * c / absPq : 0.0f;
                    const CheevjStageComplex32 s = MakeStageComplex(apq.real * scale, apq.imag * scale);

                    ApplyLocalMatrixRotation(blockReal, blockImag, localDim, p, q, app, aqq, absPq, t, c, s);
                    RotateLocalColumns(unitaryReal, unitaryImag, localDim, p, q, c, s.real, s.imag);
#endif
