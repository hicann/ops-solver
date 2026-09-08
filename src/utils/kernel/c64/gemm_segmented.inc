/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline bool IsValidSegmentedProcess(int offsetA, int offsetB, int offsetC, int blockM, int blockN,
                                                   int blockK) const
    {
        // This path intentionally mirrors the verified FP32 Cheevj panel backend.
        // FP32 NZ uses C0=8, while an FP32 L0C row uses C0=16.
        return sizeof(T) == sizeof(float) && K == 32 && blockK == 32 && blockM == 32 && segmentedLeftDim == 16 &&
               segmentedRightDim == 16 && N > 0 && segmentedMatrixLeadingDim >= N && segmentedLeftBegin >= 0 &&
               segmentedRightBegin >= 0 && segmentedLeftBegin + segmentedLeftDim <= N &&
               segmentedRightBegin + segmentedRightDim <= N && offsetA == 0 && offsetB >= 0 && offsetC >= 0 &&
               blockN > 0 && blockN <= 128 && (blockN & 7) == 0 && offsetB + blockN <= N && offsetC + blockN <= N;
    }

    __aicore__ inline void ProcessSegmented(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        if (!IsValidSegmentedProcess(offsetA, offsetB, offsetC, blockM, blockN, blockK))
        {
            return;
        }

        static constexpr uint8_t padList[]{0, 0, 0, 0};
        AscendC::Load3DSetFMatrixCal(1, 32, padList);
        AscendC::Load3DSetFMatrixBCal(1, 32, padList);

        CopyInSegmentedA(inQueueA1Real, aRealGlobal, offsetA);
        CopyInSegmentedA(inQueueA1Imag, aImagGlobal, offsetA);
        CopyInSegmentedA(inQueueA1Imag, aImagNegGlobal, offsetA);
        CopyInSegmentedB(inQueueB1Real, bRealGlobal, offsetB, blockN);
        CopyInSegmentedB(inQueueB1Imag, bImagGlobal, offsetB, blockN);

        LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
        LocalTensor<T> a1ImagLocal = inQueueA1Imag.DeQue<T>();
        LocalTensor<T> a1ImagNegLocal = inQueueA1Imag.DeQue<T>();
        LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
        LocalTensor<T> b1ImagLocal = inQueueB1Imag.DeQue<T>();
        LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
        LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();

        // (Ar + i Ai)(Br + i Bi):
        // Cr = ArBr + (-Ai)Bi, Ci = ArBi + AiBr.
        ComputeSegmented(a1RealLocal, b1RealLocal, c1RealLocal, blockN, true);
        ComputeSegmented(a1RealLocal, b1ImagLocal, c1ImagLocal, blockN, true);
        ComputeSegmented(a1ImagLocal, b1RealLocal, c1ImagLocal, blockN, false);
        ComputeSegmented(a1ImagNegLocal, b1ImagLocal, c1RealLocal, blockN, false);

        // Each CO1 result feeds two disjoint Fixpipes. Explicit events are
        // required here; the legacy single-output unit-flag protocol is not.
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(segmentedEventMToFix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(segmentedEventMToFix);
        CopyOutSegmented(c1RealLocal, cRealGlobal, offsetC, blockN);
        CopyOutSegmented(c1ImagLocal, cImagGlobal, offsetC, blockN);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(segmentedEventFixToM);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(segmentedEventFixToM);

        inQueueA1Real.FreeTensor(a1RealLocal);
        inQueueA1Imag.FreeTensor(a1ImagLocal);
        inQueueA1Imag.FreeTensor(a1ImagNegLocal);
        inQueueB1Real.FreeTensor(b1RealLocal);
        inQueueB1Imag.FreeTensor(b1ImagLocal);
    }

    __aicore__ inline void CopyInSegmentedA(TQue<QuePosition::A1, 2> &queue, GlobalTensor<T> source, int offset)
    {
        LocalTensor<T> destination = queue.AllocTensor<T>();
        Nd2NzParams params{1, 32, 32, 0, 32, 32, 1, 0};
        DataCopy(destination, source[offset], params);
        queue.EnQue(destination);
    }

    __aicore__ inline void CopyInSegmentedB(TQue<QuePosition::B1, 2> &queue, GlobalTensor<T> source, int tileOffset,
                                            int tileN)
    {
        LocalTensor<T> destination = queue.AllocTensor<T>();
        Nd2NzParams params{1, 16, static_cast<uint16_t>(tileN), 0, static_cast<uint16_t>(segmentedMatrixLeadingDim), 32,
                           1, 0};
        DataCopy(destination, source[segmentedLeftBegin * segmentedMatrixLeadingDim + tileOffset], params);
        DataCopy(destination[16 * 8], source[segmentedRightBegin * segmentedMatrixLeadingDim + tileOffset], params);
        queue.EnQue(destination);
    }

    __aicore__ inline void LoadSegmentedA2(const LocalTensor<T> &a1Local)
    {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro params;
        params.channelSize = 32;
        params.extConfig = (static_cast<uint64_t>(32) << 16) | static_cast<uint64_t>(32);
        LoadData<T>(a2Local, a1Local, params);
        inQueueA2.EnQue(a2Local);
    }

    __aicore__ inline void LoadSegmentedB2(const LocalTensor<T> &b1Local, int tileN)
    {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        LoadData3DParamsV2Pro params;
        params.channelSize = tileN;
        params.extConfig = (static_cast<uint64_t>(32) << 16) | static_cast<uint64_t>(tileN);
        params.fMatrixCtrl = true;
        LoadData<T>(b2Local, b1Local, params);
        inQueueB2.EnQue(b2Local);
    }

    __aicore__ inline void ComputeSegmented(const LocalTensor<T> &a1Local, const LocalTensor<T> &b1Local,
                                            const LocalTensor<T> &c1Local, int tileN, bool initialize)
    {
        LoadSegmentedA2(a1Local);
        LoadSegmentedB2(b1Local, tileN);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();

        MmadParams params;
        params.m = 32;
        params.n = tileN;
        params.k = 32;
        params.unitFlag = 0;
        if (initialize)
        {
            Mmad(c1Local, a2Local, b2Local, params);
        }
        else
        {
            params.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, params);
        }
        if (((32 * tileN) >> 8) < 10)
        {
            PipeBarrier<PIPE_M>();
        }
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOutSegmented(const LocalTensor<T> &source, GlobalTensor<T> destination, int tileOffset,
                                            int tileN)
    {
        FixpipeParamsV220 params;
        params.nSize = tileN;
        params.mSize = 16;
        params.srcStride = 32;
        params.dstStride = segmentedMatrixLeadingDim;
        params.unitFlag = 0;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(destination[segmentedLeftBegin * segmentedMatrixLeadingDim + tileOffset], source, params);
        Fixpipe(destination[segmentedRightBegin * segmentedMatrixLeadingDim + tileOffset], source[16 * 16], params);
    }
