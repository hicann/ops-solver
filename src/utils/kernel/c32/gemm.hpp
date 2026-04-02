/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file gemm.hpp
 * \brief
 */

#ifndef _GEMM_HPP_
#define _GEMM_HPP_

#include <cstdint>
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

using namespace AscendC;
using namespace matmul;

#define debug(x) printf("%s: %d\n", #x, x)
#define debugf(x) printf("%s: %f\n", #x, x)
#define puts(x) printf("%s\n", x)

template<typename T>
class MatmulCustom {
    GlobalTensor<T> aGlobal, bGlobal, cGlobal;
    TQue<QuePosition::A1, 2> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1;
    int N, K;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int stepM, stepN;
    int stepKa, stepKb;
    int sizeA, sizeB;
    int alignedN;
    int stepK;
public:
    __aicore__ inline MatmulCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        this->stepK = 4;

        pipe->InitBuffer(inQueueA1, 2, 131072);
        pipe->InitBuffer(inQueueB1, 2, 131072);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1, 131072);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aGlobal, GlobalTensor<T> bGlobal, GlobalTensor<T> cGlobal, int N, int K) {
        this->N = N;
        this->K = K;
        this->aGlobal = aGlobal;
        this->bGlobal = bGlobal;
        this->cGlobal = cGlobal;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;

        baseN = min(singleCoreN, 256);
        int nn = 128;
        while (nn < baseN) nn <<= 1;
        baseM = min(singleCoreM, 32768 / nn);
        int mm = 16;
        while (mm < baseM) mm <<= 1;
        baseK = min(singleCoreK, min(8192 / nn, 8192 / mm));

        int nFullBlocks = singleCoreN / baseN;
        int TailN = singleCoreN % baseN;
        alignedN = baseN <= 32 ? baseN : (baseN + 63) / 64 * 64;

        int mFullBlocks = singleCoreM / baseM;
        int TailM = singleCoreM % baseM;

        int kBlocks = (singleCoreK + baseK - 1) / baseK;

        int TailK = singleCoreK % baseK;
        if (!TailK) TailK = baseK;

        sizeA = baseM * baseK;
        sizeB = baseK * alignedN;

        static constexpr uint8_t padList[]{0, 0, 0, 0};

        int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

        for (int i = 0; i < nFullBlocks; ++i) {
            Load3DSetFMatrixCal(1, baseM, padList);
            for (int j = 0; j < mFullBlocks; ++j) {
                int offsetC_1 = offsetC + j * baseM * N + i * baseN;
                LocalTensor<T> c1Local = outBufCO1.Get<T>();
                for (int k = 0; k < kBlocks; k += stepK) {
                    int cnt = min(kBlocks - k, stepK);
                    Load3DSetFMatrixBCal(1, baseK * cnt, padList);
                    int offsetA_1 = offsetA + j * baseM * K + k * baseK;
                    int offsetB_1 = offsetB + k * baseK * N + i * baseN;
                    CopyInA(offsetA_1, cnt);
                    CopyInB(offsetB_1, cnt);
                    LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
                    LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
                    for (int x = 0; x < cnt; ++x) {
                        SplitA(a1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                        SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                        Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
                    }
                    inQueueA1.FreeTensor(a1Local);
                    inQueueB1.FreeTensor(b1Local);
                }
                CopyOut(c1Local, offsetC_1);
            }
            if (!TailM) continue;
            int _offsetA = offsetA + mFullBlocks * baseM * K;
            int _offsetC = offsetC + mFullBlocks * baseM * N;
            int _baseM = baseM;
            baseM = TailM;
            sizeA = baseM * baseK;
            Load3DSetFMatrixCal(1, baseM, padList);
            int offsetC_1 = _offsetC + i * baseN;
            LocalTensor<T> c1Local = outBufCO1.Get<T>();
            for (int k = 0; k < kBlocks; k += stepK) {
                int cnt = min(kBlocks - k, stepK);
                Load3DSetFMatrixBCal(1, baseK * cnt, padList);
                int offsetA_1 = _offsetA + k * baseK;
                int offsetB_1 = offsetB + k * baseK * N + i * baseN;
                CopyInA(offsetA_1, cnt);
                CopyInB(offsetB_1, cnt);
                LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
                LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
                for (int x = 0; x < cnt; ++x) {
                    SplitA(a1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                    SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                    Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
                }
                inQueueA1.FreeTensor(a1Local);
                inQueueB1.FreeTensor(b1Local);
            }
            CopyOut(c1Local, offsetC_1);

            baseM = _baseM; 
            sizeA = baseM * baseK;
        }

        if (!TailN) return;
        offsetB += nFullBlocks * baseN;
        offsetC += nFullBlocks * baseN;
        baseN = TailN;
        alignedN = baseN;
        if (alignedN > 32) alignedN = (alignedN + 63) / 64 * 64;
        sizeB = baseK * alignedN;

        Load3DSetFMatrixCal(1, baseM, padList);
        for (int j = 0; j < mFullBlocks; ++j) {
            int offsetC_1 = offsetC + j * baseM * N;
            LocalTensor<T> c1Local = outBufCO1.Get<T>();
            for (int k = 0; k < kBlocks; k += stepK) {
                int cnt = min(kBlocks - k, stepK);
                Load3DSetFMatrixBCal(1, baseK * cnt, padList);
                int offsetA_1 = offsetA + j * baseM * K + k * baseK;
                int offsetB_1 = offsetB + k * baseK * N;
                CopyInA(offsetA_1, cnt);
                CopyInB(offsetB_1, cnt);
                LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
                LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
                for (int x = 0; x < cnt; ++x) {
                    SplitA(a1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                    SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                    Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
                }
                inQueueA1.FreeTensor(a1Local);
                inQueueB1.FreeTensor(b1Local);
            }
            CopyOut(c1Local, offsetC_1);
        }
        if (!TailM) return;
        int _offsetA = offsetA + mFullBlocks * baseM * K;
        int _offsetC = offsetC + mFullBlocks * baseM * N;
        int _baseM = baseM;
        baseM = TailM;
        sizeA = baseM * baseK;
        Load3DSetFMatrixCal(1, baseM, padList);
        int offsetC_1 = _offsetC;
        LocalTensor<T> c1Local = outBufCO1.Get<T>();
        for (int k = 0; k < kBlocks; k += stepK) {
            int cnt = min(kBlocks - k, stepK);
            Load3DSetFMatrixBCal(1, baseK * cnt, padList);
            int offsetA_1 = _offsetA + k * baseK;
            int offsetB_1 = offsetB + k * baseK * N;
            CopyInA(offsetA_1, cnt);
            CopyInB(offsetB_1, cnt);
            LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
            LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
            for (int x = 0; x < cnt; ++x) {
                SplitA(a1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
            }
            inQueueA1.FreeTensor(a1Local);
            inQueueB1.FreeTensor(b1Local);
        }
        CopyOut(c1Local, offsetC_1);

        baseM = _baseM; 
        sizeA = baseM * baseK;
    }
private:
    __aicore__ inline void CopyInA(int offsetA_1, int cnt) {
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();
        Nd2NzParams intriParamsMatA {
            1,
            static_cast<uint16_t>(baseM),
            static_cast<uint16_t>(baseK * cnt),
            0,
            static_cast<uint16_t>(K),
            static_cast<uint16_t>(baseM),
            1,
            1
        };
        DataCopy(a1Local, aGlobal[offsetA_1], intriParamsMatA);
        inQueueA1.EnQue(a1Local);
    }
    __aicore__ inline void CopyInB(int offsetB_1, int cnt) {
        LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
        Nd2NzParams intriParamsMatB {
            1,
            static_cast<uint16_t>(baseK * cnt),
            static_cast<uint16_t>(alignedN),
            0,
            static_cast<uint16_t>(N),
            static_cast<uint16_t>(baseK * cnt),
            1,
            1
        };
        DataCopy(b1Local, bGlobal[offsetB_1], intriParamsMatB);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx, int TailK) {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        if (baseM == 16) {
            LoadData2DParams loadL0AParams;
            loadL0AParams.repeatTimes = baseK / 8;
            loadL0AParams.srcStride = 1;
            LoadData(a2Local, a1Local[splitIdx * sizeA], loadL0AParams);
        } else {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = TailK;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                    ((uint64_t)baseM << 16) | (uint64_t)TailK;
            LoadData<T>(a2Local, a1Local[splitIdx * sizeA], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx, int TailK) {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        if (baseK == 16) {
            LoadData2dTransposeParams loadDataParams {
                0,
                static_cast<uint8_t>(baseN / 16),
                1,
                0,
                static_cast<uint16_t>(baseN / 16 - 1)
            };
            LoadDataWithTranspose(b2Local, b1Local[splitIdx * sizeB], loadDataParams);
        } else {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sBL1NOffset_ = 0;
            loadData3DV2.channelSize = baseN;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) |
                                    ((uint64_t)TailK << 16) | (uint64_t)baseN;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<T>(b2Local, b1Local[splitIdx * baseK * 8], loadData3DV2);
        }
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<T> &c1Local, int first, const int unitFlag, const int TailK) {
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        MmadParams mmadParams;
        mmadParams.m = baseM;
        mmadParams.n = baseN;
        mmadParams.k = TailK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first) {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        } else {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (baseM * baseN >> 8 < 10) 
            PipeBarrier<PIPE_M>();
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1) {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = baseN;
        fixpipeParams.mSize = baseM;
        fixpipeParams.srcStride = baseM;
        fixpipeParams.dstStride = N;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        Fixpipe(cGlobal[offsetC_1], c1Local, fixpipeParams);
    }
};

#endif  // _GEMM_HPP_