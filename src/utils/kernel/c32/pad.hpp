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
 * \file pad.hpp
 * \brief
 */

#ifndef _PAD_HPP_
#define _PAD_HPP_

#include <cstdint>
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

using namespace AscendC;
using namespace matmul;

#define debug(x) printf("%s: %d\n", #x, x)
#define debugf(x) printf("%s: %f\n", #x, x)
#define puts(x) printf("%s\n", x)

template<typename T>
class PadCustom {
    static constexpr int TILE_LENGTH = 8192;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> queBind;
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    GlobalTensor<T> scrGlobal, dstGlobal;
    int N;
    int alignedN;
    int srcN;
    int dstN;
    int elementsPerBlock;
    int padN;
public:
    __aicore__ inline PadCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe) {
        pipe->InitBuffer(queBind, 2, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> scrGlobal, GlobalTensor<T> dstGlobal, int N, int srcN, int dstN) {
        this->scrGlobal = scrGlobal;
        this->dstGlobal = dstGlobal;
        this->N = N;
        this->srcN = srcN;
        this->dstN = dstN;
        elementsPerBlock = 32 / sizeof(T);
        this->alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        padN = (elementsPerBlock - (N % elementsPerBlock)) % elementsPerBlock;
    }
    __aicore__ inline void PadCustomWork(int M, int p, int id) {
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN, (k * id + min(id, r)) * dstN, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines) {
        int alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        int linesPerIter = TILE_LENGTH / alignedN;
        int loopCount = lines / linesPerIter;
        for (int i = 0; i < loopCount; ++i) {
            CopyLines(srcOffset, dstOffset, linesPerIter);
            srcOffset += linesPerIter * srcN;
            dstOffset += linesPerIter * dstN;
        }
        lines -= linesPerIter * loopCount;
        if (!lines) return;
        CopyLines(srcOffset, dstOffset, lines);
    }
private:
    __aicore__ inline void CopyLines(int srcOffset, int dstOffset, int lines) {
        LocalTensor<T> bindLocal = queBind.AllocTensor<T>();
        DataCopyExtParams copyInParams {
            static_cast<uint16_t>(lines),
            static_cast<uint16_t>(N * sizeof(T)),
            static_cast<uint16_t>((srcN - N) * sizeof(T)),
            0,
            0
        };
        DataCopyPadExtParams<T> padParams {false, 0, static_cast<uint8_t>(padN), 0};
        DataCopyPad(bindLocal, scrGlobal[srcOffset], copyInParams, padParams);
        queBind.EnQue(bindLocal);
        bindLocal = queBind.DeQue<T>();
        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(lines),
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            static_cast<uint16_t>((dstN - N) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobal[dstOffset], bindLocal, copyOutParams);
        queBind.FreeTensor(bindLocal);
    }
};

__aicore__ inline void pad(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, GM_ADDR A_org, GM_ADDR A_work) {
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));

    PadCustom<float> oppad;
    oppad.Init(&tbufPool);
    oppad.SetMatrix(aGlobalOrg, aGlobal, orgN, orgN, strideN);
    oppad.PadCustomWork(orgM, blockNum << 1, blockIdx);
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
}

__aicore__ inline void restore(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, GM_ADDR A_org, GM_ADDR A_work) {
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));

    PadCustom<float> oppad;
    oppad.Init(&tbufPool);
    oppad.SetMatrix(aGlobal, aGlobalOrg, orgN, strideN, orgN);
    oppad.PadCustomWork(orgM, blockNum << 1, blockIdx);

    tbufPool.Reset();
}

#endif  // _PAD_HPP_