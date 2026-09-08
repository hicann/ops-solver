/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    AscendC::GlobalTensor<float> tridiagonalEigenvectorsGlobal;
    AscendC::GlobalTensor<float> reflectorRealGlobal;
    AscendC::GlobalTensor<float> reflectorImagGlobal;
    AscendC::GlobalTensor<float> tauRealGlobal;
    AscendC::GlobalTensor<float> tauImagGlobal;
    AscendC::GlobalTensor<float> eigenvectorRealGlobal;
    AscendC::GlobalTensor<float> eigenvectorImagGlobal;
    AscendC::GlobalTensor<int32_t> infoGlobal;

    __aicore__ inline void BindCommonGlobals(GM_ADDR tridiagonalEigenvectorsRowMajor, GM_ADDR reflectorReal,
                                             GM_ADDR reflectorImag, GM_ADDR tauReal, GM_ADDR tauImag,
                                             GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag, GM_ADDR info)
    {
        tridiagonalEigenvectorsGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(tridiagonalEigenvectorsRowMajor));
        reflectorRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reflectorReal));
        reflectorImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reflectorImag));
        tauRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tauReal));
        tauImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tauImag));
        eigenvectorRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectorReal));
        eigenvectorImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectorImag));
        infoGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(info));
    }

    __aicore__ inline void IgnoreInitArgs(AscendC::TPipe *pipe, GM_ADDR tridiagonalEigenvectorsRowMajor,
                                          GM_ADDR reflectorReal, GM_ADDR reflectorImag, GM_ADDR tauReal,
                                          GM_ADDR tauImag, GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag,
                                          GM_ADDR info)
    {
        (void)pipe;
        (void)tridiagonalEigenvectorsRowMajor;
        (void)reflectorReal;
        (void)reflectorImag;
        (void)tauReal;
        (void)tauImag;
        (void)eigenvectorReal;
        (void)eigenvectorImag;
        (void)info;
    }

    __aicore__ inline void InitCommonBuffers(AscendC::TPipe *pipe, int matrixN, int reductionLanes)
    {
        pipe->InitBuffer(reflectorRealBuf, matrixN * sizeof(float));
        pipe->InitBuffer(reflectorImagBuf, matrixN * sizeof(float));
        pipe->InitBuffer(tauRealBuf, matrixN * sizeof(float));
        pipe->InitBuffer(tauImagBuf, matrixN * sizeof(float));
        pipe->InitBuffer(scratch0Buf, matrixN * sizeof(float));
        pipe->InitBuffer(scratch1Buf, matrixN * sizeof(float));
        pipe->InitBuffer(scratch2Buf, matrixN * sizeof(float));
        pipe->InitBuffer(reductionBuf, reductionLanes * sizeof(float));
        pipe->InitBuffer(infoBuf, 8 * sizeof(int32_t));
    }

    __aicore__ inline int32_t LoadInfo()
    {
        auto info = infoBuf.Get<int32_t>();
        AscendC::DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> pad{true, 0, 7, 0};
        AscendC::DataCopyPad(info, infoGlobal, copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        return info.GetValue(0);
    }

    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float> &destination,
                                  const AscendC::GlobalTensor<float> &source, int count)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(destination, source, copy, noPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOut(AscendC::GlobalTensor<float> destination, int offset,
                                   const AscendC::LocalTensor<float> &source, int count)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(destination[offset], source, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
