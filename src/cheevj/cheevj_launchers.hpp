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
 * \file cheevj_launchers.hpp
 * \brief Internal launch interfaces shared by the Cheevj host and device translation units.
 */

#ifndef CHEEVJ_LAUNCHERS_HPP
#define CHEEVJ_LAUNCHERS_HPP

#include <cstdint>

void cheevj_kernel_do(uint8_t* sync, uint8_t* a, uint8_t* w, uint8_t* info, uint8_t* workspace, uint8_t* tiling,
                      uint32_t numBlocks, void* stream);

void cheevj_fixed_kernel_do(
    uint8_t* matrixReal, uint8_t* matrixImag, uint8_t* packedReal, uint8_t* packedImag, uint8_t* panelVReal,
    uint8_t* panelVImag, uint8_t* panelWReal, uint8_t* panelWImag, uint8_t* wHReal, uint8_t* wHImag,
    uint8_t* wHImagNeg, uint8_t* updateReal, uint8_t* updateImag, uint8_t* diagonal, uint8_t* offDiagonal,
    uint8_t* tauReal, uint8_t* tauImag, uint8_t* panelWorkspace, uint8_t* barrierWorkspace, uint8_t* reflectorReal,
    uint8_t* reflectorImag, uint8_t* bounds, uint8_t* eigenvalues, uint8_t* lowWorkspace, uint8_t* highWorkspace,
    uint8_t* tridiagonalEigenvectorsRowMajor, uint8_t* commandWorkspace, uint8_t* eigenvectorReal,
    uint8_t* eigenvectorImag, uint8_t* info, int n, bool computeVectors, void* stream);

void cheevj_fixed_vectors_do(uint8_t* diagonal, uint8_t* offDiagonal, uint8_t* eigenvalues,
                             uint8_t* tridiagonalEigenvectors, uint8_t* reflectorReal, uint8_t* reflectorImag,
                             uint8_t* tauReal, uint8_t* tauImag, uint8_t* eigenvectorReal, uint8_t* eigenvectorImag,
                             uint8_t* info, int n, void* stream);
void cheevj_fixed_panel_do(uint8_t* packedReal, uint8_t* packedImag, uint8_t* panelVReal, uint8_t* panelVImag,
                           uint8_t* panelWReal, uint8_t* panelWImag, uint8_t* wHReal, uint8_t* wHImag,
                           uint8_t* wHImagNeg, uint8_t* diagonal, uint8_t* offDiagonal, uint8_t* tauReal,
                           uint8_t* tauImag, uint8_t* panelWorkspace, uint8_t* barrierWorkspace, int n, int activeN,
                           void* stream);
void cheevj_fixed_sturm_do(uint8_t* diagonal, uint8_t* offDiagonal, uint8_t* bounds, uint8_t* eigenvalues,
                           uint8_t* lowWorkspace, uint8_t* highWorkspace, int n, void* stream);
void cheevj_fixed_finalize_do(uint8_t* diagonal, uint8_t* offDiagonal, uint8_t* tauReal, uint8_t* tauImag,
                              uint8_t* matrixReal, uint8_t* matrixImag, uint8_t* bounds, uint8_t* info, int n,
                              void* stream);
void cheevj_fixed_backtransform_do(uint8_t* tridiagonalEigenvectors, uint8_t* reflectorReal, uint8_t* reflectorImag,
                                   uint8_t* tauReal, uint8_t* tauImag, uint8_t* eigenvectorReal,
                                   uint8_t* eigenvectorImag, uint8_t* info, int n, void* stream);

#endif  // CHEEVJ_LAUNCHERS_HPP
