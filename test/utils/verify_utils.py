# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared accuracy verification helpers for solver matrix-inverse ST scripts."""

import logging
import sys

import numpy as np

# 输出格式与既有验证脚本保持一致（纯消息文本，无日志级别前缀）
logging.basicConfig(level=logging.INFO, format="%(message)s")

RELATIVE_TOL = 5e-3
ABSOLUTE_TOL = 5e-3
ERROR_TOL = 1e-4
MAX_PRINTED_MISMATCHES = 32
DEFAULT_MATRIX_ORDER = 32
DTYPE_FLOAT32 = np.float32
DTYPE_COMPLEX64 = np.complex64


def verify_inverse(input_path, inv_path, order, dtype, format_mismatch):
    """校验 inv_path 矩阵是否为 input_path 矩阵的逆（A @ Ainv ≈ I）。

    差异元素由 format_mismatch(index, expected, actual) 负责格式化记录，
    最多记录 MAX_PRINTED_MISMATCHES 条；返回是否满足误差容限。
    """
    a = np.fromfile(input_path, dtype=dtype).reshape(order, order)
    inv_a = np.fromfile(inv_path, dtype=dtype).reshape(order, order)
    output = np.dot(a, inv_a).reshape(-1)
    golden = np.eye(order, dtype=dtype).reshape(-1)
    close = np.isclose(
        output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=True
    )
    mismatch_indexes = np.flatnonzero(~close)
    for printed, index in enumerate(mismatch_indexes):
        if printed >= MAX_PRINTED_MISMATCHES:
            break
        logging.info("%s", format_mismatch(index, golden[index], output[index]))
    error_ratio = mismatch_indexes.size / golden.size
    logging.info("error ratio: %.4f, tolerance: %.4f", error_ratio, ERROR_TOL)
    return error_ratio <= ERROR_TOL


def run_verification(input_path, inv_path, dtype, format_mismatch):
    """ST 验证入口：可选命令行参数指定矩阵阶数（默认 32），返回进程退出码。"""
    try:
        order = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MATRIX_ORDER
        passed = verify_inverse(input_path, inv_path, order, dtype, format_mismatch)
    except Exception as err:
        logging.error("%s", err)
        return 1
    if passed:
        logging.info("[Success] Case accuracy verification passed.")
        return 0
    logging.info("[Failed] Case accuracy verification failed.")
    return 1
