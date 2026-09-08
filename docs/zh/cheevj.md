## Cheevj

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 开发中 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Ascend 950PR/Ascend 950DT | × |

## 功能说明

`aclsolverCheevj` 计算单精度复数 Hermitian 矩阵的特征值，并可选计算特征向量：

$$
A V = V \Lambda
$$

其中 $A$ 为 $n \times n$ 复数 Hermitian 矩阵，$\Lambda$ 为实数特征值组成的
对角矩阵，$V$ 为特征向量矩阵。

- `W` 按升序返回 $n$ 个单精度实数特征值。
- `jobz=ACLSOLVER_EIG_MODE_VECTOR` 时，`A` 原地返回按列存储的特征向量。
- `jobz=ACLSOLVER_EIG_MODE_NOVECTOR` 时，`A` 保持不变。
- `uplo` 指定只读取 `A` 的上三角或下三角，另一三角不参与计算。
- `A` 使用 column-major 布局，`lda` 为相邻两列起始位置之间的元素跨度。

## 函数原型

```cpp
aclError aclsolverCheevj(
    aclsolverHandle_t handle,
    aclsolverEigMode_t jobz,
    aclsolverFillMode_t uplo,
    const int64_t n,
    std::complex<float> *A,
    const int64_t lda,
    float *W,
    int32_t *info);
```

## 参数说明

| 参数 | 输入/输出 | 描述 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建的 solver handle |
| `jobz` | 输入 | 仅计算特征值，或同时计算特征值和特征向量 |
| `uplo` | 输入 | 指定读取 `A` 的上三角或下三角 |
| `n` | 输入 | 非负矩阵阶数；可执行规模由主机和设备可用内存决定 |
| `A` | 输入/输出 | COMPLEX64、column-major；向量模式下原地输出特征向量 |
| `lda` | 输入 | leading dimension，满足 `lda >= max(1, n)` |
| `W` | 输出 | 长度为 `n` 的 FLOAT32 升序特征值 |
| `info` | 输出 | `0` 表示成功；负值表示参数非法；正值表示数值求解未完成 |

枚举定义：

```cpp
typedef enum {
    ACLSOLVER_EIG_MODE_NOVECTOR = 0,
    ACLSOLVER_EIG_MODE_VECTOR = 1
} aclsolverEigMode_t;

typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,
    ACLSOLVER_FILL_MODE_UPPER = 1
} aclsolverFillMode_t;
```

## 算子约束

- 输入矩阵必须为复数 Hermitian 矩阵。
- 数据类型仅支持 COMPLEX64 输入和 FLOAT32 特征值输出。
- `n=0` 时允许 `A` 和 `W` 为空，接口直接成功返回。
- 接口不设置固定矩阵阶数上限；内存或设备索引资源不足时返回错误。
- 调用返回 `ACL_SUCCESS` 后仍应检查 `info` 是否为 `0`。

## 调用示例

完整测试入口见
[`test/cheevj/cheevj_test.cpp`](../../test/cheevj/cheevj_test.cpp)。

```cpp
#include <complex>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_solver.h"

int64_t n = 8;
std::vector<std::complex<float>> a(n * n);
std::vector<float> w(n);
int32_t info = 0;

// 按 column-major 填充 a 的指定三角区域。
aclError status = aclsolverCheevj(
    handle,
    ACLSOLVER_EIG_MODE_VECTOR,
    ACLSOLVER_FILL_MODE_LOWER,
    n,
    a.data(),
    n,
    w.data(),
    &info);
```

构建、契约测试、完整正确性门禁和性能测试命令见
[`test/cheevj/README.md`](../../test/cheevj/README.md)。
