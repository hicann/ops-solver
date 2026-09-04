# Atlas 950 单精度复数Cholesky分解、求解和批量接口 任务书

## 1. 任务概述

在昇腾 NPU 上基于 AscendC/CATLASS，开发稠密 Hermitian 正定线性求解功能以及配套批量接口，具体 API 包括 `aclsolverCpotrf`、`aclsolverCpotrs`、`aclsolverCpotri`、`aclsolverCpotrfBatched`、`aclsolverCpotrsBatched`。接口参考 NVIDIA cuSolver DN 稠密 Cholesky 接口（分解 `potrf`、求解 `potrs`、求逆 `potri`）以及批量接口（`potrfBatched`、`potrsBatched`）。要求接口、功能、参数、约束与对标 CUDA 接口对齐，精度达到 CANN 生态算子开源标准。完成设计、开发、测试全流程，验收通过后合入 https://gitcode.com/cann/ops-solver 。

本任务适配硬件为 **Ascend 950PR（Atlas 950）**。五个接口作为**同一社区任务**一并交付，不允许拆分验收。ops-solver 仓库当前无对应复数 Cholesky 原型，须全新实现。

## 2. 核心开发要求

### 2.1 功能实现要求

对标基线如下：

| NPU 交付接口 | 对标 CUDA 接口 | 对标文档 |
|---|---|---|
| `aclsolverCpotrf` / `aclsolverCpotrf_bufferSize` | `cusolverDnCpotrf` / `cusolverDnCpotrf_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverCpotrs` | `cusolverDnCpotrs` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverCpotri` / `aclsolverCpotri_bufferSize` | `cusolverDnCpotri` / `cusolverDnCpotri_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverCpotrfBatched` | `cusolverDnCpotrfBatched` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-potrfbatched |
| `aclsolverCpotrsBatched` | `cusolverDnCpotrsBatched` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-potrsbatched |

#### 数学公式

1. **aclsolverCpotrf（Hermitian 正定 Cholesky 分解）**

`uplo = LOWER` 时：

```
A = L · L^H
```

`uplo = UPPER` 时：

```
A = U^H · U
```

其中 A 为 n × n 单精度复 Hermitian 正定矩阵（COMPLEX64）。仅处理 `uplo` 指定的三角部分，分解结果原地覆盖该三角部分。对角元为实数。对齐 `cusolverDnCpotrf`：未使用的另一半三角可作为 workspace 被破坏。

2. **aclsolverCpotrs（基于 Cholesky 因子求解）**

```
A · X = B
```

其中 A 为已由 `aclsolverCpotrf` 分解的因子，B 为 n × nrhs 右端项，求解后 X 原地覆盖 B。`uplo` 必须与分解时一致。

3. **aclsolverCpotri（基于 Cholesky 因子求逆）**

```
A^{-1} · A = I
```

输入 A 为 `aclsolverCpotrf` 得到的三角因子，输出原地覆盖为 Hermitian 逆矩阵的对应三角部分。语义对齐 `cusolverDnCpotri`。

4. **aclsolverCpotrfBatched（批量 Cholesky 分解）**

对 i = 0, ..., batchSize-1，对每个 A[i] 执行与 `aclsolverCpotrf` 相同的分解。`infoArray[i] = k > 0` 表示第 i 个矩阵的 k 阶顺序主子式不正定。

5. **aclsolverCpotrsBatched（批量 Cholesky 求解）**

对 i = 0, ..., batchSize-1：

```
A[i] · X[i] = B[i]
```

A[i] 须先由 `aclsolverCpotrfBatched` 分解。求解原地覆盖 B[i]。**仅支持 nrhs = 1**（对齐 `cusolverDnCpotrsBatched`）。`info` 为标量：仅用于报告非法参数（`-i`），正定性由 `potrfBatched` 的 `infoArray` 反映。

#### 算法说明

1. 数据布局必须为 **列主序（column-major）**。`lda`、`ldb` 为 leading dimension，须满足 `lda >= max(1, n)` 等对标约束，禁止要求 `lda == n`。
2. 矩阵、info、workspace 均为 **Device 指针**；Host 侧仅传入标量维数、枚举和 handle。
3. `aclsolverCpotrf` / `aclsolverCpotri` 须提供 `_bufferSize`；计算接口带 `Lwork`，对齐 `cusolverDnCpotrf` / `cusolverDnCpotri`。
4. `devInfo` / `info` 必须真实写入：`0` 成功；`-i` 表示第 i 个参数非法（不计 handle）；`i > 0` 表示第 i 阶顺序主子式不正定。批量 `potrf` 对每个 batch 独立写 `infoArray[i]`。
5. 须复用已有 handle 管理接口：`aclsolverCreate`、`aclsolverDestroy`、`aclsolverSetStream`、`aclsolverGetStream`。计算走调用方 stream，禁止无必要的 Host 同步。
6. 核心计算必须在 NPU AI Core 上完成，不允许 CPU fallback 代替 NPU 实现。
7. **确定性计算要求**：相同输入、相同 stream 串行多次执行，输出（含 `info`）bit-wise 一致。
8. 批量接口的 `Aarray` / `Barray` 为 **Device 上的指针数组**，不是 `[batch, n, n]` 连续张量。
9. 复数类型须在 `cann_ops_solver_common.h` 中定义与 `cuComplex` 布局一致的 `aclFloatComplex`（两个连续 FLOAT32：real、imag）。公开头文件须为 C 可调用。

### 2.2 算子工程模式

使用 **ops-solver Host C API + AscendC/CATLASS Kernel 直调** 工程模式（非 aclnn 两段式、非 PyTorch 接口）。

- 公开头文件：https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver.h 、https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver_common.h
- Host：参数校验、workspace 计算、kernel 下发
- Kernel：基于 AscendC/CATLASS 实现复数 POTRF / TRSM / POTRI / 批量接口
- 返回值：计算接口统一返回 `aclsolverStatus_t`，状态码语义对齐 cuSolver。若需短暂兼容现存 `aclError` 原型，须在头文件与文档中给出对照表，且最终验收以 `aclsolverStatus_t` 公开接口为准

### 2.3 接口定义

须在 `cann_ops_solver_common.h` 中补充：

```
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,  /* 下三角，对齐 CUBLAS_FILL_MODE_LOWER */
    ACLSOLVER_FILL_MODE_UPPER = 1   /* 上三角，对齐 CUBLAS_FILL_MODE_UPPER */
} aclsolverFillMode_t;
```

公开计算接口如下。

```
aclsolverStatus_t aclsolverCpotrf_bufferSize(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCpotrf(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    aclFloatComplex *Workspace,
    int Lwork,
    int *devInfo);

aclsolverStatus_t aclsolverCpotrs(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    int nrhs,
    const aclFloatComplex *A,
    int lda,
    aclFloatComplex *B,
    int ldb,
    int *devInfo);

aclsolverStatus_t aclsolverCpotri_bufferSize(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCpotri(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    aclFloatComplex *Workspace,
    int Lwork,
    int *devInfo);

aclsolverStatus_t aclsolverCpotrfBatched(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

aclsolverStatus_t aclsolverCpotrsBatched(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    int nrhs,
    aclFloatComplex *Aarray[],
    int lda,
    aclFloatComplex *Barray[],
    int ldb,
    int *info,
    int batchSize);
```

参数名、顺序、含义、Device/Host 归属不得擅自增删或调换。维数类型与对标 CUDA legacy API 一致，使用 `int`（32-bit）。

### 2.4 参数说明

非法类型 / shape / dtype / 排布 / 值域须返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在 `devInfo`/`info` 可写时写入 `-i`。空问题（n=0 / nrhs=0 / batchSize=0）成功返回。`aclsolverCpotrsBatched` 仅支持 `nrhs = 1`，`nrhs ≠ 1` 且 n>0 时必须报错。

#### 2.4.1 aclsolverCpotrf / aclsolverCpotrf_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| uplo | 输入 | 使用下三角或上三角 | attr | aclsolverFillMode_t | - | - | LOWER、UPPER | 非法枚举报错 |
| n | 输入 | 方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| A | 输入/输出（原地） | Hermitian 正定矩阵；输出为 Cholesky 因子 | tensor | COMPLEX64 | 列主序 ND | lda × n | 有限复数；对角须为实数语义 | 空指针且 n>0 时报错；lda < max(1,n) 报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| Lwork | 输出（仅 bufferSize） / 输入（仅 potrf） | workspace 元素个数 | scalar | int* / int | - | [1] | Lwork >= 0 | bufferSize 时指针为空报错 |
| Workspace | 输入（仅 potrf） | Device 工作空间 | tensor | COMPLEX64 | ND | [Lwork] | - | 指针为空且 Lwork>0 时报错 |
| devInfo | 输出 | 分解信息 | tensor | INT32 | ND | [1] | 0 / -i / i | 空指针报错 |

#### 2.4.2 aclsolverCpotrs

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| uplo | 输入 | 须与 potrf 一致 | attr | aclsolverFillMode_t | - | - | LOWER、UPPER | 非法枚举报错 |
| n | 输入 | 方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| nrhs | 输入 | 右端项列数 | scalar | int | - | - | nrhs >= 0 | nrhs < 0 报错 |
| A | 输入 | Cpotrf 得到的三角因子，只读 | tensor | COMPLEX64 | 列主序 ND | lda × n | 有限复数 | 空指针且 n>0 时报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| B | 输入/输出（原地） | 输入为 B，输出为 X | tensor | COMPLEX64 | 列主序 ND | ldb × nrhs | 有限复数 | 空指针且 n,nrhs>0 时报错；ldb < max(1,n) 报错 |
| ldb | 输入 | B 的 leading dimension | scalar | int | - | - | ldb >= max(1,n) | 不满足报错 |
| devInfo | 输出 | 求解信息 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |

#### 2.4.3 aclsolverCpotri / aclsolverCpotri_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| uplo | 输入 | 须与 potrf 一致 | attr | aclsolverFillMode_t | - | - | LOWER、UPPER | 非法枚举报错 |
| n | 输入 | 方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| A | 输入/输出（原地） | 输入为三角因子，输出为逆矩阵对应三角 | tensor | COMPLEX64 | 列主序 ND | lda × n | 有限复数 | 空指针且 n>0 时报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| Lwork | 输出（仅 bufferSize） / 输入（仅 potri） | workspace 元素个数 | scalar | int* / int | - | [1] | Lwork >= 0 | bufferSize 时指针为空报错 |
| Workspace | 输入（仅 potri） | Device 工作空间 | tensor | COMPLEX64 | ND | [Lwork] | - | 指针为空且 Lwork>0 时报错 |
| devInfo | 输出 | 求逆信息 | tensor | INT32 | ND | [1] | 0 / -i / i | 空指针报错 |

#### 2.4.4 aclsolverCpotrfBatched

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| uplo | 输入 | 使用下三角或上三角 | attr | aclsolverFillMode_t | - | - | LOWER、UPPER | 非法枚举报错 |
| n | 输入 | 每个方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| Aarray | 输入/输出（原地） | Device 指针数组 | list(tensor) | COMPLEX64 | 列主序，每个矩阵 lda × n | 指针数组长度 batchSize | 有限复数 | 空指针、lda < max(1,n) 报错 |
| lda | 输入 | 每个 Aarray[i] 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| infoArray | 输出 | 每个 batch 的 info | tensor | INT32 | ND | [batchSize] | 0 / k | 空指针报错 |
| batchSize | 输入 | 矩阵个数 | scalar | int | - | - | batchSize >= 0 | batchSize < 0 报错 |

#### 2.4.5 aclsolverCpotrsBatched

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| uplo | 输入 | 须与 potrfBatched 一致 | attr | aclsolverFillMode_t | - | - | LOWER、UPPER | 非法枚举报错 |
| n | 输入 | 每个方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| nrhs | 输入 | 右端项列数，仅支持 1 | scalar | int | - | - | nrhs == 1（n=0 或 batchSize=0 时可放宽） | nrhs ≠ 1 报错 |
| Aarray | 输入 | Device 指针数组，Cholesky 因子 | list(tensor) | COMPLEX64 | 列主序，每个矩阵 lda × n | 指针数组长度 batchSize | 有限复数 | 空指针报错 |
| lda | 输入 | 每个 Aarray[i] 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| Barray | 输入/输出（原地） | Device 指针数组，输入 B、输出 X | list(tensor) | COMPLEX64 | 列主序，每个矩阵 ldb × nrhs | 指针数组长度 batchSize | 有限复数 | 空指针、ldb < max(1,n) 报错 |
| ldb | 输入 | 每个 Barray[i] 的 leading dimension | scalar | int | - | - | ldb >= max(1,n) | 不满足报错 |
| info | 输出 | 标量 info，仅报告非法参数 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |
| batchSize | 输入 | 矩阵个数 | scalar | int | - | - | batchSize >= 0 | batchSize < 0 报错 |

**规模与泛化（验收下限，允许实现支持更大范围并在文档中声明）**：

| 接口 | 维数 | 其他 |
|---|---|---|
| aclsolverCpotrf / aclsolverCpotri | n ∈ [0, 4096] | uplo 覆盖 LOWER/UPPER；lda padding |
| aclsolverCpotrs | n ∈ [0, 4096]，nrhs ∈ [0, 128] | uplo 覆盖 LOWER/UPPER |
| aclsolverCpotrfBatched | n ∈ [1, 256]，batchSize ∈ [1, 3000] | uplo 覆盖 LOWER/UPPER |
| aclsolverCpotrsBatched | n ∈ [1, 256]，batchSize ∈ [1, 3000]，nrhs=1 | nrhs=2 必须报错 |

不支持 broadcast，不要求图融合。dynamic shape：Host 按本次调用的 n/nrhs/batchSize 生成 tiling。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Ascend 950PR（Atlas 950）
- **CANN 版本**：CANN 9.0.0 及以上，须与 https://gitcode.com/cann/ops-solver 仓库 README 已验证配套版本一致
- **对标软件（性能金标采集）**：CUDA Toolkit 中的 cuSolver，GPU 为 NVIDIA A100
- **其他依赖**：ops-solver 仓编译所需 ACL / AscendC / CATLASS 工具链；精度 golden 可使用 NumPy / SciPy LAPACK（complex128）或 CPU 参考实现
- **版本记录**：自测报告须记录 CANN 版本、NPU 具体型号（950PR）、驱动版本、CUDA / cuSolver 版本

功能和精度、性能均须提交 950PR 测试结果。

### 3.2 精度要求

1. 对标接口见 2.1 节。数值正确性以更高精度 CPU 参考为单标杆：COMPLEX64 计算的 golden 使用 COMPLEX128（NumPy/SciPy `complex128` 的 `cholesky` / `cho_solve` / `inv`，或等价 LAPACK `zpotrf`/`zpotrs`/`zpotri`）。
2. NPU 输出（`aclsolverCpotrf` 还原 L Lᴴ 或 Uᴴ U 后与原 A 的指定三角比较，或 `aclsolverCpotrs` 的 X、`aclsolverCpotri` 的 A⁻¹ 及 A A⁻¹ 相对 I、批量接口对应输出）须满足生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
3. COMPLEX64 按实部、虚部分别作为 FLOAT32 做混合容差判定：

    | 数据类型 | FLOAT32（实部 / 虚部分别判定） |
    |----------|---------|
    | rtol | 2^-10 (9.77e-4) |
    | atol | 2^-16 (1.53e-5) |
    | required_matched_ratio | 0.99 |
    | max_abs_error_limit | 1e-2 or 32 * ULP |

    逐元素判定：|actual - golden| ≤ atol + rtol × |golden|。当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

4. `info` 对非正定矩阵须给出正确正值下标（第 k 阶顺序主子式不正定）。
5. 覆盖对角占优 Hermitian 正定、随机 Hermitian 正定（A = BᴴB + n I）、故意构造的非正定矩阵；INF/NAN 按精度标准文档对应规则验收。
6. 确定性：合法 Hermitian 正定用例重复执行须 bit-wise 一致。

### 3.3 性能要求

1. 性能对标为 NVIDIA A100 上同 shape、同 dtype（COMPLEX64）、同列主序、同等预热与计时口径下的 CUDA 接口：
   - aclsolverCpotrf：`cusolverDnCpotrf`（含一次 bufferSize，不计在 kernel 耗时内）
   - aclsolverCpotrs：`cusolverDnCpotrs`（potrf 时间不计入本接口）
   - aclsolverCpotri：`cusolverDnCpotri`（potrf 时间不计入本接口）
   - aclsolverCpotrfBatched：`cusolverDnCpotrfBatched`
   - aclsolverCpotrsBatched：`cusolverDnCpotrsBatched`
2. 达标判据：对下表每一个 case，NPU kernel 耗时满足 T_NPU ≤ T_A100 / 0.8，即性能倍率 ≥ 0.8 倍 A100。须在 950PR 上达标。
3. 每个 case 至少预热 10 次、正式采样 30 次，报告中位数；每轮须 Device 同步后计时。不含首次编译、数据生成、H2D/D2H。workspace / 指针数组在正式采样期间复用。
4. 开发者须在 A100 上实测填入 T_A100，并在报告中给出 CUDA 版本与采集脚本。下表 T_A100 列为待测占位。

| 编号 | 接口 | 规格 | GPU A100 性能 | NPU 耗时 | 目标 |
|---|---|---|---|---|---|
| P-01 | aclsolverCpotrf | n=1024，uplo=LOWER，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-02 | aclsolverCpotrf | n=4096，uplo=LOWER，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-03 | aclsolverCpotrf | n=2048，uplo=UPPER，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-04 | aclsolverCpotrs | n=1024，nrhs=1，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-05 | aclsolverCpotrs | n=4096，nrhs=32，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-06 | aclsolverCpotrs | n=2048，nrhs=8，uplo=UPPER | 待测 | 待测 | ≥ 0.8×A100 |
| P-07 | aclsolverCpotri | n=1024，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-08 | aclsolverCpotri | n=4096，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-09 | aclsolverCpotrfBatched | n=32，batchSize=1024，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-10 | aclsolverCpotrfBatched | n=128，batchSize=128，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-11 | aclsolverCpotrsBatched | n=32，batchSize=1024，nrhs=1，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |
| P-12 | aclsolverCpotrsBatched | n=128，batchSize=128，nrhs=1，uplo=LOWER | 待测 | 待测 | ≥ 0.8×A100 |

全部 12 条在 950PR 上均须达标，未达标不予验收。

### 3.4 内存要求

对标 cuSolver 同等规格：

1. `aclsolverCpotrf` / `aclsolverCpotri` 的 Device workspace 不超过 `bufferSize` 返回值；禁止额外分配未计入 `Lwork` 的隐蔽大块 Device 内存，并在设计文档中给出公式。
2. 批量接口除输入/输出矩阵与指针数组、info 外，临时 Device 内存须在设计文档中量化；同规格下 NPU 额外 workspace 不得超过 A100 对应接口 workspace 的 2 倍。
3. 无内存泄漏：连续创建 handle、查询 workspace、执行、销毁，重复 100 次后 Device 已用内存回到基线。

### 3.5 自验要求

本任务使用 ops-solver 仓测试工程（https://gitcode.com/cann/ops-solver/tree/master/test ）及 AscendOpTest 工具（https://gitcode.com/HIT1920/AscendOpTest ）进行自验。请根据本任务给出的自测用例和测试指导完成自测，并输出自测报告。自测用例目录：./cpotrf_family_950_testCase/

NPU 接口与 CUDA 接口参数序列一致（见 2.3），无需额外映射。差异仅在：命名前缀 `aclsolver` vs `cusolverDn`，以及 `aclsolverFillMode_t` vs `cublasFillMode_t`（枚举值 0/1 对齐）、`aclFloatComplex` vs `cuComplex`。

功能自验至少覆盖：

| 类别 | 必测场景 |
|---|---|
| 基础功能 | potrf；potrf+potrs 求解；potrf+potri 求逆；potrfBatched；potrsBatched |
| uplo | LOWER 与 UPPER |
| padding | lda/ldb > 最小合法值 |
| info | 成功 info=0；非正定 info=i；非法 lda 等 info=-i |
| 批量边界 | batchSize=1；n=1；nrhs=1 合法；potrsBatched 的 nrhs=2 报错 |
| 空问题 | n=0 或 nrhs=0 或 batchSize=0 成功返回 |
| 确定性 | 同一输入重复执行 bit-wise 一致 |
| 流 | 非默认 stream 下结果正确 |

测试用例入参生成规则：

| 参数名 | Tensor值域分布 | Attr 覆盖规则 |
|---|---|---|
| A（精度主路径，须 Hermitian 正定） | 50%：随机 B 实/虚均匀 [-5,5] 后 A = BᴴB + n I；50%：正态同样构造。另用 10% 用例构造非正定用于 info | n 覆盖 2 的幂与 2 的幂-1 |
| B | 实/虚均匀/正态各 50% | nrhs 覆盖 1、8、32、128 及边界 0；batched 仅 nrhs=1 |
| uplo | - | LOWER、UPPER 全覆盖 |
| lda/ldb | - | 等于最小合法值，以及 +8 / +32 padding |
| batchSize | - | 1、8、128、1024、3000 及 0 |

## 4. 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 需要根据每个仓库的规范提供算子readme文档 <br> |

![邀请示意](./pics/invite.jpeg)

## 5. PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的接口合入 https://gitcode.com/cann/ops-solver 。建议目录：

```
include/cann_ops_solver.h
include/cann_ops_solver_common.h
src/cpotrf/
src/cpotrs/
src/cpotri/
src/cpotrf_batched/
src/cpotrs_batched/
test/cpotrf/
test/cpotrs/
test/cpotri/
test/cpotrf_batched/
test/cpotrs_batched/
docs/zh/cpotrf.md
docs/zh/cpotrs.md
docs/zh/cpotri.md
docs/zh/cpotrf_batched.md
docs/zh/cpotrs_batched.md
docs/api_list.md
```

须同步更新 README 与接口列表。五个接口作为同一 PR 或一组关联 PR 合入，不得只合入其中部分接口。

## 6. 参考资料

1. NVIDIA cuSolver 文档（potrf / potrs / potri / potrfBatched / potrsBatched）：https://docs.nvidia.com/cuda/cusolver/index.html ；
2. ops-solver 仓库：https://gitcode.com/cann/ops-solver ；
3. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ；
4. AscendOpTest：https://gitcode.com/HIT1920/AscendOpTest ；
5. Ascend C算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
6. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ；
7. Ascend C在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713 ；
8. CATLASS：https://gitcode.com/cann/catlass 。

## 7. 特别注意事项

1. 本任务五个接口必须全部交付：aclsolverCpotrf、aclsolverCpotrs、aclsolverCpotri、aclsolverCpotrfBatched、aclsolverCpotrsBatched。缺一不可。
2. 文档中的 cuSolver 仅作为接口、功能、约束和性能标杆，NPU 实现统一使用 aclsolver + AscendC/CATLASS，禁止依赖 CUDA。
3. 必须列主序、Device 指针、真实写入 info、支持合法 lda padding。
4. `aclsolverCpotrsBatched` 仅支持 nrhs=1，nrhs≠1 必须报错。
5. `aclsolverCpotrs` / `aclsolverCpotri` / `aclsolverCpotrsBatched` 的输入是已分解的 Cholesky 因子，不是未分解的原始 A。
6. 对齐 cuSolver：未使用的另一半三角可作为 workspace 被破坏，验收时只比较 `uplo` 指定的三角部分。对角元须保持实数语义。
7. 性能门禁为 0.8 倍 A100，须在 950PR 上达标。
8. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请。
9. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 。

## 8. 环境获取（无需修改，使用模板原始内容）

 1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。
 - **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**
 	 
 	![环境截图](./pics/zaixiankaifa1.png)  
 	![环境截图](./pics/apply.png)  
 	 
2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
 	 
 	![环境截图](./pics/yunkaifa.png)
 	 
3. 如需额外环境资源，请联系昇腾CANN小助手。
