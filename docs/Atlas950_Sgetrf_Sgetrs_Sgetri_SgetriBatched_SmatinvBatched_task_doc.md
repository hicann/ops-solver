# Atlas 950 单精度实数LU分解、求解和批量求逆 任务书

## 1. 任务概述

在昇腾 NPU 上基于 AscendC/CATLASS，开发稠密线性求解功能以及配套的稠密批量求逆功能，具体 API 包括 `aclsolverSgetrf`、`aclsolverSgetrs`、`aclsolverSgetri`、`aclsolverSgetriBatched`、`aclsolverSmatinvBatched`。接口参考 NVIDIA cuSolver DN 稠密线性求解接口（LU 分解 `getrf`、三角求解 `getrs`）、配套的稠密批量求逆接口（`getriBatched`、`matinvBatched`），以及单矩阵求逆对应的 LAPACK `sgetri`（cuSolver DN 无独立 getri）。要求接口、功能、参数、约束与对标 CUDA / LAPACK 接口对齐，精度达到 CANN 生态算子开源标准。完成设计、开发、测试全流程，验收通过后合入 https://gitcode.com/cann/ops-solver 。

本任务适配硬件为 **Ascend 950PR（Atlas 950）**。五个接口作为**同一社区任务**一并交付，不允许拆分验收。

## 2. 核心开发要求

### 2.1 功能实现要求

对标基线如下：

| NPU 交付接口 | 对标 CUDA / LAPACK 接口 | 对标文档 |
|---|---|---|
| `aclsolverSgetrf` / `aclsolverSgetrf_bufferSize` | `cusolverDnSgetrf` / `cusolverDnSgetrf_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverSgetrs` | `cusolverDnSgetrs` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverSgetri` / `aclsolverSgetri_bufferSize` | LAPACK `sgetri`，调用约定对齐 cuSolver DN | https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html |
| `aclsolverSgetriBatched` | `cublasSgetriBatched` | https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getribatched |
| `aclsolverSmatinvBatched` | `cublasSmatinvBatched` | https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-matinvbatched |

仓库 https://gitcode.com/cann/ops-solver 中已有 `aclsolverSgetrf` / `aclsolverSgetri` 原型，当前与对标接口**不对齐**（Host 指针、`lda` 强制等于 `n`、`info` 未写入、无 workspace 查询、无 `Sgetrs` / 批量接口）。本任务须将上述原型升级为下文定义的公开接口，并新增 `aclsolverSgetrs`、`aclsolverSgetriBatched`、`aclsolverSmatinvBatched`；不得将上述简化语义作为最终交付。

#### 数学公式

1. **aclsolverSgetrf（LU 分解，带部分主元）**

```
P · A = L · U
```

其中 A 为 m × n 单精度实矩阵，P 为置换矩阵，L 为单位下三角（或梯形）矩阵，U 为上三角（或梯形）矩阵。分解结果原地写入 A：对角线及上方为 U，严格下方为 L 的非对角元。`devIpiv` 为 1-based 主元序列：第 i 行与第 `devIpiv(i)` 行交换。若 `devIpiv == nullptr`，则不做选主元，分解为 `A = L · U`（数值不稳定，语义对齐 `cusolverDnSgetrf`）。

2. **aclsolverSgetrs（基于 LU 因子求解）**

```
op(A) · X = B
```

其中 A 为已由 `aclsolverSgetrf` 分解的 n × n 方阵，B 为 n × nrhs 右端项，求解后 X 原地覆盖 B。`op(A)` 由 `trans` 指定：`N` 为不转置，`T` / `C` 对实数均为转置。

3. **aclsolverSgetri（单矩阵求逆）**

```
A^{-1} · A = I
```

输入 A 为 `aclsolverSgetrf` 得到的 LU 因子，`devIpiv` 为对应主元。输出原地覆盖 A 为逆矩阵。语义对齐 LAPACK `sgetri`。

4. **aclsolverSgetriBatched（批量求逆，基于已分解 LU）**

对 i = 0, ..., batchSize-1：

```
C[i] = inv(A[i])
```

其中 A[i] 必须先完成 LU 分解（可用循环调用 `aclsolverSgetrf` 构造因子；本任务不强制交付 `SgetrfBatched`）。求逆为 **out-of-place**：C[i] 与 A[i] 内存不得重叠。

5. **aclsolverSmatinvBatched（小矩阵直接批量求逆）**

对原始矩阵直接求逆，功能等价 `getrf + getri` 的短路实现。**仅支持 n ≤ 32**；n > 32 必须返回非法参数，不得静默转调 `aclsolverSgetriBatched`（对齐 `cublasSmatinvBatched`）。若 A[i] 奇异，`info[i]` 报告奇异位置，语义同 `getrf`。

#### 算法说明

1. 数据布局必须为 **列主序（column-major）**，与 cuSolver / cuBLAS 一致。`lda`、`ldb`、`ldc`、`lda_inv` 为列间距（leading dimension），须满足 `lda >= max(1, 行数)` 等对标约束，禁止再要求 `lda == n`。
2. 矩阵、主元、info、workspace 均为 **Device 指针**；Host 侧仅传入标量维数、枚举和 handle。
3. `aclsolverSgetrf` / `aclsolverSgetri` 须提供 `_bufferSize` 查询接口，调用方按返回的 `Lwork` 分配 Device workspace 后再执行计算。
4. `devInfo` / `info` 必须真实写入：`0` 成功；`-i` 表示第 i 个参数非法（不计 handle）；`i > 0` 表示 U(i,i)=0（分解失败或求逆失败）。批量接口对每个 batch 独立写 `info[i]`。
5. 须复用已有 handle 管理接口：`aclsolverCreate`、`aclsolverDestroy`、`aclsolverSetStream`、`aclsolverGetStream`。计算走调用方 stream，禁止无必要的 Host 同步。
6. 核心计算必须在 NPU AI Core 上完成，不允许 CPU fallback 代替 NPU 实现。
7. **确定性计算要求**：相同输入、相同 stream 串行多次执行，输出（含 `ipiv` / `info`）bit-wise 一致。部分主元选取须为确定规则（例如列内绝对值最大且下标最小），禁止非确定原子竞争。

### 2.2 算子工程模式

使用 **ops-solver Host C API + AscendC/CATLASS Kernel 直调** 工程模式（非 aclnn 两段式、非 PyTorch 接口）。

- 公开头文件：https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver.h 、https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver_common.h
- Host：参数校验、workspace 计算、kernel 下发
- Kernel：基于 AscendC/CATLASS 实现 LU / TRSM / 求逆 / 批量求逆
- 返回值：计算接口统一返回 `aclsolverStatus_t`，状态码语义对齐 cuSolver（`SUCCESS` / `NOT_INITIALIZED` / `INVALID_VALUE` / `ARCH_MISMATCH` / `INTERNAL_ERROR` / `ALLOC_FAILED` / `HANDLE_IS_NULLPTR` 等）。若需短暂兼容现存 `aclError` 原型，须在头文件与文档中给出对照表，且最终验收以 `aclsolverStatus_t` 公开接口为准

### 2.3 接口定义

须在 `cann_ops_solver_common.h` 中补充与 `cublasOperation_t` 对齐的枚举：

```
typedef enum {
    ACLSOLVER_OP_N = 0,   /* 不转置，对齐 CUBLAS_OP_N */
    ACLSOLVER_OP_T = 1,   /* 转置，对齐 CUBLAS_OP_T */
    ACLSOLVER_OP_C = 2    /* 共轭转置；实数语义同 T，对齐 CUBLAS_OP_C */
} aclsolverOperation_t;
```

公开计算接口如下。

```
aclsolverStatus_t aclsolverSgetrf_bufferSize(
    aclsolverHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverSgetrf(
    aclsolverHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    float *Workspace,
    int *devIpiv,
    int *devInfo);

aclsolverStatus_t aclsolverSgetrs(
    aclsolverHandle_t handle,
    aclsolverOperation_t trans,
    int n,
    int nrhs,
    const float *A,
    int lda,
    const int *devIpiv,
    float *B,
    int ldb,
    int *devInfo);

aclsolverStatus_t aclsolverSgetri_bufferSize(
    aclsolverHandle_t handle,
    int n,
    float *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverSgetri(
    aclsolverHandle_t handle,
    int n,
    float *A,
    int lda,
    int *devIpiv,
    float *Workspace,
    int Lwork,
    int *devInfo);

aclsolverStatus_t aclsolverSgetriBatched(
    aclsolverHandle_t handle,
    int n,
    const float *const Aarray[],
    int lda,
    const int *PivotArray,
    float *const Carray[],
    int ldc,
    int *infoArray,
    int batchSize);

aclsolverStatus_t aclsolverSmatinvBatched(
    aclsolverHandle_t handle,
    int n,
    const float *const A[],
    int lda,
    float *const Ainv[],
    int lda_inv,
    int *info,
    int batchSize);
```

参数名、顺序、含义、Device/Host 归属不得擅自增删或调换。维数类型与对标 CUDA legacy API 一致，使用 `int`（32-bit）。批量接口的 `Aarray` / `Carray` / `A` / `Ainv` 为 **Device 上的指针数组**（array of device pointers），不是 `[batch, n, n]` 连续张量。`aclsolverSgetri` 计算接口带 `Lwork`，对齐 LAPACK `sgetri` / MAGMA 风格；`aclsolverSgetrf` 计算接口不带 `Lwork`，对齐 `cusolverDnSgetrf`。

### 2.4 参数说明

下列表格中「内存」列：`host` 表示 Host 标量/句柄，`device` 表示 Device 缓冲区。非法类型 / shape / dtype / 排布 / 值域须返回 `ACLSOLVER_STATUS_INVALID_VALUE`（或对等错误码），并在 `devInfo`/`info` 可写时写入 `-i`。

#### 2.4.1 aclsolverSgetrf / aclsolverSgetrf_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文，由 aclsolverCreate 创建 | scalar | 句柄 | - | - | 非空 | handle 为空返回 HANDLE_IS_NULLPTR |
| m | 输入 | 矩阵 A 的行数 | scalar | int | - | - | m >= 0 | m < 0 报错；m = 0 视为空问题成功返回 |
| n | 输入 | 矩阵 A 的列数 | scalar | int | - | - | n >= 0 | n < 0 报错；n = 0 视为空问题成功返回 |
| A | 输入/输出（原地） | 列主序矩阵。输入为 A，输出为 LU 因子 | tensor | FLOAT32 | 列主序 ND | lda × n，有效区域 m × n | 有限浮点；允许构造奇异矩阵以验证 info | A 为空且 m,n>0 时报错；lda < max(1,m) 报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,m) | 不满足报错 |
| Lwork | 输出（仅 bufferSize） | workspace 元素个数 | scalar | int* | - | [1] | Lwork >= 0 | Lwork 指针为空报错 |
| Workspace | 输入（仅 getrf） | Device 工作空间，长度至少为 Lwork | tensor | FLOAT32 | ND | [Lwork] | - | 指针为空且 Lwork>0 时报错 |
| devIpiv | 输出 | 1-based 主元，长度 min(m,n)。nullptr 表示不选主元 | tensor | INT32 | ND | [min(m,n)] | 取值落在 [1, m] | 非空时必须为 Device 指针 |
| devInfo | 输出 | 分解信息，Device 标量 | tensor | INT32 | ND | [1] | 0 / -i / i | 指针为空报错 |

#### 2.4.2 aclsolverSgetrs

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| trans | 输入 | op(A)：N / T / C | attr | aclsolverOperation_t | - | - | N、T、C | 非法枚举报错 |
| n | 输入 | 方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| nrhs | 输入 | 右端项列数 | scalar | int | - | - | nrhs >= 0 | nrhs < 0 报错 |
| A | 输入 | Sgetrf 得到的 LU 因子，只读 | tensor | FLOAT32 | 列主序 ND | lda × n | 有限浮点 | 空指针且 n>0 时报错；lda < max(1,n) 报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| devIpiv | 输入 | Sgetrf 输出的主元。若 getrf 时未选主元，此处须为 nullptr | tensor | INT32 | ND | [n] | 1-based | 与 getrf 选主元约定不一致时报错 |
| B | 输入/输出（原地） | 输入为 B，输出为 X | tensor | FLOAT32 | 列主序 ND | ldb × nrhs，有效区域 n × nrhs | 有限浮点 | 空指针且 n,nrhs>0 时报错；ldb < max(1,n) 报错 |
| ldb | 输入 | B 的 leading dimension | scalar | int | - | - | ldb >= max(1,n) | 不满足报错 |
| devInfo | 输出 | 求解信息 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |

#### 2.4.3 aclsolverSgetri / aclsolverSgetri_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| n | 输入 | 方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错；n = 0 视为空问题成功返回 |
| A | 输入/输出（原地） | 输入为 LU 因子，输出为逆矩阵 | tensor | FLOAT32 | 列主序 ND | lda × n | 有限浮点 | 空指针且 n>0 时报错；lda < max(1,n) 报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| Lwork | 输出（仅 bufferSize） / 输入（仅 getri） | workspace 元素个数 | scalar | int* / int | - | [1] | Lwork >= 0 | bufferSize 时指针为空报错；getri 时 Lwork 小于查询值报错 |
| Workspace | 输入（仅 getri） | Device 工作空间 | tensor | FLOAT32 | ND | [Lwork] | - | 指针为空且 Lwork>0 时报错 |
| devIpiv | 输入 | Sgetrf 输出的主元。未选主元时须为 nullptr | tensor | INT32 | ND | [n] | 1-based | 与 getrf 选主元约定不一致时报错 |
| devInfo | 输出 | 求逆信息 | tensor | INT32 | ND | [1] | 0 / -i / i | 空指针报错 |

#### 2.4.4 aclsolverSgetriBatched

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| n | 输入 | 每个方阵阶数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| Aarray | 输入 | Device 指针数组，Aarray[i] 指向第 i 个 LU 因子 | list(tensor) | FLOAT32 | 列主序，每个矩阵 lda × n | 指针数组长度 batchSize | 有限浮点 | 空指针、重叠输出、lda < max(1,n) 报错 |
| lda | 输入 | 每个 Aarray[i] 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| PivotArray | 输入 | 各矩阵主元线性排列，长度 n*batchSize；nullptr 表示不选主元 | tensor | INT32 | ND | [n * batchSize] | 1-based | 非法主元报错 |
| Carray | 输出（独立输出） | Device 指针数组，Carray[i] = inv(A[i])，不得与 Aarray[i] 重叠 | list(tensor) | FLOAT32 | 列主序，每个矩阵 ldc × n | 指针数组长度 batchSize | 有限浮点 | 重叠或 ldc < max(1,n) 报错 |
| ldc | 输入 | 每个 Carray[i] 的 leading dimension | scalar | int | - | - | ldc >= max(1,n) | 不满足报错 |
| infoArray | 输出 | 每个 batch 的 info | tensor | INT32 | ND | [batchSize] | 0 / k | 空指针报错 |
| batchSize | 输入 | 矩阵个数 | scalar | int | - | - | batchSize >= 0 | batchSize < 0 报错 |

#### 2.4.5 aclsolverSmatinvBatched

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| n | 输入 | 每个方阵阶数 | scalar | int | - | - | 0 <= n <= 32 | n < 0 或 n > 32 报错 |
| A | 输入 | Device 指针数组，原始矩阵 | list(tensor) | FLOAT32 | 列主序，每个矩阵 lda × n | 指针数组长度 batchSize | 有限浮点 | 空指针、lda < max(1,n) 报错 |
| lda | 输入 | 每个 A[i] 的 leading dimension | scalar | int | - | - | lda >= max(1,n) | 不满足报错 |
| Ainv | 输出（独立输出） | Device 指针数组，逆矩阵，不得与 A[i] 重叠 | list(tensor) | FLOAT32 | 列主序，每个矩阵 lda_inv × n | 指针数组长度 batchSize | 有限浮点 | 重叠或 lda_inv < max(1,n) 报错 |
| lda_inv | 输入 | 每个 Ainv[i] 的 leading dimension | scalar | int | - | - | lda_inv >= max(1,n) | 不满足报错 |
| info | 输出 | 每个 batch 的 info | tensor | INT32 | ND | [batchSize] | 0 / k | 空指针报错 |
| batchSize | 输入 | 矩阵个数 | scalar | int | - | - | batchSize >= 0 | batchSize < 0 报错 |

**规模与泛化（验收下限，允许实现支持更大范围并在文档中声明）**：

| 接口 | 维数 | 其他 |
|---|---|---|
| aclsolverSgetrf | m, n ∈ [0, 4096]，须覆盖方阵与矩形（m≠n） | lda 允许大于 m 的 padding |
| aclsolverSgetrs | n ∈ [0, 4096]，nrhs ∈ [0, 128] | trans 覆盖 N/T/C |
| aclsolverSgetri | n ∈ [0, 4096] | lda 允许 padding |
| aclsolverSgetriBatched | n ∈ [1, 256]，batchSize ∈ [1, 3000] | 须覆盖 lda/ldc padding |
| aclsolverSmatinvBatched | n ∈ [1, 32]，batchSize ∈ [1, 3000] | n=33 必须报错 |

不支持 broadcast，不要求图融合。dynamic shape：Host 按本次调用的 m/n/nrhs/batchSize 生成 tiling。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Ascend 950PR（Atlas 950）
- **CANN 版本**：CANN 9.0.0 及以上，须与 https://gitcode.com/cann/ops-solver 仓库 README 已验证配套版本一致
- **对标软件（性能金标采集）**：CUDA Toolkit 中的 cuSolver / cuBLAS，GPU 为 NVIDIA A100
- **其他依赖**：ops-solver 仓编译所需 ACL / AscendC / CATLASS 工具链；精度 golden 可使用 NumPy / SciPy LAPACK（float64）或 CPU 参考实现
- **版本记录**：自测报告须记录 CANN 版本、NPU 具体型号（950PR）、驱动版本、CUDA / cuSolver / cuBLAS 版本

功能和精度、性能均须提交 950PR 测试结果。

### 3.2 精度要求

1. 对标接口见 2.1 节。数值正确性以更高精度 CPU 参考为单标杆：FLOAT32 计算的 golden 使用 FLOAT64（NumPy/SciPy `float64` 的 `lu_factor` / `lu_solve` / `inv`，或等价 LAPACK `dgetrf`/`dgetrs`/`dgetri`）。
2. NPU 输出（`aclsolverSgetrf` 还原 P⁻¹LU 后与原 A 比较，或 `aclsolverSgetrs` 的 X、`aclsolverSgetri` / `aclsolverSgetriBatched` / `aclsolverSmatinvBatched` 的 A⁻¹ 及 A A⁻¹ 相对 I）须满足生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
3. FLOAT32 混合容差如下：

    | 数据类型 | FLOAT32 |
    |----------|---------|
    | rtol | 2^-10 (9.77e-4) |
    | atol | 2^-16 (1.53e-5) |
    | required_matched_ratio | 0.99 |
    | max_abs_error_limit | 1e-2 or 32 * ULP |

    逐元素判定：|actual - golden| ≤ atol + rtol × |golden|。当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

4. `ipiv` 允许与 CUDA 在「并列最大主元」时下标选择不同，但必须满足：按 NPU 的 `ipiv` 还原 P A = L U 后，与输入 A 的混合容差通过。`info` 对奇异矩阵须给出正确正值下标。
5. 覆盖普通值、小值、正负混合、对角占优可逆矩阵、故意构造的奇异矩阵；INF/NAN 按精度标准文档对应规则验收。
6. 确定性：合法可逆用例重复执行须 bit-wise 一致。

### 3.3 性能要求

1. 性能对标为 NVIDIA A100 上同 shape、同 dtype（FLOAT32）、同列主序、同等预热与计时口径下的 CUDA / LAPACK 接口：
   - aclsolverSgetrf：`cusolverDnSgetrf`（含一次 bufferSize，不计在 kernel 耗时内）
   - aclsolverSgetrs：`cusolverDnSgetrs`（getrf 时间不计入本接口）
   - aclsolverSgetri：A100 上 `cusolverDnSgetrf` + 求解 AX=I（getrf 时间不计入本接口；cuSolver 无独立 getri，以此作为金标）
   - aclsolverSgetriBatched：`cublasSgetriBatched`
   - aclsolverSmatinvBatched：`cublasSmatinvBatched`
2. 达标判据：对下表每一个 case，NPU kernel 耗时满足 T_NPU ≤ T_A100 / 0.8，即性能倍率 ≥ 0.8 倍 A100。须在 950PR 上达标。
3. 每个 case 至少预热 10 次、正式采样 30 次，报告中位数；每轮须 Device 同步后计时。不含首次编译、数据生成、H2D/D2H。workspace / 指针数组在正式采样期间复用。
4. 开发者须在 A100 上实测填入 T_A100，并在报告中给出 CUDA 版本与采集脚本。下表 T_A100 列为待测占位。

| 编号 | 接口 | 规格 | GPU A100 性能 | NPU 耗时 | 目标 |
|---|---|---|---|---|---|
| P-01 | aclsolverSgetrf | m=n=1024，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-02 | aclsolverSgetrf | m=n=4096，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-03 | aclsolverSgetrf | m=2048，n=1024，lda=m | 待测 | 待测 | ≥ 0.8×A100 |
| P-04 | aclsolverSgetrs | n=1024，nrhs=1，trans=N | 待测 | 待测 | ≥ 0.8×A100 |
| P-05 | aclsolverSgetrs | n=4096，nrhs=32，trans=N | 待测 | 待测 | ≥ 0.8×A100 |
| P-06 | aclsolverSgetrs | n=2048，nrhs=8，trans=T | 待测 | 待测 | ≥ 0.8×A100 |
| P-07 | aclsolverSgetri | n=1024，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-08 | aclsolverSgetri | n=4096，lda=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-09 | aclsolverSgetriBatched | n=32，batchSize=1024，lda=ldc=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-10 | aclsolverSgetriBatched | n=128，batchSize=128，lda=ldc=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-11 | aclsolverSmatinvBatched | n=16，batchSize=1024，lda=lda_inv=n | 待测 | 待测 | ≥ 0.8×A100 |
| P-12 | aclsolverSmatinvBatched | n=32，batchSize=256，lda=lda_inv=n | 待测 | 待测 | ≥ 0.8×A100 |

全部 12 条在 950PR 上均须达标，未达标不予验收。

### 3.4 内存要求

对标 cuSolver / cuBLAS 同等规格：

1. `aclsolverSgetrf` / `aclsolverSgetri` 的 Device workspace 不超过 `bufferSize` 返回值；禁止额外分配未计入 `Lwork` 的隐蔽大块 Device 内存。`bufferSize` 建议不超过 max(1, m × n) 量级（对齐 cuSolver getrf 大 workspace 实现口径），并在设计文档中给出公式。
2. 批量接口除输入/输出矩阵与指针数组、info、主元外，临时 Device 内存须在设计文档中量化；同规格下 NPU 额外 workspace 不得超过 A100 对应接口 workspace 的 2 倍。
3. 无内存泄漏：连续创建 handle、查询 workspace、执行、销毁，重复 100 次后 Device 已用内存回到基线。

### 3.5 自验要求

本任务使用 ops-solver 仓测试工程（https://gitcode.com/cann/ops-solver/tree/master/test ）及 AscendOpTest 工具（https://gitcode.com/HIT1920/AscendOpTest ）进行自验。请根据本任务给出的自测用例和测试指导完成自测，并输出自测报告。自测用例目录：./sgetrf_family_950_testCase/

NPU 接口与 CUDA 接口参数序列一致（见 2.3），无需额外映射。差异仅在：命名前缀 `aclsolver` vs `cusolverDn`/`cublas`，以及 `aclsolverOperation_t` vs `cublasOperation_t`（枚举值 0/1/2 对齐）。`aclsolverSgetri` 对标 LAPACK `sgetri`，无 cuSolver 同名接口。

功能自验至少覆盖：

| 类别 | 必测场景 |
|---|---|
| 基础功能 | 方阵 getrf；getrf+getrs 求解；getrf+getri 求逆；SgetriBatched 求逆；matinv n≤32 |
| 矩形与 padding | m≠n 的 getrf；lda/ldb/ldc > 最小合法值 |
| trans | Sgetrs 的 N/T/C |
| 选主元 | 有主元 / 无主元（ipiv 为空） |
| info | 成功 info=0；奇异矩阵 info=i；非法 lda 等 info=-i |
| 批量边界 | batchSize=1；n=1；n=32 matinv；n=33 matinv 报错 |
| 空问题 | m=0 或 n=0 或 nrhs=0 或 batchSize=0 成功返回 |
| 确定性 | 同一输入重复执行 bit-wise 一致 |
| 流 | 非默认 stream 下结果正确 |

测试用例入参生成规则：

| 参数名 | Tensor值域分布 | Attr 覆盖规则 |
|---|---|---|
| A（精度主路径，须可逆） | 50% 均匀分布 [-5,5] 后加对角占优（对角 += n）；50% 正态分布 μ∈[-5,5]、σ∈[0.1,2] 后同样对角占优。另用 10% 用例不加对角占优，用于奇异/近奇异与 info 行为 | m、n 覆盖 2 的幂与 2 的幂-1，落在声明范围内 |
| B | 与 A 相同的均匀/正态各 50% | nrhs 覆盖 1、8、32、128 及边界 0 |
| trans | - | N、T、C 全覆盖 |
| lda/ldb/ldc/lda_inv | - | 等于最小合法值，以及 +8 / +32 padding |
| batchSize | - | 1、8、128、1024、3000 及 0 |
| n（matinv） | - | 1、8、16、32 合法；33 非法 |
| ipiv 空指针 | - | 有主元 / 无主元各至少一组 |

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
src/sgetrf/
src/sgetrs/
src/sgetri/
src/sgetri_batched/
src/smatinv_batched/
test/sgetrf/
test/sgetrs/
test/sgetri/
test/sgetri_batched/
test/smatinv_batched/
docs/zh/sgetrf.md
docs/zh/sgetrs.md
docs/zh/sgetri.md
docs/zh/sgetri_batched.md
docs/zh/smatinv_batched.md
docs/api_list.md
```

须同步更新 README 与接口列表。五个接口作为同一 PR 或一组关联 PR 合入，不得只合入其中部分接口。

## 6. 参考资料

1. NVIDIA cuSolver 文档（getrf / getrs）：https://docs.nvidia.com/cuda/cusolver/index.html ；
2. NVIDIA cuBLAS 文档（getriBatched / matinvBatched）：https://docs.nvidia.com/cuda/cublas/index.html ；
3. LAPACK sgetri：https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html ；
4. ops-solver 仓库：https://gitcode.com/cann/ops-solver ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ；
6. AscendOpTest：https://gitcode.com/HIT1920/AscendOpTest ；
7. Ascend C算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
8. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ；
9. Ascend C在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713 ；
10. CATLASS：https://gitcode.com/cann/catlass ；
11. ops-solver 现有单精度原型文档：https://gitcode.com/cann/ops-solver/blob/master/docs/zh/sgetrf.md 。

## 7. 特别注意事项

1. 本任务五个接口必须全部交付：aclsolverSgetrf、aclsolverSgetrs、aclsolverSgetri、aclsolverSgetriBatched、aclsolverSmatinvBatched。缺一不可。
2. 文档中的 cuSolver / cuBLAS 仅作为接口、功能、约束和性能标杆，NPU 实现统一使用 aclsolver + AscendC/CATLASS，禁止依赖 CUDA。
3. 必须列主序、Device 指针、真实写入 info、支持合法 lda padding；禁止继续使用「lda 必须等于 n、info 不写回、Host 矩阵指针」作为最终语义。
4. `aclsolverSmatinvBatched` 在 n>32 时必须报错，不得转调 `aclsolverSgetriBatched`。
5. `aclsolverSgetri` / `aclsolverSgetriBatched` 的输入是已分解的 LU 因子 + 主元，不是未分解的原始 A；端到端批量求逆须先 getrf 再 getriBatched。`aclsolverSmatinvBatched` 直接对原始矩阵求逆。
6. 性能门禁为 0.8 倍 A100，须在 950PR 上达标。
7. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请。
8. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 。

## 8. 环境获取（无需修改，使用模板原始内容）

 1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。
 - **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**
 	 
 	![环境截图](./pics/zaixiankaifa1.png)  
 	![环境截图](./pics/apply.png)  
 	 
2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
 	 
 	![环境截图](./pics/yunkaifa.png)
 	 
3. 如需额外环境资源，请联系昇腾CANN小助手。
