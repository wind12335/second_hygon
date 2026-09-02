# Phase B NVIDIA 侧移植说明（RTX 4090 / sm_89 / 4 GPUs / PCIe）

日期：2026-09-02。本目录是 `ag_gemm_phaseb.cpp`（K500SM_AI / gfx928 / 4 GPUs / PCIe）
的 NVIDIA 移植：**同一 10 路径、同一 CLI、同一 CSV schema**，父目录的
`analyze_phaseb.py` 对两侧产出零修改可用。

| 文件 | 说明 |
|---|---|
| `ag_gemm_phaseb_nv.cpp` | CUDA/NCCL/NVSHMEM/cuBLAS 版基准（10 路径） |
| `Makefile` | nvcc 单命令构建（-arch=sm_89 -rdc=true） |
| `run_phaseb_nv.sh {smoke\|formal}` | runner，目录约定/防覆盖锁/snapshot 与海光侧完全一致 |

## 1. 编译环境要求

- **CUDA ≥ 11.8**（sm_89 需要 11.8+；建议 12.x）
- **NCCL ≥ 2.19**（对外部库无特殊版本要求，纯标准 collective API）
- **NVSHMEM ≥ 2.9**：用到 `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, ...)`、
  `nvshmemx_putmem_signal_on_stream`、`nvshmemx_signal_wait_until_on_stream`、
  `nvshmemx_quiet_on_stream`、`nvshmemx_barrier_all_on_stream`、device 侧
  `nvshmem_uint64_signal`（RDC 必开）
- **OpenMPI**（runner 用 `--allow-run-as-root -mca coll ^hcoll`；二进制用
  `OMPI_COMM_WORLD_LOCAL_RANK` 选卡）

```
make                       # 默认路径
make NVSHMEM_HOME=/opt/nvshmem NCCL_HOME=/usr/local/nccl   # 非默认安装
bash run_phaseb_nv.sh smoke
```

## 2. API 映射表（逐条，与 HIP 版对照）

| HIP 侧（K500SM_AI / gfx928） | NVIDIA 侧（RTX 4090 / sm_89） | 置信度 |
|---|---|---|
| `hip/hip_runtime.h`、`hipSetDevice` 等 | `cuda_runtime.h`、`cudaSetDevice` 等 | 确定 |
| `hipInit`+`hipDeviceGet`+`hipDevicePrimaryCtxRetain`+`hipCtxSetCurrent` | **删除**。这是 DUSHMEM 需要显式主上下文的怪癖；NVSHMEM 自己建上下文，`cudaSetDevice` 即可 | 确定（行为差异，见 §4.1） |
| `rccl/rccl.h`、`ncclGetUniqueId/ncclCommInitRank/ncclAllGather/ncclCommDestroy` | `nccl.h`，函数名与签名完全相同 | 确定（RCCL 本就是 NCCL 系） |
| `rocblas_create_handle/set_stream/sgemm/destroy_handle` | `cublasCreate/cublasSetStream/cublasSgemm/cublasDestroy`；两者同为列主序，转置参数/lda 直接沿用 | 确定 |
| `dushmemx_init_attr(DUSHMEMX_INIT_WITH_MPI_COMM, &attr)`（attr.mpi_comm）| `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)`，attr.options->mpi_comm（结构分层不同） | 确定 |
| `dushmem_my_pe/n_npes` | `nvshmem_my_pe/n_npes` | 确定 |
| `dushmem_malloc/free` 对称堆（x_local/full_a/gathered/ready/credit 全部对称分配，NCCL 路径同堆公平性原则保留） | `nvshmem_malloc/nvshmem_free` | 确定 |
| `dushmemx_putmem_signal_on_stream(dst,src,size,&sig,val,DUSHMEM_SIGNAL_SET,pe,stream)` | `nvshmemx_putmem_signal_on_stream(dst,src,size,&sig,val,NVSHMEM_SIGNAL_SET,pe,stream)`，参数序相同 | 高 |
| `dushmemx_signal_wait_until_on_stream(&sig,DUSHMEM_CMP_GE,val,stream)` | `nvshmemx_signal_wait_until_on_stream(&sig,NVSHMEM_CMP_GE,val,stream)` | 高 |
| `dushmemx_signal_op_on_stream(&credit,epoch,SET,pe,stream)`（credit 回传）| **改为 1 线程 device kernel** `signal_op_kernel` 在同一 stream 上调 `nvshmem_uint64_signal(dest,val,pe)`。原因：NVSHMEM host 侧 signal 不是 stream-ordered，直接调用会在 GEMM 完成前就发 credit，破坏协议；device kernel 保持流序 | 设计决策（见 §4.2） |
| `dushmemx_fcollectmem_on_stream(TEAM_WORLD,dst,src,bytes,stream)`（返回 int） | `nvshmemx_fcollectmem_on_stream(NVSHMEM_TEAM_WORLD,dst,src,bytes,stream)`（返回 void） | **中——头号编译风险**（见 §4.3） |
| `dushmemx_quiet_on_stream` | `nvshmemx_quiet_on_stream` | 高 |
| `dushmemx_barrier_all_on_stream` | `nvshmemx_barrier_all_on_stream` | 高 |
| `dushmem_finalize` | `nvshmem_finalize` | 确定 |
| `hipMemcpyAsync(...,hipMemcpyDeviceToDevice,s)` / `hipMemsetAsync` / `hipMalloc/Free` / `hipEvent*` / `hipStreamCreateWithFlags(...,hipStreamNonBlocking)` | cuda 对应，语义逐一相同 | 确定 |
| `--dush-quiet` CLI 旗标名 | **保留不改名**（runner 兼容；语义=nvshmemx_quiet_on_stream） | - |
| 编译：`--offload-arch=gfx928 -mcode-object-version=4 -fgpu-rdc -DHIP_ENABLE_WARP_SYNC_BUILTINS`，两阶段 hipcc 链接 | `-arch=sm_89 -rdc=true`，nvcc 单命令（无需两阶段）；其余旗标不需要 | 确定 |

**CSV/口径不变项**：10 路径 CLI 值（comm/gemm/r0/rs/r1/fc/dc/d0/ds/d1）与
PATH 枚举名、全部 CSV 列名与行格式（含 `t_issue_us` 占位 0）、逐迭代校验口径
（abs 1e-2 / rel 1e-4，参考=同进程 NCCL 全量 AG+GEMM）、max-rank 聚合、
manifest 列。变化项：`gfx_arch` 列值 gfx928→sm_89、`platform_id`
K500SM_AI→RTX4090、family 值 RCCL/DUSHMEM→**NCCL/NVSHMEM**（数据保真优先）。

> ⚠️ analyze_phaseb.py 的 boundary/xcand 表里 family 标签是按路径名硬编码的
> "DUSHMEM/RCCL"（仅展示用途，不影响判定逻辑）；NVIDIA 数据过分析器时这两处
> 标签会仍显示 DUSHMEM/RCCL。合并双基座数据前把这两个字符串参数化即可。

## 3. 冒烟前必须检查

1. `nvidia-smi topo -m`：记录 4 卡互联拓扑（见风险 #1）。runner 已自动写进
   platform_facts.txt。
2. P2P 可用性：若 topo 显示 SYS/PHB（无 P2P），先跑一个简单 cudaDeviceCanAccessPeer
   / p2pBandwidthLatencyTest 并记录——这直接决定"基座能力向量"怎么解读。
3. `make` 通过后再跑 `bash run_phaseb_nv.sh smoke`（1 rep × 40 iters，~12 case）；
   smoke 全 PASS 才跑 formal（590 case）。
4. 核对 NVSHMEM 库布局：`ls $NVSHMEM_HOME/lib`；只有 `libnvshmem.so` 没有
   device/host 分库时改 Makefile LIBS（风险 #4）。

## 4. 已知风险与"不确定但已落地"的决策（人工复核清单）

### 4.1 HIP 主上下文序列的删除
HIP 版需要 `hipInit`+`hipDevicePrimaryCtxRetain`+`hipCtxSetCurrent` 才能初始化
DUSHMEM（deprecated 警告但必需）。CUDA+NVSHMEM 无此要求，`cudaSetDevice` 即可。
若 NVSHMEM 初始化报 "no CUDA context"，加 `cudaFree(0)` 强制建上下文后重试。

### 4.2 credit 回传改 device kernel（协议等价性）
HIP 版 `dushmemx_signal_op_on_stream(..., compute_stream)` 在 GEMM 之后流序执行。
NVSHMEM 无对应 stream-ordered host signal，故用 `signal_op_kernel<<<1,1,0,compute_stream>>>`
包 device 侧 `nvshmem_uint64_signal`，同 stream 保序。
备选（若不想用 device API / RDC）：`nvshmemx_putmem_signal_on_stream` 携带
size=0 只发信号——语义上等价但"size=0 仍保证投递信号"在文档里没有明说，故未采用。
**复核点**：`nvshmem_uint64_signal` 的 device API 名与签名（nvshmem.h 的
device API 节）。

### 4.3 头号编译风险：`nvshmemx_fcollectmem_on_stream`
此符号在 NVSHMEM 文档的 Stream Operations 清单里我不确定是否存在（put/get/
put_signal/wait/fence/quiet/barrier_all 的 on_stream 版本是确定的）。若编译报
undeclared：
- 备选 A：用 `nvshmem_float_fcollect(NVSHMEM_TEAM_WORLD, dst, src, nelems)`（阻塞、
  host 侧）+ 事件包裹，会轻微改变 fc/d0 的计时口径（记录进 README 再对比）；
- 备选 B：fcollect 换成"每 rank 对全队 broadcast 自己分段"的 put 组合（口径最
  接近 fcollect 语义）。
只有 fc/d0 两条路径用到它，先跑 comm/gemm/r0/rs/r1/dc/ds/d1 不受影响。

### 4.4 NVSHMEM 库名
Makefile 默认 `-lnvshmem_device -lnvshmem_host`（NVSHMEM Getting Started 的
CUDA 链接惯例）。部分发行版只提供单一 `-lnvshmem`；按需删改。

### 4.5 4090 消费级平台：P2P 大概率不可用（解读风险，非代码风险）
消费级主板 4090 之间常无 P2P（topo 显示走 root complex 的 SYS/PHB）。若 P2P 被禁：
- NVSHMEM put/signal 走 host proxy（GPU→host→GPU），单向带宽形态与
  K500SM_AI / gfx928 的 DUSHMEM 直采路径**不可直接比数值**；
- NCCL 会走 SHM/NET 传输，Ring/Simple 通道行为不同。
这正是"基座能力向量"要记录的内容而非要消除的差异——但论文对比表必须按
platform_facts 里的拓扑分层解读。冒烟前先确认 `nvidia-smi topo -m`。

### 4.6 NVSHMEM 纯 PCIe 单机 transport
无 IB 时 NVSHMEM 默认 transport 走 P2P（若有）否则 host。可调环境变量：
`NVSHMEM_DISABLE_P2P`、`NVSHMEM_REMOTE_TRANSPORT`。建议 formal 前固定并在
run_metadata 里记录当前值（本项目暂不改默认，保持出厂行为可比）。

### 4.7 对称堆大小
最大配置（m_local=K=2048, q=16）：x_local 16MiB + full_a 64MiB + gathered 共
64MiB + 信号量 ~KB ≈ 150MiB/PE。NVSHMEM 默认 1GiB 足够，但 runner 显式 export
`NVSHMEM_SYMMETRIC_SIZE=1G` 以钉死可复现性；若改形状矩阵记得复核。

### 4.8 （移植中发现，海光侧同样存在）DS 与 D1 结构性相同
HIP 源码里 `kDS`/`kD1` 分支唯一的 `if (serial...)` 块是**空的（只有注释）**，
两条路径发出的流操作序列完全一致——即当前实现里 DS 并没有"分片串行"的强制
门控（RS 路径有 `cuda/hipStreamWaitEvent(comm_stream, gemm_end[i])`，DS 没有
对应物）。本移植保持一致（faithful）。若这是有意为之（d1 与 ds 实为同一策略的
重复测量），建议在文档里写明；若是遗漏，海光侧先修，NVIDIA 侧照抄修复。
formal N512 数据 d1≈ds（8938 vs 8953us）与"结构性相同"的解读一致，而 smoke
N2048/q8 的 d1/ds 差 5.8% 存疑，需要回看。

## 5. 与海光侧结果合并注意

- `phaseb_case_summary.csv` 的 family 列两侧分别输出 NCCL/NVSHMEM 与
  RCCL/DUSHMEM（真实标注）；分析器派生表的硬编码标签见 §2 末尾的 ⚠️。
- 结果根目录不同（`results/rtx4090_4gpu/` vs `results/k500sm_ai_gfx928_4gpu/`），
  合并分析时以 platform_id/gfx_arch 列区分。
- 引用海光平台时严格写 `K500SM_AI / gfx928 / 4 GPUs / PCIe`，NVIDIA 侧写
  `RTX 4090 / sm_89 / 4 GPUs / PCIe`。
