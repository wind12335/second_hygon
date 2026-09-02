# 四卡 NCCL/RCCL 结果综合分析与第二篇论文定题建议

> 更新日期：2026-08-30  
> 覆盖对象：四卡 RTX 4090/NCCL 与四卡海光 DCU/RCCL 的 collective 微基准；现有海光 AG-GEMM/DUSHMEM 探索。  
> 目的：基于原始 CSV 和已有日志解读，判断哪些现象可信、哪些不能得出结论，并把第二篇论文收敛为一条可证伪的研究路线。  
> 结论状态：**可以启动方向验证，但尚不能宣布论文创新已经被实验确认。**

---

## 1. 执行结论

四卡 NCCL 与 RCCL 的基础实验已经完成且大部分可复核。它们证明了三件重要的事：

1. 当前两套平台上的默认 collective 策略已经非常接近孤立通信带宽最优值；仅仅搜索 `algorithm/protocol/channel`，很难形成有竞争力的论文核心，也会直接落入 AutoCCL 的问题边界。
2. `channel`、协议能力和 LL 的行为确实强烈依赖平台、collective、消息规模和实际 transport。NCCL/RTX 4090 在较大消息下通常在 `ch4` 左右已接近饱和；RCCL/当前 PCIe DCU 平台通常仍能从 `ch4` 受益到 `ch8`。这说明固定跨平台参数没有根据，但它本身仍只是一个调参现象。
3. 海光上的初步 DUSHMEM 分块 AG-GEMM 结果表明，细分 chunk **不会自动产生重叠收益**。当前 `chunks=2/4/8` 的巨大退化主要伴随分块 GEMM 自身约 `7.8--7.9x` 的退化，不能作为 DUSHMEM 或重叠机制优劣的结论。

因此，建议的第二篇论文主线不是“将 AutoCCL 移植到 RCCL”，也不是“找一个更好的 Ring/Simple/channel”，而是下面这个**待验证命题**：

> 在有数据依赖的 Collective-GEMM 中，孤立 `busbw` 和完整 collective 的完成时间不足以预测端到端最优重叠策略。可被正确消费的分片释放时刻、通信-计算资源竞争和平台能力边界共同决定最优策略；这些因素在 NVIDIA 与海光的通信基座上会表现出不同的决策边界。

推荐先将选题表述为：

```text
中文：面向异构 GPU 通信基座的完成语义与依赖释放感知
      Collective-GEMM 自适应重叠策略

英文（工作标题）：Beyond Bandwidth: Completion-Semantics-Aware
Adaptive Collective-GEMM Overlap across GPU Communication Substrates
```

这里的“跨基座”指**同一抽象方法分别在 NVIDIA 和海光同构 GPU 组上验证**，不是将两类 GPU 混入同一个 collective group。统一的是能力描述、测量方法和选择逻辑；通信实现可以是各自平台的 backend。

这个方向只有在后续端到端实验观察到“带宽最优/`T_done` 最优与 AG-GEMM 端到端最优不一致”，并且能够用合法的 release timeline 和资源竞争解释时，才值得正式定题。若没有这种现象，应及时放弃该主线，而不是用更多微基准包装成创新。

---

## 2. 本报告的证据范围与证据等级

### 2.1 已逐项阅读的本地材料

| 材料 | 用途 | 证据等级 |
|---|---|---|
| `统一实验配置-NCCL-RCCL-4090-K100AI (1).md` | 统一口径、变量和数据保存要求 | 实验设计证据 |
| `NCCL-4卡RTX4090实验执行方案.md` | RTX 4090/NCCL 执行过程和 manifest 解释 | 实验设计与日志说明 |
| `NCCL四卡RTX4090实验结果与RCCL对比解读 (1).md` | 已有共同 case 对齐、能力边界和统计解读 | 二次分析，已与 CSV 交叉核对 |
| `RCCL实验教学与结果解读 (1).md` | DCU 平台、RCCL 结果和 DUSHMEM overlap 探索 | 二次分析，关键数字已核对 |
| `nccl_rtx4090_4gpu_formal_summary.csv` | RTX 4090 的原始导出表 | **直接测量数据** |
| `rccl_k500sm_ai_4gpu_formal_summary.csv` | 海光 DCU 的原始导出表 | **直接测量数据** |
| `/root/seconde-paper/nsdi25-xu-guanbin.pdf` | AutoCCL 原文 | 已阅读的相关工作 |
| `/root/seconde-paper/第二篇文献调研.md`、`已核验研究方向分析.md`、`当前研究方向与首轮Profiling分析.md` | 已有选题边界、平台 API 风险和 profiling 设计 | 研究规划材料 |

本报告新增的派生数据由原始 CSV 生成，绝不覆盖原始文件：

| 新增文件 | 内容 | 规则 |
|---|---|---|
| `生成四卡结果审计表.py` | 可重复的聚合脚本 | Python 标准库；输入 CSV 不修改 |
| `rccl_4gpu_caseid_audit.csv` | 每个 RCCL `case_id` 的重复数、均值、范围和处置建议 | 对同 case 的重复测量保留并显式报告范围 |
| `四卡默认策略跨平台对齐汇总.csv` | 默认策略、共同尺寸的均值和比值 | 仅 `status=0`、正确性通过的记录 |
| `四卡RingSimple通道扩展汇总.csv` | Ring/Simple 的 `ch1/ch2/ch4/ch8` 曲线 | NCCL 三次测量取均值；RCCL 重复 case 取均值并回指审计表 |

重新生成派生表的命令为：

```bash
python3 /root/实验结果/生成四卡结果审计表.py
```

### 2.2 三类陈述必须分开

后续论文、汇报和代码注释应严格区分：

| 类型 | 本报告中的表达方式 | 例子 |
|---|---|---|
| 实测事实 | “数据表明”“在当前平台上测得” | 64 MiB Ring/Simple 的 channel 曲线 |
| 合理推断 | “提示”“支持优先验证” | channel 的最优区间与 transport 有关 |
| 待验证假设 | “假设”“只有满足条件才成立” | 孤立带宽排序会在 AG-GEMM 中发生反转 |

尤其不能把“两个不同硬件平台的 NCCL/RCCL 数值不同”表述为“某个库的软件实现更好”，也不能把“原语 API 名称相似”表述为“NVSHMEM 与 DUSHMEM 具有相同完成语义”。

---

## 3. 数据审计与可比性边界

### 3.1 NVIDIA CSV 不是完整 case manifest

`nccl_rtx4090_4gpu_formal_summary.csv` 有 547 条数据行，均为：

```text
ranks=4
datatype=float
status=0
correctness=PASS
wrong_count=0
```

这表示 CSV 中导出的 547 个**通过 case**均正确，不表示所有请求组合均可用。已有 NCCL manifest 记录总共 769 个 case，其中：

| 状态 | 数量 | 正确解释 |
|---|---:|---|
| 通过并进入 CSV | 547 | 可用于性能比较 |
| `invalid usage` | 222 | NCCL 不支持的组合，不是数据校验损坏 |
| 通信 correctness error | 0 | 本批通过项没有观察到 |

222 个不可用项主要来自 AllGather/ReduceScatter 的部分 Tree 协议组合。它们必须进入 capability mask，不能被当作“零性能样本”平均，也不能被写成算法 correctness 失败。

### 3.2 RCCL CSV 的重复记录不能静默丢弃

`rccl_k500sm_ai_4gpu_formal_summary.csv` 有 959 条数据行。按 `case_id` 审计后得到：

| RCCL 四卡记录 | 数量 |
|---|---:|
| 原始数据行 | 959 |
| 唯一 `case_id` | 750 |
| 额外重复行 | 209 |
| 发生重复的阶段 | 全部在 `channels` |
| `status=0` | 958 |
| `status=UNKNOWN` 且 `wrong_count=0` | 1 |

此前已有报告以“保留最终有效记录”的方式获得 750 个唯一 case；但 CSV 没有足以证明先后顺序的时间戳，且同 `case_id` 的性能并非字节级相同。因此本报告的派生统计采用更可审计的规则：

1. 原始 CSV 永久保留，不覆盖也不删除重复行；
2. 相同 `case_id` 的性能均值仅用于趋势图/汇总；
3. `rccl_4gpu_caseid_audit.csv` 始终保留 `original_row_count`、最小值、最大值和相对范围；
4. 最终论文数值应回到原始日志，重跑高波动 case，而不是用平均值掩盖波动；
5. `status=UNKNOWN` 的记录不进入默认策略和 Ring/Simple 主汇总，必须先回查日志。

唯一 `UNKNOWN` 记录是：

```text
channels_r2_allreduce_8M_ring_ll_ch2
```

它的 `wrong_count=0` 不等于执行状态已经被严格确认。由于它不是本报告 Ring/Simple 主曲线的一部分，不影响下文的主结论。

### 3.3 必须重跑的 RCCL 高波动 case

审计表以同 case 的 `busbw` 相对范围大于等于 10% 标记 `RERUN_REQUIRED_HIGH_VARIATION`。当前共有 6 个：

| case_id | 平均 busbw (GB/s) | 相对范围 | 处置 |
|---|---:|---:|---|
| `channels_r1_allgather_1M_tree_ll_ch8` | 3.371 | 16.19% | 重跑 |
| `channels_r1_allgather_8M_ring_simple_ch1` | 4.480 | 11.46% | 重跑 |
| `channels_r1_allreduce_1M_ring_simple_ch2` | 5.748 | 19.53% | 重跑 |
| `channels_r1_allreduce_1M_tree_simple_ch2` | 3.922 | 10.38% | 重跑 |
| `channels_r2_allreduce_1M_ring_ll_ch8` | 4.798 | 13.75% | 重跑 |
| `channels_r2_allreduce_1M_ring_simple_ch8` | 6.981 | 22.16% | 重跑 |

最后一项的原始值为 `7.754884` 和 `6.207853 GB/s`。这类波动说明 1 MiB 小消息的运行状态、频率、CPU 绑核、PCIe 负载或运行顺序仍可能影响结果。它不是论文现象本身，不能用于宣称算法差异。

### 3.4 平台标签存在一处必须澄清的不一致

用户当前口述为“4 卡 K100_ai”，但全部 RCCL 原始路径、CSV 文件名和实验教学文档写的是：

```text
K500SM_AI x4
gfx928
PCIe
```

本报告因而使用 `K500SM_AI/gfx928` 作为数据标签，**不擅自将其改名为 K100_ai**。在任何论文图表、caption、artifact 或摘要中引用该数据前，必须在实际机器上执行并保存设备事实，例如：

```bash
rocminfo | sed -n '1,220p'
rocm-smi --showproductname --showuniqueid --showtopo
```

然后统一更正实验标签。GPU/DCU 型号不是排版细节，会直接影响硬件、互联和相关工作对比的可信度。

### 3.5 目前是“跨平台基线对照”，不是库实现对照

| 维度 | NVIDIA 平台 | 海光平台 |
|---|---|---|
| 加速器 | RTX 4090 x4 | K500SM_AI/gfx928 x4（待再次核验型号） |
| 通信库 | NCCL 2.30.7 | RCCL 2.22.3-HEAD |
| 软件栈 | CUDA 12.6.85，Open MPI 4.1.2 | ROCm/HIP 6.3，Open MPI 5.0.3 |
| 已观测路径 | SHM/direct，P2P read/write: CNS | PCIe P2P 可用 |
| 拓扑信息 | GPU0/1 位于 NUMA0；GPU2/3 位于 NUMA1 | 当前材料表明 PCIe，完整拓扑图应随日志存档 |

GPU 架构、PCIe/NUMA 路径、驱动、MPI、library version 和实际 transport 都不同。因此 `NCCL/RCCL` 的绝对带宽比值只能描述“当前两套机器上的基线差异”，不能归因为 NCCL/RCCL 源码质量。

对于第二篇论文，还有一个更关键的边界：若论文最终声称在 NVIDIA A100 与海光 K100_ai/K500SM_AI 上验证，那么当前 RTX 4090 表只能作为开发和方向筛查数据。A100 必须至少复现后文 Phase A 的核心 AG-GEMM 矩阵，不能用 4090 代替 A100 完成最终论文验证。

---

## 4. 当前实验的可靠发现

### 4.1 默认策略在孤立通信中已接近 oracle

下表来自新增的 `四卡默认策略跨平台对齐汇总.csv`。1 MiB、8 MiB、64 MiB、256 MiB 使用各自三次 DEFAULT 重复的均值；1 GiB 是当前单次 pilot。单位为 `busbw GB/s`。

| Collective | Size | RTX 4090/NCCL | K500SM_AI/RCCL | NCCL/RCCL |
|---|---:|---:|---:|---:|
| AllGather | 1 MiB | 8.357 | 6.441 | 1.297x |
| AllGather | 8 MiB | 13.260 | 11.258 | 1.178x |
| AllGather | 64 MiB | 16.743 | 12.240 | 1.368x |
| AllGather | 256 MiB | 16.810 | 12.538 | 1.341x |
| AllReduce | 1 MiB | 9.350 | 7.942 | 1.177x |
| AllReduce | 8 MiB | 14.600 | 12.005 | 1.216x |
| AllReduce | 64 MiB | 17.563 | 12.832 | 1.369x |
| AllReduce | 256 MiB | 17.633 | 13.028 | 1.353x |
| ReduceScatter | 1 MiB | 8.670 | 6.032 | 1.437x |
| ReduceScatter | 8 MiB | 11.767 | 10.818 | 1.088x |
| ReduceScatter | 64 MiB | 15.873 | 11.691 | 1.358x |
| ReduceScatter | 256 MiB | 15.910 | 11.970 | 1.329x |

1 GiB pilot 的 bus bandwidth 分别为：

| Collective | RTX 4090/NCCL | K500SM_AI/RCCL | 比值 |
|---|---:|---:|---:|
| AllGather | 16.800 | 12.805 | 1.312x |
| AllReduce | 17.800 | 13.146 | 1.354x |
| ReduceScatter | 15.930 | 12.222 | 1.303x |

这些表明当前 RTX 4090 主机的绝对有效带宽更高，但正确结论到此为止。它们不能证明 NCCL 在其他硬件上优于 RCCL，也不能证明海光硬件本身的上限更低。

更重要的是，在代表性策略矩阵中，DEFAULT 到每个 collective/size 的已测孤立通信 best case 的余量很小：

| 平台 | 观察到的最大 DEFAULT-to-oracle 改善 | 解释 |
|---|---:|---|
| NCCL/RTX 4090 | AllGather 256 MiB Ring/Simple 约 +0.26%；AllReduce 8 MiB 约 +2.17%；ReduceScatter 8 MiB 约 +3.31% | 默认策略基本已选到强路径 |
| RCCL/K500SM_AI | AllGather 最大约 +0.13%；AllReduce Ring/Simple 最大约 +1.44%；ReduceScatter 最大约 +0.21% | 同样没有显著的固定策略缺口 |

**研究含义：**“给所有 case 强制 Ring/Simple”无法形成主方法。即使确有个位数百分点，也既缺乏普适性，又与 AutoCCL 的低层参数在线调优重叠。

### 4.2 LL 的孤立吞吐低，但不能据此否定端到端价值

在当前四卡、64 MiB/256 MiB 的孤立通信中，Ring/LL 明显慢于 DEFAULT：

| 平台 | Collective | 64 MiB DEFAULT | 64 MiB Ring/LL | 256 MiB DEFAULT | 256 MiB Ring/LL |
|---|---|---:|---:|---:|---:|
| NCCL/RTX 4090 | AllGather | 16.743 | 5.120 | 16.810 | 5.227 |
| NCCL/RTX 4090 | AllReduce | 17.563 | 6.537 | 17.633 | 6.430 |
| RCCL/K500SM_AI | AllGather | 12.240 | 2.016 | 12.538 | 2.020 |
| RCCL/K500SM_AI | AllReduce | 12.832 | 3.479 | 13.028 | 3.483 |

当前可以严谨地说：

> 在本轮中大消息的独立 collective benchmark 中，LL 的吞吐低于 DEFAULT/Simple，且退化幅度具有 backend/平台依赖性。

当前不能说：

> 因此 LL 在 AG-GEMM 或训练 step 中一定更慢。

端到端重叠可能受启动延迟、分片最早可消费时间、同步开销、SM/CU 占用和 GEMM 效率共同影响。LL 的低带宽正是需要用 release timeline 去检验而不是凭直觉淘汰的候选；但只有合法、正确的实现才能进入比较。

### 4.3 channel 是稳定且跨平台差异明显的条件变量

64 MiB、Ring/Simple、`ch1/ch2/ch4/ch8` 的均值如下。RCCL 的重复 case 已按本报告规则平均，具体范围见审计 CSV。

| Collective | NCCL ch1 | ch2 | ch4 | ch8 | RCCL ch1 | ch2 | ch4 | ch8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AllGather | 13.903 | 16.753 | 17.510 | 17.500 | 4.956 | 9.874 | 12.187 | 12.712 |
| AllReduce | 14.817 | 17.457 | 19.190 | 19.413 | 5.407 | 10.533 | 12.779 | 13.086 |
| ReduceScatter | 14.223 | 15.880 | 17.253 | 17.583 | 5.110 | 9.643 | 11.599 | 12.428 |

可复核的趋势为：

- NCCL/RTX 4090 的 64 MiB Ring/Simple 从 `ch1` 提升到 `ch4`，AllGather/AllReduce/ReduceScatter 分别约为 `1.26x/1.30x/1.21x`；`ch4` 后大多趋于饱和。
- RCCL/当前 PCIe DCU 从 `ch1` 提升到 `ch8`，三类 collective 分别约为 `2.57x/2.42x/2.43x`；`ch4 -> ch8` 仍有可见收益。
- 1 MiB 小消息中并行度并非越高越好。NCCL 中 AllGather `ch8/ch1` 约 `0.96x`、AllReduce `ch8/ch1` 约 `0.86x`；额外 channel 的调度和同步开销可以抵消并行度。

正确的解释是：**最优 channel 是 `backend + collective + message size + transport` 的函数。**这里的“backend”不是唯一因果变量，NCCL 侧的 SHM/direct 与 RCCL 侧的 PCIe P2P 也在变化。

这组现象可以成为后续 selector 的输入特征和 candidate 维度，但不能单独成为论文创新。

### 4.4 rank mapping 目前不是强信号

NCCL 的 64 MiB mapping 测量为：

| Collective | `0123` | `0213` | `0321` |
|---|---:|---:|---:|
| AllGather | 16.78 | 16.71 | 16.71 |
| AllReduce | 17.58 | 17.53 | 17.46 |

差异约小于 1%。已有 RCCL 解读中 mapping 的主要差异同样多在 5% 内。有限实验预算不应优先投入“rank mapping 自动搜索”。

只有在出现明显 PCIe root complex/NUMA 差异、NVLink/HYLink 路径差异、跨节点网络，或扩大至更多 GPU 后，mapping 才有重新成为研究变量的理由。届时应把实际 ring route、NUMA、链路和 repetition 一起记录，而不是只比较三个均值。

### 4.5 capability mask 是必要条件，不是独立贡献

当前已观测的能力边界包括：

| backend | 已测边界 |
|---|---|
| NCCL/RTX 4090 | AllGather + Tree/Simple、AllGather + Tree/LL、ReduceScatter + Tree/Simple、ReduceScatter + Tree/LL 不可用；AllReduce 的部分 Tree 组合可用；LL128 能力依 collective/algorithm 而变 |
| RCCL/gfx928 | 已有实验中 LL128 probe 不可用，未被放入可比主矩阵 |

因此所有后续枚举都应遵守：

```text
capability discovery -> candidate filtering -> performance measurement -> selection
```

而不是遍历所有 `algorithm/protocol/channel` 后将报错混入训练样本。这个 capability layer 是方法可移植性和工程正确性的必要组成，但本身不足以构成论文主贡献。

### 4.6 现有 DUSHMEM overlap 结果是“方法警报”，不是负面结论

海光文档中的探索性实验比较了：

```text
RCCL 完整 AllGather + 整块 rocBLAS GEMM
vs.
DUSHMEM 分块通信 + 分块 rocBLAS GEMM
```

记录结果：

| 方案 | 端到端时间 |
|---|---:|
| RCCL baseline，chunks=1 | 约 184.0 ms |
| DUSHMEM，chunks=1 | 约 217.2 ms |
| DUSHMEM，chunks=2 | 约 1292.1 ms |
| DUSHMEM，chunks=4 | 约 1293.6 ms |
| DUSHMEM，chunks=8 | 约 1291.2 ms |
| 整块 GEMM | 约 159 ms |
| 分块 GEMM | 约 1247--1248 ms |

可得出的唯一可靠结论是：**本实现中的 chunked GEMM 形状/调用方式严重改变了计算效率，因而不能用端到端时间评价 DUSHMEM 重叠本身。**

下轮设计必须为每个 `q`（chunk 数）分别测量：

```text
T_comm_only(q)
T_gemm_only(q)
T_serial(q) = T_comm_only(q) + T_gemm_only(q) 的实际串行版本
T_overlap(q)
GEMM_TFLOPS(q) / GEMM_TFLOPS(whole)
```

任何“重叠带来多少收益”都必须相对于**同一 q、同一 GEMM partition、同一正确性路径**的串行基线，不能拿 `whole GEMM` 与 `chunked GEMM` 直接比。

---

## 5. 这些结果没有回答的问题

目前所有可靠主表都是 isolated collective `busbw`。它们还没有测量下面五个变量：

| 未测量变量 | 为什么它决定论文是否成立 |
|---|---|
| `T_done` | 完整 collective 何时完成；用于证明“只看完整完成也不够” |
| `R_i` | 第 `i` 个 GEMM consumer tile **合法且正确**开始读取所需输入的最早时刻 |
| `T_overlap` | 对应策略下真实 AG-GEMM 端到端时间 |
| GEMM slow-down | 通信与计算并发时，GEMM 是否因 SM/CU、HBM、cache 或 launch 资源竞争变慢 |
| primitive capability/semantic cost | NVSHMEM/DUSHMEM 的 put/get/signal/wait/fence/quiet 在目标软件版本上究竟可否使用、何时可见、同步成本多高 |

其中 `R_i` 是方向的关键。它必须定义为：

```text
t_issue       : 对一个 slice 的通信操作被提交的时刻
t_release(i)  : slice i 对应的 consumer GEMM tile 在完成正确同步后，
                第一次被允许读取的时刻
R_i           : t_release(i) - t_issue
t_done        : 完整 collective/通信 epoch 的完成时刻
T_done        : t_done - t_issue
```

`R_i` 绝不是：

- sender API 返回的时刻；
- 远端 buffer 中“看起来已经有数值”的非同步轮询时刻；
- 完整 collective 返回的时刻；
- 仅有 `cudaEvent`/`hipEvent` 但没有证明依赖和可见性合法的时刻。

对于 CCL stream 版本，先采用保守且合法的定义：分块 collective 的 stream event 完成后，消费该 chunk 的 GEMM stream 才能启动。对于 NVSHMEM/DUSHMEM，只有完成 payload 写入、顺序保证、远端 ready notification 和 consumer wait 的最小正确性程序都通过后，才可把通知时刻定义为 `t_release(i)`。

---

## 6. 与已有工作的边界：为什么不能重做 AutoCCL、FLUX 或 FiCCO

逐篇的来源等级、DOI、已覆盖优化对象和全文排重缺口见
`第二篇论文相关工作排重与创新证据图谱.md`。本节保留与当前实验决策直接相关的摘要。

### 6.1 已核验的相关工作

以下信息以官方页面或 arXiv 官方摘要为依据；正式论文写作前仍应精读全文并保存 BibTeX/PDF 版本。

| 工作 | 已核验的核心范围 | 对本项目的影响 |
|---|---|---|
| AutoCCL, NSDI 2025 | 针对 NCCL collective 的低层配置调优；拆分搜索空间并进行在线 tuning，也评估了通信与计算并发 | 不能把“自动选 algorithm/protocol/channel”当新意；必须证明联合依赖释放/重叠与其不同 |
| FLUX, arXiv:2406.06858 | 将依赖通信和计算过度分解并 kernel fusion，以隐藏通信；摘要报告了 AG/RS 类通信重叠收益 | 不能仅以“更细粒度 AllGather-GEMM 融合/流水”作为贡献 |
| FiCCO, arXiv:2512.10236 | 更细粒度的 compute-communication overlap 设计空间；分析 decomposition/contention slowdown，并用 DMA 和 heuristic 选择 schedule | 不能仅以“DMA + chunk + 形状启发式/策略选择”作为贡献；需要明显不同的第一性变量与证据 |
| Demystifying NVSHMEM, arXiv:2606.05951 | 系统层分析 NVSHMEM 的 symmetric memory 和 device-initiated one-sided operations | 可借鉴其原语/路径 profiling 方法，但不能声称该工作已经验证 DUSHMEM 或本项目的跨基座模型 |
| The Landscape of GPU-Centric Communication, arXiv:2409.09874 | 梳理 GPU-centric communication 的库、机制、挑战和开放问题 | 可用于领域术语和 motivation，不能代替直接机制实验 |

官方来源：

- AutoCCL: <https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin>
- FLUX: <https://arxiv.org/abs/2406.06858>
- FiCCO: <https://arxiv.org/abs/2512.10236>
- Demystifying NVSHMEM: <https://arxiv.org/abs/2606.05951>
- The Landscape of GPU-Centric Communication: <https://arxiv.org/abs/2409.09874>

### 6.2 与 AutoCCL 的明确分界

AutoCCL 的直接问题是“在一个 NCCL backend 中如何高效搜索 collective 的低层参数”。当前微基准也表明 DEFAULT 已近似孤立通信 oracle，因此即使把 AutoCCL 的 plugin 机制移植到 RCCL，也很可能只是工程移植，且收益空间有限。

本项目若成立，问题必须变为：

```text
不是：哪个通信参数令 isolated busbw 最大？

而是：对于一个有数据依赖的 AG-GEMM，哪个正确的通信路径、分片、
     ready-notification/window、collective 参数和 GEMM 切分组合，
     能使可消费数据最早到达且整体临界路径最短？
```

AutoCCL 可以是强基线或候选生成器：它给出通信参数候选，但不能代替对 `R_i`、GEMM fragmentation 和 contention 的建模。

### 6.3 与 FLUX/第一篇论文的明确分界

第一篇论文已经建立了 NVSHMEM、Triton/Triton-dist 与 AG-GEMM 细粒度重叠的资产。第二篇不能将“又实现一个 fused kernel”当作主要结果，否则在方法层与 FLUX、第一篇均重叠。

第二篇应把实现定位为**受控验证载体**：可以复用 CUDA C++/HIP C++、CUTLASS/rocBLAS、NCCL/RCCL、NVSHMEM/DUSHMEM，但贡献应是一个能解释并选择不同正确策略的跨基座决策模型，而不是某一 vendor 专属 kernel 的手工优化。

### 6.4 与 FiCCO 的明确分界

FiCCO 的官方摘要已经覆盖了细粒度 schedule、decomposition/contention slowdown、DMA offload 与 heuristic。因而下列路线不应再作为本论文的中心：

```text
“更细 chunk 一定更好”
“用 DMA/copy engine 便可消除计算干扰”
“按 M/K 或静态 shape 写一个固定 rule”
```

本项目可保留这些量作为 candidate 或 baseline，但需要新增下列不可替代的证据链：

1. 通过正确同步测出不同 backend/path 的 release curve；
2. 证明 `busbw`/`T_done` 不能充分解释 `T_e2e` 的排序；
3. 用 capability-aware、release-aware 和 contention-aware 的模型选策略；
4. 同一抽象在 NVIDIA 与海光两套基座上成立，而非将某一实现硬搬到另一平台；
5. 证明选择器在未见 shape/粒度上接近 exhaustive oracle。

FiCCO、FLUX 和 AutoCCL 的全文应在正式定题前精读，并做逐段 related-work evidence map。还需继续检索并精读与 CCL resource scheduling、Collective-GEMM overlap、PGAS completion semantics 最接近的论文，特别是可能名为 ResCCL、CoCoNet 的工作；在拿到原文前，不应对其具体机制作事实性表述。

---

## 7. 建议的论文创新结构

### 7.1 不是预设贡献，而是一条可证伪假设

建议把论文核心写成待验证假设 `H1`：

> 对依赖型 Collective-GEMM，孤立 collective bandwidth 和完整通信完成时间都可能无法预测端到端最优策略；不同策略的分片可释放曲线以及通信-计算资源竞争会改变排序，且该决策边界依赖通信基座能力。

若后续 sweep 没有出现排序反转，`H1` 被否证，应该停止而不是牵强解释。一个诚实的 no-go 结论比建立在错误前提上的数月实现更有价值。

### 7.2 若 H1 成立，最小但完整的方法可以由三部分组成

#### A. Capability Profile：先确定“什么是合法候选”

对每个平台生成机器可读 profile：

```text
backend / library version / accelerator architecture
collective algorithm-protocol-channel availability
actual transport and P2P reachability
NVSHMEM/DUSHMEM primitive availability
payload + signal + wait 的正确性协议
支持的 memory/synchronization 路径
```

这层的价值不是“发现 API 不同”，而是让选择器永不推荐非法、错误或无法证明可见性的候选。

#### B. Release-and-Contention Profile：测量“数据何时可以正确消费”

对候选策略

```text
pi = (path, collective parameters, partition axis, q, window,
      notification mode, GEMM tile/resource mode)
```

测量：

```text
R_0, R_1, ..., R_(q-1)
T_done
T_comm_only
T_gemm_only
T_serial
T_overlap
GEMM efficiency loss
```

并用一条半解析临界路径模型，而不是只用 `busbw` 排序：

```text
T_hat(pi) = T_startup(pi)
          + T_pipeline_steady(pi, {R_i})
          + T_drain(pi)
          + T_contention(pi)
```

模型不需要假装预测一切硬件细节。它的贡献应是把 `R_i` 与 GEMM fragmentation/争用作为首等变量，并用少量 calibration 修正参数。

#### C. Adaptive Selector：在每个平台选择候选，而非强行统一实现

输入应至少包括：

```text
collective, message bytes, M/N/K, datatype, rank count,
backend/platform, actual transport, capability mask,
primitive synchronization cost, GEMM efficiency curve,
communication-compute contention profile
```

输出为当前平台上合法候选集合中的 `top-1` 或 `top-2`。评估应同时报告：

```text
prediction error
oracle top-1 / top-2 hit rate
normalized regret relative to exhaustive oracle
selector calibration and selection overhead
end-to-end step/operator speedup relative to honest baselines
```

### 7.3 建议的主工作负载与泛化范围

| 工作负载 | 定位 | 原因 |
|---|---|---|
| AllGather-GEMM | 主实验 | 与 tensor parallel 直接相关；每个 remote slice 的依赖释放可定义；和第一篇已有资产衔接 |
| ReduceScatter-GEMM | 少量泛化实验 | 检验方法不只适配一个 collectives，但不要在第一阶段扩张复杂度 |
| AllReduce | 通信/能力基线 | 保留微基准，但不应成为主重叠路径 |

GEMM 形状要同时来自真实 LLM trace 和受控合成矩阵。应覆盖 decode/prefill、不同 TP degree 下的 M/N/K，而不是只扫消息大小。真实 workload 才能显示“通信相近但 GEMM efficiency/依赖结构不同”的条件。

---

## 8. Go / No-Go 判据：何时真正确定研究方向

### 8.1 Go 条件

在 NVIDIA 与海光各自平台上，至少应满足以下大部分条件后再将该方向定为第二篇论文：

1. 各平台至少有两种正确、稳定、可比的 AG-GEMM 策略，例如 CCL stream-event pipeline 与经验证的 PGAS ready-notification pipeline；
2. 在 10--20 个真实/代表性形状中，最优策略随 `M/N/K`、通信量、q/window、计算强度或平台发生变化；
3. 至少存在可重复的 case，其中 `busbw` 最优不等于 `T_e2e` 最优，或相近 `T_done` 对应明显不同的 `R_i`/`T_e2e`；
4. 排序反转超过重复噪声。建议每个端点配置至少 5 次独立 process run，报告均值、标准差、median、p5/p95 或 95% CI；
5. profiler timeline 能将差异归因到 release、GEMM efficiency 或 contention，而不是无法解释的抖动；
6. capability-aware 模型经少量 calibration 后，在留出的 shape 上能接近 exhaustive oracle；
7. 两平台的共同抽象都成立，哪怕具体最优实现不同。

### 8.2 No-Go 或收缩条件

| 观察结果 | 决策 |
|---|---|
| `busbw`、`T_done`、`T_e2e` 在全部 case 始终同序 | 放弃“完成语义/依赖释放感知选择器”主线；它没有必要性 |
| DUSHMEM/NVSHMEM primitive 无法以正确且稳定的方式接入目标 pipeline | PGAS 不作为主贡献；先保留 CCL stream path |
| 只有一个平台存在效果 | 不能宣称跨基座方法；可考虑收缩为单平台 runtime/同步机制问题，但需重新排重 |
| 任何收益都来自 chunked GEMM 计算本身变化 | 重做公平基线；当前没有通信优化结论 |
| 仅有 1--3% 结果且未超过波动 | 不足以支撑系统论文主张 |
| 只在 4090 上成立，A100 上未验证 | 不满足若论文承诺 A100+DCU 的平台范围 |

---

## 9. 下一轮实验：从现象发现到可发表证据

### Phase 0：冻结平台事实与数据协议（1--2 天）

目标：清除标签、版本和数据管理歧义。

1. 在两台机器分别保存 GPU/DCU 型号、driver、CUDA/DTK、NCCL/RCCL、NVSHMEM/DUSHMEM、MPI、CPU/NUMA、P2P matrix、topology dump 和实际加载库路径；
2. 解析并固定 K500SM_AI/K100_ai 的真实型号；
3. 为每次 run 分配唯一 `run_id`，不能再以相同 `case_id` 覆盖不同时刻的结果；
4. 所有 CSV 都存储每次 raw run，另建派生 aggregate CSV，绝不混写；
5. 对上文 6 个 RCCL 高波动 case 单独运行 10 次，记录 CPU 绑核、GPU clock、温度、空闲状态、process placement。

建议每个 raw row 至少包含：

```text
run_id, timestamp, platform_id, device_model, backend, library_version,
topology_id, transport, ranks, collective, bytes, dtype,
algo, protocol, channels, mapping, path, partition_axis, q, window,
gemm_M, gemm_N, gemm_K, gemm_impl,
warmup, iterations, repetition_index,
time_us, algbw, busbw, T_done, T_serial, T_overlap,
R_0...R_(q-1), gemm_tflops, correctness, status, log_path
```

`R_i` 可拆为另一个长表，避免在 q 可变时形成大量稀疏列：

```text
run_id, strategy_id, slice_index, t_issue_us, t_release_us, release_latency_us, correctness
```

### Phase A：先用 CCL stream-event 建立公平 AG-GEMM 基线（约 2 周）

目标：在不依赖未核验 PGAS 语义的情况下，回答“isolated bandwidth 是否足以选择端到端策略”。

实现原则：

1. 使用 CUDA C++ + NCCL + CUTLASS/cuBLASLt，HIP C++ + RCCL + rocBLAS/hipBLASLt 的小型同构 harness；
2. 先实现安全的 CCL 分块方式。每个 chunk 的 collective 在 communication stream 发起，完成 event 后由 compute stream 消费对应 GEMM chunk；
3. 消费前必须 `streamWaitEvent` 或等价的合法同步；不允许直接读取尚未同步的 collective output；
4. 每个 `q` 保持总数学计算、datatype、布局、输出校验和 tile ownership 一致；
5. 先控制 q 为有限集合，例如 `q in {1, 2, 4, 8, 16}`，window 为 `{1, 2, 4}`。如 q=16 已使 GEMM 显著碎片化，可以停止扩张；
6. 先只用 capability mask 中稳定可用的 DEFAULT 和 Ring/Simple 的少量 channel 候选，如 `DEFAULT`、`Ring/Simple ch1/ch4/ch8`；Tree/LL128 等不支持或不稳定项不进入第一轮主矩阵。

每一组 `shape x strategy` 运行顺序：

```text
1. correctness / deterministic reference
2. T_comm_only(q)
3. T_gemm_only(q)
4. T_serial(q)
5. T_overlap(q)
6. event-based R_i and T_done
7. GPU profiler timeline on representative cases
```

必须同时保留：

```text
whole-GEMM serial baseline
same-partition serial baseline
same-partition overlap implementation
```

这样可以把“算法重叠收益”和“切分导致的 GEMM 效率损失”分开。

### Phase B：NVSHMEM/DUSHMEM 原语与完成语义 profile（约 1--2 周）

目标：验证能否把 PGAS path 作为合法 candidate，而不是提前假定它更快。

每个平台分别做最小程序和 payload 校验，至少覆盖：

```text
symmetric allocation / peer reachability
put and get
payload write + fence/ordering + signal/doorbell + remote wait
quiet / barrier / sync（只测试已在 installed headers 和运行时确认的 API）
thread / warp / block / on-stream variants（若该版本提供）
multiple outstanding operations and window depth
```

每项都要记录以下时刻和正确性：

```text
local issue return
local source safe-to-reuse
remote payload delivered
remote notification observed
consumer begins a checksum/GEMM read
complete epoch finish
```

推荐使用 epoch/doorbell 防止“旧数据恰好满足等待条件”。consumer 必须验证 payload checksum 或完整 AG-GEMM 输出，才可将 notification 视为 release。禁止把未同步的远端 memory poll 解释为数据早到。

Phase B 的正确产物是 `capability profile + primitive microbenchmark + correctness log`，不是马上将一个单边 put 实现宣布为论文方案。

### Phase C：穷举 oracle 与 selector 验证（约 2--3 周）

当 Phase A/B 已证实存在策略反转后，建立有限而完整的 candidate matrix：

```text
path          : CCL event pipeline; PGAS path（仅通过 Phase B 的路径）
collective    : DEFAULT; Ring/Simple ch1/ch4/ch8
partition     : 合法的 M-side/consumer-side partition
q             : 1/2/4/8/16（视 GEMM 退化裁剪）
window        : 1/2/4
GEMM mode     : 保持数学等价的少量 tile/resource modes
```

建议 shape 选择：

- 从第一篇的真实 LLM trace 抽取至少 10 个 AG-GEMM；覆盖 prefill 与 decode；
- 另加少量控制形状，使通信字节近似但 M/N/K 和 compute intensity 不同；
- NVIDIA 与海光使用同一数学形状、TP degree 和 datatype；
- 若最终论文目标是 A100，核心形状必须在 A100 重跑；RTX 4090 仅作为开发扩展平台。

对每个配置先穷举得到 oracle。随后使用一部分 shape 做 calibration，其余 shape 做留出验证。不要让同一 shape 的不同重复同时出现在训练和测试集合中。

建议报告而非承诺的目标门槛：

```text
top-1 hit rate                 : 目标 >= 80%
top-2 hit rate                 : 目标 >= 90%
median normalized regret       : 目标 <= 5%
95th-percentile regret         : 明确报告，不以均值掩盖
selection/calibration overhead : 必须低于可节省的运行时间
```

这些是研究门槛，不是当前已经取得的结果。

### Phase D：论文级消融、泛化与规模（约 2--3 周）

只有 Phase C 成功后再投入：

1. 关闭 capability mask、release features、contention features，分别测 selector 退化；
2. 比较 isolated-bandwidth best、`T_done` best、固定 chunk、固定 channel、手工策略、exhaustive oracle 和本方法；
3. 报告 AllGather-GEMM 为主结果，ReduceScatter-GEMM 为有限泛化；
4. 测 2 卡与 4 卡以判断现象是否随 scale 改变；如有更大同构平台，再测 8 卡，但不应因为暂时没有 8 卡阻塞主线；
5. 在 A100 与真实海光型号上复核主结论；保存完整 platform facts；
6. 对有代表性的正例、负例分别给 timeline，证明模型为何选择不同策略。

---

## 10. 对当前实验脚本和日志体系的具体要求

1. 原始日志必须保留。`tee` 保存控制台输出还不够，必须保存 manifest、环境快照、topology、stdout/stderr、每个 case 的 run ID 和解析 CSV。
2. raw CSV 必须允许同一个逻辑 case 有多行，但这些行需通过 `run_id` 和 timestamp 区分；绝不能再依赖“最后写入的一行就是最终值”。
3. 汇总表必须写明 aggregation rule：mean/median、过滤的 status、重复次数和 error bar。
4. 正确性必须分层：collective buffer 正确、AG-GEMM output 正确、跨 chunk 累加/拼接正确。只看到 `wrong_count=0` 并不足以证明后续 GEMM 依赖合法。
5. 每个平台至少保留一条 profiler trace，包含 communication stream、compute stream、event/wait 和 kernel 名称；否则无法区分真的 overlap 与 host launch 排队。
6. clock、功耗、CPU 绑核、NUMA 绑核、GPU visibility、batch/sequence length 也要入环境快照。小消息中 10--20% 的波动足以淹没弱优化。
7. `DEFAULT`、手工 Ring/Simple、PGAS 和不同 q 的 case 必须在随机化或轮转顺序下运行，避免温度/频率导致策略顺序偏差。

---

## 11. 最终决策建议

### 11.1 现在可以确定的“研究方向”

可以确定进入验证阶段的方向是：

> **面向 NVIDIA 与海光 GPU 通信基座的、完成语义与依赖释放感知的 Collective-GEMM 自适应重叠。**

它满足项目的技术栈要求：实现可以使用 CUDA C++/HIP C++、NCCL/RCCL、CUTLASS/rocBLAS（或 hipBLASLt）以及经过验证后可加入的 NVSHMEM/DUSHMEM；但创新不依赖重写 NCCL/RCCL 内核，也不依赖实现一个 vendor-specific 的新 GEMM kernel。

### 11.2 现在不能确定的“创新点”

下列说法目前没有实验证据，不能写入摘要或开题结论：

```text
“我们已经证明 busbw 不能预测 overlap。”
“NVSHMEM 与 DUSHMEM 的完成语义导致策略反转。”
“模型能跨基座选到最优策略。”
“DUSHMEM 分块重叠比 RCCL 快。”
“NCCL 比 RCCL 强。”
```

这些都是后续 Phase A--C 要检验的命题。

### 11.3 能成为论文创新的最小闭环

如果得到正结果，一篇完整的系统论文应形成这条证据链：

```text
观察：isolated bandwidth / T_done 与 AG-GEMM 的 T_e2e 存在可重复排序反转
  -> 归因：正确 release curve 与 GEMM fragmentation/contention 解释反转
  -> 方法：capability-aware, release-aware, contention-aware cost model
  -> 决策：轻量 selector 在合法候选中选策略
  -> 结果：接近 exhaustive oracle，优于 DEFAULT/固定策略/只看带宽的选择器
  -> 泛化：NVIDIA 和海光分别复现同一抽象，但最优路径可不同
```

其中任一箭头缺失，都会削弱创新的可辩护性。反过来说，这也是一个范围可控的计划：主场景只锁定 AllGather-GEMM，候选集合有限，先用安全 CCL pipeline 获得事实，再决定是否值得引入 NVSHMEM/DUSHMEM 的复杂路径。

---

## 12. 后续阅读与排重清单

在正式写 proposal 前，建议按以下优先级完成全文精读和 evidence map：

1. AutoCCL：确认它在“通信和计算并发”中的具体模型、控制粒度、候选参数、实验 workload；
2. FLUX：确认 AG/RS-GEMM 数据依赖、同步协议、kernel fusion 条件和适用拓扑；
3. FiCCO：确认其 design space、DMA 路径、contention 模型、shape heuristic 和实验拓扑，逐项与本方法作表；
4. NVSHMEM/DUSHMEM 官方编程手册与当前安装头文件：确认每个拟用 primitive 的 host/device/on-stream 可用性、完成/可见性、fence/quiet/signal/wait 规则；
5. 继续检索 ResCCL、CoCoNet 及 2025--2026 年 Collectives-GEMM overlap/runtime scheduling 工作。对每篇建立“覆盖的问题、假设、控制粒度、平台、与本方法差异、必须避免复现的贡献”的一页表。

审稿人会首先追问：这是否只是 FLUX/FiCCO/AutoCCL 的重新组合？答案不能靠措辞，而必须靠上述端到端证据链、明确的完成语义定义和跨基座验证来给出。
