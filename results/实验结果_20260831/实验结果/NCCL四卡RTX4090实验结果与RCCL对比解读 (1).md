# NCCL 四卡 RTX 4090 实验结果与 RCCL 对比解读

## 1. 结论摘要

`results/rtx4090_sm89_4gpu/20260826T222600Z_debug/manifest.tsv` 确实是四卡 RTX 4090 的 NCCL 实验结果目录，覆盖了 preflight、代表性 case、channel 扫描、rank mapping、1 GiB pilot 和 LL128 探测等阶段。

但它与 `formal_summary.csv` 中的 RCCL 实验并不是逐行完全相同的实验集合：两边有 750 个可对齐的四卡基础 case；NCCL 额外包含 1 个 transport probe 和 18 个 LL128 probe，而 RCCL 的 gfx928 结果没有纳入 LL128 探测。因此，比较性能时应使用四卡、去重后的共同 case，并把 LL128 能力差异单独分析。

在共同的 DEFAULT、out-of-place、三次重复口径下，RTX 4090/NCCL 的绝对 bus bandwidth 高于 K500SM_AI/RCCL。但这不能直接解释为 NCCL 软件优于 RCCL，因为两侧 GPU、拓扑和 transport 不同：NCCL 侧是 RTX 4090 + SHM/direct，RCCL 侧是 K500SM_AI + PCIe P2P。

两套 backend 都显示出以下规律：

- DEFAULT 与 Ring/Simple 在中大消息上通常接近，固定选择 Ring/Simple 没有普适收益。
- LL 在孤立通信测试中明显慢于 DEFAULT，且退化程度依赖 backend、消息大小和 transport；因此不能只按孤立带宽判断 LL 的端到端价值。
- channel 的最优值依赖 backend、collective、消息大小和 transport。NCCL 通常在 ch2/ch4 后趋于饱和，RCCL 在当前 PCIe P2P 平台上继续受益于 ch8。
- 当前三种 rank mapping 的差异很小，尚不足以支持“固定 mapping 最优”的结论。

最有价值的研究方向不是寻找一个跨平台固定参数，而是构建 backend、collective、消息规模、transport 和能力约束感知的自适应策略选择器，并在通信计算重叠场景中验证其端到端收益。

## 2. 实验对象与结果文件

### 2.1 NCCL 四卡 RTX 4090

结果目录：

```text
/data/coding/comm-study-migration-rtx4090-20260826/results/rtx4090_sm89_4gpu/20260826T222600Z_debug/
```

关键文件：

- `manifest.tsv`：case 清单及每个 case 的配置和状态。
- `formal_master.log`：主运行日志。
- `summary.txt`：实验摘要。
- `environment.txt`、`platform/platform.txt`：软件、GPU、拓扑和环境信息。
- `logs/*.log`：各 case 的 NCCL 输出和带宽结果。

平台信息：

| 项目 | NCCL 实验 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 ×4 |
| 架构 | sm_89 |
| 显存 | 24564 MiB/卡 |
| CUDA | 12.6.85 |
| NCCL | 2.30.7 |
| MPI | Open MPI 4.1.2 |
| NUMA | GPU0/GPU1 在 NUMA 0，GPU2/GPU3 在 NUMA 1 |
| GPU 间连接 | `nvidia-smi topo -p2p r/w` 均为 CNS |
| NCCL transport | 日志显示 `via SHM/direct` |

### 2.2 RCCL 四卡 K500SM_AI

主要数据文件：

```text
/data/coding/formal_summary.csv
```

该文件混合了 2 卡和 4 卡数据，四卡分析必须先筛选 `ranks == 4`。RCCL 四卡平台为：

| 项目 | RCCL 实验 |
|---|---|
| GPU | K500SM_AI ×4 |
| 架构 | gfx928 |
| 连接 | PCIe，P2P 可用 |
| RCCL | 2.22.3-HEAD |
| HIP/ROCm | 6.3 |
| MPI | Open MPI 5.0.3 |

## 3. NCCL manifest 的实验矩阵

`manifest.tsv` 共 770 行，其中 1 行是表头，因此共有 769 个数据 case：

| 阶段 | case 数 |
|---|---:|
| transport probe | 1 |
| preflight | 9 |
| representative | 270 |
| channel scan | 432 |
| rank mapping | 36 |
| 1 GiB pilot | 3 |
| LL128 probe | 18 |
| 合计 | 769 |

状态统计：

- `status=0, correctness=PASS`：547 个。
- `status=3, correctness=FAIL`：222 个。
- 222 个非通过项均是 NCCL 对不支持组合返回的 `invalid usage`，不是通信结果校验错误，也不是随机数据损坏。

失败按阶段分布为 representative 72 个、channel 144 个、LL128 6 个。失败集中在 NCCL 不支持的 Tree 算法组合，而不是 Ring/Simple 主路径本身。

## 4. 与 RCCL 实验口径是否一致

### 4.1 可以对齐的共同集合

`formal_summary.csv` 共 1079 条数据行，包含 2 卡和 4 卡记录。筛选 `ranks == 4` 后，channel 扫描存在重复记录：原始 channel 行 641 条，但唯一 `case_id` 只有 432 个。因此，比较前需要按 `case_id` 去重，并保留最终有效记录。

去重后 RCCL 四卡有效 case 为 750 个：

| 阶段 | RCCL 四卡有效 case |
|---|---:|
| preflight | 9 |
| representative | 270 |
| channels | 432 |
| mapping | 36 |
| 1 GiB pilot | 3 |
| 合计 | 750 |

这 750 个 case 与 NCCL manifest 中上述同名阶段的基础矩阵是一致的，因而可以作为共同实验集合进行比较。

### 4.2 不完全一致的部分

NCCL 比共同集合多出：

- 1 个 transport probe，用来确认 RTX 4090 上的 NCCL transport 路径。
- 18 个 LL128 probe，用来探测协议能力和性能。

RCCL 文档和实验记录表明，gfx928 RCCL 的 LL128 探测不可用，`formal_summary.csv` 没有对应的 LL128 失败记录。因此不能把 RCCL 的“没有 LL128 行”解释成 LL128 性能为零，而应解释为该 backend/架构组合没有进入可比的 LL128 测量路径。

结论是：

> 这是同一套基础实验设计在两个 backend 上的迁移，而不是两个平台逐行完全相同的日志集合。共同的 750 个 case 可直接对齐；transport probe 和 LL128 probe 必须作为平台能力差异单独报告。

## 5. 数据清洗和比较口径

本文所有跨平台数值比较均遵循以下口径：

1. 仅使用四卡记录：`ranks == 4`。
2. 对 RCCL channel 重复记录按唯一 `case_id` 去重，保留最终有效记录。
3. 选择 out-of-place、DEFAULT 配置、三次重复结果的 `busbw` 均值。
4. 单位统一为 GB/s。
5. 将 1 MiB、8 MiB、64 MiB、256 MiB 和 1 GiB 作为主要观察点。

`busbw` 是 collective benchmark 的有效通信带宽，不等于 PCIe 链路原始物理带宽，也不等于应用端到端吞吐。

## 6. 四卡 DEFAULT 性能对比

### 6.1 共同消息规模的带宽

| Collective | Size | NCCL RTX 4090 | RCCL K500SM_AI | NCCL/RCCL |
|---|---:|---:|---:|---:|
| AllGather | 1 MiB | 8.36 | 6.44 | 1.30 |
| AllGather | 8 MiB | 13.26 | 11.26 | 1.18 |
| AllGather | 64 MiB | 16.74 | 12.24 | 1.37 |
| AllGather | 256 MiB | 16.81 | 12.54 | 1.34 |
| AllReduce | 1 MiB | 9.35 | 7.94 | 1.18 |
| AllReduce | 8 MiB | 14.60 | 12.00 | 1.22 |
| AllReduce | 64 MiB | 17.56 | 12.83 | 1.37 |
| AllReduce | 256 MiB | 17.63 | 13.03 | 1.35 |
| ReduceScatter | 1 MiB | 8.67 | 6.03 | 1.44 |
| ReduceScatter | 8 MiB | 11.77 | 10.82 | 1.09 |
| ReduceScatter | 64 MiB | 15.87 | 11.69 | 1.36 |
| ReduceScatter | 256 MiB | 15.91 | 11.97 | 1.33 |

### 6.2 1 GiB pilot

| Collective | NCCL RTX 4090 | RCCL K500SM_AI |
|---|---:|---:|
| AllGather | 16.80 | 12.8045 |
| AllReduce | 17.80 | 13.1464 |
| ReduceScatter | 15.93 | 12.2217 |

NCCL 在当前机器上的四卡 1 GiB pilot 分别达到约 16.80、17.80、15.93 GB/s；RCCL 分别为约 12.80、13.15、12.22 GB/s。中大消息上 NCCL/RCCL 比值约为 1.33--1.37，小消息比值约为 1.09--1.44。

### 6.3 对绝对值的正确解释

这些结果说明两台机器在各自当前软件栈和连接条件下的有效带宽不同，但不能单独归因于 NCCL 实现优于 RCCL。影响因素至少包括：

- RTX 4090 与 K500SM_AI 的 GPU 计算和内存系统不同。
- NCCL 通过 SHM/direct，RCCL 通过 PCIe P2P；transport 不同。
- CUDA/NCCL 与 HIP/ROCm/RCCL 版本不同。
- NUMA、PCIe 拓扑、驱动和 kernel 调度不同。

如果要形成“实现优化带来多少收益”的因果结论，需要在同一硬件上比较 backend 版本，或固定 transport 后进行受控消融；当前结果更适合用于发现策略选择规律和平台依赖性。

## 7. DEFAULT、Ring/Simple 与 LL 协议

### 7.1 NCCL Ring/Simple 相对 DEFAULT

在 NCCL representative case 中，Ring/Simple 与 DEFAULT 基本接近：

- AllGather：约为 DEFAULT 的 0.97--1.00 倍。
- ReduceScatter：约为 DEFAULT 的 1.00--1.04 倍。
- AllReduce：1 MiB 约 0.93 倍，8 MiB 约 1.02 倍，64 MiB 及以上约 0.99--1.00 倍。

这表示 DEFAULT 在这些 case 上已经能选到接近 Ring/Simple 的路径。固定 Ring/Simple 只会减少选择空间，不能据此宣称一定更快。

RCCL 侧也观察到 DEFAULT 与 Ring/Simple 接近，因此该结论不是某一 backend 的偶然现象。

### 7.2 NCCL Ring/LL 相对 DEFAULT

NCCL 的 Ring/LL 在孤立通信吞吐上明显退化：

| Collective | 1 MiB | 8 MiB | 64 MiB | 256 MiB |
|---|---:|---:|---:|---:|
| AllGather | 约 0.41 倍 | 约 0.36 倍 | 约 0.31 倍 | 约 0.31 倍 |
| AllReduce | 约 0.43 倍 | 约 0.31 倍 | 约 0.36 倍 | 约 0.37 倍 |
| ReduceScatter | 约 0.39 倍 | 约 0.36 倍 | 约 0.32 倍 | 约 0.35 倍 |

RCCL 侧 LL 的退化更明显，尤其在 64 MiB：AllGather 的 Ring/LL 约为 DEFAULT 的 0.16 倍，AllReduce 和 ReduceScatter 约为 0.27 倍。

### 7.3 协议结论的边界

LL 协议通常为低延迟和通信计算重叠设计；孤立 benchmark 中吞吐低，不等于训练 step 一定更慢。当前数据只能证明：

> 在当前四卡平台、当前消息范围和当前 transport 下，LL 的孤立通信带宽低于 DEFAULT/Simple，且退化具有 backend 依赖性。

还不能证明 LL 在 AG-GEMM、梯度同步或其他重叠工作负载中的端到端收益。这个差距正是后续研究需要补齐的实验环节。

## 8. algorithm/protocol 能力边界

NCCL 的 222 个失败 case 主要集中于以下组合：

- AllGather + Tree/Simple
- AllGather + Tree/LL
- ReduceScatter + Tree/Simple
- ReduceScatter + Tree/LL

LL128 探测结果为：

| Collective | Ring/LL128 | Tree/LL128 |
|---|---|---|
| AllGather | 通过 | 不可用 |
| AllReduce | 通过 | 通过 |
| ReduceScatter | 通过 | 不可用 |

这些 `invalid usage` 是能力边界，不应作为低性能样本或 correctness 失败样本混入平均值。实验系统应在性能搜索前先建立 capability mask，例如：

```text
NCCL + AllGather + Tree       -> 排除
NCCL + ReduceScatter + Tree   -> 排除
RCCL gfx928 + LL128           -> 排除
```

该能力层对跨 backend 自动调参尤其重要：如果先盲目遍历所有算法/协议组合，搜索时间会被非法配置占用，还会把 backend 的能力差异误判为性能差异。

## 9. channel 扫描结果

以下为 64 MiB、Ring/Simple、三次重复的平均 `busbw`，单位 GB/s：

| Collective | Backend | ch1 | ch2 | ch4 | ch8 |
|---|---|---:|---:|---:|---:|
| AllGather | NCCL | 13.90 | 16.75 | 17.51 | 17.50 |
| AllGather | RCCL | 4.96 | 9.87 | 12.19 | 12.71 |
| AllReduce | NCCL | 14.82 | 17.46 | 19.19 | 19.41 |
| AllReduce | RCCL | 5.41 | 10.53 | 12.78 | 13.09 |
| ReduceScatter | NCCL | 14.22 | 15.88 | 17.25 | 17.58 |
| ReduceScatter | RCCL | 5.11 | 9.64 | 11.60 | 12.43 |

观察结果：

- NCCL 从 ch1 增加到 ch2/ch4 有明显收益；ch4 到 ch8 多数 collective 已接近饱和。
- RCCL 在当前 PCIe P2P 平台上从 ch4 增加到 ch8 仍有收益，说明其最优 channel 区间与 NCCL 不同。
- 小消息下 channel 并非越多越好；NCCL 的 1 MiB case 中，ch8 可能低于 ch4，额外 channel 的调度开销会抵消并行度收益。

因此 channel 不能脱离上下文设为全局常数。绝对带宽差异仍不能直接用来比较 NCCL 和 RCCL 的 channel 机制，因为两侧 transport 不同；可以比较的是各自 backend 内部随 channel 变化的趋势。

## 10. rank mapping 结果

NCCL 64 MiB 的三种 mapping 结果如下，单位 GB/s：

| Collective | 0123 | 0213 | 0321 |
|---|---:|---:|---:|
| AllGather | 16.78 | 16.71 | 16.71 |
| AllReduce | 17.58 | 17.53 | 17.46 |

三种 mapping 的差异小于 1%，当前拓扑下 mapping 不是强信号。1 MiB AllGather 存在重复运行波动，不能据此过度解读某一种排列。后续如果研究 mapping，需要扩大拓扑差异、重复次数和消息范围，并同时记录 PCIe/NUMA 路径，而不是只比较一个带宽均值。

## 11. 可以从实验中得出的结论

### 11.1 跨平台固定策略不可靠

DEFAULT、Ring/Simple、LL 和 channel 扫描共同表明：最优策略随 backend、collective、消息规模和 transport 改变。一个在 RTX 4090 上接近最优的参数，不一定在 K500SM_AI 上成立；一个在孤立通信中最快的参数，也不一定在重叠场景中最快。

### 11.2 能力约束必须先于性能搜索

Tree/Simple、Tree/LL 和 gfx928 LL128 的不可用组合说明，调优器需要先做能力过滤，再在合法集合中测量性能。能力 mask 本身也是跨 NCCL/RCCL 迁移时必须维护的知识。

### 11.3 transport 是重要的条件变量

NCCL 的 SHM/direct 和 RCCL 的 PCIe P2P 产生了不同的 channel 扩展曲线和 LL 退化曲线。研究报告中应把 transport 作为显式特征，而不是把 backend 名称当作所有差异的替代变量。

### 11.4 当前 mapping 信号弱于 channel 和协议信号

本轮四卡拓扑中，mapping 变化小于 1%，而 channel 和 LL 的影响可达几十个百分点。因此有限实验预算应优先用于 protocol/channel/overlap 的联合搜索，再扩大 mapping 研究。

## 12. 不能从当前实验直接得出的结论

以下结论目前证据不足：

- 不能说 NCCL 在所有硬件上都优于 RCCL。
- 不能说 Ring/Simple 在所有消息大小和 collective 上都优于 DEFAULT。
- 不能说 LL 因为孤立带宽低，就一定不适合训练。
- 不能说 ch8 是 NCCL 或 RCCL 的全局最优 channel。
- 不能把 `invalid usage` 当作通信 correctness 错误或性能异常。
- 不能仅凭当前 4 卡、单一拓扑推断 8 卡、多 NUMA 或 NVLink/Infinity Fabric 场景的规律。

## 13. 对第二篇论文的研究启示

建议将创新点定义为“跨 backend、跨 transport 的上下文感知 collective 策略选择”，而不是简单提出一个固定的 Ring/Simple 或 channel 配置。

### 13.1 三层策略框架

1. **能力掩码层**：输入 backend、GPU 架构、collective 和协议，排除 `invalid usage` 组合及不支持的 LL128。
2. **性能选择层**：在合法组合中，根据 collective、消息大小、rank 数、backend、transport、channel 和 rank mapping 选择算法/协议/channel。
3. **端到端验证层**：在通信计算重叠工作负载中验证选择结果，而不是只优化 microbenchmark 的 `busbw`。

### 13.2 可形成的研究问题

核心问题可以表述为：

> 孤立通信最快的策略，是否也是通信计算重叠场景下端到端最快的策略？

由此可以比较两种选择器：

- 仅以 NCCL/RCCL microbenchmark 带宽为目标的静态选择器。
- 使用 transport 和 overlap 特征、以通信暴露时间或 step time 为目标的上下文感知选择器。

如果第二种选择器能在 NCCL RTX 4090 和 RCCL K500SM_AI 上同时降低通信暴露时间，并减少非法配置搜索，就具备清晰的跨平台研究贡献。

## 14. 后续实验建议

后续实验应保持 NCCL/RCCL 的共同 case 定义，并新增统一的 AG-GEMM overlap harness，至少记录：

```text
T_comm
T_gemm
T_serial
T_overlap
communication exposed time
steady-state step time
```

推荐的消融维度：

- collective：AllGather、AllReduce、ReduceScatter。
- message size：从小消息到 1 GiB，覆盖协议切换区间。
- algorithm/protocol：DEFAULT、Ring/Simple、Ring/LL，以及各 backend 可用的 LL128。
- channels：ch1/ch2/ch4/ch8，必要时扩展更大范围。
- transport：显式记录 SHM/direct、PCIe P2P 等路径。
- mapping：至少覆盖 NUMA/PCIe 路径明显不同的排列。
- backend：NCCL 与 RCCL 的 capability mask 和选择结果分别报告。

最终报告应同时给出孤立 `busbw` 和端到端 step time，避免把 microbenchmark 最优误写成训练最优。

## 15. 一句话结论

本轮实验最可靠的结论是：**四卡 NCCL 与四卡 RCCL 的共同基础矩阵可以对齐，但性能和协议行为明显受硬件、transport 与 backend 能力约束；因此第二篇论文应研究上下文感知、自适应的跨 backend 策略选择，而不是寻找一个固定的全局最优参数。**
