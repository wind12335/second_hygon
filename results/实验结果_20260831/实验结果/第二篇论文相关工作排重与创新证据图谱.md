# 第二篇论文相关工作排重与创新证据图谱

> 更新日期：2026-08-30  
> 用途：为第二篇论文定题服务，不是论文正文。  
> 核心原则：只将原文、官方页面、DOI 元数据或厂商手册直接支持的内容记为事实；“研究空白”和“可投稿创新”均须通过后续实验和全文排重验证。

---

## 1. 本轮排重后的结论

当前最可行的候选方向仍是：

> **完成语义与依赖释放感知的 Collective-GEMM 自适应重叠策略。**

但它的创新不能是以下任何一项的简单重述：

```text
更细粒度地切分通信和 GEMM
把通信和计算融合/流水化
为 collective 搜索 algo/protocol/channel/chunk
为通信 primitive 做 thread-block/SM 资源调度
用 DMA/copy engine 与计算重叠
为异构 GPU cluster 合成 collective schedule
```

这些方向分别已有 FLUX、CoCoNeT、AutoCCL、ResCCL、FiCCO、HeteCCL/TACCL/MSCCL 等直接邻居。

当前唯一仍有机会形成独立系统问题的表述是：

```text
在依赖型 Collective-GEMM 中，何时一个到达的分片可以被正确消费？
这个“合法数据释放”时间、通信-计算资源竞争和平台能力约束，
能否比 isolated bandwidth 或整个 collective 完成时间更好地选择重叠策略？
```

这里的跨 NVIDIA/海光验证是**必要的证据维度**，不是创新本身。只在两家平台各跑一次相同 benchmark，不会自动产生论文 novelty。

---

## 2. 检索范围与证据等级

### 2.1 本轮实际核验的来源

| ID | 工作 | 本轮核验来源 | 证据等级 |
|---|---|---|---|
| CCL-1 | AutoCCL, NSDI 2025 | USENIX 官方页面的题名、摘要和评估描述；本地 PDF | 高 |
| OVL-1 | FLUX, arXiv:2406.06858 | arXiv 官方摘要；本地 PDF | 高（摘要范围） |
| OVL-2 | FiCCO, arXiv:2512.10236 | arXiv 官方摘要；本地 PDF | 高（摘要范围） |
| OVL-3 | CoCoNeT, ASPLOS 2023, DOI `10.1145/3567955.3567959` | Crossref 正式元数据；本地文献台账 | 中，仅题名/元数据 |
| CCL-2 | ResCCL, SIGCOMM 2025, DOI `10.1145/3718958.3750514` | Crossref 元数据、Semantic Scholar 摘要；本地文献台账 | 中，摘要级 |
| CCL-3 | HeteCCL, NSDI 2026 | USENIX 官方页面的摘要与平台说明 | 高 |
| PGAS-1 | Demystifying NVSHMEM, arXiv:2606.05951 | arXiv 官方摘要；本地 PDF | 高（摘要范围） |
| PGAS-2 | Unified GPU-Aware OpenSHMEM, arXiv:2607.08006 | 本地文献台账与 PDF | 中，本轮未重读全文 |
| DTK-1 | DUSHMEM/RCCL/DUMMA 手册 | 已有本地手册核验与证据台账 | 高于 API 文字；实际可用性仍需实测 |

### 2.2 证据等级的含义

| 等级 | 可以写入当前定题文档 | 不能据此做的事 |
|---|---|---|
| 高 | 题目、摘要明确陈述的优化对象、方法大类、公开实验范围 | 推断摘要没有写出的具体 kernel/同步细节 |
| 中 | 题目、DOI、会议、元数据，以及标明“摘要级”的范围 | 将其精确实现细节、限制或性能机制当作事实 |
| 待实测 | 厂商手册定义的语义可指导最小实验 | 假定当前安装版本、当前硬件路径已经支持该 API 或具有同样性能 |

在论文正文中，所有“已有工作没有……”的强排他性语言都必须等到 CoCoNeT、ResCCL 及后续最新工作全文读完后再写。现在只能写“本轮已核验材料尚未显示……”。

---

## 3. 直接相关工作逐篇边界

### 3.1 AutoCCL：collective 低层参数的自动调优

**来源**：USENIX NSDI 2025 官方页面；`/root/seconde-paper/AutoCCL_NSDI_2025.pdf`。  
**链接**：<https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin>

官方摘要明确说明：

- 目标是 collective communication library 的低层参数选择；
- 为处理配置搜索空间爆炸，区分 implementation-related 参数与影响搜索空间的参数，并采用 divide-and-conquer；
- 采用考虑 communication-computation interference 的 online tuning；
- 实现于 NCCL；
- 评估包含 isolated microbenchmark、并发计算和端到端 DNN 训练。

官方页面报告的上限结果包括：相对 NCCL microbenchmark `1.24--1.29x`，并发计算时最高 `1.80x`，端到端 iteration 最高 `1.07--1.32x`。这些是 AutoCCL 的结果，不是本项目预期结果。

**对本选题的直接限制：**

```text
不能把“在线测量后选择 Ring/Tree、LL/Simple、channel、chunk”当作核心方法；
不能只用 isolated busbw 或 concurrent-compute throughput 作为方法目标；
不能因为 AutoCCL 只基于 NCCL 就把“移植到 RCCL”称为研究创新。
```

**仍可借鉴的部分：**

- 有限候选集的在线/低开销 calibration；
- 将通信和计算共同出现时再测量，避免 isolated benchmark 偏差；
- capability mask 后的候选探索。

**本方向必须额外证明的差异：**

```text
AutoCCL 选择的是 collective 配置；
本方向必须选择“合法分片释放 + GEMM 消费”组成的端到端策略，
并显式解释 R_i、GEMM fragmentation 和 contention。
```

若最终实验发现最佳端到端策略永远等于 AutoCCL/DEFAULT 选出的最佳 collective 配置，本方向没有足够独立性，应停止。

### 3.2 FLUX：通过 kernel fusion 实现细粒度通信计算重叠

**来源**：arXiv 官方摘要，`arXiv:2406.06858`。  
**链接**：<https://arxiv.org/abs/2406.06858>

官方摘要明确说明：

- 面向分布式深度学习中的依赖通信和计算；
- 将通信和计算过度分解为更细粒度操作；
- 将其融合进更大的 kernel，以隐藏通信且尽量不损失 kernel efficiency；
- 报告的上限包括训练相对 Megatron-LM `1.24x`，推理 prefill/decoding 相对 vLLM `1.66x/1.30x`。

**对本选题的直接限制：**

```text
“切 chunk + 让 GEMM 等待 chunk + 实现一个 fused CUDA/HIP kernel”已经不是新的问题；
把第一篇的 Triton/NVSHMEM 实现换成 CUDA C++ 或 HIP C++ 也不是新贡献。
```

**可能的差异空间：**

- FLUX 的摘要没有提供本项目提出的“跨通信基座合法 release profile”或 capability-aware selector 的证据；
- 但这不等于 FLUX 一定没有类似机制，必须全文确认其 AG-GEMM 数据流、同步协议、autotuning 维度和平台假设；
- 因此，最终论文不能以“比 FLUX 更细粒度”作为卖点，只能以可测的策略排序反转和决策模型作为卖点。

### 3.3 FiCCO：DMA-based finer-grain overlap 与 schedule heuristic

**来源**：arXiv 官方摘要，`arXiv:2512.10236`。  
**链接**：<https://arxiv.org/abs/2512.10236>

官方摘要明确说明：

- 研究数据依赖的 compute-communication overlap；
- 相对传统 shard-level overlap，采用更细一级粒度；
- 构造更广的 schedule design space；
- 将 decomposition 和 contention slowdown 作为性能限制进行表征；
- 使用 GPU DMA engines 以减轻 overlap 的 contention inefficiency；
- 使用 static operator sizes 设计 schedule heuristic；
- 报告最高 `1.6x` 加速和未见场景 `81%` 的最优 schedule 选择准确率。

**对本选题的直接限制：**

```text
不能将“DMA offload”“更细 chunk”“分解 + contention 的启发式”作为本项目独有贡献；
不能只报告一个 shape rule 或一张 bandwidth/roofline 图就宣称自适应；
不能把 DUSHMEM/NVSHMEM 一概视为无竞争的 copy engine 路径。
```

**可能仍成立的差异：**

```text
FiCCO 的摘要描述的是 schedule/decomposition/contention 选择；
本项目的前提是额外验证 completion/visibility/notification 形成的合法 release curve
是否造成“相近 T_done、不同 T_e2e”或“bandwidth 排名反转”。
```

不过，“跨厂商”本身不能代替这个差异。若本项目最终只做 `M/N/K -> q` 的静态启发式，审稿人会合理地认为它只是 FiCCO 的平台迁移。

### 3.4 CoCoNeT：dependent computation 通过 decomposition 的 overlap

**正式题名**：*Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models*。  
**出版信息**：ASPLOS 2023，DOI `10.1145/3567955.3567959`，Wang et al.  
**链接**：<https://doi.org/10.1145/3567955.3567959>

本轮通过 Crossref 核验了题名、作者、出版会议、页码 `93--106` 和 CC-BY 许可元数据；现有环境未取得可读全文。因此目前仅能做如下保守判断：

```text
其题名已经证明“为 dependent computation 做 decomposition overlap”是已有研究对象。
```

**禁止的推断：**

- 不得在尚未阅读全文前断言它用/不用某种 P2P、PGAS、event 或 cost model；
- 不得给出与本项目的细粒度机制差异；
- 不得将其未取得的 PDF 当成“已精读”。

**定题行动：**通过学校/图书馆访问获得全文后，优先补齐以下字段：计算图切分维度、通信 API、ready 条件、是否测 first-consumable data、autotuning/heuristic、硬件/拓扑、主要反例。这是正式投稿前的 P0 缺口。

### 3.5 ResCCL：collective backend 内部资源调度

**正式题名**：*ResCCL: Resource-Efficient Scheduling for Collective Communication*。  
**出版信息**：SIGCOMM 2025，DOI `10.1145/3718958.3750514`，Liu et al.  
**链接**：<https://doi.org/10.1145/3718958.3750514>

Crossref 已核验出版元数据；Semantic Scholar 摘要描述其：

- 以新的 CCL backend 为目标；
- 在 primitive（例如 `send`、`recvReduceCopy`）级别调度；
- 支持灵活 thread-block allocation；
- 生成轻量通信 kernel；
- 以降低 SM resource overhead、提高 TB 利用率和通信带宽为目标；
- 报告相对 NCCL/MSCCL 的带宽与端到端训练提升。

摘要中列出的结果是 ResCCL 的结果：最高 `2.5x` bandwidth、`77.8%` 更低 SM resource overhead、最高 `39%` Megatron throughput 提升。

**对本选题的直接限制：**

```text
不得将“调整 collective kernel 的 TB 数量/SM 占用、为通信 primitive 做资源调度”作为主创新；
不得为获得资源控制而把项目扩大为重写 NCCL/RCCL backend。
```

**尚未确认的关键问题：**ResCCL 全文是否直接研究 Collective-GEMM 并发、是否提出类似 release-time 指标、是否覆盖 dependent compute。全文未获得前，不能宣称本项目已经与 ResCCL 完全排重。该 PDF 同样是 P0 缺口。

### 3.6 HeteCCL：异构 NVIDIA cluster 的 collective schedule synthesis

**来源**：USENIX NSDI 2026 官方页面。  
**链接**：<https://www.usenix.org/conference/nsdi26/presentation/hei>

官方摘要说明：

- 面向 heterogeneous GPU clusters；
- 显式建模 topology 与 link bandwidth；
- 对 chunk 进行 schedule-step 级量化；
- 将调度建模为 weighted directed graph 上的并行传输问题；
- 使用 SMT 和 counterexample-guided inductive synthesis；
- 官方实验为由 `32 H20 + V100` 构成的异构 NVIDIA testbed。

**对本选题的直接限制：**

```text
不应将项目扩张为跨设备 collective schedule synthesis、全局 routing 或 SMT 搜索；
不应把 NVIDIA 与海光各自的独立同构实验误说成一个 heterogeneous collective group。
```

HeteCCL 可以作为“平台和链路特征进入模型”的相关工作，但问题对象与本项目不同：它解决 heterogeneous cluster 内 collective schedule，不是 dependent AG-GEMM 的分片合法消费和通信-计算重叠选择。

### 3.7 PGAS 与完成语义：NVSHMEM/DUSHMEM 不是可直接等同的 API

**Demystifying NVSHMEM** 的官方摘要将 NVSHMEM 定位为基于 symmetric memory 的 GPU-initiated one-sided communication，并分析 device-side operations。它支持将 PGAS 视为与 host-launched CCL 不同的候选路径，但不证明 DUSHMEM 与 NVSHMEM 的 API、内存序或性能相同。

现有 DTK DUSHMEM 手册的已核验语义更具体：

```text
put 的本地返回不等于远端交付/可消费；
连续 put 的顺序需要相应的 fence/ordering 语义；
get 的本地目标完成语义不同；
barrier、sync、quiet、wait 的含义不可混用。
```

这正是本项目可以测量而不是假定的对象：

```text
local submission -> remote delivery -> ready notification -> consumer release -> full epoch completion
```

**重要限制：**只有当前环境的 installed headers、最小程序和 payload checksum 都证明某条 NVSHMEM/DUSHMEM 路径可用后，它才可进入第二篇论文的实现和候选集合。手册中的接口名不等于当前版本、当前拓扑和当前 device-side context 中可用。

---

## 4. 现有工作覆盖图

表中“覆盖”只表达从已核验材料能确认的主优化对象，不代表所有实现细节。

| 优化对象 | AutoCCL | FLUX | FiCCO | CoCoNeT | ResCCL | HeteCCL | 候选论文应如何处理 |
|---|---|---|---|---|---|---|---|
| collective algo/protocol/channel 搜索 | 直接覆盖 | 非主重点 | 非主重点 | 未全文确认 | 非主重点 | schedule synthesis | 仅作为有限 candidate，不做核心 |
| collective kernel TB/SM 调度 | 未作为主声明 | 未作为主声明 | 非主重点 | 未全文确认 | **直接覆盖** | 非主重点 | 明确排除 |
| topology/routing schedule synthesis | 非主重点 | 非主重点 | 非主重点 | 未全文确认 | 非主重点 | **直接覆盖** | 明确排除 |
| kernel fusion 的 overlap | 评估并发影响 | **直接覆盖** | 可能有实现路径但摘要重点不同 | 题名指向 overlap/decomposition | 未全文确认 | 非主重点 | 只用于受控实现，不作主贡献 |
| finer decomposition / schedule space | 可调 chunk | 细粒度 | **直接覆盖** | 题名直接涉及 decomposition | 未全文确认 | step-level chunk | 不能以“切更细”为新意 |
| DMA offload | 未作为主声明 | 未由摘要确认 | **直接覆盖** | 未全文确认 | 未由摘要确认 | 未由摘要确认 | 只可当候选路径/消融 |
| completion/visibility/notification 的合法 release curve | 未从摘要确认 | 未从摘要确认 | 未从摘要确认 | 未全文确认 | 未全文确认 | 不处理 AG-GEMM 依赖 | **必须实测后才可能成为差异** |
| 跨 NVIDIA 与海光的同一决策抽象 | 否，NCCL | 未从摘要确认 | 未从摘要确认 | 未全文确认 | 未从摘要确认 | NVIDIA H20/V100 heterogeneous | 需要两平台均有证据；不能仅口头声称 |

这张表的关键不是宣称“最后两行无人做过”，而是定义了本项目必须补的实验事实。任何一篇 P0 缺口全文若已覆盖 release-aware selection，应立即调整方向。

---

## 5. 可辩护的创新点应满足什么条件

### 5.1 第一创新：以“合法 release”而非带宽定义优化对象

候选指标：

```text
R_i = t_release(i) - t_issue
```

其中 `t_release(i)` 是第 `i` 个 consumer GEMM tile 被**正确同步地允许读取**的最早时刻。它不是 API return、未同步的 buffer poll，亦不是 whole collective completion。

该指标只有在下列现象被实测后才有贡献价值：

```text
1. T_done 相近但 R_i 曲线不同；或
2. busbw 最优、T_done 最优与 T_e2e 最优不同；并且
3. 差异超过重跑噪声，能够由正确 timeline 与 profiler 解释。
```

### 5.2 第二创新：将 capability、release 和 contention 联合用于选择

候选策略不只是 collective config，而是：

```text
pi = (backend/path, algo, protocol, channels,
      partition axis, q, window, notification mode, GEMM mode)
```

一个最小模型可以是：

```text
T_hat(pi) = T_startup(pi)
          + T_steady({R_i}, q, window)
          + T_drain(pi)
          + T_contention(pi, GEMM mode)
```

它必须做两件当前 AutoCCL/Fine-grain heuristic 不能替代的事：

1. 在 capability mask 后只评估语义上正确的候选；
2. 把 GEMM 切分造成的计算损失与通信路径带来的早释放/同步代价一起估计。

若这个模型最后只以 bytes 或 `M/K` 选择 q，则不够；那只是 FiCCO 类 shape heuristic 的一个变体。

### 5.3 第三创新：跨基座是“验证压力测试”，不是卖点本体

只有下列事实同时成立，NVIDIA+海光才有论文价值：

```text
相同数学工作负载和统一指标；
不同平台因 capability/transport/semantic/contension profile 得到不同最优策略；
同一抽象模型能解释两边的决策边界；
每边都优于合理固定基线，且不是由不公平的 GEMM 切分引起。
```

不应追求“同一套 NVSHMEM 代码不修改地跑海光”。这既不现实，也不是所需科学结论。正确的可移植层是 strategy contract，例如：

```text
payload(slice i) 完成
-> 顺序/可见性得到保证
-> ready(i) 被消费者观测
-> consumer GEMM tile i 可合法开始
```

NCCL/RCCL event、NVSHMEM、DUSHMEM 可以各自实现这一 contract，前提是语义和正确性已验证。

---

## 6. 研究问题、反例与基线

### 6.1 应回答的研究问题

| RQ | 具体问题 | 必需证据 |
|---|---|---|
| RQ1 | isolated `busbw` 和 `T_done` 能否预测 AG-GEMM 的 `T_e2e`？ | 同一候选集的 rank correlation、排序反转表、置信区间 |
| RQ2 | `R_i` 是否比 `T_done` 更能解释端到端结果？ | 合法事件/doorbell timeline、回归或误差比较、profiler trace |
| RQ3 | 哪些 capability/contension 特征决定策略切换？ | mask、primitive profile、GEMM efficiency curve、消融 |
| RQ4 | 同一抽象能否在 NVIDIA 与海光两侧工作？ | 两边同形状、同指标、分别验证的 decision boundary |
| RQ5 | 选择器是否值得其 calibration 开销？ | top-1/top-2、normalized regret、冷启动/在线开销 |

### 6.2 必须包含的诚实基线

| 基线 | 目的 |
|---|---|
| `serial-whole` | 完整 CCL + 整块 GEMM，给出无重叠基线 |
| `serial-same-partition(q)` | 分块通信 + **同一分块 GEMM** 串行执行，隔离 fragmentation 成本 |
| `overlap-fixed(q)` | 固定 q/window/channel 的典型 pipeline |
| `bandwidth-best` | 仅按 isolated `busbw` 选择的策略，直接检验 H1 |
| `T_done-best` | 仅按完整通信结束选择的策略，直接检验 release 指标价值 |
| `collective-config oracle` | 在固定 partition/GEMM 下穷举 CCL config 的最好结果，约束 AutoCCL 类空间 |
| `full oracle` | 所有合法 path/config/q/window/GEMM candidate 的穷举结果 |
| `proposed selector` | 少量 calibration 后的选择结果 |

对无法在目标平台复现的 FLUX、FiCCO、CoCoNeT 或 ResCCL，不应杜撰数值对比。可用公开代码且符合平台时才作为运行基线；否则应做**设计匹配的开源基线**并在论文中明确说明差异。

### 6.3 必须主动展示的反例

正例不足以让方法可信。论文至少应展示：

```text
过细 q 使 GEMM 效率下跌，selector 回退到较粗粒度；
高 busbw 的策略因为 release 更晚而输给另一个策略；
较早 release 的策略因为 contention 太强而输给简单策略；
某个 PGAS path 被 capability/correctness mask 排除；
NVIDIA 与海光在相同 shape 下选择不同策略，但模型能解释原因。
```

如果没有反例，选择器很可能只是对一个单一平台默认参数做重复测量。

---

## 7. 当前的实现边界

为避免与上述工作重合且控制个人研究工作量，建议保留以下边界：

```text
不重写 NCCL/RCCL data-plane kernel；
不开发新的 collective DSL、编译器、全局 topology route synthesizer；
不实现以 TB/SM allocation 为核心的新 CCL backend；
不把 DUMMA 称为 DMA/copy-engine；
不将“更细 chunk”本身作为优化结果；
不在未验证 memory order/notification 的情况下消费 remote payload；
不把跨厂商 GPU 放在同一个 collective group。
```

允许的实现范畴：

```text
CUDA C++ / HIP C++ 的小型 AG-GEMM harness；
NCCL/RCCL public APIs 和 stream/event；
CUTLASS/cuBLASLt 与 rocBLAS/hipBLASLt 的等价 GEMM 版本；
通过最小正确性实验后的 NVSHMEM/DUSHMEM path；
capability/release/contention profile 与轻量 selector。
```

---

## 8. 定题前仍必须完成的 P0 工作

### 文献 P0

1. 获得并精读 CoCoNeT DOI `10.1145/3567955.3567959` 全文；
2. 获得并精读 ResCCL DOI `10.1145/3718958.3750514` 全文；
3. 对 FLUX、FiCCO、AutoCCL 的全文各产出一页“问题、方法、候选控制项、同步语义、平台、已覆盖/未覆盖”表；
4. 检索 2026 年之后或投稿前最新的 `dependent collective GEMM overlap`、`completion semantics`、`PGAS adaptive scheduling` 工作；
5. 阅读时不要只看实验数字，优先确认它是否已经定义了 first-consumable data 或等价指标。

### 实验 P0

1. 固化海光设备真实型号，解决 K100_ai/K500SM_AI 标签冲突；
2. 用 stream-event CCL pipeline 先测 `R_i`、`T_done`、`T_serial(q)`、`T_overlap(q)`；
3. 在相同 q 下测 chunked GEMM 的 TFLOPS，修正已有 DUSHMEM exploratory comparison 的不公平基线；
4. 对 NVIDIA、海光各自验证 `put/get/fence/quiet/signal/wait` 的当前版本可用性与 payload correctness；
5. 在至少 10--20 个真实/代表性 AG-GEMM shape 中寻找可重复的排序反转；
6. 若论文最终承诺 A100，必须在 A100 重跑核心矩阵，RTX 4090 不能替代最终平台验证。

---

## 9. 结论

最新排重使路线更窄，也更可辩护：

> 不做通信库内部调参、通信 kernel 资源调度、DMA 粗细粒度分解或 collective schedule 合成；先实测并证明“合法分片释放与计算争用”这一变量是否让 bandwidth/T_done 的策略选择失效。只有这个现象存在，才实现 capability-aware adaptive selector，并在 NVIDIA 与海光平台上用同一抽象验证。

这是一条明确的 go/no-go 路线，而不是先写一个大系统再寻找故事。当前四卡 NCCL/RCCL 微基准已经给出候选空间、能力边界和平台差异；下一轮 AG-GEMM release experiment 才决定第二篇论文的核心创新是否成立。
