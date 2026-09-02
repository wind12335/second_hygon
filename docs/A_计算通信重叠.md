# A. 计算-通信重叠（Communication-Computation Overlap）—— 2025–2026 文献调研

- 调研日期：2026-09-02
- 覆盖子方向：(a) AllGather/ReduceScatter 等 collective 与 GEMM 的流水线/融合重叠（collective-GEMM overlap / AG-GEMM）；(b) NVSHMEM/rocSHMEM/DUSHMEM 等单边 PGAS 通信做细粒度 ready-signal 驱动的重叠；(c) TP/SP/PP 中的通信隐藏（Megatron overlap part、async TP、推理引擎通信重叠）
- 检索渠道：WebSearch 20+ 次不同 query（关键词覆盖 overlap+LLM、NVSHMEM+GEMM、collective GEMM fusion、sequence parallelism overlap、SC/MLSys/PPoPP/EuroSys/OSDI/ASPLOS/ISCA/HPCA+overlap、MoE fusion、FlashInfer/TRT-LLM 融合、DUSHMEM/国产基座等）；MLSys/PPoPP/EuroSys/OSDI/SC 官方 program 页与 GitHub README 交叉核实录用状态
- 相关性基准：本组第二篇方向 **"Beyond Bandwidth：完成语义/依赖释放感知的跨基座 Collective-GEMM 自适应重叠策略选择"**（中心假设：isolated collective 最优 ≠ 端到端 AG-GEMM T_e2e 最优；跨基座 = NVIDIA NCCL/NVSHMEM vs 海光 DCU RCCL/DUSHMEM）
- 与同目录文件分工：**B 文件** = 集合通信自动调优与策略选择轴（AutoCCL/Lagom/Theseus 等深条目在 B）；**C 文件** = GPU 单边/PGAS 通信与完成语义（Demystifying NVSHMEM、MoE-Hub、NCCLX 等深条目在 C）；**本文件** = 重叠本体：kernel 级细粒度重叠、重叠调度/策略、并行维度（TP/SP/PP/MoE）通信隐藏。交叉条目此处只留一段话 + 指针。
- 标注约定：只收录实际检索确认存在的论文；录用状态经官方 program 页 / GitHub README / ACM DL 核实；未能核实处标 **待确认**。

---

## 第一梯队：kernel/tile 级细粒度 collective-GEMM 重叠（子方向 a+b 核心，按相关度排序）

### 1. Syncopate（前称 AutoOverlap）: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap
- **Venue/年份**：**OSDI 2026 录用**（USENIX 官方 presentation 页确认）；arXiv 2601.20595（2026-01，v1 名 AutoOverlap）
- **链接**：https://arxiv.org/abs/2601.20595 ｜ https://www.usenix.org/conference/osdi26/presentation/qiang
- **核心思路**：编译器 + 运行时，把细粒度计算-通信重叠自动化到**单个融合 kernel 内部**。引入 communication chunk 抽象，**解耦通信粒度与 kernel 结构、后端机制**；底层统一抽象三类原语：async_memcpy（CE）、async_send/recv（NVSHMEM）、load/store（SM 同步路径），按 chunk 状态机交错发射，实现"声明重叠意图、编译器选实现"。
- **与本方向关系**：★★★★★ (a)(b) 的 2026 最新占位者。其"三原语统一抽象"与本组 triton-distributed 多后端 shmem 抽象层同构，是第二篇迁移工作的**直接对照系**。但注意：其重心是"把重叠自动化"而非"在多个重叠策略/完成语义之间做端到端选择"；且仅 NVIDIA 基座、不跨厂商。与"完成语义感知策略选择"正交——它可以成为我们的下层机制之一。
- **Repo**：https://github.com/tie-pilot-qxw/syncopate（实验性编译器，含 artifact 复现说明）
- **交叉引用**：C 文件 #8（单边语义视角）

### 2. FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation
- **Venue/年份**：**EuroSys 2026 录用**（官方 GitHub README 确认"accepted by EuroSys'26"）；arXiv 2504.19519（2025-04）
- **链接**：https://arxiv.org/abs/2504.19519 ｜ https://github.com/infinigence/FlashOverlap
- **核心思路**：**signaling-based 设计**：CUTLASS GEMM kernel 在部分输出 tile 算完时主动发 signal 触发对应通信（all-gather/reduce-scatter），不需要打断计算主体、不占额外 kernel。三特性：tile-wise overlapping、interference-free computation（计算不被通信拖慢）、communication agnosticism（不绑定具体通信实现）。可达理论重叠收益的 69–98%，端到端最高 1.65×。
- **与本方向关系**：★★★★★ **"ready-signal 驱动重叠"最直接的公开实现**——正是子方向 (b) 的机制原型。但其 signal 机制绑定 CUTLASS/NVIDIA GEMM，单一 tile-wise 模式，不做策略选择、不做多基座完成语义适配。可作为"release 点显式化"的引用支点：我们研究的是 release 语义不同（NVSHMEM quiet/flush vs stream event vs CE 完成通知）时端到端选哪种结构。
- **Repo**：https://github.com/infinigence/FlashOverlap（Infinigence，开源）

### 3. TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives
- **Venue/年份**：**MLSys 2025 录用**（ByteDance-Seed/Triton-distributed GitHub News "05/12/2025 accepted by MLSys 2025"）；arXiv 2503.20313
- **链接**：https://arxiv.org/abs/2503.20313 ｜ https://github.com/ByteDance-Seed/Triton-distributed
- **核心思路**：在 Triton 编译器中扩展 **tile 级通信原语**（与 tile-centric 计算原语同构），让编译器自动生成重叠 kernel（例：inter-node AllGather 与 Flash-Attention 的重叠、AG-GEMM），在 Hopper 上接近手工融合 kernel 性能。是 triton-distributed 项目的论文版。
- **与本方向关系**：★★★★★ (a) 的编译器路线代表，且**后端已覆盖 NVSHMEM（NVIDIA）+ rocSHMEM（AMD）+ NCCL**——是目前公开工作中"跨后端统一 shmem 抽象"走得最远的。第二篇"迁移到国产基座（DUSHMEM 后端）"的直接基线与出发点。缺口：它优化的是单个重叠 kernel 的生成，不做"策略选择"，也没有端到端语义感知。
- **Repo**：https://github.com/ByteDance-Seed/Triton-distributed（开源，活跃）

### 4. Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler
- **Venue/年份**：arXiv 2504.19442（2025-04，**仍在 CoRR，正式 venue 待确认**；清华 Zhai 组 + ByteDance，Size Zheng 等）
- **链接**：https://arxiv.org/abs/2504.19442
- **核心思路**：首个原生支持重叠优化的分布式 AI 编译器：Python 层 task/chunk 描述 + shmem 抽象层（NVSHMEM/rocSHMEM/NCCL 多后端）+ 依赖驱动调度，自动生成 AllGather-GEMM 等重叠程序（论文图 4 即 inter-node AG-GEMM）。
- **与本方向关系**：★★★★★ 本组第一篇工作的直接基础（memory 中的 triton-distributed-arch）；第二篇的"多后端→双基座"扩展叙事必须引用并区别于它：它做了**后端可插拔**，但没做**基于基座完成语义差异的策略选择**，也没验证国产 GPU 后端。
- **Repo**：https://github.com/ByteDance-Seed/Triton-distributed

### 5. COMET: Fine-grained Computation-Communication Overlapping for Mixture-of-Experts
- **Venue/年份**：**MLSys 2025 录用，Outstanding Paper Honorable Mention**（MLSys 官方 awards 页确认）
- **链接**：https://mlsys.org/virtual/2025/poster/3246 ｜ https://openreview.net/pdf?id=fGgQS5VW09
- **核心思路**：针对 MoE 的 dispatch/combine 通信：通过**细粒度数据依赖分析 + 通信任务重调度/重排**，把 token 级通信与 expert 计算、内存搬运精确重叠，消除 SM 空转；与 DeepEP 等组合显著提升 MoE 训练吞吐（NVIDIA 侧已集成进 Megatron 生态）。
- **与本方向关系**：★★★★☆ (a)(c) MoE 侧细粒度重叠的标杆（获奖佐证该轴热度）。其"依赖分析决定何时可通信/可计算"与我们的 release 感知同源，但目标是 MoE 调度正确性+吞吐，不涉及跨基座完成语义。
- **Repo**：开源（NVIDIA Megatron-COMET，已并入 Megatron-LM 生态；具体 repo 链接待确认）

### 6. FLUX: Fast Software-based Communication Overlap on GPUs through Kernel Fusion
- **Venue/年份**：arXiv 2406.06858（**2024 年末里程碑**，MSRA+PKU，Li-Wen Chang 等；正式 venue 待确认）
- **链接**：https://arxiv.org/abs/2406.06858 ｜ https://github.com/bytedance/flux
- **核心思路**：纯软件 chunk 化 kernel fusion：把 all-gather/reduce-scatter 拆块嵌入 FP8 GEMM（生产者-消费者式逐块发送），融合 kernel 内可重叠高达 96% 通信，训练较 Megatron-LM 最高 1.24×。
- **与本方向关系**：★★★★☆ (a) 的奠基之作；2025–2026 一批 chunk-scheduling 工作（FlashOverlap/Syncopate/Lagom/CommFuse）的共同前身。作为背景引用。
- **Repo**：https://github.com/bytedance/flux（开源，FP8 dense/MoE）
- **交叉引用**：B 文件 #10、C 文件 #13

### 7. Design Space Exploration of DMA-based Finer-Grain Overlap（DSE）
- **Venue/年份**：arXiv 2512.10236（2025-12，**venue 待确认**；标题以 arXiv 页为准）
- **链接**：https://arxiv.org/html/2512.10236
- **核心思路**：系统探索用 **DMA（CE）路径做 GEMM 与 all-gather/all-to-all 等 collective 的更细粒度重叠**的设计空间（vs SM 路径 kernel fusion），量化不同粒度/路径组合的收益。
- **与本方向关系**：★★★★☆ 直接对应"**通信路径/完成机制差异影响重叠收益**"这一我们的核心变量（CE 异步完成 vs SM 同步 vs NVSHMEM 单边），是最接近"按机制选策略"的探索性工作；但仍在单一硬件（NVIDIA）上，且是 kernel 级 DSE 而非端到端策略选择框架。
- **Repo**：待确认

### 8. CommFuse: Hiding Tail Latency via Communication Decomposition and Overlap
- **Venue/年份**：arXiv 2604.24013（2026，**venue 待确认**）
- **链接**：https://arxiv.org/html/2604.24013
- **核心思路**：指出 SOTA overlap 方法存在**尾部延迟**（最后一块通信/计算收尾串行化），通过通信分解（decomposition）+ 重组的重叠消除 tail。
- **与本方向关系**：★★★☆☆ (a) 轴上 2026 的新问题定义（tail effect 此前多见于 NVIDIA 论坛与 FSDPv2 dev-discuss 的工程抱怨：重叠使 MFU 从 ~75–80% 掉到 ~45–50%）。佐证"重叠结构选择影响端到端"的动机，但解法是又一个固定 kernel 方案。
- **Repo**：待确认

### 9. Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping
- **Venue/年份**：arXiv 2409.15241（**2024 年末里程碑**）
- **链接**：https://arxiv.org/abs/2409.15241
- **核心思路**：通用张量切片方案：把输入/权重切分成 slice，让 partial collective（分块 AG/RS）与依赖计算逐片重叠，理论上隐藏绝大部分 TP/SP 通信。
- **与本方向关系**：★★★☆☆ (a) 中"tensor slicing + partial collective"通用框架的代表作（与 MegaScale/Concerto/Centauri 同族），为 2025 年 chunk 化工作的直接前置。
- **Repo**：待确认

---

## 第二梯队：单边 PGAS / 持久 kernel 融合与推理侧重叠（子方向 b + c 推理）

### 10. FlashMoE: Fast Distributed MoE in a Single Kernel
- **Venue/年份**：**NeurIPS 2025 录用**（poster 119124）
- **链接**：https://arxiv.org/abs/2506.04667 ｜ https://neurips.cc/virtual/2025/poster/119124
- **核心思路**：首个**全融合分布式 MoE 算子**：expert 计算与跨 GPU 通信全部融进一个 persistent GPU kernel，用 **device-initiated 单边通信**（GPU 线程直接发起远端读写，不经 host）；消除 kernel 边界后 tensor core 利用率大幅提升，通信效率 >89%（约为基线 4×）。
- **与本方向关系**：★★★★☆ (b) 的极端形态——把"通信完成后立即可算"做到单 kernel 内自然成立（生产者-消费者在同一 kernel 内）。说明单边 PGAS + kernel fusion 是重叠的终局形态之一；但实现极度绑定 NVIDIA 架构，无策略选择可言。
- **Repo**：https://github.com/osayamenja/FlashMoE（开源）

### 11. Harnessing Inter-GPU Shared Memory for Seamless MoE Communication-Computation Fusion（CCFuser）
- **Venue/年份**：**PPoPP 2025 录用**（ACM DL 确认，DOI 10.1145/3710848.3710868）
- **链接**：https://dl.acm.org/doi/10.1145/3710848.3710868
- **核心思路**：基于 inter-GPU shared memory（单边共享地址空间）做 MoE serving 的通信-计算融合（框架名 CCFuser），在 token 调度上重叠 expert 计算与 inter-GPU 通信，避免 GPU 空转。
- **与本方向关系**：★★★★☆ (b)(c) 中"共享内存抽象承载融合重叠"的会议级代表；与 FlashMoE 同轴（one-sided + fusion），区别在 serving 场景与框架化。
- **Repo**：待确认

### 12. TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference
- **Venue/年份**：**MLSys 2026 录用（oral）**（mlsys.org/virtual/2026/oral/3744 确认，PMLR v8）；arXiv 2505.11329
- **链接**：https://arxiv.org/abs/2505.11329 ｜ https://github.com/microsoft/tokenweave
- **核心思路**：TP 推理的重叠：把 token 序列切分为子集，一个子集通信时另一个子集计算；配合自定义 fused kernel（AR+RMSNorm，TMA + NVSHMEM 融合，通信 stall 由计算 warp 自动填充），即使 token 数小到 1024/单 token decode 也有效；prefill 相比 Megatron-LM 延迟约 -64%，fused kernel 1.34–1.39×。
- **与本方向关系**：★★★★☆ (c) 推理侧重叠的当前 SOTA（Microsoft）。其 token 切分思想与训练侧 chunk 化对偶；同样绑定 NVIDIA（TMA/NVSHMEM），不做基座选择。
- **Repo**：https://github.com/microsoft/tokenweave（开源）
- **交叉引用**：B 文件 #51、C 文件 #21

### 13. AsyncTP（异步张量并行，PyTorch/TorchTitan 生产实现）
- **Venue/年份**：**无独立论文**——官方博客 + TorchTitan 论文章节（arXiv 2410.06511 §2.2.3）；NVIDIA/Microsoft 合作，约 2024 年落地
- **链接**：https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 ｜ https://arxiv.org/abs/2410.06511
- **核心思路**：把 TP 的 GEMM 分块（matmul over split K 或 column-parallel 分块），前块结果即可发起 async all-gather/reduce-scatter，与后块计算重叠；叠加 stream 调度使 overlap 延伸到相邻 sub-layer 的 partial wave。
- **与本方向关系**：★★★☆☆ (c) 中 Megatron "overlap part" 思想的 PyTorch 官方化/产品化，属于工程基线（我们实验对照物），也说明该轴已成生产标配。
- **Repo**：pytorch/pytorch（torch.distributed.tensor 的 async TP 实现）

### 14. MoE-Hub: Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication
- **Venue/年份**：**ISCA 2026 录用**；arXiv 2605.05888
- **链接**：https://arxiv.org/abs/2605.05888
- **核心思路**：指出 MoE 动态 token-expert 映射与 GPU 静态地址中心通信模型的**抽象失配**是重叠难写的根因；提出 Load-Dispatch-Apply（LDA）范式，NVL72 上用 TMA 优化缓冲与硬件加速路径，对比 SM-based UCX/NVSHMEM 实现。
- **与本方向关系**：★★★★☆ 论证"**通信基座抽象决定重叠软件复杂度**"——与我们"完成语义感知抽象"的动机同源（硬件路线）。
- **Repo**：待确认
- **交叉引用**：C 文件 #9（深条目）

### 15. Collective Communication for 100k+ GPUs（NCCLX）
- **Venue/年份**：arXiv 2510.20171（2025-10，venue 待确认）
- **链接**：https://arxiv.org/html/2510.20171
- **核心思路**：面向 10 万卡级（MNNVL/多机 NVLink 域）的通信库：用 **NVSHMEM 单边通信**构建 collective，**不占用 SM 的 streaming 资源**即可与计算重叠。
- **与本方向关系**：★★★☆☆ 基座层走向"单边、不占 SM、天然可重叠"的证据（与 Demystifying NVSHMEM、NCCL GIN 一起构成 C 文件主线）。
- **交叉引用**：C 文件 #16

---

## 第三梯队：重叠调度与策略层（重叠"怎么选/怎么排"，与中心假设最近）

### 16. Mist: Efficient Distributed Training of LLMs via Memory-Parallelism Co-Optimization
- **Venue/年份**：**EuroSys 2025 录用**（Session 9.2 LLM Training；ACM 3689031.3717461）；arXiv 2503.19050
- **链接**：https://dl.acm.org/doi/10.1145/3689031.3717461 ｜ https://arxiv.org/abs/2503.19050
- **核心思路**：**overlap-centric 调度**：以最大化"计算-通信-内存操作"三重重叠为目标，协同编排并行策略（TP/PP/SP）与内存优化（offload/recompute/checkpoint）的相对顺序；比手工优化系统平均 1.28×/1.27×（最高 1.73×/2.04×）。
- **与本方向关系**：★★★★☆ "以 overlap 为中心做全局策略编排"的 EuroSys 2025 代表——但它是**算子图/内存级**调度，通信被当作黑盒（NCCL 语义不感知），也不跨基座。与我们"通信基座完成语义作为一等变量"互补。
- **Repo**：待确认

### 17. Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training
- **Venue/年份**：arXiv 2602.20656（2026-02，AutoCCL 原班人马 USTC Guanbin Xu/Cheng Li 等；**venue 待确认**）
- **链接**：https://arxiv.org/abs/2602.20656
- **核心思路**：（组内已确认情报）核心发现：**isolated 调参 ≠ 端到端最优**——AutoCCL 激进调参（NC 8→61）在 compute-bound 重叠场景因 SM/带宽竞争反而端到端劣化为 NCCL 的 0.87×；提出统一 overlap 代价模型（makespan = max(Σ计算, Σ通信)）、两类竞争建模与线性搜索，对 NCCL 的 NC/NT/chunk 协同调参，端到端比 NCCL/AutoCCL 提升 1.07–1.33×/1.03–1.27×（A40，FSDP/TP/EP）。
- **与本方向关系**：★★★★★ **中心假设的实验证明者 + 最近竞品**。其解空间限于 NCCL 资源参数轴、overlap 结构固定 stream 级、仅 NVIDIA；"分片何时可被 GEMM 合法消费（release/完成语义）"维度与跨基座均为空白。详见 B 文件 #1。
- **Repo**：论文称将开源（截至调研日未见）
- **交叉引用**：B 文件 #1（深条目）、C 文件 #20

### 18. AutoCCL: Automated Collective Communication Tuning（背景条目）
- **Venue/年份**：**NSDI 2025 录用**
- **链接**：https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin
- **一句话**：NCCL fork + tuner plugin 对六参数在线调优，**目标函数是孤立 collective 的 busbw/T_done**——正是被 Lagom 推翻的目标函数；作为我们"isolated vs 端到端"叙事的靶子。
- **Repo**：https://github.com/gbxu/autoccl
- **交叉引用**：B 文件 #2（深条目）

### 19. Helix: Automating Communication-Computation Overlap with Dependency Graph Scheduling
- **Venue/年份**：SIGCOMM 2024（**2024 里程碑**；清华 Zhiyao Li 等）
- **链接**：https://yezhisheng.me/post/helix/（技术笔记；ACM DL 条目待确认）
- **核心思路**：编译器式依赖图调度：把 n-D 模型并行下所有通信与计算建成依赖图，自动乱序/流水，最大化通算重叠（对 Megatron 手工 overlap part 的泛化）。
- **与本方向关系**：★★★☆☆ "自动找重叠调度"的图调度路线（与 Tessel 同族）；通信仍为黑盒算子，无完成语义感知。
- **Repo**：待确认

### 20. Tessel: Boosting Distributed Execution of Large DNN Models via Flexible Schedule Search
- **Venue/年份**：**HPCA 2024 录用**（Microsoft，Saeed Maleki 等；**2024 里程碑**）
- **链接**：https://www.semanticscholar.org/（检索"Tessel HPCA 2024"；ResearchGate 379518508）
- **核心思路**：两阶段（重复模式构造 + schedule 补全）自动搜索分布式训练/推理的算子放置与调度，使通信被计算覆盖。
- **与本方向关系**：★★★☆☆ "schedule 搜索"轴的会议级先例；搜索空间是算子顺序/放置，不进入通信基座内部语义。
- **Repo**：待确认

### 21. Optimizing Intra-Layer Parallel Communication for LLM Training on Fully-Connected Mesh GPU Topology
- **Venue/年份**：**HPCAsia 2026 录用**（ACM 3773656.3773675；Tokyo Tech Hosoki/Sato/Endo）
- **链接**：https://dl.acm.org/doi/10.1145/3773656.3773675
- **核心思路**：在全连接 mesh 拓扑（光互联 Tsubame 类系统）上做层内并行通信优化：**partial-collective 分解**（把 collective 切片成子集合）配合分块流水重叠，继承 Domino/MegaScale/Concerto 思路并适配 mesh 拓扑。
- **与本方向关系**：★★★☆☆ (a) 轴在非常规拓扑上的延展；说明"chunk 化 partial collective + 重叠"已被当作通用配方，但策略选择仍手工。
- **Repo**：待确认

### 22. ICCL: An Efficient, Reliable and Observable Collective Communication Library
- **Venue/年份**：arXiv 2510.00991（2025-10，venue 待确认；阿里系工作，待确认作者归属）
- **链接**：https://arxiv.org/html/2510.00991
- **核心思路**：通信库支持把更多 SM 资源用于 P2P 通信以加速前/反向中的重叠（资源分配轴），兼顾可观测性。
- **与本方向关系**：★★☆☆☆ 通信库资源轴（与 Lagom/Resource-aware 同轴），佐证资源竞争轴在拥挤。
- **Repo**：待确认

### 23. DITRON: A Flexible and Versatile Distributed Tensor Program Optimizer
- **Venue/年份**：OpenReview 在审/公开 PDF（id qLfScqAzkd，**venue 待确认**）
- **链接**：https://openreview.net/pdf?id=qLfScqAzkd
- **核心思路**：Triton 系分布式张量程序优化器（文中以 TileLink 为相关工作）：编译期优化含通信的分布式 kernel（含重叠生成），目标是通用分布式张量程序性能。
- **与本方向关系**：★★★☆☆ (a) 编译器轴的并行竞品（尚未见正式录用），说明"Triton+通信+优化"赛道 2025–2026 快速填充。
- **Repo**：待确认

### 24. Resource-aware Computation-Communication Overlap for Multi-GPU ML Workloads
- **arXiv 2606.09200（2026）**：用可移植运行时旋钮（计算 kernel occupancy shaping + 通信流提优先级）缓解通算资源竞争，执行时间 -25.5%。资源竞争轴新条目，详见 C 文件 #12。
- **DeFT（FGCS 2026，arXiv 2503.16815）**：树形通信-计算重叠的 decision framework（细节待确认）——**"决策框架"措辞与本组方向最接近的标题，需精读确认其决策对象**（B 文件 #56）。
- **HiPIPE（IWQoS 2025）**：adaptive-chunk 流水线重叠（细节待确认；B 文件 #54）。
- **Concerto（ASPLOS 2025）/ Centauri（ASPLOS 2024）**：通信切分+自动调度的代表（B 文件 #57/#58）。
- **CAIS（Zhiyao Li 等，venue 待确认）**：改进 collective 使端到端 LLM 训练较 NVLS-enabled SOTA 平均 1.38×（Helix 同组后续；作者主页 https://ziolee.xyz/publication/）。

---

## 第四梯队：SP/长序列/流水线维度通信隐藏（子方向 c 训练+推理）

### 25. BurstEngine: an Efficient Distributed Framework for Training LLMs on Long-Sequence Data
- **Venue/年份**：**SC 2025 录用**（pap290；THUNLP）；arXiv 2509.19836
- **链接**：https://arxiv.org/abs/2509.19836 ｜ https://github.com/thunlp/BurstEngine
- **核心思路**：1M+ token 长序列训练框架：通信-计算重叠 + sequence-level selective checkpointing，长序列下系统级优化。
- **与本方向关系**：★★★☆☆ (c) SP 维度 SC25 录用代表；长序列场景 AG-GEMM/SP 重叠的需求来源。
- **Repo**：https://github.com/thunlp/BurstEngine（开源）

### 26. Training Ultra Long Context Language Models with Fully Pipelined Distributed Transformers（TPDT）
- **Venue/年份**：**MLSys 2025 录用**（DistFlashAttn 谱系，UCSD Hao Zhang 组）
- **链接**：https://proceedings.mlsys.org/paper_files/paper/2025/file/d5a655b8b373737b4f2aea8f78e5e754-Paper-Conference.pdf
- **核心思路**：序列并行下把 KV/attention 分块做成完全流水线的分布式执行，隐藏跨卡通信与负载不均。
- **与本方向关系**：★★★☆☆ (c) SP 流水线隐藏通信的会议级代表。
- **Repo**：待确认

### 27. Tetris: Chunkwise Dynamic Sequence Parallelism for Long-Context LLM Serving
- **Venue/年份**：arXiv 2511.06247（2025-11，venue 待确认）
- **链接**：https://arxiv.org/html/2511.06247
- **核心思路**：面向 serving 的 chunkwise 动态序列并行（CDSP），按上下文动态分块调度，隐藏长上下文推理的序列维通信。
- **与本方向关系**：★★☆☆☆ (c) 推理 SP 侧新条目。
- **Repo**：待确认

### 28. ISO: Overlap of Computation and Communication within Sequence（2024 里程碑）
- **arXiv 2409.11155**：推理侧 op 级序列分块重叠（attention/GEMM 与 SP 通信交错的 schedule）。
- **LoongServe: Elastic Sequence Parallelism（SOSP 2024 录用，arXiv 2404.09526）**：弹性序列并行，长上下文 serving 5.81×；其 KV 重分布涉及通信但主打弹性调度。
- 以上两条作为 2024 上下文收录。

### 29. MEPipe: Democratizing LLM Training with Memory-Efficient Slice-Level Pipelining
- **Venue/年份**：**EuroSys 2025 录用**
- **链接**：https://2025.eurosys.org/accepted-papers.html
- **核心思路**：slice 级流水线化协调内存受限环境下的训练（与 Mist 同 session）。
- **与本方向关系**：★★☆☆☆ 重叠思想的内存化变体（细节待确认）。
- **Repo**：待确认

### 30. Cross-region Model Training with Communication-Computation Overlapping and Delay Compensation
- **Venue/年份**：arXiv 2504.17672（2025-04，venue 待确认）
- **链接**：https://arxiv.org/abs/2504.17672
- **核心思路**：跨地域（广域网）训练：通算重叠 + 梯度延迟补偿（overlap 引入 staleness 的算法侧配套）。
- **与本方向关系**：★★☆☆☆ 说明 overlap 在 WAN 场景的算法代价维度（我们 scale-up/scale-out 场景无此问题）。
- **Repo**：待确认

---

## 第五梯队：推理引擎生态与工业实践（非论文，证明"缺乏统一抽象层"的现状证据）

1. **FlashInfer comm 模块**（https://docs.flashinfer.ai/api/comm.html）：trtllm_allreduce_fusion 系列（IPC/symmetric memory 工作区、MNNVL one-shot allreduce fusion）；**issue #1605（open）**：请求 TokenWeave 式 fused AR+GEMM，讨论指出 multimem 需要对称内存、应由 FlashInfer 统一分配——**fused AR+GEMM 在主流推理 kernel 库仍是未完成的 feature**。
2. **SGLang issue #8728**：基于 **Triton-Distributed** 实现 FP8 GEMM+AR fusion（grouped GEMM + collective fusion，Qwen 类模型）——triton-distributed 抽象已被下游推理引擎用作融合底座（对我们第二篇的工程影响力佐证）。
3. **vLLM collective_fusion**（v0.12.0 docs）：torch.compile pass 将 AR+RMSNorm(+residual) 替换为 FlashInfer 融合实现——目前只覆盖 norm 级融合，未到 GEMM。
4. **PyTorch Symmetric Memory**（https://docs.pytorch.org/docs/stable/symmetric_memory.html + dev-discuss #2798）：把 NVLink/NVLS 多播、NVSHMEM 单边 RDMA 暴露成对称内存 API——"对称内存成为细粒度重叠基础设施"的官方信号（论文层面由 Demystifying NVSHMEM 承接，见 C 文件 #1）。
5. **NVIDIA TRT-LLM MultiShot AllReduce**（developer blog：NVSwitch multicast 3× AR）与 **AWS Neuron compute-comm overlap** 文档：硬件/厂商侧路径。
6. **Stanford Hazy Research "One Kernel for All Your GPUs"（2025-09 博客）**：NVLink/NVSwitch 可编程性上做融合通信 kernel 的探索性工作（PGL），无正式论文——**待确认**后续发表。
7. **NVIDIA 开发者论坛 tail effect 报告 + PyTorch dev-discuss FSDPv2 帖**：重叠使 GEMM MFU 75–80%→45–50% 的工程实证，作为"重叠有代价、需策略选择"的社区证据（论文层面由 arXiv 2507.03114 量化，ISPASS 2025，见 C 文件 #11）。

---

## 国产基座与跨厂商检索情况（对第二篇空白判断的关键输入）

1. **DUSHMEM / 海光 DCU 重叠工作：英文公开文献检索为零。** WebSearch 以 "DUSHMEM Hygon DCU shmem overlap"、"DUSHMEM arxiv"、"Hygon DCU OpenSHMEM deep learning" 等多组关键词检索，**未检索到任何 DUSHMEM 公开论文**（仅有飞桨 DCU 适配文档、HAMi 共享调度等行业资料）——与 C 文件 #24 的结论一致：国产基座上的细粒度通算重叠在公开文献层面是空白。
2. **JCST 海光 DCU 论文**（DOI 10.1007/s11390-025-4285-7，PDF 预览标注 "For Review Only"）：报告在 Hygon DCU 上达到 113.77 GB/s 的通信相关工作；**标题/主题/是否涉及重叠待确认**（检索仅见预览片段）。
3. **跨厂商统一抽象的公开进展**：Triton-distributed/TileLink 覆盖 **NVIDIA(NVSHMEM)+AMD(rocSHMEM)+NCCL**；Syncopate 抽象 CE/NVSHMEM/SM 三原语但仅 NVIDIA；**没有任何公开工作覆盖国产 GPU（DCU/DUSHMEM 等）并与 NVIDIA 基座统一到同一重叠抽象下**。规格层面 "Toward a Unified GPU-Aware OpenSHMEM Specification"（arXiv 2607.08006，见 C 文件 #15）在推动 GPU-centric OpenSHMEM 标准化，但无 LLM 训推重叠落地。
4. **AMD 侧动态**：AMD 2025 年为 DeepEP 提供 ROCm 后端（替代 NVSHMEM 依赖，SemiAnalysis 报道）；AMD Developer Challenge 2025 有 AllGather-GEMM fused kernel 赛题（Yotta Labs 解读）——ROCm 生态在补齐，但仍是"单厂商各自为政"。
5. **Huawei CloudMatrix384 serving 论文**（C 文件 #23）：国产算力系统级工作的代表，但通信是总线级汇聚、非 shmem 单边重叠路线。

---

## 研究空白观察

### 1. "通信基座完成语义 / release 感知的端到端重叠策略选择"——没有公开工作占据
- **最接近的三条线及其边界**：
  - **Lagom（arXiv 2602.20656）**：证明了 isolated 调参 ≠ 端到端最优（我们的中心假设），但其搜索空间是 **NCCL 资源参数轴**（NC/NT/chunk），overlap 结构固定为 stream 级，无"分片何时可被 GEMM 合法消费"的语义维度，仅 NVIDIA；
  - **FlashOverlap（EuroSys 2026）**：signaling/ready-trigger 机制与"release 感知"最接近，但绑定 CUTLASS+NVIDIA、单一 tile-wise 模式、**不做策略选择**；
  - **Mist（EuroSys 2025）/ Helix（SIGCOMM 2024）/ Tessel（HPCA 2024）/ Concerto（ASPLOS 2025）**：做"调度/放置级"自动重叠，但通信一律是黑盒算子，不进入基座完成语义（quiet/flush 语义 vs stream event vs CE 完成通知 vs ready-flag）。
- **结论**：（基座完成语义差异）×（重叠结构选择）×（端到端 T_e2e 目标）这一交叉格**无人占据**；B 文件对"策略选择轴"（Theseus/TuCCL 等）的结论同样支持：它们的自变量是 workload/硬件演化或调参维度，均非"与下游 GEMM 的依赖释放关系"。
- 注意一个**术语风险**：DeFT（FGCS 2026）标题含 "decision framework"，必须精读确认其决策对象，避免 novelty 表述撞车。

### 2. 跨 NVIDIA + 国产 GPU 双基座统一重叠抽象——公开文献空白
- TileLink/Triton-distributed 做到了 NVIDIA+AMD+NCCL 多后端，**未触及国产 GPU**；DUSHMEM 在英文文献中检索不到（本文件与 C 文件双重检索确认）；
- 因此"在同一重叠编程抽象下接入 DUSHMEM/RCCL，并按基座完成语义差异做端到端策略选择"没有先例；这正是第二篇"跨基座"叙事的空地。风险在于：这是**工程可达性**问题（triton-distributed 抽象层已预留后端位），需以"语义差异导致的策略反转"实验（对应 PhaseA 的 STRONG_REVERSAL 证据链）作为学术贡献而非仅移植报告。

### 3. 重叠的"代价"已被量化，但没人把它变成"选择依据"
- ISPASS 2025 表征（arXiv 2507.03114）：重叠使计算平均 -18.9%（最大 -40%），且 NVIDIA/AMD 干扰形态不同——**跨厂商完成机制差异影响重叠收益的直接证据**，但该文止步于测量；
- 各家单点缓解：FlashOverlap（interference-free）、Resource-aware（arXiv 2606.09200，occupancy shaping）、ICCL（SM 分配）、CommFuse（tail latency）——**都在修一种结构，没有人在结构之间做感知式选择**。

### 4. 推理侧 fused collective-GEMM 缺统一抽象层（时间窗口证据）
- FlashInfer #1605（fused AR+GEMM 仍是 open feature request）、SGLang 自行基于 Triton-Distributed 实现（#8728）、vLLM 只到 AR+RMSNorm——**kernel 库层没有统一的"通信-GEMM 融合 + 完成语义"接口**，第二篇的抽象若同时覆盖训/推两侧更有说服力。

### 5. 拥挤轴规避（写作层面）
- 2025–2026 已快速拥挤的轴：资源竞争参数轴（Lagom/Resource-aware/ICCL）、编译器自动重叠轴（Syncopate/TileLink/DITRON）、MoE 融合轴（FlashMoE/CCFuser/COMET/MoE-Hub）、调度图轴（Mist/Helix/Tessel/Concerto）；
- 空着的轴：**完成语义结构轴 + 跨基座（含国产）+ 端到端目标**——与 B/C 文件结论互相印证。

---

## 附：检索记录（WebSearch 主要 query，≥10 次）

1. communication computation overlap LLM training 2025 arxiv
2. NVSHMEM one-sided communication GEMM overlap distributed training 2025
3. collective GEMM fusion all-gather overlap 2025 paper
4. sequence parallelism communication overlap arxiv 2025 LLM
5. SC25 paper communication computation overlap distributed training supercomputing
6. async tensor parallelism LLM inference communication overlap ASPLOS MLSys 2025
7. MLSys 2025 accepted papers communication overlap
8. PPoPP 2025 communication computation overlap GEMM accepted paper
9. OSDI 2025 / EuroSys 2025 LLM training communication overlap accepted paper
10. TokenWeave distributed LLM inference compute communication overlap
11. Syncopate chunk-based compute communication overlap OSDI 2026
12. rocSHMEM AMD GPU overlap GEMM training kernel 2025
13. BurstEngine SC25 long sequence LLM training overlap
14. NVSHMEM delayed scaling all-gather GEMM FP8 overlap kernel
15. AutoCCL communication library cost model LLM training overlap 2025
16. FlashMoE one-sided communication fused kernel MoE NeurIPS 2025
17. "Harnessing Inter-GPU Shared Memory" MoE fusion PPoPP 2025
18. PyTorch symmetric memory one-sided communication 2025
19. ISCA 2025 HPCA 2025 communication computation overlap accelerator LLM
20. FlashInfer TensorRT-LLM fused allreduce GEMM symmetric memory overlap inference 2025
21. LoongServe Tessel elastic sequence parallelism communication overlap
22. Lagom distributed LLM training overlap AutoCCL 2602.20656
23. adaptive overlap strategy selection LLM training autotuning 2025/2026
24. DUSHMEM Hygon DCU shmem communication overlap training paper
25. Triton-distributed overlapping kernels accepted venue
26. "Characterizing Compute-Communication Overlap" venue 2507.03114
27. "Optimizing Intra-Layer Parallel Communication" Domino partial collectives
28. FlashOverlap infinigence accepted conference（确认 EuroSys 2026）

（共 28 次 query；另以 MLSys/PPoPP/EuroSys/OSDI/SC 官方页、GitHub README、ACM DL 交叉核实录用状态。）
