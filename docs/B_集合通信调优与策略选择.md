# B. 集合通信自动调优与自适应策略选择（2025–2026 文献调研）

- 调研日期：2026-09-02
- 覆盖子方向：(a) NCCL/RCCL algorithm/protocol/channel/chunk 自动调优；(b) 能力感知策略选择与解析代价模型；(c) 异构 GPU 集群通信调度；(d) 跨厂商/跨通信基座统一抽象与选择框架；(e) GPU 通信库内部机制（内核设计、拓扑感知）
- 检索渠道：DBLP API（177 篇 2024–2026 候选）、OpenAlex API、WebSearch（≥10 次不同 query）、arXiv 全文抓取（webReader）
- 相关性基准：本组第二篇方向 **"Beyond Bandwidth：完成语义/依赖释放感知的跨基座 Collective-GEMM 自适应重叠策略选择"**（中心假设：isolated collective 最优 ≠ 端到端 AG-GEMM T_e2e 最优；跨基座 = NVIDIA NCCL/NVSHMEM vs 海光 DCU RCCL/DUSHMEM，不混 group）
- 标注约定：所有论文均经 DBLP/OpenAlex/出版社页面/arXiv 全文实际核实；未能核实细节处标 **待确认**

---

## 第一梯队：直接决定第二篇 novelty 边界的论文（按相关度排序）

### 1. Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training
- **Venue/年份**：arXiv 2602.20656（2026-02，未见正式会议版本——待确认投稿去向）
- **链接**：https://arxiv.org/abs/2602.20656
- **核心思路**：AutoCCL 原班人马（USTC Guanbin Xu、Cheng Li 等）的后续工作。核心发现：在 compute-bound 场景下，**AutoCCL 激进调参（如把 NC 从 8 提到 61）会因 SM 竞争与全局带宽竞争拖慢重叠的计算，端到端反而劣化为 NCCL 的 0.87×**。提出统一 overlap 代价模型（makespan = max(Σ计算, Σ通信)）、两类竞争（SM 竞争 / 全局资源竞争）建模、优先级指标 H 与线性复杂度搜索，对 NCCL 的 NC/NT/Chunk 做资源感知协同调参，比 NCCL/AutoCCL 端到端提速 1.07–1.33×（8–16×A40，FSDP/TP/EP）。
- **与本方向关系**：**最关键竞品 + 最佳 motivation 引用**。它用实验证明了我们的中心假设"isolated collective 最优 ≠ 端到端最优"，但其解空间仍限于 NCCL 资源参数（algo/protocol/transport + NC/NT/C），overlap 结构固定（stream 级），无"分片何时可被 GEMM 合法消费（release/完成语义）"维度，不跨基座、不跨厂商。差异必须写成"资源参数轴 vs 完成语义结构轴"。
- **Repo**：论文称"will be open-sourced soon"（截至调研日尚未放出）

### 2. AutoCCL: Automated Collective Communication Tuning for Accelerating Distributed and Parallel DNN Training
- **Venue/年份**：NSDI 2025（USENIX）
- **链接**：https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin
- **核心思路**：NCCL fork + tuner plugin，对六参数（algorithm、protocol、transport、channel 数、thread 数、chunk 大小）做分而治之在线调优：实现类参数划分子空间、资源类参数子空间内在线采样搜索；以孤立 collective 的 busbw/T_done 为目标函数。
- **与本方向关系**：直接对标基线。其目标函数是孤立 collective 性能，正是我们要推翻的对象（Lagom 已给出实验证据）。
- **Repo**：https://github.com/gbxu/autoccl（master 分支；LD_PRELOAD build/lib/libnccl.so + NCCL_TUNER_PLUGIN + TUNER_COORDINATOR/WORLDSIZE/ROLE 环境变量）

### 3. Theseus: Runtime-Adaptive GPU Collective Communication with Hot-Swappable Schedules
- **Venue/年份**：SIGCOMM 2026（DOI 10.1145/3789240.3829134）
- **链接**：https://doi.org/10.1145/3789240.3829134
- **核心思路**：现有 CCL 的 schedule 与选择逻辑在 communicator 初始化时固化，无法适应运行时条件演化（workload 特征、硬件健康状态），长时间作业数小时/数天后性能次优。Theseus 提供 schedule 级运行时自适应：接受用户自定义 schedule 与选择策略，依据"CCL 内部属性之外的集群级运行时属性"热切换 schedule。
- **与本方向关系**：子方向 (a)(b) 的 2026 最新"自适应"占位者。但其自变量是 workload/硬件健康演化（运维视角），不是"与下游 GEMM 的依赖关系"；切换对象是等价 schedule 而非不同完成语义的基座原语。
- **Repo**：待确认

### 4. TuCCL: Enhancing Collective Communication through Co-Optimizing Algorithms and Runtime Configurations
- **Venue/年份**：ICNP 2025（DOI 10.1109/ICNP65844.2025.11192370）
- **链接**：https://doi.org/10.1109/ICNP65844.2025.11192370
- **核心思路**：批评现有工作"suboptimal isolated configuration optimization"——逐参数孤立调优忽略算法与 runtime 配置（SM/channel/queue 等资源）的耦合，提出算法与 runtime 配置联合协同优化。
- **与本方向关系**：其"isolated"批评针对**逐参数调优**（调参维度孤立），目标仍是孤立 collective 的性能；与我们"isolated **collective**（通信目标 vs 端到端 AG-GEMM 目标）"不是一回事。引用其措辞时须精确区分，避免审稿人混淆。
- **Repo**：待确认

### 5. HetCCL: Enabling Portable Collective Communication on Heterogeneous GPU Systems（arXiv 版标题：HetCCL: Accelerating LLM Training with Heterogeneous GPUs）
- **Venue/年份**：ACM Compute Frontiers (CF) 2026（DOI 10.1145/3801488.3806379）；arXiv 2601.22585（2026-01）与 2605.31000
- **链接**：https://arxiv.org/abs/2601.22585 ；https://doi.org/10.1145/3801488.3806379
- **核心思路**：SNU/KAIST（Jaejin Lee 组）。统一厂商后端（NVIDIA NCCL + AMD RCCL），通过两个新机制实现跨厂商 RDMA P2P 通信而无需改驱动，使 **NVIDIA 与 AMD GPU 混布在同一 collective group** 训练成为可能；同构场景下性能持平 NCCL/RCCL。
- **与本方向关系**：子方向 (d) 跨厂商最接近的工作。但它解决"互操作"（跨厂商 P2P transport 桥接、混 group），我们的设定是"不混 group、统一能力抽象/候选策略表达/选择逻辑"——是选择框架问题而非互操作问题。必须在 related work 明确切割。
- **Repo**：待确认（论文未在 abs 页给出）

### 6. HeteCCL: Synthesizing Near-Optimal Collective Communication Schedules for Heterogeneous GPU Clusters
- **Venue/年份**：NSDI 2026（东北大学 + 阿里云）
- **链接**：https://www.usenix.org/conference/nsdi26/presentation/hei （PDF: https://www.usenix.org/system/files/nsdi26-hei.pdf）
- **核心思路**：异构集群（如 32×H20 + 32×V100 混布、同厂商不同代）上合成近优 collective schedule：详细建模拓扑与链路带宽、schedule-step 级 chunk 量化、把调度形式化为加权有向图上最大并行传输问题，用 SMT 编码 + 反例引导归纳综合（CEGIS）加速合成；比 NCCL/TACCL/TE-CCL 带宽高至 2.8×/4.4×/2.6×，合成快两个数量级，端到端训练提速 23–37%。
- **与本方向关系**：子方向 (c) 异构调度 2026 代表作（TACCL 谱系）。注意与 HetCCL（CF'26，跨厂商互操作）是**两篇不同论文**，名字极易混淆。它做 schedule 合成（离线），不做运行时策略选择、不做完成语义、不跨厂商库。
- **Repo**：待确认

### 7. NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication
- **Venue/年份**：arXiv 2603.11438（2026-03，投稿去向待确认）
- **链接**：https://arxiv.org/abs/2603.11438
- **核心思路**：把用户态 eBPF 运行时（bpftime）嵌入 NCCL 现有 tuner/profiler/net plugin 接口：加载期静态验证防崩溃、typed map 跨 plugin 共享状态实现闭环自适应、原子热替换策略（1.07 µs）。8×B300 NVLink 上每次决策开销 80–130 ns（<0.03%），消息尺寸感知策略在 4–128 MiB 比 NCCL 默认 NVLS 提升 AllReduce 吞吐至 27%。**相关工作章节明确批评 AutoCCL "automates tuning via search but uses native code without verification"**；讨论章节明确指出"RCCL 暴露类似 plugin 架构，跨厂商移植可行"。
- **与本方向关系**：AutoCCL 后续生态（引用者）+ 其 RCCL 可移植性论述直接支撑我们"跨基座 tuner"路线的可行性；其自适应是策略热更新，选择目标仍是孤立 collective 性能。
- **Repo**：待确认（论文基于 NCCL 2.29.7 + bpftime）

### 8. Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication
- **Venue/年份**：arXiv 2606.05951（2026，ETH Zürich + NVIDIA，Hoefler 组；期刊去向待确认）
- **链接**：https://arxiv.org/abs/2606.05951
- **核心思路**：对 NVSHMEM 3.3.9 的源码级剖析：对称堆基于 CUDA VMM（VA 预留 + 按需 commit 物理页）、fast path（P2P 映射直访）/ slow path（IBGDA 或 host proxy）；给出 collective 算法表（NVLS one-/two-shot、LL/LL128 协议、pSync 双缓冲、多 CTA 仅 NVLS 门控）；AllReduce 实测：on-stream 多 CTA 路径 264 GB/s 接近 NCCL NVLS，但 device 单 CTA block 路径仅 30 GB/s；DeepEP 案例（HT/LL kernel 的 warp specialization 与 IBGDA 用法）。
- **与本方向关系**：子方向 (e) + 我们 NVIDIA 侧基座（NVSHMEM）的机制参照系。其"NVSHMEM 集体实现不能充分利用 GPU 并行度（多 CTA 支持受限）"的结论，正是 put_signal 细粒度重叠路线的论据；对 DUSHMEM 能力对比（wave64、signal_wait_until 语义）提供测量口径模板。
- **Repo**：分析对象为 NVIDIA/NVSHMEM 公开源码

### 9. MSCCL++: Rethinking GPU Communication Abstractions for AI Inference
- **Venue/年份**：ASPLOS 2026（Hwang, Cheng, Dathathri, Jangda, Maleki, Musuvathi, Saarikivi 等，Microsoft 系）
- **链接**：待确认（从 NCCLbpf 参考文献确认录用 ASPLOS'26；ACM DL 链接待补）
- **核心思路**：面向 AI 推理重设计 GPU 通信抽象，提供 GPU 驱动（device-initiated）的原语层（经 NCCLbpf 引文确认："provides GPU-driven primitives at the mechanism layer"）。
- **与本方向关系**：子方向 (e)：把"通信机制层"与"策略层"解耦的同潮流——策略层（选什么）与机制层（怎么发）分离正是我们选择框架的架构前提。细节待确认（尚未取得全文）。
- **Repo**：https://github.com/microsoft/MSCCL（MSCCL++ 在该仓库内——待确认）

### 10. FLUX: Fully-Communication-Efficient Overlap for Distributed LLM Training（2024 里程碑，上下文）
- **Venue/年份**：arXiv 2406.06858（2024）
- **链接**：https://arxiv.org/abs/2406.06858
- **核心思路**：把通信 kernel 与 GEMM kernel 融合为通信-计算 overlap 的 fused kernel（AG-GEMM/RS-GEMM），细粒度切换通信与计算，显著降低通信暴露时间。
- **与本方向关系**：AG-GEMM 重叠的 SOTA baseline（我们第一篇已对标）；证明"kernel 级融合"路线的收益上限，但策略是手工设计而非自动选择、单基座（NVSHMEM/NCCL）。注意与 IWQoS'25 的同名 Flux（multi-tenant 调度）是不同工作。
- **Repo**：待确认

---

## 第二梯队：五个子方向的核心论文

### (a) algorithm/protocol/channel/chunk 自动调优

11. **Practical Machine Learning Autotuning for Large-Scale Collective Communication** — TPDS 2026（DOI 10.1109/TPDS.2026.3661876）。ML 驱动的大规模 collective 调优（摘要未获取，细节**待确认**）。与我们关系：子方向 (a) 期刊线代表作。
12. **COCCL: A Collective Communication Library Supporting Easy Integration and Configuration of Customized Compression for Scalable LLM Training** — PPoPP 2026（DOI 10.1145/3774934.3786432，被引 3）。NCCL 之上构建压缩感知 collective 库：压缩算法与通信算法协同设计 + runtime overlap 机制，3D 并行 GPT/Qwen-7B 训练吞吐 +1.24×；含自动调优机制按环境特征选通信算法（引用 AutoCCL）。与我们关系：调优对象扩展到"压缩×算法"的组合空间，仍单基座 NCCL。
13. **OptiFlow: Towards LLM-Driven Optimization of Collective Communication Algorithms** — APNet 2026（DOI 10.1145/3820441.3820452）。用 LLM 驱动 collective 算法优化（摘要未获取，**待确认**；经 WebSearch 确认主题为 LLM 驱动调优，被引 0）。
14. **SmartCCL: Learn to Schedule Near-Optimal Collective Communication for GPU Clusters** — IEEE SECON 2026（DOI 10.1109/SECON68281.2026.11579060）。学习式调度近优 collective（无摘要，细节**待确认**）。
15. **SyCCL: Exploiting Symmetry for Efficient Collective Communication Scheduling** — SIGCOMM 2025（DOI 10.1145/3718958.3750499，被引 14；阿里 + 清华，Ennan Zhai 组）。利用 collective 与拓扑对称性分解需求为子需求、分治合成 schedule：32×A100 实测性能至 +127%，合成时间比 TECCL/TACCL 类 MILP 方法少 2–4 个数量级（分钟级）。与我们关系：schedule 合成加速线；PDF: https://ennanzhai.github.io/pub/sigcomm25-syccl.pdf
16. **OptCCL** — SIGCOMM 2026（DOI 10.1145/3789240.3829207）。按硬件配置与拓扑自动合成最优 collective 算法：数百 GPU 规模数十分钟内完成，支持多条并发 collective。与我们关系：合成目标为孤立 collective 带宽，同 Theseus 前的"静态合成"一极。
17. **ForestColl: Throughput-Optimal Collective Communications on Heterogeneous Network Fabrics** — NSDI 2026（arXiv 2402.06787，2024 预印本 → NSDI'26 录用）。任意拓扑上生成吞吐最优 schedule（广播/聚合生成树），多项式时间、理论最优性证明；AMD MI250 + A100/H100 集群验证。与我们关系：子方向 (c)(a) 交叉；"理论最优 schedule"仍以通信吞吐为目标函数。
18. **Exploring NCCL Tuning Strategies: A Measurement Study on Hybrid Parallel DNN Training** — IPDPS Workshop 2025（TU Wien：Salimi Benji、Laso、Cosenza、Benkner、Hunold）。对 NCCL 调参策略（env 变量/通道/协议）在混合并行训练下的测量研究。与我们关系：调参空间的实证测绘，可用作 (a) 的 measurement 引用。
19. **HiCCL: A Hierarchical Collective Communication Library** — IPDPS 2025。分层可组合 collective 库（NVLink/Infinity Fabric/PCIe/节点间）。与我们关系：(e) 库内部实现线。

### (b) 能力感知策略选择 / 解析代价模型

20. **Revisiting the Time Cost Model of AllReduce**（2024 上下文）— arXiv 2409.04202。系统修正 AllReduce 解析代价模型（ring/tree 等在真实 GPU 互联上的偏差）。与我们关系：代价模型基线之一。
21. **Characterizing Compute-Communication Overlap in Large-Scale Model Training** — ISPASS 2025（arXiv 2507.03114）。对训练中 overlap 的微架构级表征（SM 占用、带宽竞争）。与我们关系：为"通信拖慢计算"提供机理证据，与 Lagom 的竞争建模互补。
22. **Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms** — arXiv 2507.04786（2025，ETH + NVIDIA；HOTI 2025 版本待确认）。NCCL 协议（LL/LL128/Simple）与算法（Ring/Tree/NVLS）的深度解析。与我们关系：(e) 基准参照系，选择空间的机制解释。
23. **Parameterized Algorithms and Parameter Selection for AllReduce**（MASCOTS 2025，具体细节待确认）— AllReduce 参数化算法与参数选择。与我们关系：(b) 代价模型 + 参数选择线。
24. **An autotuning approach to select the inter-GPU communication library on heterogeneous systems** — J. Supercomputing（2024 在线/2025 卷，DOI 10.1007/s11227-024-06794-3，被引 1）。**自动选择"用哪个通信库"**（CUDA-Aware MPI vs NCCL）+ GPU 数 + 负载划分，分层执行时间建模（实验 + 理论结合）做决策。与我们关系：**"选库"方向目前最直接的先行工作**——但粒度是整库二选一、单机多 GPU、以 kernel/routine 时间为目标，无端到端 AG-GEMM 目标、无能力向量抽象、无策略族（同库内不同完成语义路径）选择。
25. **The Landscape of GPU-Centric Communication**（2024 上下文）— arXiv 2409.09874（ACM 出版版 DOI 10.1145/3813799）。GPU 中心通信全景综述，把 AutoCCL 等归入"online parameter tuning"类。与我们关系：定位我们工作的分类学框架。

### (c) 异构 GPU 集群通信调度

26. **HeteCCL**（见第 6 条，本子方向代表作）。
27. **XTree on EquiMesh: Case for Direct Interconnect for Scalable Training** — DATE 2026。EquiMesh 直连拓扑上的 XTree collective。与我们关系：拓扑-算法协同设计线（与 NSDI'25 "Efficient Direct-Connect Topologies for Training LLMs" 同谱系）。
28. **TidalMesh** — HPCA 2025。拓扑感知通信调度/网络架构（细节待确认）。
29. **Bine Trees for AllReduce** — SC 2025。Bine 树 AllReduce 算法（拓扑感知树结构，细节待确认）。
30. **Optimizing Allreduce for Heterogeneous Architectures with Multiple Processes per GPU** — arXiv 2508.13397（2025）。异构架构 + 每 GPU 多进程的 AllReduce 优化。与我们关系：异构 + 资源划分视角。
31. **Optimizing Intra-Layer Parallel Communication for LLM Training on Systems with Fully-Connected Mesh GPU Topology** — 2026（DOI 10.1145/3773656.3773675，venue **待确认**——OpenAlex 无 source 信息，推测 ASPLOS/EuroSys'26 系列）。full-mesh 拓扑上提出 AG-WGRAD overlap：**推迟无下游依赖的 weight-gradient 计算使之与 AllGather 并发**，避免切分 collective 的低效，不改通信 kernel。与我们关系：AG-GEMM overlap 近邻——同是"重排计算以配合 AG"，但为固定拓扑手工技巧，无策略选择器、单基座；其"避免 partitioned collectives"立场与我们"分片释放感知"路线相反，可作对比讨论。

### (d) 跨厂商/跨通信基座统一抽象与选择框架

32. **HetCCL**（见第 5 条）。
33. **Toward a Unified GPU-Aware OpenSHMEM Specification** — arXiv 2607.08006（2026）。OpenSHMEM 1.x 内存模型缺乏加速器可移植语义，**现有 GPU 版实现（NVSHMEM/rocSHMEM/Intel SHMEM）在内存管理、能力发现、操作语义上互不兼容**；提出统一 GPU-aware 规范方向。与我们关系：**直接支撑"跨基座能力向量 B"的必要性**的规范性证据；但它做 API/规范统一，不做策略选择，不覆盖 DUSHMEM。
34. **UNICONN: A Unified Communication Framework** — IEEE Cluster 2025。统一 NVSHMEM/rocSHMEM/Intel SHMEM 的通信框架（PDF: beyondmoore.com，细节待确认）。与我们关系：PGAS 后端统一抽象先行者。
35. **UCC (Unified Collective Communication)** — IEEE Micro 2025。统一 collective 通信运行时（多框架/多网络）。与我们关系：工业界统一层参照。
36. **GPU-Initiated Networking for NCCL（NCCL Device API / GIN）** — arXiv 2511.15076（2025，NVIDIA；Hamidouche 等）。NCCL 设备端发起通信（对称内存 + device API），DeepEP 上与 NVSHMEM 版本差距 1–2% 内。与我们关系：(e)(d)——NCCL 与 NVSHMEM 两大基座在 device-initiated 语义上正在收敛，直接影响我们"基座能力差异"论述的时间有效性（需强调 RCCL/DUSHMEM 侧无对应物）。
37. **GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems** — arXiv 2604.22126（2026）。Slingshot 网络上基于 triggered operations 的 GPU 发起通信运行时。与我们关系：(d) 能力感知（网络触发能力）的实例。
38. **Unified Designs of Multi-Rail-Aware MPI Allreduce/Alltoall across GPU+Interconnect Systems** — IPDPS 2025。GPU + 多网络 rail 的统一 Allreduce/Alltoall 设计。与我们关系：多基座（multi-rail）统一实现线。
39. **TCCL: Distributed ML Communication via Modern Clusters' Topologies** — EuroMLSys 2024（上下文）。Tencent 拓扑感知 CCL。与我们关系：(e) 拓扑感知参照。

### (e) GPU 通信库内部机制（内核设计、拓扑感知）

40. **NCCLX: Collective Communication for 100k+ GPUs** — arXiv 2510.20171（2025，Meta）。100k+ GPU（Llama 4）规模的 NCCL 演进：通信群组、拓扑适配、拥塞控制等生产级机制。与我们关系：超大规模工业视角，策略选择问题的规模化形态。
41. **NCCLZ** — HPDC 2026（细节待确认）。NCCL 相关系统优化。
42. **NIXT** — arXiv 2608.01449（2026）。NCCL 可观测性（observability）工具（细节待确认）。
43. **Comprehensive Deadlock Prevention for GPU Collective Communication** — EuroSys 2025。GPU collective 死锁预防的形式化方法。与我们关系：(e) 正确性线——我们第一篇的合法策略集合概念可援引其正确性框架。
44. **Multipath Collective Communication** — EuroSys 2026（DOI 10.1145/3767295.3769330）。多路径 collective（细节待确认）。
45. **Exploiting Multicast for Accelerating Collective Communication** — arXiv 2605.22428（2026）。多播能力（NVLS/SHARP 类）加速 collective。与我们关系：能力向量 B 中 multicast 维度的对标（DCU HYLink 无等价物——第一篇跨基座三缺口之二）。
46. **Don't Let a Few Network Failures Slow the Entire AllReduce** — arXiv 2606.01680（2026）。网络局部故障下的 AllReduce 韧性（细节待确认）。
47. **NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL** — arXiv 2603.13606（2026-03，NVIDIA）。完全基于 NCCL Device API 的 MoE 通信库：统一 ncclEpDispatch/ncclEpCombine 原语（C+Python），LL 模式（1–128 tokens，RDMA+NVLink mesh、双缓冲重叠 dispatch/combine）与 HT 模式（4096+ tokens，NVLink 域内聚合后跨节点 RDMA）；H100 集群 + vLLM 端到端验证。与我们关系：(e) device-initiated 潮流的 MoE 实例；证明"基座原语级 API + 上层策略"分层正在成为 NVIDIA 官方路线。
48. **Collective Communication for Distributed LLM Systems: Planning, Runtime Adaptation, and Computation Coordination** — IEEE Network 2026（综述，DOI 10.1109/mnet.2026.3724863）。按 planning/runtime adaptation/computation coordination 三分法综述。与我们关系：我们的选题正好落在这三分法的交叉点，可用其分类法定位。
49. **Demystifying NVSHMEM / Demystifying NCCL**（见第 8、22 条）。
50. **SwiftEP: Accelerating MoE Inference with Buffer Fusion** — NSDI 2026（https://www.usenix.org/system/files/nsdi26-li-xingyi.pdf）。MoE 推理 buffer fusion，引用 AutoCCL 为自动化调优基础工作。与我们关系：AutoCCL 引用网络成员；MoE 通信优化线。

### 邻近子方向：重叠策略与细粒度通信-计算协同（与中心假设直接相关）

51. **TokenWeave: Efficient Compute-Communication Overlap in Distributed LLM Inference** — MLSys 2026（oral；arXiv 2505.11329）。token 级细粒度任务分解实现推理中通信-计算重叠。与我们关系：细粒度释放思想的推理侧对应物。
52. **COMET: Fine-Grained Communication-Memory-Computation Overlap for MoE** — MLSys 2025（最佳论文提名，细节待确认）。MoE 细粒度通信-内存-计算三重重叠。
53. **TileLink: Enabling Efficient Fine-Grained Communication for Distributed Transformers** — MLSys 2025（arXiv 2503.20313）。细粒度链接式通信。与我们关系：细粒度 chunk 化通信的机制先例。
54. **HiPIPE** — IWQoS 2025。adaptive-chunk 流水线重叠（细节待确认）。
55. **Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads** — arXiv 2606.09200（2026）。用可移植运行时控制（shared-memory 驱动的 occupancy shaping + 通信流提优先级）实现并发 overlap，至 -25.5% 执行时间。与我们关系：与 Lagom 同轴（资源竞争），证明该轴正在快速拥挤。
56. **DeFT** — FGCS 2026（arXiv 2503.16815）。通信-计算重叠的决策框架（细节待确认；标题意为 decision framework for tree-based communication-computation overlap）。
57. **Concerto: Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning** — ASPLOS 2025（从 Lagom 参考文献确认）。自动通信优化与调度。
58. **Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning** — ASPLOS 2024（上下文）。通信切分 + 调度实现 overlap 的代表工作。
59. **AutoOverlap: Enabling Fine-Grained Overlap of Computation and Communication with Chunk-Based Scheduling** — 2025（ResearchGate 检索到，venue **待确认**）。通信 chunk 抽象解耦通信粒度与 kernel 结构。
60. **TACOS: Guiding Collective Algorithm Synthesis using Communication Sketches** — NSDI 2023 / 期刊版 2023-2024（上下文）；**Swing** — NSDI 2024（上下文，容错 collective schedule）；**Efficient Direct-Connect Topologies for Training LLMs**（Zhao）与 **OptiReduce** — NSDI 2025（上下文/并行线）。
61. **ZipCCL / Trivance / PReCCL / LEVELLER / CCSwitch / Harvest** — SIGCOMM 2026（DBLP 确认存在，主题细节**待确认**，初步判断与本方向相关度低于上述各条）。
62. **XTree/HiCCL/NCCLZ** 等已列各处；**PML-MPI** — IPDPS-W 2024（上下文，通信-计算重叠 MPI 线）；**DSE: finer-grain DMA overlap** — arXiv 2512.10236（2025-12，细节待确认）。

---

## 研究空白观察

### 问题一：有没有人做"端到端（非孤立 collective）的策略选择"？

**部分被占，但只占了"资源竞争轴"。**

- **Lagom（arXiv 2602.20656，2026-02）已经把"端到端目标函数"的高地占了一半**：它是 AutoCCL 原班人马（USTC Xu/Li），用实验证明"AutoCCL 激进调孤立参数 → compute-bound 场景端到端反而比 NCCL 慢（0.87×）"。这等于替我们证明了中心假设的前半句（isolated 最优 ≠ 端到端最优）——**可作为 motivation 引用，但也意味着"我们发现孤立调参伤端到端"本身不能再当贡献写**。
- 但 Lagom 的解空间仍是 **NCCL 资源参数**（algo/protocol/transport + NC/NT/chunk），overlap 结构固定（双 stream、collective 整体切分由 NCCL 内部决定）。它不问"分片何时可被 GEMM 合法消费（release/完成语义）"，不在"NCCL collective vs NVSHMEM put-signal vs fcollect"这类**不同完成语义的基座原语**之间做选择，更不跨厂商。
- TuCCL（ICNP'25）的"isolated"批评针对逐参数调优 → 算法+runtime 协同，目标函数仍是孤立 collective 性能；Theseus（SIGCOMM'26）做运行时热切换，自变量是 workload/硬件健康演化；AG-WGRAD（2026）是单拓扑手工重排；Resource-aware overlap（2606.09200）与 Lagom 同属资源轴。
- **结论：「以 T_e2e 为目标 + 选择变量 = 完成语义/释放结构（策略结构本身）+ 双基座能力」的组合无人做。**"端到端"作为 motivation 已有 Lagom 背书（降低我们的实验举证负担），但必须把贡献切割到"语义结构轴"：同一资源配比下，put-signal 分片释放 / fcollect 串行 / NCCL 分片重叠等**结构不同**的策略仍会发生排序反转（这正是 Phase B 实验要抓的）。

### 问题二：有没有人做"跨厂商双基座"？

**"互操作"和"API 统一"都有人做，"能力抽象 + 选择器"没人做。**

- **HetCCL（CF'26）**做跨厂商，但方向相反：让 NV 与 AMD GPU 混进**同一 collective group**（P2P transport 桥接互操作）；我们的设定是"不混 group，统一的是能力抽象、测量方法、候选策略表达和选择逻辑"。
- **HeteCCL（NSDI'26）**是异构集群 schedule 合成（同厂商不同代 GPU 混布，SMT+CEGIS），不涉及库级抽象与选择。
- **UNICONN（Cluster'25）与 Unified GPU-Aware OpenSHMEM（arXiv 2607.08006）**统一 PGAS API/规范，后者明确指出各厂商实现在 memory management、capability discovery、operation semantics 上不统一——**直接支撑"能力向量 B"的必要性**，但都止步于 API 层，不做策略选择，且均不覆盖海光 DUSHMEM。
- **J. Supercomputing 2024/25 的"选库 autotuning"**（MPI vs NCCL）是最接近"选择"二字的先行者，但整库二选一、单机、以 routine 时间为目标。
- **NCCLbpf 作者明确提出"RCCL 有类似 plugin 架构、跨厂商移植可行"但没人做**；NCCL GIN（2511.15076）显示 NCCL/NVSHMEM 在 NVIDIA 侧正在收敛，**反而抬高了"基座差异"论述在国产侧（RCCL/DUSHMEM）的独特性**。
- **海光 DCU/DUSHMEM/HCCL 的英文学术论文检索结果为零**（HCCL 在英文检索中会撞名 Intel Habana HCCL；海光生态仅有中文综述提及，如 CCCF《国产AI芯片软件生态的架构体系与产业实践》）。双刃剑：无对标 = novelty 无冲突，但审稿人无参照系，论文必须自建测量口径与能力对比表（可借 Demystifying NVSHMEM 的分析框架做 DUSHMEM 侧对应物，这本身即是贡献）。

### 问题三：AutoCCL 之后，这个 niche 有没有被占？

**参数调优赛道已经拥挤，剩余空隙在"语义结构 + 双基座"。**

- 已确认的 AutoCCL 后续/引用生态（截至 2026-09）：**Lagom**（同组，资源感知协同调参）、**NCCLbpf**（安全可验证的 policy 执行，点名批评其无验证）、**COCCL**（PPoPP'26，压缩×算法协同）、**OptiFlow**（APNet'26，LLM 驱动调优）、**SwiftEP**（NSDI'26，MoE 场景引用）、**Theseus**（运行时热切换 schedule）、以及 Landscape 综述归类。**SmartCCL（SECON'26）/TPDS'26 autotuning** 细节待确认，大概率同属参数/调度调优赛道。
- 这些工作全部位于"NCCL 的 algo/protocol/channel/chunk/资源参数"或"schedule 合成"框架内，**无人把完成语义/依赖释放作为一等选择变量，无人做 NVIDIA + 国产双基座**。
- **风险评估**：(1) Lagom 与我们共享"端到端"motivation，且同属中国高校体系、审稿人池可能重叠——related work 必须第一段就切割；(2) 资源竞争轴（Lagom、2606.09200、ISPASS'25 表征）两年内可能饱和，我们的选择变量必须明确超出资源配比；(3) Theseus 证明"运行时自适应"已有 SIGCOMM'26 占位，我们的"自适应"须限定为"依赖释放感知的策略选择"而非泛泛 runtime adaptation。

---

## 附：调研方法与覆盖度说明

- DBLP 16 组 query（collective communication / autotuning / NCCL / RCCL / NVSHMEM / rocSHMEM / allreduce / scheduling / topology-aware / heterogeneous / overlap / HCCL / cost model / algorithm selection / GPU communication optimization），year≥2024，得 177 篇候选，人工筛出上述高相关集合。
- OpenAlex 补充摘要与被引（Theseus、OptCCL、TuCCL、SyCCL、COCCL、J.Supercomput 等）；arXiv 全文抓取确认 Lagom、NCCLbpf、Demystifying NVSHMEM、NCCL EP、HetCCL、HeteCCL。
- Semantic Scholar API 全程 429（10 次重试失败），AutoCCL 引用网络改经 OpenAlex/论文参考文献交叉确认。
- 已执行 ≥10 次不同 WebSearch（中英文），含 AutoCCL 引用追踪、Hygon/HCCL 定向检索、完成语义感知重叠检索。
- 未覆盖/待深挖：SmartCCL、TPDS'26 autotuning、OptiFlow、ZipCCL 等 SIGCOMM'26 短文的具体机制（取得 PDF 后补）；MSCCL++（ASPLOS'26）全文。
