# C · GPU 单边/PGAS 通信与细粒度同步语义 —— 2025–2026 文献调研

> 调研日期:2026-09-02。检索手段:WebSearch(10+ 次不同查询)+ DBLP/Semantic Scholar API 交叉核实。
> 覆盖主题:(a) NVSHMEM/rocSHMEM/OpenSHMEM GPU 版系统与建模;(b) device-initiated / kernel 内通信 / TMA/multicast 新硬件能力;(c) fine-grained overlap 设计空间(DMA/CE vs SM load/store);(d) 完成语/可见性/内存序;(e) 国产 GPU/DCU 通信栈。
> 所有条目均为实际确认存在的论文/文档;不确定处已标"待确认"。按与本方向相关性排序。

---

## 一、核心相关(单边通信系统分析与跨基座比较)

### 1. Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication
- **venue/年份**:arXiv 2606.05951,2026(ETH Zurich + NVIDIA;Yijun Ma、Siyuan Shen、Torsten Hoefler 等)
- **链接**:https://arxiv.org/abs/2606.05951
- **思路**:对 NVSHMEM 3.3.9 做源码级系统解剖:对称堆基于 CUDA VMM(VA 预留 + 按需 commit 物理页、P2P peer 映射、offset 对称寻址);单边 RMA 分快路径(P2P 可达 → SM 直接 load/store 远端映射地址)与慢路径(IBGDA 设备侧构造 RDMA WQE,或回退 host proxy 线程);集合通信用 pSync + LL/LL128 协议 + NVLS 算法,规则树式选择;barrier 与 sync 的区别在 quiet/__threadfence_system 的完成保证。H200 微基准:节点内 bulk put 313 GB/s / get 141 GB / scalar p 172 GB/s / **scalar g <9 GB/s**(远端 load 无法流水),节点间 IBGDA ~48 GB/s、延迟 ~9.5 μs(scalar g 25.3 μs);单 CTA device 集合只有 30 GB/s vs on-stream 多 CTA 264 GB/s。DeepEP 案例分析 HT/LL 内核如何用 nvshmemi_ibgda_put_nbi_warp + 原子计数信用。
- **与本方向关系**:★★★★★ 就是"Demystifying NVSHMEM 一类"的正式论文(第一篇四支柱框架的直接对标/竞品)。它把 SM-path 与 CE/网络路径、fence/quiet 语义都讲清了,但**只在 NVIDIA 一个生态内**做分析——第二篇"跨基座最优策略漂移"正好补位。
- **repo**:分析基于 NVIDIA/nvshmem 公开源码(https://github.com/NVIDIA/nvshmem);论文自身 benchmark 代码未见独立发布。

### 2. GPU-Initiated Networking for NCCL(NCCL GIN / Device API)
- **venue/年份**:arXiv 2511.15076,2025(NVIDIA;K. Hamidouche、J. Bachan、S. Jeaugey、J. Dinan 等)
- **链接**:https://arxiv.org/abs/2511.15076
- **思路**:NCCL 2.28 引入 Device API(三种模式),让 CUDA kernel 内直接发起集合通信。提供两种后端:GDAKI(DOCA GPUNetIO / IBGDA,GPU 直驱 NIC、零 CPU 介入)与 Proxy(CPU 代理转发)。核心动机是 MoE 类负载的细粒度 device 侧控制。DeepEP 对比实验:NCCL GIN 与 NVSHMEM 版本性能差 1–2%,但 GIN 提供层级 communicator 而 NVSHMEM 无。
- **与本方向关系**:★★★★★ NVSHMEM 的官方"竞品基座"。NCCL GIN(NVLink 域内对称内存 + IBGDA)与 NVSHMEM(平坦 PGAS)语义差异正是"同一算法在不同通信基座上的策略选择"的现成实验场。
- **repo**:实现在 https://github.com/NVIDIA/nccl(2.28+ Device API)。

### 3. CPU- and GPU-initiated Communication Strategies for Conjugate Gradient Methods on Large GPU Clusters
- **venue/年份**:SC 2025 Inno4Scale Workshop(ACM DOI 10.1145/3712285.3759774);J. D. Trotter 等(Argonne)
- **链接**:https://dl.acm.org/doi/10.1145/3712285.3759774(预印本:https://beyondmoore.com/_file/assets/preprint-pdfs/SC25_Inno4Scale_aCG.6fc2df87.pdf)
- **思路**:同一算法(CG / pipelined CG)在多通信基座上重实现:GPU-aware MPI、NCCL/RCCL、NVSHMEM,分别覆盖 CPU-initiated 与 GPU-initiated 两种发起方式;并提出"monolithic"变体——整个 CG 循环常驻 GPU、全程 device-initiated(NVSHMEM)。底层比较 IBRC(RDMA for CUDA)与 IBGDA(GPUDirect Async)两种传输。
- **与本方向关系**:★★★★★ 目前最接近"同一算法跨基座策略漂移"的公开工作,但结论停留在"哪个快用哪个"的性能对比层面,**没有提炼跨基座可预测的机理模型,也没有涉及完成语义维度**——这是它留给我们的空白。
- **repo**:待确认(beyondmoore.com 有预印本;代码未确认公开)。

### 4. Redesigning GROMACS Halo Exchange: Improving Strong Scaling with GPU-initiated NVSHMEM
- **venue/年份**:SC 2025 Workshops(PAW-ATM),ACM DOI 10.1145/3731599.3767508;arXiv 2509.21527;Doijade、Alekseenko、Brown、Gray、Páll
- **链接**:https://arxiv.org/abs/2509.21527
- **思路**:把 GROMACS 域分解 halo 交换从 MPI(CPU 中心)重写为 NVSHMEM kernel-initiated:打包与通信融合进单个 kernel、利用硬件 latency hiding;跨通信阶段的 kernel 融合;**显式混合使用 NVLink 上的异步 copy engine 与 SM 路径**优化延迟与带宽;分层同步把"接收方通知"融合进数据搬运 kernel。强扩展提升 1.5x(节点内)/ 2x(多节点 NVLink)/ 1.3x(NVLink+IB)。
- **与本方向关系**:★★★★☆ (c) 两条数据路径(CE vs SM)在真实延迟敏感应用里混合调度的代表作,也是"完成通知融合进数据路径"的工程样板。
- **repo**:GROMACS 主线(https://gitlab.com/gromacs/gromacs,NVSHMEM 后端已合入实验分支;细节待确认)。

### 5. Demystifying NCCL: An In-Depth Analysis of GPU Communication Protocols and Algorithms
- **venue/年份**:arXiv 2507.04786,2025;IEEE HotI 2025(Hu、Shen、Bonato、Jeaugey、Dinan、Hammond、Hoefler 等)
- **链接**:https://arxiv.org/abs/2507.04786
- **思路**:源码级解剖 NCCL:Simple/LL/LL128 协议、channel 编排、SM/CE(NVLS/ST/CTA pairs)内存搬运机制、算法选择。揭示 LL128 依赖 128B 原子写故仅 NVLink 安全等语义-硬件耦合。
- **与本方向关系**:★★★★☆ 与 #1 成对,提供 NCCL 侧的透明度;"协议安全性依赖互连语义"是其与完成语义主题的交叉点。
- **repo**:无独立 repo(分析对象为 NVIDIA/nccl)。

### 6. GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems
- **venue/年份**:HPDC 2026;arXiv 2604.22126(B. Shan、M. Araya-Polo、B. Chapman;Stony Brook + TotalEnergies)
- **链接**:https://arxiv.org/abs/2604.22126
- **思路**:面向 HPE Slingshot(OFI/CXI,Top500 前三系统)的 GPU-initiated 通信运行时:利用 CXI **triggered operations** 让 NIC 预置工作、GPU 端有界(bounded)机制跨多次 kernel 触发回收预置 NIC work,消除 host 进度线程,实现亚微秒级 kernel 内跨节点协调(0.88 μs allreduce 等)。
- **与本方向关系**:★★★★☆ 第三条生态路径(IBGDA 之外):"预触发/网卡侧状态机"提供的完成语义与 IBGDA 的门铃模型不同,是"基座语义差异导致策略漂移"的又一例证源。
- **repo**:待确认(论文页未见到公开代码链接)。

---

## 二、细粒度重叠设计空间与延迟下界(c/b 方向)

### 7. Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives
- **venue/年份**:arXiv 2607.16100,2026(venue 待确认,疑为 SC26 投稿)
- **链接**:https://arxiv.org/abs/2607.16100
- **思路**:研究 scale-up 域内小消息集合如何逼近硬件 Speed-of-Light 下限:原则包括 **barrier-free 同步、对称内存 + multicast 的高效利用**;基于 NCCL device API 构建自定义集合接口,小/中消息距绝对 SoL 下限仅 7%。
- **与本方向关系**:★★★★☆ 直接论证"同步/完成语义强度决定小消息延迟下限"——release 语义与重叠策略耦合的最新证据。
- **repo**:待确认。

### 8. Syncopate(前称 AutoOverlap):Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap
- **venue/年份**:arXiv 2601.20595,2026(Xinwei Qiang、Yue Guan、Yufei Ding、Adnan Aziz 等,UCSB;v1 名为 AutoOverlap)
- **链接**:https://arxiv.org/abs/2601.20595
- **思路**:编译器 + 运行时,把细粒度计算-通信重叠自动化到**单个融合 kernel 内部**:引入 communication chunk 抽象,解耦通信调度与 kernel 结构;底层同时抽象 async_memcpy(CE)、async_send/recv(NVSHMEM)、load/store(SM 同步)三类原语,按 chunk 状态机交错发射。
- **与本方向关系**:★★★★☆ (c) 的编译器视角系统化;其"三原语统一抽象"与 triton-distributed 的多后端 shmem 抽象层同构,是第二篇迁移工作的直接参照。
- **repo**:待确认。

### 9. MoE-Hub: Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- **venue/年份**:ISCA 2026;arXiv 2605.05888
- **链接**:https://arxiv.org/abs/2605.05888
- **思路**:指出 MoE token-expert 动态映射与 GPU 静态地址中心通信模型的"抽象失配"是重叠难写的根因;提出 Load-Dispatch-Apply(LDA)范式 + 域特定抽象,在 NVL72 上用 TMA 优化的缓冲与硬件加速路径,对比 SM-based 的 UCX/NVSHMEM 实现,兼顾性能与可编程性。
- **与本方向关系**:★★★★☆ (b)(c) 硬件加速(TMA/CE)路径 vs SM 路径的直接对比实验所在;论证"通信基座抽象决定软件复杂度"。
- **repo**:待确认。

### 10. Exploiting Multicast for Accelerating Collective Communication(MultiWrite)
- **venue/年份**:arXiv 2605.22428,2026
- **链接**:https://arxiv.org/abs/2605.22428
- **思路**:针对 AllGather/AlltoAll(dispatch)的 unicast 冗余拷贝问题,提出 many-to-many 的 MultiWrite 传输语义:吸收 multicast 原理但解决传统 multicast 在 AI 负载上的局限,消除重复包直接降低算子时延。
- **与本方向关系**:★★★☆☆ (b) multicast 语义的新作;与 NVLS/SHARP、TMA multicast 一并构成"多播原语进入通信库"趋势。
- **repo**:待确认。

### 11. Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications
- **venue/年份**:ISPASS 2025;arXiv 2507.03114
- **链接**:https://arxiv.org/abs/2507.03114
- **思路**:在 H100/A100(NVIDIA)与 MI250/MI210(AMD)上系统测量 overlap 的代价:重叠使计算平均减速 18.9%、最高 40%(SM/内存系统资源争用),并分析精度、专用核、power cap 的影响;NVIDIA 与 AMD 的重叠干扰形态不同。
- **与本方向关系**:★★★☆☆ (c) 的定量基线:"SM 路径通信吃掉计算资源"这一漂移因素的跨厂商证据。
- **repo**:待确认(基准脚本未见发布)。

### 12. Resource-aware Computation-Communication Overlap for Multi-GPU ML Workloads
- **venue/年份**:arXiv 2606.09200,2026
- **链接**:https://arxiv.org/abs/2606.09200
- **思路**:用两个可移植的运行时旋钮做并发重叠:共享内存驱动的计算 kernel occupancy shaping + 通信 kernel 提升调度优先级,缓解通信与计算的资源竞争。
- **与本方向关系**:★★★☆☆ (c) 资源感知重叠的轻量方法,可借用到"SM 路径 vs CE 路径"的资源仲裁上。
- **repo**:待确认。

### 13. FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion
- **venue/年份**:arXiv 2406.06858,2024 年末里程碑(Li-Wen Chang 等,MSRA + 北大)
- **链接**:https://arxiv.org/abs/2406.06858
- **思路**:纯软件路线:把通信拆成 chunk 融合进计算 kernel(生产者-消费者),分块依赖的 token 逐块发送,消除通信 kernel 启动与同步;在 LLM 推理/训练上取得接近理想重叠的收益。是 2025–2026 一批 chunk-scheduling 工作(Syncopate、Lagom 等)的先行者。
- **与本方向关系**:★★★☆☆ (c) SM 路径 kernel-fusion 重叠的奠基工作。
- **repo**:待确认(GitHub 检索 pku-liang/FLUX 已 404,现状不明)。

### 14. Fantasy: Efficient Large-scale Vector Search on GPU Clusters with GPUDirect Async
- **venue/年份**:arXiv 2512.02278,2025
- **链接**:https://arxiv.org/abs/2512.02278
- **思路**:GPU 集群向量检索系统,用 IBGDA 做 kernel-initiated 数据加载(计算 kernel 直接拉远端向量),消除 CPU-GPU 加载停滞;单边 get 语义支撑图遍历式不规则访问。
- **与本方向关系**:★★★☆☆ (a) IBGDA 单边通信在非 HPC 模板应用上的落地案例。
- **repo**:待确认。

---

## 三、PGAS/OpenSHMEM 语义与生态(a/d 方向)

### 15. Toward a Unified GPU-Aware OpenSHMEM Specification
- **venue/年份**:arXiv 2607.08006,2026
- **链接**:https://arxiv.org/abs/2607.08006
- **思路**:论证 OpenSHMEM 1.x 的内存模型在 GPU 上缺乏可移植性:各家(NVSHMEM/rocSHMEM/Intel SHMEM)对 fence/quiet、原子、同步的语义实现不一致;提出统一 GPU-aware 规范方向,覆盖紧耦合 device 域(内存 fabric)上的细粒度远程操作。
- **与本方向关系**:★★★★☆ (d) 的规格层论文:官方承认"各 GPU shmem 完成语/内存序不统一",是"可见性语义影响上层策略"命题的直接文献支撑。
- **repo**:规范文本随 OpenSHMEM 社区(https://openshmem.org)。

### 16. Collective Communication for 100k+ GPUs(NCCLX)
- **venue/年份**:arXiv 2510.20171,2025(Meta)
- **链接**:https://arxiv.org/abs/2510.20171
- **思路**:Meta 的 NCCLX:面向 10 万+ GPU 全生命周期(训练低时延同步 ↔ 推理低延迟),支持 lazily-initiated communication、内存复用、**基于 NVSHMEM 的低延迟动态通信路径**,与 PyTorch 深度集成。
- **与本方向关系**:★★★☆☆ 工业级系统里 NVSHMEM(NVLink 域)+ NCCL(跨域)混合基座的实例;说明"按域选基座"已是生产实践,但论文未给出可迁移的选择准则。
- **repo**:部分在 https://github.com/facebookincubator/NCCLX(开源版本;待确认)。

### 17. Intel SHMEM: GPU-initiated OpenSHMEM using SYCL
- **venue/年份**:SC 2024 Workshops;arXiv 2409.20476(A. Brooks、P. Marshall 等,Intel)
- **链接**:https://arxiv.org/abs/2409.20476
- **思路**:把 GPU-initiated OpenSHMEM 移植到 SYCL/oneAPI 生态:device 侧 API 支持从 kernel 内发起 put/get/原子/同步,适配 Intel GPU 内存模型与互连。证明 device-initiated PGAS 是跨厂商趋势(NVSHMEM/rocSHMEM/ishmem 三家)。
- **与本方向关系**:★★★☆☆ (a)(e):第三家厂商的移植样本;"同一编程模型、不同硬件语义"的实证材料。
- **repo**:待确认(GitHub 上 intel/ishmem、intel/intel-shmem 均未检索到,2026-09 状态不明)。

### 18. Weak Scaling of NVSHMEM Applied to Hashed Distributed Structured Data(HashBrick)
- **venue/年份**:SC 2025 Workshops;ACM DOI 10.1145/3731599.3767506(Davis 等)
- **链接**:https://dl.acm.org/doi/10.1145/3731599.3767506
- **思路**:brick 化 + 哈希索引的非规则块结构数据布局,用单边 NVSHMEM 直接搬 ghost brick,**免打包通信缓冲**;GH200 集群上高阶 CFD/Jacobi 弱扩展,通信随规模亚线性增长。
- **与本方向关系**:★★★☆☆ (a) 单边通信 + 细粒度数据布局协同的应用论文;"免打包"依赖 put 的直写语义。
- **repo**:待确认。

### 19. The Landscape of GPU-Centric Communication
- **venue/年份**:ACM Computing Surveys(2024 投稿/在线,arXiv 2409.09874;D. Unat 等,Koc 大学)
- **链接**:https://arxiv.org/abs/2409.09874
- **思路**:GPU 中心通信的分类学综述:按 CPU 介入程度、编程模型(message passing/RMA/共享内存)、硬件路径(Issue:SM vs CE/DMA)建立 taxonomy,覆盖 2024 前系统。
- **与本方向关系**:★★★☆☆ 领域地图;第二篇 related work 的骨架来源,但无 2025–2026 新系统与语义量化。
- **repo**:无。

### 20. Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training
- **venue/年份**:arXiv 2602.20656,2026
- **链接**:https://arxiv.org/abs/2602.20656
- **思路**:LLM 训练重叠框架,延续 chunk-based 调度思想,把通信切片与计算切片做细粒度配对调度。
- **与本方向关系**:★★☆☆☆ (c) 的 LLM 应用层;不涉及完成语义。
- **repo**:待确认。

### 21. TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference
- **venue/年份**:arXiv 2505.11329,2025(Microsoft)
- **链接**:https://arxiv.org/abs/2505.11329
- **思路**:推理侧 token 级 All2All 与专家计算重叠:动态 batch 切分 + 提前派发。
- **与本方向关系**:★★☆☆☆ 应用层重叠,参考其 chunk 粒度选择经验。
- **repo**:待确认。

### 22. Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC
- **venue/年份**:arXiv 2604.07609,2026
- **链接**:https://arxiv.org/abs/2604.07609
- **思路**:把整个推理 serving 栈下放:GPU 负责 kernel 与调度,SmartNIC 负责网络与 KV 传输,CPU 完全退出关键路径。
- **与本方向关系**:★★☆☆☆ (b) device-initiated 极致化(SmartNIC 路径)。
- **repo**:待确认。

---

## 四、国产 GPU/DCU 通信栈(e 方向)

### 23. Serving Large Language Models on Huawei CloudMatrix384
- **venue/年份**:arXiv 2506.12708,2025(华为)
- **链接**:https://arxiv.org/abs/2506.12708
- **思路**:目前国产平台上最详尽的公开系统论文:CloudMatrix384(384 昇腾 910C,UB 总线 scale-up 全互联 + 400G IB scale-out);详述 ascend 上的算子库(AOL)、**HCCL 集合通信**在 scale-up/scale-out 两级域的工程形态、通信-计算协同的推理优化。给了 P/D 分离等场景下通信模式的量化数据。
- **与本方向关系**:★★★☆☆ (e) 国产栈对标 NCCL 生态的最完整论文素材;但只谈集合通信,无 PGAS/单边/kernel-initiated 语义。
- **repo**:HCCL 开源在 https://gitee.com/ascend/cann-hccl。

### 24. 海光 DUSHMEM / DCU 通信栈
- **现状**:截至 2026-09,**未检索到任何公开论文或 arXiv 预印本**(WebSearch 中英文多轮 + DBLP 均无)。海光 DCU 通信相关公开物仅限于 ROCm 生态移植的工程文档(rocSHMEM 的 DCU 分支/DTK)与 PaddlePaddle/飞桨适配文档(https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/dcu/support_cn.html)。
- **与本方向关系**:★★★★☆(作为空白)这正是第二篇的护城河:国产 DCU 上的 device-initiated PGAS 通信抽象与建模无人发表,抢先发表即是首个公开测量。
- **repo**:无公开 repo(内部库)。

### 25. 其他国产生态参考(非论文,工程/行业资料)
- **VCCL**(创智研究院/基流,2025-09 开源):GPU 集合通信库增强方案(https://www.sii.edu.cn/2025/0922/c27a466/page.htm)——开源项目,未见论文。
- **InfiniCCL**(沐曦 Metax,与开源 InfiniTensor 配套):https://www.metax-tech.com/ndetail/12613.html——国产 GPU 集合通信移植案例。
- **星脉网络**(中兴通讯技术 2025):端侧集合通信库与集中式路由控制器协同优化,10 万+ GPU 集群(https://www.zte.com.cn/content/zte-site/www/zte-com-cn/china/about/magazine/zte-communications/2025/cn202502/specialtopic/cn202502002.html)——期刊工程论文,偏网络侧。
- **《9 款国产 GPU 芯片低时延通信技术研究》**(电子工程专辑,2025,https://www.eet-china.com/mp/a501822.html)——行业测评文章,非学术。
- **《国产 AI 芯片构建集合通信库的工作分析与建议》**(知乎专栏,https://zhuanlan.zhihu.com/p/1971319954541348800)——梳理 NCCL 移植中国产芯片的 IPC(shm.cc)与 RDMA(net.cc)双路径替换(NVLink→HCCS/KBW 等)。
- 综述背景:**XCCL: a survey of industry-led collective communication libraries**(JCST 2023,较旧,作背景)。

### 26. 生态里程碑(官方文档/演讲,非论文)
- **NVSHMEM TMA 接口**:官方文档已提供 TMA 后端的 nonblocking put(cp.async.bulk 提交、bulk-group 完成语;远端可见性需 group wait)——https://docs.nvidia.com/nvshmem/api/latest/tma.html 。说明 NVSHMEM 正式把 TMA 硬件通路纳入单边 RMA(b 方向)。
- **PyTorch Symmetric Memory + NCCL Window/Device API**:PyTorch 2.7+ 的 torch.distributed._symmetric_memory(后端 "cuda"/"nccl"/"nvshmem")+ NCCL 2.28+ Window Registration(ncclCommWindowRegister)与 ncclDevKernel——文档 https://docs.pytorch.org/docs/stable/symmetric_memory.html 、https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html ;PyTorch Conf Europe 演讲 "A New Path Towards Multi-GPU Kernels"(Ke Wen & Sylvain Jeaugey)。**尚无正式论文**,是当前 kernel 内多 GPU 编程的事实接口。
- **DeepEP**(DeepSeek,2025):NVSHMEM 之上的 MoE dispatch/combine 通信库,V2 转向 NCCL GIN——https://github.com/deepseek-ai/DeepEP 。与 #1、#2 构成完整叙事链。
- **PGL / "One Kernel for All Your GPUs"**(Stanford Hazy Research,2025-09 博客):https://hazyresearch.stanford.edu/blog/2025-09-22-pgl ——跨 GPU 单 kernel 融合的工程探索,明确讨论 copy engine(不占 SM、可完全重叠)vs SM load/store(细粒度但抢计算资源)取舍;正式论文待确认。

---

## 五、研究空白观察

**问题 1:有没有人系统研究"同一算法在不同通信基座上的最优策略漂移"?**
- **没有系统研究,只有碎片。** 最接近的是:(i) Trotter 等 SC25-W Inno4Scale(#3)——CG 在 MPI/NCCL/RCCL/NVSHMEM × CPU/GPU-initiated 上对比,含 IBRC vs IBGDA,但止于性能横评;(ii) NCCL GIN 论文(#2)——DeepEP 在 NVSHMEM vs NCCL GIN 上 1–2% 差距,单一负载;(iii) MoE-Hub(#9)——硬件加速路径 vs SM-based UCX/NVSHMEM 的开发代价与性能对比;(iv) ISPASS 2025(#11)——NVIDIA/AMD 上重叠干扰的跨厂商测量;(v) GICC(#6)——证明 Slingshot 触发式语义要求完全不同的 runtime 结构。
- **共同缺口**:没有任何工作把"基座差异(IBGDA 门铃 vs P2P 直访 vs triggered ops vs 集合式 GIN)→ 语义/资源差异 → 最优 chunk 粒度、路径选择(CE/SM)、同步插入点、warp 划分的系统性漂移"建模成可预测的框架。Trotter 的数据甚至暗示漂移存在(CG 上 monolithic 变体只在部分基座/规模占优),但无人提炼规律。**这正是 LYC 第二篇可以占据的位置**:以同一组 triton-distributed kernel 在 NVSHMEM / NCCL-GIN / DUSHMEM(国产)上的策略漂移为实验主体。
- 佐证生态事实:NCCLX(#16)在生产中已按域混用 NVSHMEM 与 NCCL,但选择靠手调。

**问题 2:有没有人系统研究"release/可见性语义对重叠策略的影响"?**
- **没有量化研究,只有定性陈述与规格层讨论。** 相关碎片:(i) Demystifying NVSHMEM(#1)讲清了 fence(顺序)/quiet(完成+远端可见)/put_signal(数据+信号原子绑定)三档语义及其实现成本,但未把"语义强度"当作自变量做重叠收益实验;(ii) NVIDIA/nvshmem GitHub issue #60 记录了 nvshmem4py quiet 的"本地完成 vs 远端可见"语义歧义——社区层面的痛点实证;(iii) Unified GPU-Aware OpenSHMEM 规范(#15)明言各 GPU shmem 内存序不可移植,但给的是规范倡议而非性能后果;(iv) Every Microsecond Matters(#7)证明 barrier-free 同步是小消息逼近 SoL 的必要条件(同步语义强度的极端效应);(v) GROMACS(#4)把接收通知融合进数据搬运 kernel、"弱化同步"换取重叠;(vi) NVSHMEM TMA 文档区分"提交完成"与"远端可见"两档 bulk-group 语义。
- **共同缺口**:无人绘制"语义强度 × 重叠收益"的权衡曲线(如:数据到达即消费 vs 等远端可见 vs 全序,各换来多少 chunk 流水深度与 SM 占用),也无人分析信号通路与数据通路分离(put_signal 的 flag 走原子网络 vs 数据走 RDMA)对吞吐的影响规律。第二篇若把"完成语义作为可调维度"引入重叠策略空间(不同基座语义档位不同 → 最优策略随之漂移),即同时缝合两个空白。
- 国产维度:DUSHMEM 无任何公开论文,国产 DCU 上的单边语义测量完全空白——既是风险(没有可引用的国产对标)也是机会(首篇)。

**风险提示**:Demystifying 系列(ETH+NVIDIA)动作很快(NCCL 2025-07、NVSHMEM 2026-06),若其下一步做"NVSHMEM vs GIN vs rocSHMEM 跨基座建模",会直接威胁第二篇 novelty;建议尽快把"跨基座策略漂移 + 完成语量化"的 framing 钉死,并把国产 DCU 基座(他们无法覆盖)作为差异化护城河。

---

## 附:检索记录(WebSearch 主查询,≥10 次)
1. "NVSHMEM 2025 paper GPU one-sided communication" 2. "rocSHMEM paper 2025 SC symposium" 3. "GPU initiated communication SC 2025 paper kernel NVSHMEM IBGDA" 4. "fine-grained communication overlap GPU 2025 2026 copy engine SM load store" 5. "symmetric memory GPU kernel 2025 PyTorch"(限流重试)6. "Hygon DCU communication library paper DUSHMEM"(限流重试)7. "TMA multicast tensor memory accelerator GPU communication ASPLOS 2025"(限流)8. "HCCL Huawei Ascend collective communication paper 2025" 9. "NVSHMEM fence quiet completion semantics visibility analysis 2026" 10. "Tessellate vectorized message passing microsecond interconnects GPU multicast TMA arXiv 2025"(未确认存在,已弃)11. "国产 GPU 通信栈 论文 2025 海光 昇腾 寒武纪"(中文)12. "PyTorch symmetric memory NCCL window API device communication 2025" 13. "NVSHMEM put_signal remote completion notification overlap kernel synchronization 2025 2026"。另有 DBLP/Semantic Scholar API 批量核实(NVSHMEM/rocSHMEM/OpenSHMEM/GPUDirect/GPU communication overlap/SmartNIC/symmetric memory/PGAS/Hygon/Ascend/Cambricon/FLUX/AutoOverlap/Trotter 等 15+ 查询)。
