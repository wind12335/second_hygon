# 海光 RCCL-DUSHMEM 依赖释放感知 AllGather-GEMM 实验执行文档

> 文档版本：v1.0  
> 日期：2026-08-31  
> 交接对象：海光 DCU 平台实验人员  
> 文档性质：可执行实验协议。本文不声称任何预设结论，目标是用可复现实验验证或否定研究假设。  
> 适用范围：单机同构海光 DCU，首轮正式实验使用 4 卡；可先以 2 卡完成编译、语义和正确性排障。

## 1. 任务、边界与最终交付

### 1.1 要回答的唯一问题

本实验研究有数据依赖的 AllGather-GEMM，而不是孤立 AllGather 带宽调优。

需要验证的假设是：对于一个 AllGather-GEMM 操作，只以 isolated collective 的 busbw 或所有通信完成时间 T_done 选择策略，可能无法得到最短端到端时间 T_e2e；分片的合法可消费释放时间、分块 GEMM 的效率损失，以及通信和计算的资源竞争，可能改变策略排序。

该假设必须允许被否定。若所有合法候选在 busbw、T_done 和 T_e2e 上总是同序，或差异落在重复测量波动内，则本实验的结论应为“不支持该假设”。

### 1.2 本轮不做什么

- 不运行 NVIDIA 实验，也不要求 NVIDIA 侧有 4 张 RTX 4090。海光侧可以独立完成本协议。
- 不把 NVIDIA 与海光的绝对带宽或延迟当成 NCCL 与 RCCL 的库间排名。硬件、拓扑、驱动、互联、P2P 与软件栈均不同。
- 不把 RCCL 的 algorithm/protocol/channel 参数扫描本身当作论文创新。
- 不把 DUSHMEM、put、ready signal、细分 chunk、window/slot 或通信-GEMM 重叠本身当作论文创新。第一篇工作已涉及 one-sided、ready 单元、细粒度依赖重叠、staging/window 和候选短测量。
- 不重写 RCCL 或 DUSHMEM，不构造大规模运行时系统。
- 不把 DUSHMEM 接在 RCCL AllGather 后再复制同一份 payload，当成“融合”。这通常只是重复数据搬运。

### 1.3 需要交付的结果包

完成后请按 run_id 归档以下材料：

- 机器与软件栈事实快照、环境变量与实际动态库路径。
- RCCL 能力矩阵与 DUSHMEM 能力矩阵。
- B0、B1、H0 与通过准入后的 H1 原始逐次 CSV；不可只保留均值。
- 每种候选策略的 RCCL 日志，记录实际 algorithm、protocol、channel 与 transport。
- 所有正确性报告、超时报告、失败日志与复现命令。
- 通信主导、计算主导、平衡、疑似排序反转四类代表形状的 profiler trace。
- 统计汇总、保持集结果和 Go/No-Go 结论。

建议的结果目录：

~~~text
results/<run_id>/
  manifest.json
  platform/
  capability/
  raw/
  rccl_logs/
  correctness/
  traces/
  summary/
  reproduce.sh
~~~

## 2. 已知事实与设计约束

### 2.1 已有数据能说明什么

现有四卡资料包含 NVIDIA NCCL 与海光 RCCL 的 isolated collective 测试。它们用于挑选合法且稳定的通信候选，不足以回答有依赖计算时的端到端最优性。

- NVIDIA 四卡 CSV 有 547 个 PASS 导出案例；另有 222 个无效组合，主要是 AllGather/ReduceScatter 的 Tree 组合。无效组合代表 capability mask，不能补成零性能点。
- 海光 RCCL 原始 CSV 有 959 行、750 个唯一 case_id；channels 阶段另有 209 条重复，且有 6 个高波动案例需重跑。
- 海光原始结果将平台标记为 K500SM_AI、gfx928、PCIe。交接人员不能仅依据口头描述把实际设备写成 K100_ai，必须先在目标机重新核验型号和拓扑。
- 旧海光 DUSHMEM 分块 AG-GEMM 探索在 chunks=2、4、8 时出现明显变慢；同一时期分块 GEMM 大约比整块 GEMM 慢 7.8-7.9 倍。该观察说明旧结果混入了 GEMM fragmentation loss，不能说明 DUSHMEM 较慢，也不能说明 overlap 无效。

### 2.2 本协议的核心控制原则

对任意分块数 q，B1、H0、H1 必须使用完全相同的：

- dtype、M/N/K、rank 数和 rank mapping；
- q、每个 slice 的字节数、输入与输出 layout；
- GEMM 实现、GEMM epilogue 与输出 scatter/unpack；
- 初始化、warmup、计时方法、正确性比较和 CPU 亲和性；
- 需要复用 slot 时的 window 协议。

这样 B1 到 H0/H1 的差别才可归因于并发与通信数据路径，而不是某一路径使用了更快的 GEMM、较少的拷贝或不同输出布局。

## 3. 统一算子、布局与时间语义

### 3.1 统一 AllGather-GEMM

令 rank 数为 P。每张卡 r 拥有本地 activation：

~~~text
X_r: [m_local, K]
W:   [K, N]

AllGather(X_r) -> X: [P * m_local, K]
Y = X @ W
~~~

沿 m_local 的行方向将每张卡本地输入等分为 q 个 slice：

~~~text
m_chunk = m_local / q
X_r_i: [m_chunk, K]
G_i:   [P * m_chunk, K]
Y_i = G_i @ W
~~~

q 必须整除 m_local。若真实 workload 不能整除，统一用零填充到可整除尺寸，并把 valid rows 记入 manifest；各路径均使用同一填充规则。

最终输出固定为 rank-major：

~~~text
Y[producer_rank][local_row][N]
~~~

任何实现中为了 GEMM 连续性所做的 pack、scatter 或 unpack 都必须计入 T_e2e，且所有可比路径使用等价的输出语义。

### 3.2 固定工程 smoke case：可立即执行，但不进入论文主表

为让海光侧无需等待 NVIDIA 机器或完整 trace 即可验证程序，先固定一个仅用于编译、同步、内存和时间线排障的 case。它来自已有统一实验配置，不能标注为真实模型 trace。

| 字段 | 固定值 |
|---|---:|
| case_id | SMOKE_AGGEMM_8192 |
| P | 4 |
| global M | 8192 |
| m_local | 2048 |
| N | 8192 |
| K | 8192 |
| dtype | FP16；若目标机 FP16 GEMM 不稳定，可暂用 FP32 排障并单列，不与 FP16 数据比较 |
| q | 1、2、4、8 |
| 每 rank 输入 payload | 2048 x 8192 x 2 B = 32 MiB |
| 每 slice payload | q=1/2/4/8 分别为 32/16/8/4 MiB |
| 初始 window | q |
| candidate | 仅先 C0；C1/C2 通过 capability 表后再加入 |

执行顺序固定为：B0 correct -> B1 correct -> H0 correct -> H0 timeline -> DUSHMEM P0/P1/P2 -> H1 correct。这个 case 的任何结果只能用于工程准入，不能用作通信主导、平衡或计算主导的论文证据。

### 3.3 正式 shape manifest：运行前冻结的真实 trace 清单

正式数据必须从第一篇已运行过的模型 trace 中导出，且在开始 P=4 discovery 之前创建 results/shape_manifest.csv。当前材料没有保存该 trace 的完整 M/N/K 列表，因此海光侧不得自行把合成 shape 替换为“真实 workload”。项目方提供 trace 后，按下面的固定契约冻结 10 个 shape：6 个 discovery、4 个 holdout；每类至少 3 个，允许一条 shape 同时被标记为边界类但必须说明分类依据。

~~~text
shape_id,split,workload_source,model_or_layer,phase,
global_M,N,K,m_local,rank_count,dtype,
ag_payload_bytes_per_rank,gemm_flops,category,
category_basis,valid_q,notes

D01,discovery,<trace-id>,<layer>,prefill,...,communication_dominant,...
D02,discovery,<trace-id>,<layer>,prefill,...,communication_dominant,...
D03,discovery,<trace-id>,<layer>,decode,...,balanced,...
D04,discovery,<trace-id>,<layer>,prefill,...,balanced,...
D05,discovery,<trace-id>,<layer>,decode,...,compute_dominant,...
D06,discovery,<trace-id>,<layer>,prefill,...,compute_dominant,...
H01,holdout,<trace-id>,<layer>,decode,...,communication_dominant,...
H02,holdout,<trace-id>,<layer>,prefill,...,balanced,...
H03,holdout,<trace-id>,<layer>,decode,...,compute_dominant,...
H04,holdout,<trace-id>,<layer>,prefill,...,<boundary-case>,...
~~~

分类不能只凭 M/N/K 的直觉。先在 C0、q=1 下测 T_comm_only 和 T_gemm_only，计算 comm_to_gemm_ratio = T_comm_only / T_gemm_only，并将实际值写入 manifest：大于 1.5 标为 communication_dominant，小于 0.67 标为 compute_dominant，介于两者之间标为 balanced。这个分类只用于覆盖工作负载，不用于事后挑选有利结果。

### 3.4 合法释放时间

本实验刻意区分“通信操作提交”“数据真正可消费”“通信完全结束”。对一次操作定义：

~~~text
t_issue       = 通信工作提交给 comm stream 或 device work queue 的时间
t_release(i)  = 第 i 个 G_i 经正确同步后首次允许其 GEMM 读取的时间
t_done        = 本轮所有通信分片完成时间
t_last_end    = 最后一个 GEMM 输出完成时间

R_i           = t_release(i) - t_issue
T_done        = t_done - t_issue
T_e2e         = t_last_end - t_issue
~~~

t_release(i) 只可由已经证明 payload 可见且具有正确 happens-before 关系的机制定义：

- H0：RCCL AllGather 所在 stream 写出的 HIP event，compute stream 显式 wait 后的时刻；
- H1：DUSHMEM payload 的远端发布已经确认，且对应 epoch signal 已对 consumer 可见、所有 producer 的 signal 满足条件后的时刻。

以下都不能当作 t_release：host API 返回、发送端 put 发起时间、未同步读取到“像是正确”的字节，或没有验证 payload 可见性的 signal 到达。

## 4. 实验路径及其“融合”关系

这里的融合指的是：通信路径的一个 slice 一旦满足合法可消费条件，就只触发依赖该 slice 的 GEMM；它不表示 RCCL 与 DUSHMEM 对同一数据做两遍传输。

### 4.1 B0：完整 AllGather 后完整 GEMM

~~~text
RCCL AllGather(full X) -> HIP event/同 stream 顺序 -> full GEMM
~~~

B0 是输出参考、整任务锚点和无局部依赖暴露的串行基线。建议通信和 GEMM 先置于同一 stream，消除意外并发。

### 4.2 B1：同分块严格串行控制组

~~~text
AG_0 -> GEMM_0 -> AG_1 -> GEMM_1 -> ... -> AG_(q-1) -> GEMM_(q-1)
~~~

B1 必须用和 H0/H1 一样的 q、slice、GEMM 和 layout。GEMM i 结束后记录 event，comm stream 必须 wait 该 event 后才允许提交 AG i+1。不能只将所有 AG 排队后称为“串行”。

B1 的作用是隔离 fragmentation loss：

~~~text
B0 与 B1 的差异 = 由切片、pack/scatter 和小 GEMM 引入的代价
B1 与 H0/H1 的差异 = 在相同切片下允许重叠后得到或失去的效果
~~~

### 4.3 H0：RCCL 数据路径加 HIP event 依赖释放

~~~text
comm_stream:
  RCCL AllGather(slice_i) -> record E_release[i]

compute_stream:
  wait E_release[i] -> GEMM(G_i, W) -> record E_gemm_done[i]
~~~

H0 是正式的 overlap 基线，也是 DUSHMEM 接入前必须先做好的路径。对 q 个 slice，所有 AllGather 与 GEMM 可以有依赖式流水；不要让 compute 在未 wait E_release[i] 时读取 G_i。对每个 i 记录 E_release[i]，不得仅记录首尾两个 release event。

### 4.4 H1：DUSHMEM 单独承担数据平面

H1 中 DUSHMEM 替代 RCCL 作为 slice 的数据路径。它与 H0 是完整路径的二选一候选，而不是 H0 后面的附加通知或额外拷贝。

推荐的抽象对称内存布局如下：

~~~text
gather_slot[slot][producer_rank][m_chunk][K]
ready[slot][producer_rank]
free_or_credit[slot][consumer_rank]    # 仅 window < q 时需要
epoch[operation][slice]
~~~

初始实现必须使用 window=q，即单次操作每个 slice 使用独立 slot，不复用缓冲区。先获得正确、可复现的数据，再测试 window 为 1、2、4。

每一个 producer 对某 consumer 的概念协议是：

~~~text
1. 将 X_producer_i put 到 consumer 的 gather_slot[slot][producer_rank]
2. 执行已验证的 fence / quiet / remote completion / publish 操作
3. 向 consumer 的 ready[slot][producer_rank] 写入本次 epoch

consumer：
4. 等待 ready[slot][0..P-1] 均等于本次 epoch
5. 此刻定义 t_release(i)，随后发起 GEMM(gather_slot[slot], W)
6. 仅当 GEMM i 完成后，若该 slot 要复用才写回 credit
~~~

步骤 2 与步骤 3 的精确 API 绝不能根据 NVSHMEM 名字猜测。必须按目标机安装的 DUSHMEM headers、DTK 26.04 手册和最小验证程序确认其内存顺序和远端完成语义。payload 可见性与 epoch 可见性是同一个正确性原子条件的一部分。

### 4.5 H2：RCCL payload 加 DUSHMEM 通知，仅作负对照

~~~text
RCCL AllGather(slice_i) -> HIP event -> DUSHMEM signal/wait -> GEMM
~~~

H2 不属于主实现。对于同机 HIP stream，RCCL completion event 已经提供合法依赖顺序；额外 DUSHMEM signal 通常不能改善 payload 的到达，反而增加同步和诊断复杂度。只有 profiler 和最小程序确认存在真实 event/host/device 调度限制时，才可把 H2 作为负对照运行。H2 的测量不得被宣传为 RCCL-DUSHMEM 协同数据传输。

### 4.6 H3：完整后端选择器，后置实验

只有 H0 和 H1 均已正确、稳定并有可比样本后，才构造 H3。H3 的一次选择只能在 H0 和 H1 这两条完整数据路径中选一条：

~~~text
profile(shape, P, q, capability, release curve, fragmentation, contention)
  -> choose H0 or H1
~~~

同一个 collective 的同一 slice 不允许部分 rank 选 H0、部分 rank 选 H1；所有 rank 必须通过统一控制面选择同一 backend、q、window 和 epoch。禁止 RCCL 与 DUSHMEM 对同一 destination buffer 同时写入。

## 5. CUTLASS、DUMMA 与 GEMM 实现的决策

### 5.1 首轮实现选择

海光端第一阶段固定使用 rocBLAS 或 hipBLASLt 的一个版本作为 gemm_impl。选择哪一个由本机已安装版本和最小正确性程序决定，但一旦进入正式比较，同一个 case 的 B0、B1、H0、H1 必须完全一致。

建议 manifest 中记录：

~~~text
gemm_impl = rocblas | hipblaslt
gemm_impl_version
math_mode
transpose_A
transpose_B
leading_dimensions
workspace_bytes
epilogue
~~~

### 5.2 CUTLASS 不进入海光实验

CUTLASS 是 NVIDIA CUDA 生态的 GEMM/kernel 框架，不是海光 DCU 端的依赖，也不用于本轮 Hygon 实验。NVIDIA 日后补测时可单独讨论 CUDA GEMM 实现，但不能影响海光侧协议。

### 5.3 DUMMA 不是首轮必需项

DUMMA 不是 DMA 引擎、不是 DUSHMEM 的替代品，也不是通信库。DUMMA 只能作为以后验证“库级小 GEMM 是否掩盖通信现象”的 GEMM microkernel ablation。

启用 DUMMA 的前提必须全部满足：

1. B1 显示分片 GEMM 造成显著且可复现的 fragmentation loss；
2. H0/H1 的 release timeline 已正确，且仍无法判断瓶颈来自通信还是 GEMM；
3. 同一形状已有 rocBLAS/hipBLASLt 完整结果；
4. DUMMA 版本可以保持输出 layout、dtype、精度策略和 epilogue 等价；
5. DUMMA 结果单列 gemm_impl，不得与库 GEMM 样本混合计算收益。

因此首轮不需要 CUTLASS，也不需要 DUMMA。首先用固定库 GEMM 把通信路径、发布语义和端到端时间量清楚；若发现碎片化 GEMM 遮蔽了现象，再做单独的 DUMMA 消融。

## 6. Phase 0：平台、拓扑与能力核验

### 6.1 目标机事实快照

在任何性能实验前，将以下输出保存到 results/<run_id>/platform。命令中的结果是证据，不能只在报告中手工填写型号。

~~~bash
rocminfo
rocm-smi --showproductname --showuniqueid --showtopotype
rocm-smi --showpids
hipcc --version
lscpu
numactl --hardware
env | sort
ldd ./ag_gemm_rccl
ldd ./ag_gemm_dushmem
~~~

记录并人工核验：

- 实际 DCU 产品名、gfx 架构、每卡 UUID、显存与卡数；
- rank-to-DCU mapping、HIP_VISIBLE_DEVICES、CPU NUMA 绑核与进程绑定；
- PCIe/HYLink/P2P 可达性、IOMMU 状态及是否有其他 GPU/CPU 作业；
- DTK、HIP、RCCL、DUSHMEM、rocBLAS、hipBLASLt、DUMMA 的版本和真实加载路径；
- 用于 RCCL 的全部 NCCL_ 前缀环境变量，以及 DUSHMEM 相关环境变量；
- GPU 频率、电源限制、散热状态与系统时间。

RCCL 实验必须收集实际日志：

~~~bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,TASK
export NCCL_DEBUG_FILE=results/<run_id>/rccl_logs/rccl.%h.%p.log
~~~

若 DTK/RCCL 版本支持拓扑 dump，也保存拓扑 dump 文件。日志中读取实际 algorithm、protocol、channel 与 transport；不能只记录请求值。

### 6.2 RCCL capability 表

先对目标 P=2 和 P=4 建立 capability CSV。每一行是一个真实被执行的配置：

~~~text
platform_id,rank_count,collective,dtype,message_bytes,
requested_algo,requested_proto,requested_channels,
launch_status,correctness,actual_algo,actual_proto,actual_channels,
transport,rccl_log,status,notes
~~~

首轮候选固定为三项，只有在 capability 表确认合法时才运行：

~~~text
C0 = DEFAULT
C1 = Ring / Simple / channels=4
C2 = Ring / Simple / channels=8
~~~

若某配置不支持，CSV 写明 UNSUPPORTED 或 INVALID 及日志路径；不要把它伪造为零性能结果，也不要强制加入 Tree、LL 或其他协议凑矩阵。

### 6.3 DUSHMEM capability 表

在写 H1 之前逐项验证安装版本的实际能力。每项都填入“机制、声明位置、是否能编译、最小程序结果、备注”，不允许根据 NVSHMEM 的 API 名或网络教程推断。

~~~text
feature,required_for_H1,actual_api_or_mechanism,header_or_manual_location,
build_result,minimal_test_result,semantics_verified,notes

symmetric_allocation,yes
device_or_onstream_put,yes
remote_payload_completion_or_publish,yes
fence,conditional
quiet_or_equivalent,conditional
epoch_signal_or_atomic,yes
device_or_stream_wait,yes
remote_pointer_or_direct_access,no
host_progress_requirement,yes
~~~

如果没有 device/on-stream put、能证明远端 payload 完成的 publish 机制、epoch signal 和设备侧或 stream 侧 wait，则 H1 不准进入性能比较。应记录 No-Go，保留 H0 结果即可。

## 7. Phase 1：RCCL-HIP event 基线实验

### 7.1 开发顺序

严格按下列顺序推进。前一项的正确性、日志与计时不合格，不得跳到后续矩阵。

1. P=2，一个小型合成 shape，q=1、2，仅 C0。完成编译、输出检查、event 依赖和日志采集。
2. P=2，三个 trace-derived shape，q=1、2、4，运行 C0/C1/C2 的合法子集。
3. P=4，六个 discovery shape，q=1、2、4、8，运行 C0/C1/C2 的合法子集，每个配置 5 次独立进程重复。
4. P=4，保持集 holdout shapes。候选只允许使用 discovery 阶段事先确定的每形状 top-2 合法候选；不能看到 holdout 数据后重新搜索。

shape 应从第一篇或目标模型 trace 中选取，而不是只挑一个理想方阵。每个 shape 应记录来源层、M/N/K、dtype、P、局部 M、activation 字节数和 W 字节数。

### 7.2 每个 case 的执行量

~~~text
warmup iterations                 20
timed iterations / process run    50
independent process repetitions    5
统计单元                          每次 timed iteration 的原始样本
~~~

一次独立 process repetition 要重新启动全部 P 个 rank，重新写 run_id，避免只在一个长进程里把自相关样本当独立样本。若系统允许，交替运行候选顺序，避免热态或频率漂移总偏向某一个候选。

### 7.3 必测路径

每个形状、合法 candidate 与 q 的正式样本至少包含：

~~~text
T_gemm_only(q)    同一 q 下仅做 q 个 GEMM slice、相同 pack/scatter
T_comm_only(q)    同一 q 下仅做 q 个 AllGather slice、相同 comm stream 顺序
B0                full AllGather + full GEMM
B1                same-partition strict serial
H0                RCCL data path + HIP event overlap
~~~

T_gemm_only(q) 和 B1 是强制项目。没有它们就无法判断“重叠失败”究竟是通信问题还是小 GEMM 损失。

### 7.4 H0 的实现检查清单

- comm_stream 与 compute_stream 必须不同；对照 B1 需明确禁止并发。
- 每个 i 在 RCCL AllGather 后记录 E_release[i]，compute_stream 在 GEMM i 前 wait E_release[i]。
- 对 B1，compute stream 的 E_gemm_done[i] 必须让 comm stream wait 后才发起 AG i+1。
- event 与 device timestamp 的参考点必须一致；不要把 host wall-clock 直接当作 device release time。
- 输出布局和 B0 一致；必要的 scatter/unpack 必须真正执行且计时。
- 每次 operation 使用新的输入 pattern 或 epoch 标记，避免上轮残留数据通过正确性检查。

## 8. Phase 2：完整 release curve、资源竞争与正确性

### 8.1 逐 slice 时间线

现有只记录首/尾 release 的最小 harness 不可用于正式数据。正式 HIP 程序必须为每个 slice 产生一个时间线条目：

~~~text
run_id,rank,path,candidate_id,q,slice_index,slice_bytes,
t_issue_us,t_release_us,release_latency_us,
t_gemm_start_us,t_gemm_end_us,
correctness,status,log_path
~~~

对 H0，t_release 来自 compute stream wait 对应事件后的合法时刻。对 H1，t_release 来自全 producer epoch 均满足且 payload 发布已被验证的时刻。

### 8.2 主结果 CSV

每条 timed iteration 记录以下字段，禁止只保存 mean、min 或“最佳值”：

~~~text
run_id,timestamp_utc,platform_id,device_model,gfx_arch,
rank,rank_count,rank_mapping,topology_id,transport,
backend,path,collective,dtype,M,N,K,q,window,slice_bytes,
gemm_impl,requested_algo,requested_proto,requested_channels,
actual_algo,actual_proto,actual_channels,
warmup,iteration_index,
t_issue_us,t_release_first_us,t_release_last_us,t_done_us,
gemm_first_start_us,gemm_last_end_us,e2e_us,
gemm_tflops,correctness,max_abs_error,max_rel_error,status,
log_path,notes
~~~

PGAS 路径额外记录：

~~~text
publish_protocol,put_variant,fence_or_quiet_variant,
signal_variant,wait_variant,credit_variant,
t_src_safe_us,t_remote_delivery_us,t_signal_visible_us
~~~

要求每个 rank 均有记录。最终汇总一轮操作时，要明确采用 max-rank e2e 作为分布式操作完成时间，不能只报告最快 rank。

### 8.3 派生指标

用原始样本计算以下量：

~~~text
R_i = t_release(i) - t_issue
T_done = t_done - t_issue
T_e2e = t_last_compute_end - t_issue

TFLOPS_solo(q) = GEMM flops / T_gemm_only(q)
TFLOPS_overlap(q) = GEMM flops / measured overlapping GEMM interval
contention_loss = 1 - TFLOPS_overlap(q) / TFLOPS_solo(q)
first_release_window = (T_done - R_0) / T_done
release_jitter(i) = R_i - R_(i-1)
fragmentation_loss(q) = B1(q) / B0 - 1
overlap_gain_vs_B1(q) = B1(q) / Hx(q) - 1
~~~

其中 TFLOPS 的 flops 定义必须统一为 2 * M * N * K，若含 beta、bias 或特殊 epilogue 需另行注明。不要把 T_comm_only 反推出来的带宽当作 H0/H1 中真实通信性能；并发时应以 timeline 和 profiler 观察 contention。

### 8.4 正确性规则

B0 生成 reference output。每个 B1、H0、H1 输出与 B0 比较，且每轮使用确定性输入模式：

- activation 值至少编码 producer_rank、slice_index、local row 与 K 维位置；
- 每个 slot 带 guard 区和 epoch，防止旧 epoch 内容被误读；
- 每个 rank 比较自身完整 rank-major 输出，不仅比较 checksum；
- 同时保留 max_abs_error、max_rel_error、错误元素数量和第一个错误位置；
- 容忍阈值依 dtype 和 GEMM 数值模式预先写入 manifest，所有路径一致；
- 任一次 timed iteration 正确性失败、未完成或异常退出，整个 case 标为 FAIL，不可从均值中静默剔除。

对整数或精确 pattern 可增加逐字节 checksum；这只能辅助，不能替代浮点输出逐元素比较。

### 8.5 profiler 取样

保存下列四类 case 的完整 profiler trace：

1. communication-dominant：T_comm_only 明显大于 T_gemm_only；
2. compute-dominant：T_gemm_only 明显大于 T_comm_only；
3. balanced：两者同量级；
4. suspected ranking reversal：isolated communication 最优候选不是 H0 T_e2e 最优候选。

trace 至少应能显示 RCCL kernel、GEMM kernel、HIP event/wait、内存拷贝以及两类 stream 的时间关系。若版本支持 rocprof 或 roctracer，记录使用的命令和版本；若只能使用其他平台工具，也必须保留原始 trace。

## 9. Phase 3：DUSHMEM 原语准入实验

H1 不可直接从“大型 AG-GEMM”开始。先用最小程序证明数据发布协议正确。

### 9.1 P0：对称内存与拓扑启动

目标：验证 P=2 和 P=4 的 rank 初始化、对称 allocation、地址/slot 一致性、rank mapping 和退出路径。

通过条件：所有 rank 成功启动、对称对象可访问、未出现 allocator/registration 错误，进程能够有序退出。

### 9.2 P1：payload 发布正确性

目标：对每个 producer、consumer、slot、epoch，用 DUSHMEM 传输带 rank/slice/epoch 编码的 payload，并在 remote completion/publish 后由 consumer 校验。

payload 大小：

~~~text
4 KiB, 64 KiB, 1 MiB, 8 MiB, 64 MiB
~~~

每个 P x payload x 协议变体至少运行 10,000 epochs。记录 payload checksum errors、guard errors、epoch errors、stale-slot reads、timeouts 与每种事件的时间戳。

### 9.3 P2：源安全、远端到达和通知可见性的分离

目标：区分以下时间点，而不是把它们混为“put 完成”：

~~~text
t_src_safe          本地源缓冲可重用
t_remote_delivery   payload 已到达或满足 DUSHMEM 定义的远端完成
t_signal_visible    consumer 可见 epoch signal
~~~

对每个 API 组合明确其语义依据：安装 header、官方手册章节、最小正确性测试。只有存在 payload -> publish -> signal 的已验证 happens-before，才可将 t_signal_visible 当成 H1 的 release 依据。

### 9.4 P3：slot 复用和 credit

初始 H1 使用 window=q，因此一次操作没有 slot 复用。后续测试 window=1、2、4 时必须实现 credit：consumer 只有在 GEMM i 完成后才对 producer 发出本 slot 可复用的 credit。producer 看见正确 epoch 的 credit 前不得覆盖该 slot。

额外检查：

- 使用单调递增 epoch，避免仅用 0/1 标志造成 ABA 问题；
- 使 producer 故意快于 consumer，以主动触发复用压力；
- 使 consumer 故意延迟，以检验 credit 是否在 GEMM 前错误发出；
- 超时后打印每个 rank/slot 的 ready、credit、epoch、guard 和最后进度位置。

### 9.5 P4：progress、可终止性与错误处理

确认 DUSHMEM 是否需要 host progress、特定 stream、线程级别或 barrier。所有 wait 必须支持诊断超时，不允许在正式运行中静默无限等待。超时是 FAIL，不可重试后只上报成功样本。

### 9.6 H1 准入门槛

任一 P、payload 或进入 H1 的协议变体必须满足：

~~~text
payload checksum errors = 0
epoch / guard errors = 0
timeouts = 0
stale-slot reads = 0
10,000 epochs 全部完成
~~~

有任一项不满足，H1 不进入性能比较。保存故障证据，继续完成 B0/B1/H0；这属于有效的 No-Go 结果，不得用未经证明的 signal 代替同步。

## 10. Phase 4：H1 DUSHMEM AG-GEMM 性能实验

### 10.1 进入条件

只有同时满足以下条件才开始：

1. Phase 3 对目标 P 和 payload 范围通过；
2. H0 至少有两个正确、稳定、合法的 RCCL candidate；
3. B1/H0 已能分别报告 fragmentation 和 overlap 效果；
4. H1 与 B1/H0 使用同一个 gemm_impl、layout、dtype 和正确性判定；
5. 初始 H1 使用 window=q 且未复用 slot。

### 10.2 首轮矩阵

先选 Phase 2 的三个代表 shape：通信主导、计算主导、平衡各一个。对 q=1、2、4、8 运行；若 q 导致 m_chunk 非法、显存不足或库 GEMM 不支持，记为 INVALID 并说明原因。

初始比较必须包含：

~~~text
H1(q, window=q) vs B1(q)      DUSHMEM 与相同分块严格串行的差异
H0(q)           vs B1(q)      RCCL event overlap 的效果
H1(q, window=q) vs H0(q)      两条完整数据路径的可比端到端差异
~~~

禁止只把 H1 与 B0 比较，因为 B0 没有相同 q 的 fragmentation loss。H1 正确稳定后，才将 window 扩展为 1、2、4，并严格遵守 P3 credit 协议。

### 10.3 H1 的公平性清单

- DUSHMEM 的 payload 是 H1 的唯一 gather payload；RCCL 不能暗中为 H1 预填同一个 input buffer。
- H1 的每个 consumer 均收集 P 个 producer 的 slice，不能只验证本地或一部分 rank。
- H1 使用的 DUSHMEM signal 必须与对应 payload 和 epoch 同步；不能用仅本地可见的 completion 当远端可读证明。
- H1 计入 pack、publish、wait、GEMM、scatter/unpack 和必要 credit，不能只记录 put 时间。
- 所有 rank 对同一 operation 使用同一 backend；不允许局部 rank 独自退回 RCCL。
- 若 DUSHMEM 的对称内存限制导致 workspace 更大，按实际占用记入 manifest，不以“忽略内存”比较性能。

## 11. 候选选择、统计分析与判定

### 11.1 Discovery 与 holdout

将 trace-derived shape 划分为 discovery 与 holdout 两部分，划分表在运行前写入仓库且不得改动。discovery 阶段可以比较 C0/C1/C2、q 和 H0/H1；holdout 阶段不再按结果扩大搜索空间。

对 holdout，每个 shape 在 discovery 已选择的 top-2 合法 H0 candidate 上重跑，H1 只运行已经通过原语准入的固定协议。这样可区分“到处搜索后偶然最快”与可泛化的选择依据。

### 11.2 汇总方法

统计以每个独立 process repetition 内 50 个 timed iteration 的原始样本为基础。报告：

- 每个 rank 的原始样本和每轮 max-rank T_e2e；
- 每个 process run 的 median、p05、p95、mean、标准差和 CV；
- 5 个独立 process run 的 median-of-medians，以及 bootstrap 或 paired 95% CI；
- 成对比较 H0/B1、H1/B1、H1/H0 时的绝对 us 差、相对百分比和置信区间；
- 所有 FAIL、TIMEOUT、INVALID、UNSUPPORTED 的总数与原因。

候选 A、B 的配对必须在同一 shape、P、q、rank mapping、运行批次和尽可能相邻的运行时段中完成。不要把不同夜间/不同机器状态下的两个均值直接相减并赋予统计解释。

### 11.3 支持研究假设的标准

要把结果作为第二篇论文候选证据，至少同时满足：

1. 至少一个 holdout trace shape 中，isolated communication 最优 candidate 不等于 T_e2e 最优 candidate；
2. 两者的 median T_e2e 差异至少为 5%；
3. 配对 95% CI 不跨过零；
4. release curve、fragmentation_loss 或 contention_loss 至少有一个能解释排序反转；
5. H0 与 H1 都存在至少一个合法、正确、可比较的案例，或 H1 失败原因被原语准入实验严格定位；
6. 结论不是由 dtype、GEMM 实现、output layout、额外拷贝或不同显存策略造成。

这里的 5% 是进入论文论证的工程阈值，不是通过所有实验的硬性能 KPI；必须同时看变异和证据链。

### 11.4 No-Go 条件与下一步

出现下列任一情形，应明确记录 No-Go，而不是通过调整叙述强行保留方向：

- 所有合法候选的 busbw、T_done 和 T_e2e 始终同序；
- H1 无法证明 remote payload publication 正确；
- 所谓收益来自不同 dtype、不同 GEMM、不同 layout 或未计入的拷贝；
- 任何收益仅 1-3% 且低于或接近重复测量变异；
- 需要修改 RCCL/DUSHMEM 内部实现或构造大型 runtime 才能得到结果；
- 旧分块慢速无法被 B1、T_gemm_only(q) 和 profile 解释。

若 No-Go 是“无排序反转”，则保留完整基线和负结果，下一轮可将问题收缩为已证实的具体瓶颈，例如分块 GEMM 效率或 RCCL release pattern，而不是声称完成语义感知选择器。

## 12. 执行清单与交接顺序

### 12.1 第一天：只做事实和语义

1. 建立 run_id 目录，采集 Phase 0 平台快照。
2. 生成 RCCL capability CSV，确认 C0/C1/C2 中哪些合法。
3. 填写 DUSHMEM capability 表；通过 header、手册、编译和最小程序确认 API 语义。
4. 实现 P0/P1 及 P2 的一个小 payload 路径；尚不测 AG-GEMM 性能。

### 12.2 第二阶段：完成无 PGAS 的可信基线

1. 实现并检查 B0 reference。
2. 实现 B1，确认 E_gemm_done 确实阻塞下一 AG。
3. 实现 H0，确认每个 E_release[i] 都控制正确 GEMM i。
4. 先在 P=2 小形状比对，再扩到 P=4 discovery 矩阵。
5. 输出逐 slice release curve，保存代表 trace。

### 12.3 第三阶段：再接入 H1

1. 完成 P1-P4 且达到 10,000 epoch 零错误准入。
2. 以 window=q 实现 H1，运行三个代表 shape。
3. 按 H1/B1、H0/B1、H1/H0 的顺序比较。
4. 只有证据显示必要时，测试小 window credit 和 DUMMA GEMM 消融。
5. 最后才按 discovery/holdout 协议分析是否存在可泛化排序反转。

## 13. 建议的 manifest 与状态码

### 13.1 manifest 必填字段

~~~text
run_id
git_commit
build_command
launch_command
platform_id
device_model
gfx_arch
rank_count
rank_mapping
topology_id
software_versions
loaded_library_paths
environment_variables
shape_source
dtype
M,N,K
q
window
backend
candidate_id
gemm_impl
measurement_protocol
correctness_tolerance
~~~

### 13.2 统一状态码

~~~text
PASS          正确且完成，纳入统计
FAIL          输出、guard 或 epoch 正确性失败
TIMEOUT       wait、collective 或进程在规定时间内未完成
INVALID       当前 shape/q/layout/资源不满足执行前提
UNSUPPORTED   库、协议或硬件能力不支持
ENV_ERROR     版本、动态库、rank mapping 或系统环境错误
~~~

任何非 PASS 原始记录都必须保留。汇总图可只绘制 PASS，但图注和汇总表必须给出排除计数与原因。

## 14. 最终报告应回答的具体问题

海光侧最终报告不应只写“某带宽更高”或“DUSHMEM 更快/更慢”，而必须逐条回答：

1. 目标机实际是什么 DCU、gfx、拓扑、P2P 状态和软件版本？
2. 哪些 RCCL C0/C1/C2 组合在真实目标机合法，实际采用了什么 algorithm/protocol/channel/transport？
3. B0 到 B1 的代价是多少，分块 GEMM 是否仍是主要瓶颈？
4. H0 对每个 q 的 R_i、T_done、T_e2e 和 contention_loss 是什么？
5. 是否存在 busbw 或 T_done 更优、但 T_e2e 更差的合法候选？该现象是否通过保持集验证？
6. DUSHMEM 的 payload-publication-signal 语义是否被 10,000 epoch 原语测试证明？
7. 在相同 q 和 gemm_impl 下，H1 相对 B1、H0 的差异是什么，且差异来自哪一段时间线？
8. 是否满足本文件的 Go 标准；若不满足，No-Go 的可复现原因是什么？

这些回答完整后，才可决定第二篇论文是否围绕“依赖释放感知的后端/策略选择”继续推进。没有上述证据时，不应把 RCCL 参数调优、DUSHMEM 接入或一个单点加速结果写成主要创新。

## 15. 与现有材料的对应关系

本执行文档基于并细化以下已有材料：

- 下一阶段实验计划-统一Collective-GEMM基线与NCCL-NVSHMEM融合路线.md
- RCCL实验教学与结果解读 (1).md
- 四卡NCCL-RCCL结果综合分析与第二篇论文定题建议.md
- 第二篇论文相关工作排重与创新证据图谱.md
- 当前进展与Phase1启动交接.md

若本文与实际 DTK 26.04 DUSHMEM API 文档、已安装头文件或目标机能力发生冲突，以目标机可验证的语义为准，并在 capability 表中记录差异；不得为符合本文伪代码而自行假设 API 保证。
