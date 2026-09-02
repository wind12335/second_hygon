# 下一阶段实验计划：统一 Collective-GEMM 基线与 NCCL/NVSHMEM 融合路线

> 更新日期：2026-08-31  
> 文档性质：阶段性实验方案，不是已经得到的论文结论  
> 适用平台：NVIDIA GPU（开发阶段 RTX 4090，论文目标 A100）与海光 DCU（目标 K100_ai；现有原始数据文件标记为 K500SM_AI/gfx928，型号必须重新核验）  
> 主问题：孤立 collective 的带宽和完成时间，是否足以预测有数据依赖的 Collective-GEMM 端到端最优执行策略？

---

## 1. 先明确这阶段要回答什么

目前已经完成的是四卡 NCCL/RCCL 的**孤立 collective 基线**。这些实验测量了不同通信操作、消息大小、算法、协议、channel 和 mapping 下的 `algbw/busbw`，并且基本证明当前默认策略已经接近孤立通信性能的强候选。

但这还没有回答第二篇论文真正需要回答的问题：

```text
通信产生的数据什么时候能够被后续计算合法地消费？
通信与 GEMM 同时运行时，哪一种策略的端到端时间最短？
孤立通信最优配置是否仍然是 Collective-GEMM 最优配置？
这个判断在 NVIDIA 与海光平台上是否具有不同的边界？
```

本阶段的工作假设是：

> 对于有数据依赖的 AllGather-GEMM，平均 busbw 和完整 collective 的 `T_done` 可能不足以描述端到端性能。分片的合法释放时刻、同步语义、GEMM 分块效率以及通信-计算资源竞争，可能使策略排序发生反转。

这是一个**待证伪假设**，不能直接写成论文创新。只有在后续实验中观察到稳定、可解释、跨平台可复现的现象，才继续发展成正式方法。

### 1.1 本阶段的最小成功条件

至少需要得到以下证据：

1. NVIDIA 和海光平台各自都存在两个或更多正确、稳定、可比较的 AG-GEMM 策略。
2. 至少有一批形状或分块参数，使 isolated busbw 最优策略不是端到端 `T_e2e` 最优策略。
3. 这个排序反转超过重复测量噪声，并且能由 release timeline、GEMM slow-down 或同步开销解释。
4. 同一抽象实验方法可以在两个平台分别运行；不要求两个平台选择完全相同的底层策略。
5. 如果进一步加入 NVSHMEM/DUSHMEM，PGAS 路径必须在 payload、通知和消费三个层面都通过正确性验证。

如果 2 或 3 不成立，就不能强行提出“完成语义感知选择器”。应把论文方向收缩为实验中真正观察到的单一瓶颈，或者及时更换问题。

---

## 2. 当前已知事实与不能越过的边界

### 2.1 已有四卡结果说明了什么

当前四卡资料表明：

- NCCL/RTX 4090 与 RCCL/海光的 DEFAULT 策略已经接近各自当前测试矩阵中的 isolated communication oracle。
- 在大消息上，Ring/Simple 和较多 channel 有时更快，但收益通常很小，且受平台、transport、消息大小和 collective 类型影响。
- NCCL 侧当前观察到 SHM/direct 和 P2P 路径；海光侧原始材料显示 PCIe P2P。两边硬件、互联、驱动、通信库版本不同，不能把绝对带宽差异归因于 NCCL 或 RCCL 本身。
- 海光原有 DUSHMEM 分块 AG-GEMM 探索中，分块 GEMM 比整块 GEMM 慢很多。因此，任何重叠收益都必须与**相同分块方式的串行基线**比较。

详细审计见：

- `四卡NCCL-RCCL结果综合分析与第二篇论文定题建议.md`
- `第二篇论文相关工作排重与创新证据图谱.md`
- `四卡默认策略跨平台对齐汇总.csv`
- `四卡RingSimple通道扩展汇总.csv`

### 2.2 当前不能声称的内容

以下说法目前都没有足够证据：

```text
NCCL 软件实现一定比 RCCL 好；
NVSHMEM 一定比 NCCL 快；
DUSHMEM 的单边通信一定能改善 RCCL；
分块越细，重叠越充分；
LL 虽然 isolated busbw 低，但一定不适合 overlap；
把 NCCL 与 NVSHMEM 同时调用本身就是创新；
当前方案已经是第二篇论文的最终贡献。
```

### 2.3 与第一篇和已有工作的硬边界

本阶段使用 AG-GEMM、分片、NVSHMEM/DUSHMEM、ready signal，并不自动构成第二篇论文的创新。尤其需要避免与第一篇论文重复：现有第一篇方法已经包含 NVSHMEM one-sided、细粒度 ready 单元、producer/consumer 的优先调度、有限对称内存的 staging/window 以及约束感知的短测量搜索。因此下面这些都只能作为第二篇的候选实现或实验控制变量：

```text
把 AllGather 切成更小 chunk；
使用 put + ready signal；
设置 window/slot/frontier；
用 CUDA/HIP C++ 重写第一篇的 Triton 部分；
把 NVSHMEM 路径移植到 DUSHMEM；
仅从多个 chunk 配置中选择一个更快的配置。
```

已有工作进一步限定了问题空间：

| 工作 | 已覆盖或强相关的内容 | 本阶段必须避免的重复 |
|---|---|---|
| 第一篇论文 | one-sided、ready 单元、细粒度 data-dependent overlap、staging/window、候选短测量 | 再提出一个 chunk/ready/window 的重叠机制 |
| AutoCCL，NSDI 2025 | collective 的 algorithm/protocol/thread/channel/chunk 低层调优与在线试探 | 把 NCCL/RCCL 参数扫描或调参器作为主贡献 |
| FLUX，arXiv:2406.06858 | software kernel fusion 的通信-计算重叠 | 只声称“融合通信和 GEMM” |
| FiCCO，arXiv:2512.10236 | DMA-based finer-grain overlap、schedule space、静态 shape heuristic | 只声称“DMA/更细 chunk/shape heuristic” |
| CoCoNet，ASPLOS 2023 | dependent computation overlap via decomposition | 只声称“把 dependent compute 分解后重叠” |
| ResCCL，SIGCOMM 2025 | primitive-level collective scheduling 和通信 kernel 资源开销 | 只做通信 kernel 的 TB/SM 资源调度 |

因此，本计划把 PGAS 融合定位为**实验证据和候选策略**，不是预设贡献。第二篇若能成立，中心必须是：

> 不同通信基座的 capability 与完成/可见/通知语义如何改变一个已有 AG-GEMM overlap 机制的可行策略集合和端到端最优性；在不重写通信库的约束下，如何根据这种差异用少量校准正确选择路径。

换句话说，真正要证明的是：

```text
不是“细粒度重叠有用”，
而是“只看 busbw 或 collective 完成时间会选错策略；
必须显式建模合法依赖释放和资源竞争，且该边界随通信基座变化”。
```

---

## 3. 统一实验对象：AllGather-GEMM

### 3.1 为什么先选 AllGather-GEMM

AllGather-GEMM 是一个适合做机制验证的对象：每个 rank 先拥有一段本地 activation，AllGather 后得到完整输入，GEMM 的不同输出 tile 只依赖部分输入。于是可以明确区分：

```text
完整 AllGather 完成
vs.
某一个输入分片已经传输、可见、通知并可以被对应 GEMM tile 消费
```

它同时具备三个优点：

1. 能复用第一篇工作中的 AG-GEMM workload 和 shape 来源。
2. 能用 NCCL/RCCL 的 stream-event 路径先建立不依赖 PGAS 的正确基线。
3. 后续可以在相同依赖图上比较 NCCL/RCCL、NVSHMEM/DUSHMEM 和混合策略。

第一阶段不要把 AllReduce、跨节点、多机拓扑、完整训练框架和所有模型算子一起纳入。ReduceScatter-GEMM 可以作为后续有限泛化，但不是本阶段的必要条件。

### 3.2 统一抽象和符号

两个平台分别运行自己的同构 GPU/DCU 通信组，不把 NVIDIA GPU 和海光 DCU 混在一个 collective group 中。

| 符号 | 含义 |
|---|---|
| `p` | rank/GPU 数量，开发时可为 2，正式首轮优先为 4 |
| `M,N,K` | GEMM 形状，采用 `C = A x B` 的统一逻辑定义 |
| `q` | 将可消费输入沿依赖方向切成的 slice 数 |
| `slice_bytes` | 单个通信分片的实际字节数 |
| `window` | 同时在途或已提交但未消费的分片数量 |
| `T_comm` | communication-only 时间 |
| `T_gemm` | GEMM-only 时间 |
| `T_serial` | 同一分块方式下通信和 GEMM 串行执行的实际时间 |
| `T_overlap` | 通信流和计算流并发时的端到端时间 |
| `T_done` | 完整 collective 或所有分片通信完成的时间 |
| `R_i` | 第 `i` 个分片合法可消费的 release latency |
| `T_e2e` | 从本轮开始到最后一个计算结果完成的端到端时间 |

定义：

```text
t_issue       = 通信操作被提交到对应 stream 或 device work queue 的时间
t_release(i)  = 第 i 个分片在正确同步后首次允许对应 GEMM tile 读取的时间
R_i           = t_release(i) - t_issue
T_done        = t_done - t_issue
T_e2e         = t_last_compute_end - t_issue
```

`t_release(i)` 不能用下面的事件替代：

- host API 返回时间；
- 未同步地读取远端 buffer 看到“像是正确”的数值的时间；
- 发送端认为 put 已发出的时间；
- 没有证明 payload 可见性的 signal 到达时间。

---

## 4. 统一平台事实表

在任何正式性能运行前，分别在 NVIDIA 和海光机器保存一份机器事实快照。文件名必须包含 `platform_id`、日期和 run id。

### 4.1 NVIDIA 侧

至少保存：

```bash
nvidia-smi -L
nvidia-smi
nvcc --version
python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count())'
git -C /path/to/nccl rev-parse HEAD
git -C /path/to/nvshmem rev-parse HEAD
ldd ./ag_gemm_nccl
```

同时记录：

- GPU 型号、显存、驱动和 CUDA 版本；
- NCCL、NVSHMEM、cuBLASLt、CUTLASS 版本或 git commit；
- `CUDA_VISIBLE_DEVICES`、rank 到 GPU 映射；
- `nvidia-smi topo -m` 和 GPU NUMA 关系；
- P2P 可达性、实际 transport 和 NCCL INFO 日志；
- 是否锁定频率、功耗和应用时钟；
- 是否有其他作业占用 GPU 或 PCIe/CPU。

论文目标平台是 A100 时，RTX 4090 只能作为开发和方向筛选平台。最终核心矩阵必须在 A100 重复，不能用 4090 替代 A100 结果。

### 4.2 海光侧

必须在实际机器重新核验用户口述的 K100_ai 与原始文件中的 K500SM_AI/gfx928 是否为同一设备：

```bash
rocminfo | sed -n '1,220p'
rocm-smi --showproductname --showuniqueid --showtopo
hipcc --version
```

另外保存：

- DTK/ROCm、HIP、RCCL、DUSHMEM、rocBLAS/hipBLASLt、DUMMA 版本；
- `HIP_VISIBLE_DEVICES`、rank 到 DCU 映射；
- PCIe/HYLink/P2P/NUMA 拓扑；
- RCCL `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=TASK,GRAPH` 日志；
- 实际选中的 algorithm、protocol、channel；
- `NCCL_TOPO_DUMP_FILE` 或 DTK 对应拓扑 dump；
- 频率、电源、其他作业和系统负载。

设备型号、GFX 架构和互联不是文档排版问题。它们必须在论文表格、CSV、图注和实验脚本中统一。

---

## 5. 基线设计：必须同时保留五类基线

所有策略都必须输出正确性状态和逐次时间。不能只保留最好的一个结果。

### 5.1 B0：完整通信后完整 GEMM

```text
一次完整 AllGather
        |
        v
完整 GEMM
```

实现：

- NVIDIA：CUDA C++ + NCCL `ncclAllGather` + cuBLASLt 或 cuBLAS；
- 海光：HIP C++ + RCCL AllGather + rocBLAS 或 hipBLASLt。

通信和计算可以先放在同一 stream 中，形成最清楚的非重叠基线。记录：

```text
T_comm_only_whole
T_gemm_only_whole
T_serial_whole
T_e2e_whole
correctness
```

B0 回答的是：如果不主动暴露局部依赖，完整任务需要多长时间。

### 5.2 B1：相同分块方式、完全串行

把 B0 的输入沿依赖方向切成 `q` 个 slice，但每个 slice 仍然按以下顺序执行：

```text
AllGather(slice 0) -> GEMM(slice 0)
AllGather(slice 1) -> GEMM(slice 1)
...
```

它不是重叠策略，而是控制实验。B1 的作用是测出分块和小 GEMM 带来的真实损失：

```text
partition_penalty(q) = T_gemm_only_partition(q) / T_gemm_only_whole
```

如果没有 B1，就无法知道所谓“重叠收益”是否只是被 chunked GEMM 的效率退化吞掉。

### 5.3 B2：NCCL/RCCL 分块通信 + stream-event 重叠

通信流负责分片 collective，计算流在对应 event 后执行分片 GEMM：

```text
communication stream:
    allgather(slice 0) -> record event 0
    allgather(slice 1) -> record event 1
    ...

compute stream:
    wait event 0 -> GEMM tile 0
    wait event 1 -> GEMM tile 1
    ...
```

NVIDIA 侧使用 CUDA event；海光侧使用 HIP event。这里的 release 时刻由通信 stream event 和 compute stream wait 共同定义，属于 library/stream 语义允许的消费时刻。

B2 是本阶段最重要的第一个重叠基线，因为它不需要假定 NVSHMEM/DUSHMEM 的远端内存可见性，也不需要读取尚未同步的通信输出。

### 5.4 B3：固定 DEFAULT 的重叠策略

在 B2 中固定每个平台通信库默认选择的路径，不手动指定 algorithm/protocol/channel。它回答：

```text
现有库默认策略直接配合分片 pipeline，能否得到收益？
```

B3 是后续所有自适应方法的最重要工程 baseline。

### 5.5 B4：有限通信配置扫描

不要把 AutoCCL 的完整搜索问题重新做一遍。只保留经过 capability discovery、正确性测试和预实验筛选的少量候选：

```text
DEFAULT
Ring/Simple/ch1
Ring/Simple/ch4
Ring/Simple/ch8
```

如果某平台的接口或库版本不支持某候选，则记录 `UNSUPPORTED`，不能填 0，也不能把错误当作慢样本。

必要时再加入：

```text
Tree/Simple
Ring/LL
Tree/LL
```

但只有在该组合真实支持且 B2 能正确运行时才加入。当前四卡数据已经表明 capability mask 必须是第一步。

### 5.6 B5：独立 GEMM 参考实现

对每一个 `q` 分别测量 GEMM：

- whole GEMM；
- same-partition GEMM；
- 与通信并发时的 GEMM。

这组数据用来计算：

```text
gemm_partition_loss
gemm_contention_loss
```

没有 B5，通信和计算的性能变化无法分解。

---

## 6. 初始统一实验矩阵

本矩阵分为“必须运行”和“条件扩展”。先完成必须部分，再决定是否扩大。

### 6.1 维度一：GPU 数量

```text
开发调试：p = 2
第一轮正式：p = 4
论文目标：按 A100/K100_ai 实际可用规模复现，优先保留 p = 4
```

2 卡用于快速发现编程、同步和 buffer 布局错误；不能把 2 卡结果直接替代 4 卡论文结论。正式比较中两平台应尽量使用相同 `p`，并报告拓扑差异。

### 6.2 维度二：消息和分块规模

孤立 primitive 首轮使用：

```text
4 KiB, 16 KiB, 64 KiB, 256 KiB,
1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB
```

AG-GEMM 首轮使用实际 workload 的 `M,N,K`，并记录每个 slice 的字节数。建议先固定总输入量，再改变 `q`，否则总工作量变化会破坏比较。

### 6.3 维度三：q 和 window

必须先测：

```text
q      = 1, 2, 4, 8, 16
window = 1, 2, 4
```

满足以下条件时才加入 `q=32`：

- slice 仍然满足 dtype、地址和 collective 对齐要求；
- 分块 GEMM 没有完全退化；
- 内存占用和 event 数量可控；
- 运行没有超时或死锁。

`q=1` 不是可忽略的点，它对应整块通信/整块 GEMM，也是 B0/B2 的共同锚点。

### 6.4 维度四：GEMM 形状

从第一篇论文实际使用过的 shape 中挑选，至少覆盖三类：

| 类别 | 特征 | 用途 |
|---|---|---|
| 通信主导 | GEMM 较轻，通信占比高 | 检查能否隐藏通信 |
| 平衡型 | 通信和计算时间接近 | 最可能出现排序反转 |
| 计算主导 | GEMM 较重 | 检查通信是否被自然隐藏、分块是否伤害计算 |

不要一开始凭空选择一个只对某个策略有利的形状。每类至少 3 个 shape；方向验证阶段建议 10--20 个真实或代表性 shape，按 shape 划分发现集和留出集。

### 6.5 维度五：数据类型

先选 NVIDIA 和海光都稳定支持、且库实现一致的 FP16 或 BF16。每种 dtype 单独报告：

- accumulator dtype；
- GEMM 输出 dtype；
- NCCL/RCCL datatype；
- correctness tolerance。

不要因为某个平台支持某种低精度格式就把另一平台不支持的格式混入“统一比较”。

---

## 7. 每一个实验要记录什么

### 7.1 主结果 CSV：一行一个重复样本

建议文件：

```text
results/ag_gemm_runs.csv
```

最少字段：

```text
run_id,timestamp,platform_id,host_id,
device_model,gpu_count,rank_count,rank_mapping,
driver_version,cuda_or_rocm_version,
nccl_or_rccl_version,nvshmem_or_dushmem_version,
collective,dtype,M,N,K,
strategy_id,backend,api,algo,proto,nchannels,
q,window,slice_bytes,partition_axis,
gemm_impl,gemm_tile,comm_stream_id,compute_stream_id,
warmup,iterations,repetition_index,
t_issue_us,t_release_first_us,t_release_last_us,t_done_us,
e2e_us,comm_only_us,gemm_only_us,serial_same_partition_us,
gemm_tflops,comm_algbw_GBs,comm_busbw_GBs,
correctness,status,error_code,timeout,
profiler_trace,library_log,topology_file,notes
```

规则：

- 一次重复一行，不能只保存均值；
- 原始日志路径和 profiler trace 路径必须写进 CSV；
- `UNSUPPORTED`、`CORRECTNESS_FAIL`、`TIMEOUT`、`RUNTIME_ERROR` 都保留；
- 不能把失败样本删除后只报告成功率；
- 不适用字段使用空值，不使用 0 伪装为“耗时为零”；
- 每次正式运行生成唯一 `run_id`，不能覆盖相同 case 的旧结果。

### 7.2 Release CSV：一行一个 slice

建议文件：

```text
results/ag_gemm_release_timeline.csv
```

字段：

```text
run_id,strategy_id,platform_id,rank,
slice_index,slice_bytes,
t_issue_us,t_payload_or_event_us,
t_release_us,release_latency_us,
t_gemm_start_us,t_gemm_end_us,
notification_kind,ordering_kind,
correctness,status,notes
```

对于 B2：

```text
notification_kind = nccl_or_rccl_stream_event
ordering_kind = stream_event_wait
```

对于 PGAS 路径，只有实际证明 payload 可见和消费者合法读取后，才填写 `t_payload_or_event_us`；否则留空。

### 7.3 Capability CSV：记录库的可用能力

建议文件：

```text
results/capability_matrix.csv
```

字段：

```text
platform_id,device_model,backend,collective,algo,proto,
nchannels,dtype,min_bytes,max_bytes,
host_api_supported,device_api_supported,
stream_api_supported,requires_alignment,
correctness,status,log_path,notes
```

策略选择前的顺序必须是：

```text
capability discovery
        -> correctness filter
        -> performance measurement
        -> candidate selection
```

---

## 8. 实验执行顺序

### Phase 0：环境和数据清理，1--2 天

交付物：

1. NVIDIA 与海光各一份平台事实 JSON/Markdown。
2. K100_ai/K500SM_AI 型号冲突的核验结果。
3. 原有 RCCL 高波动 case 的重跑结果。
4. NCCL/RCCL/NVSHMEM/DUSHMEM 的实际加载库路径。
5. 一份统一 `run_manifest.csv`，描述每个 case 的参数和状态。

通过条件：

- rank 数、GPU 数、dtype、shape 和环境变量均可从日志恢复；
- 两个平台没有把不同硬件的绝对带宽误标为软件差异；
- 失败和不支持组合都有状态。

### Phase 1：NCCL/RCCL stream-event AG-GEMM，1--2 周

此阶段**不使用 NVSHMEM/DUSHMEM**。目的是先回答：仅依靠正确的分块 collective 和 stream-event，是否已经存在端到端策略反转。

实现三个最小程序或三个 strategy mode：

```text
mode=whole
mode=serial_partition
mode=overlap_event
```

NVIDIA 侧：

```text
CUDA C++
NCCL ncclAllGather
CUDA stream/event
cuBLASLt 或 cuBLAS GEMM
```

海光侧：

```text
HIP C++
RCCL AllGather
HIP stream/event
rocBLAS 或 hipBLASLt GEMM
```

每个平台先做：

```text
p=2, q=1/2/4, 3 个 shape, 20 warmup + 30 正式重复
```

稳定后扩大到：

```text
p=4, q=1/2/4/8/16, 10--20 个 shape, 至少 50 正式重复
```

注意：每个 rank 必须以完全一致的 collective 顺序调用分片 AllGather。不能某个 rank 先执行下一片、另一个 rank 还停留在上一片，否则容易把正常的 collective 顺序错误误判为硬件或库死锁。

### Phase 2：release curve 与资源竞争，约 1 周

对 Phase 1 中的稳定候选测量：

```text
R_0 ... R_(q-1)
T_done
T_serial_same_partition
T_overlap
GEMM TFLOPS
```

同时做三个计算负载：

1. 无 GEMM：测纯通信；
2. 轻 GEMM：测弱资源竞争；
3. 目标 GEMM tile：测真实竞争。

重点观察：

- 第一个 slice 是否明显提前释放；
- release curve 是否均匀，还是前快后慢；
- `T_done` 最短的配置是否仍然最早释放第一批有用数据；
- stream-event 等待是否造成空洞；
- 通信并发是否使 GEMM TFLOPS 降低；
- `q` 增大后，释放收益是否被 GEMM 分块损失抵消。

### Phase 3：NVIDIA NVSHMEM 与海光 DUSHMEM 原语验证，约 3--5 天

该阶段先做独立 microbenchmark，不直接接 AG-GEMM。每个平台分别验证：

```text
symmetric allocation
put / put_nbi
get / get_nbi
fence
quiet
signal
wait
stream-ordered variant
multiple outstanding operations
payload checksum
```

NVIDIA 当前安装的 NVSHMEM 3.8.0 头文件中已经确认存在：

```text
nvshmemx_putmem_on_stream
nvshmemx_putmem_nbi_on_stream
nvshmemx_putmem_signal_on_stream
nvshmemx_putmem_signal_nbi_on_stream
nvshmemx_quiet_on_stream
nvshmemx_signal_op_on_stream
nvshmemx_signal_wait_until_on_stream
nvshmem_putmem_nbi
device-side wait_until APIs
```

本机头文件还明确区分了：

```text
flush_on_stream：保证源 buffer 可以复用，不保证远端 destination 已可读；
quiet_on_stream：用于更强的远端完成/可见性语义；
signal/wait：需要和 payload 的 ordering 一起设计，不能单独当作数据就绪。
```

因此第一个 PGAS 正确性协议应是：

```text
write/put payload
        -> 正确的 fence 或 quiet
        -> signal ready(epoch)
        -> consumer wait ready(epoch)
        -> consumer 读取 payload
        -> checksum/guard 校验
```

每个 epoch 的 signal 值必须递增或带有 slot/epoch 信息，不能重复使用同一个值而不清理，否则会把上一轮 ready 误认为当前轮 ready。

海光侧不能因为 DUSHMEM 有相同名字就直接复制 NVSHMEM 的语义。必须以 DTK 26.04 手册、installed headers 和运行结果确认：

- `put` 返回是否只表示本地源可以复用；
- 何种操作保证远端 payload 可见；
- `barrier` 和 `sync` 是否保证 RMA 完成；
- stream 版本是否存在以及是否是 host enqueue；
- device-side wait 是否能够在目标环境稳定推进。

### Phase 4：逐步加入 PGAS，约 1--2 周

只有 Phase 3 的原语正确性和 Phase 1 的 NCCL/RCCL event baseline 均稳定，才进入此阶段。

---

## 9. NCCL 与 NVSHMEM 如何“巧妙地”融合

这里最容易产生误解。合理融合不是让 NCCL 和 NVSHMEM 对同一 payload 各传一遍，也不是在 NCCL 完整 AllGather 后随便发一个 signal。合理的融合必须让两个库承担不同职责，并且保证数据路径只有一个权威来源。

### 9.1 路线 H0：NCCL/RCCL 负责数据，event 负责依赖

```text
NCCL/RCCL AllGather slice
        -> CUDA/HIP event
        -> compute stream wait
        -> GEMM slice
```

这是必须先完成的基线，不算 NVSHMEM 融合。它回答：NCCL/RCCL 自己的 stream 语义是否已经足够支持安全的分片 overlap。

### 9.2 路线 H1：PGAS 负责分片数据，NCCL/RCCL 保留为完整/大块 collective 候选

这是最直接、也最容易形成清楚对照的混合候选：

```text
大消息或不规则阶段：NCCL/RCCL collective
细粒度、具有明确 owner/consumer 的阶段：NVSHMEM/DUSHMEM put/get
GEMM：cuBLASLt/rocBLAS/hipBLASLt 或受控自定义 kernel
同步：PGAS signal/quiet/wait 或 stream event
```

例如 AllGather 的一部分数据由每个 rank 的 owner 通过 PGAS 写入对称接收区；当某个 slice 的 payload 经过合法 ordering 后，发送 ready signal，消费者只启动依赖该 slice 的 GEMM tile。剩余大块数据仍可用 NCCL/RCCL 批量完成。

但这条路线成立必须满足：

1. PGAS 对应的通信拓扑和带宽不能比 NCCL/RCCL 差到完全抵消粒度收益。
2. PGAS payload 不与 NCCL/RCCL 同时写同一 destination 区域。
3. 同一个 logical slice 只有一个传输 owner，避免重复写和竞态。
4. signal 发布晚于 payload 达到文档保证的可见状态。
5. 所有 rank 的 epoch、slot 和同步顺序一致。

### 9.3 路线 H2：NCCL/RCCL 负责 bulk，PGAS 只负责低成本通知

这条路线看似巧妙，但不能默认可行。

```text
NCCL/RCCL 传输 payload
        -> 确认某个分片对应的合法完成边界
        -> PGAS signal/doorbell 通知 consumer
        -> consumer wait
        -> GEMM
```

问题在于：NCCL 标准 `ncclAllGather` 通常以整个调用为完成单位。除非程序把 collective 拆成多个明确的 slice 调用，并在每个 slice 上记录 event，否则 NVSHMEM signal 没有可靠依据表明“某个 NCCL 内部片段已经到达且可读”。

因此，第一版不要做“窥探 NCCL 内部进度”。只允许：

```text
一个 slice 的 NCCL 调用完成
        -> event 记录
        -> 必要时由 host/device 侧发出 PGAS notification
```

如果 PGAS signal 只是在 event 之后增加一次通知，而计算流本来就可以直接 wait event，那么 H2 可能没有性能价值。它只有在后续 profiling 证明 PGAS wait 能减少 CPU 调度、减少跨 stream 管理或更好地连接 device-side consumer 时才值得保留。

### 9.4 路线 H3：按阶段切换 backend，而不是每个 slice 混合写入

更稳妥的混合策略是把策略空间定义成：

```text
whole-NCCL
chunked-NCCL-event
whole-NVSHMEM-stream
chunked-NVSHMEM-signal
bulk-NCCL + tail-NVSHMEM
```

selector 根据 shape、消息规模、GEMM 计算强度、平台 capability 和已测 release curve 选择一条路径。这样可以避免“同一 slice 同时属于两个通信库”的所有权混乱。

在论文方法上，真正可能有价值的是：

```text
根据通信基座能力和合法完成语义选择执行策略
```

而不是：

```text
调用两个库
```

### 9.5 推荐的实际推进顺序

```text
H0  NCCL/RCCL + event
  -> H1  PGAS standalone put/signal/wait
  -> H3  backend-level selection
  -> H2  只在有明确必要性时测试
```

暂不建议一开始直接实现 H2。它最容易把 NCCL 的 collective completion、NVSHMEM 的 signal completion 和远端内存可见性混为一谈。

---

## 10. CUDA/HIP、CUTLASS、cuBLASLt、rocBLAS 各自负责什么

### 10.1 CUDA C++ 和 HIP C++：实验控制层

本项目需要的底层控制包括：

- 分配和布局输入、输出、对称内存；
- 创建 communication stream 和 compute stream；
- 调用 NCCL/RCCL；
- 创建 event、记录 event、建立 wait dependency；
- 启动分片 GEMM 或 PGAS kernel；
- 记录时间、校验输出和写 CSV；
- 处理 capability、超时和错误状态。

所以 CUDA/HIP C++ 是必要的宿主和 kernel 控制层，但它本身不是创新。它的价值是让依赖、同步和时间线可见，避免 PyTorch 调度层隐藏关键事实。

### 10.2 cuBLASLt 和 hipBLASLt/rocBLAS：第一优先级 GEMM 基线

第一版应优先使用成熟库：

```text
NVIDIA: cuBLASLt，必要时 cuBLAS
海光：hipBLASLt 或 rocBLAS，按目标 DTK 支持情况选择
```

原因：

- 先确认问题到底来自通信、同步还是 GEMM 分块；
- 减少自写 GEMM kernel 的 correctness 和性能变量；
- 方便计算 `GEMM_TFLOPS(q)` 和 `gemm_partition_loss`；
- 两个平台都能使用相应厂商成熟库。

cuBLASLt/hipBLASLt 还可以提供 heuristic/algo 选择，但本阶段不要把 GEMM 算法搜索和通信策略搜索同时扩大到全空间。先固定一个稳定、高性能、可重复的 GEMM 实现。

### 10.3 CUTLASS：不是第一版必须项

CUTLASS 是 NVIDIA CUDA C++ 的可组合 GEMM 和数据移动抽象，适合：

- 自定义 tile shape；
- 控制 shared memory、寄存器和 pipeline stage；
- 写带有特殊输入布局或 epilogue 的 GEMM；
- 在后续将通信到达的 slice 直接接入自定义 mainloop；
- 研究通信和计算资源竞争的细节。

但是 CUTLASS 只适用于 NVIDIA CUDA 路径，不能直接作为海光端统一实现。海光侧对应候选可能是：

```text
rocBLAS/hipBLASLt
DUMMA
Composable Kernel 或 DTK 官方矩阵内核
HIP C++ 自定义 kernel
```

因此，第一轮不要因为“底层论文”就强行加入 CUTLASS。推荐顺序是：

```text
cuBLASLt/rocBLAS 基线
        -> 确认 q 对 GEMM 的影响
        -> 仅对关键 shape 做 CUTLASS/DUMMA microkernel
        -> 比较 library GEMM 与 custom tiled GEMM
```

### 10.4 DUMMA 的正确定位

DUMMA 应被看作海光 DCU Tensor Core/MMA 相关的 GEMM 微内核接口，不是通信库，也不能在没有证据时称为 DMA 或 copy engine。

它只有在以下问题出现时才加入：

```text
rocBLAS/hipBLASLt 的分块 GEMM 退化过大；
需要固定 tile 形状以保证每个到达 slice 可独立计算；
需要显式控制 CU/寄存器/shared memory 占用；
需要与 NVIDIA CUTLASS 做“同一抽象、不同后端”的对照。
```

DUMMA 的使用必须单独报告：

- tile shape；
- accumulator 类型；
- occupancy 和寄存器使用；
- whole 与 partitioned GEMM TFLOPS；
- correctness；
- 是否与通信并发。

---

## 11. 参考实现的逻辑结构

建议先做一个轻量、独立于 PyTorch 的 benchmark，而不是修改 NCCL/RCCL 内部或构建大型运行时。逻辑结构如下：

```text
ag-gemm-bench/
  common/
    config schema
    CSV/JSONL logger
    correctness checker
    timeout/error state
  cuda/
    nccl_backend
    nvshmem_backend
    cublaslt_gemm
    optional_cutlass_gemm
  hip/
    rccl_backend
    dushmem_backend
    rocblas_or_hipblaslt_gemm
    optional_dumma_gemm
  scripts/
    collect_platform_facts.sh
    run_phase1_matrix.sh
    aggregate_results.py
  results/
    raw logs
    csv
    topology
    profiler traces
```

统一接口只描述抽象，不要求两个 backend 内部相同：

```cpp
capability query(const CaseConfig&);
void allocate(const CaseConfig&);
void submit_collective(const Slice&, Stream);
void submit_pgas_transfer(const Slice&, Stream);  // 条件路径
void record_release(const Slice&, Stream);
void wait_release(const Slice&, Stream);
void launch_gemm(const Slice&, Stream);
void check_result();
```

NVIDIA 与海光可以使用不同的内部调用和不同的 capability mask，但必须输出相同抽象字段，才方便做跨平台分析。

---

## 12. 正确性、死锁和超时规则

通信-计算 overlap 最容易出现“程序看起来卡住”，所以必须把正确性和进度测试设计在性能实验之前。

### 12.1 Collective 顺序规则

对于每一个 `q`：


```text
rank 0: slice 0 -> slice 1 -> slice 2 -> ...
rank 1: slice 0 -> slice 1 -> slice 2 -> ...
...
```

所有 rank 必须执行相同数量、相同顺序、相同 datatype、相同 count 的 collective。不能让某个 rank 因为本地计算快就提前调用下一轮 collective。

### 12.2 Stream 依赖规则

事件关系必须明确写在代码和日志中：

```text
comm_stream:   collective_i -> event_i
compute_stream: wait(event_i) -> gemm_i
```

计算 stream 不能在 event 前访问 slice 输出。host 不能在没有 stream synchronize 的情况下释放或复用 device buffer。

### 12.3 PGAS payload/notification 规则

ready notification 必须带 epoch/slot：

```text
slot = slice_index % window
epoch = global_iteration * q + slice_index
ready[slot] = epoch
```

consumer 必须等待目标 epoch，而不是只等待 `ready != 0`。数据校验至少包含：

- payload checksum；
- 每一轮 epoch 的 guard value；
- GEMM 输出与 reference 的误差；
- 多轮重复后是否出现上一轮数据。

### 12.4 超时和异常

每个 case 设置 wall-clock timeout，例如首轮 60 秒或根据 B0 的若干倍设置。超时后：

1. 保存所有 rank 的 stdout/stderr；
2. 记录最后一个已提交 collective、slice、epoch 和 stream；
3. 保存进程状态和库日志；
4. 标记 `TIMEOUT`；
5. 不把它当成性能结果；
6. 用缩小到 2 rank、`q=1`、单次迭代的复现程序定位。

---

## 13. Profiling 与分析方法

### 13.1 时间线工具

NVIDIA：

```text
Nsight Systems：看 NCCL、CUDA stream、event、GEMM 的时间关系
Nsight Compute：看关键 GEMM 的 Tensor Core、occupancy、memory 指标
```

海光：

```text
hipprof --print-trace 或 DTK/ROCm 实际可用 profiler：看 HIP stream、RCCL、GEMM
RCCL TASK/GRAPH 日志：确认实际 algorithm/protocol/channel
```

不要要求两套 profiler 输出完全相同的硬件计数器。统一的是语义字段和时间线，不是强行把 NVIDIA 指标名称映射成海光指标。

### 13.2 必须画出的图

首轮结果至少生成：

1. `T_e2e` 随 `q` 的曲线；
2. `T_serial_same_partition` 与 `T_overlap` 对比；
3. `T_done`、`R_0`、`R_last` 和 `T_e2e` 的并列图；
4. 每个 slice 的 release curve；
5. whole/partitioned GEMM TFLOPS；
6. communication-only 与 communication+GEMM 的带宽/时间变化；
7. 同一 shape 上 isolated best、`T_done` best、`T_e2e` best 的策略排名；
8. NVIDIA 与海光各自 capability mask 和最终可行策略集合。

### 13.3 关键派生指标

```text
overlap_gain(q) =
  (T_serial_same_partition(q) - T_overlap(q)) /
  T_serial_same_partition(q)

partition_penalty(q) =
  (T_gemm_only_partition(q) - T_gemm_only_whole) /
  T_gemm_only_whole

first_release_window =
  (T_done - R_0) / T_done

release_jitter(i) =
  R_i - R_(i-1)

contention_loss =
  (T_gemm_only_partition - T_gemm_with_comm) /
  T_gemm_only_partition

normalized_regret(strategy) =
  (T_strategy - T_oracle) / T_oracle
```

这里的 `overlap_gain` 必须对比同一个 `q` 的串行分块基线，不能用 whole GEMM 作为分块 overlap 的分母。

---

## 14. 统计规则与实验划分

### 14.1 重复次数

建议：

```text
预热：20 次
方向探索：30 次以上
正式稳定点：50 次以上
```

每个点报告：

- median；
- P10/P90 或 P25/P75；
- mean 作为补充；
- 95% bootstrap CI；
- 异常和超时数量。

### 14.2 发现集和留出集

不能在所有 shape 上搜索完最优策略，再在同一批 shape 上声称 selector 泛化。建议：

```text
discovery set：约 60% shape，寻找 release/竞争现象
validation set：约 40% 未参与规则设计的 shape，验证选择方法
```

留出集还可以包含未参与调参的 `q` 或消息大小。若样本太少，至少按 shape 留一法报告，而不是只随机打散重复样本。

### 14.3 反例判据

一个有效反例必须同时满足：

```text
同一平台、同一 shape、同一 workload、同一正确性条件；
策略 A isolated busbw 更高或 T_done 更短；
策略 B T_e2e 更短；
差异超过重复测量 CI/噪声；
release curve、GEMM contention 或同步开销给出机制解释。
```

只看到两个硬件平台的数值不同，不算策略反例。

---

## 15. Go/No-Go：什么时候才加入 NVSHMEM/DUSHMEM 和 selector

### 15.1 Go 条件

继续发展“完成语义/依赖释放感知”方向，当且仅当：

1. 两个平台的 H0/B2 至少有两个稳定可行策略。
2. 10--20 个 shape 中有可重复的排序反转或近似带宽下明显不同的 release curve。
3. 该现象不是单纯由错误、频率波动、拓扑变化或分块 GEMM 退化造成。
4. profiler 能解释通信、计算和同步的时间关系。
5. PGAS 原语能在至少一个受控 workload 中正确推进，并且不会把 signal 提前误当 payload ready。
6. 一个离线 profile 驱动的轻量 selector 在留出 shape 上接近 exhaustive oracle。

### 15.2 No-Go 条件

出现以下情况时，应暂停该方向：

- isolated bandwidth、`T_done`、`T_e2e` 排名始终一致；
- 所有 overlap 方案都被 chunked GEMM 的巨大损失压倒；
- 只有某一个偶然 shape 有收益；
- NVSHMEM/DUSHMEM 只能运行但不能获得可验证的消费语义；
- 只有平台名不同，没有平台内部的策略变化；
- 几乎所有候选都不支持，无法形成策略选择问题；
- 为了得到收益必须重写 NCCL/RCCL/NVSHMEM/DUSHMEM 内部或构建庞大运行时。

No-Go 不是失败，而是避免在缺乏现象时人为包装论文题目。

---

## 16. 预期的论文方法形态，但现在不要提前承诺

如果 Go 条件成立，最终方法可以收敛成一个轻量的 capability-aware policy：

```text
输入：
  platform capability
  topology/transport
  collective type
  message size
  M,N,K,dtype
  q, window
  primitive release profile
  GEMM partition efficiency

过滤：
  不支持的算法/协议/channel
  不满足对齐、显存、同步和 dtype 约束的策略

预测：
  first-release cost
  release curve
  steady-state pipeline cost
  GEMM contention/partition cost
  drain cost

选择：
  在可行候选中最小化预测 T_e2e
```

可以使用如下抽象模型：

```text
T_hat(strategy) =
    startup_release_cost
  + steady_state_pipeline_cost
  + drain_cost
  + synchronization_cost
  + gemm_partition_cost
  + communication_compute_contention_cost
```

selector 的输入必须来自真实可采集、可验证的 profile，不要把无法实时获取的硬件内部状态写成模型必需输入。

正式评价应比较：

```text
whole-NCCL/RCCL
fixed DEFAULT overlap
isolated-bandwidth-best
T_done-best
exhaustive strategy oracle
proposed selector
```

至少报告：

- top-1 oracle hit rate；
- top-2 coverage；
- normalized regret；
- prediction error；
- profile/selection overhead；
- 两个平台的端到端 speedup；
- 去掉 release feature、去掉 contention feature、去掉 capability mask 的消融。

---

## 17. 本阶段最终交付清单

### 必须交付

```text
1. 两个平台事实快照
2. capability_matrix.csv
3. whole / serial_partition / overlap_event 的正确性程序
4. ag_gemm_runs.csv
5. ag_gemm_release_timeline.csv
6. 每个 case 的原始 stdout/stderr 和库日志
7. 至少三类 GEMM shape 的 whole/partitioned/overlap 对照
8. T_done、R_i、T_e2e、GEMM TFLOPS 分析图
9. 失败、超时和不支持组合审计表
10. Go/No-Go 判定记录
```

### 条件交付

只有 Phase 1/2 出现明确现象后才交付：

```text
11. NVSHMEM put/signal/wait 正确性 microbenchmark
12. DUSHMEM 对应原语正确性 microbenchmark
13. PGAS chunked path
14. bulk-NCCL + tail-PGAS path
15. capability-aware selector
16. 留出 shape 上的 oracle/regret 验证
```

---

## 18. 本文档使用的事实来源

### 本地材料

- `四卡NCCL-RCCL结果综合分析与第二篇论文定题建议.md`
- `第二篇论文相关工作排重与创新证据图谱.md`
- `统一实验配置-NCCL-RCCL-4090-K100AI (1).md`
- `../seconde-paper/plan/experiment-protocol.md`
- `../seconde-paper/DTK 26.04 RCCL库使用手册.pdf`
- `../seconde-paper/DTK 26.04 DUSHMEM库使用手册.pdf`
- `../seconde-paper/DTK 26.04 DUMMA使用手册.pdf`
- `/root/comm-study/install/nvshmem/include/host/nvshmemx_api.h`
- `/root/comm-study/install/nvshmem/include/host/nvshmem_api.h`

### 官方在线资料

- NCCL CUDA Stream Semantics：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html>
- NCCL collective API：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html>
- NCCL environment variables：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
- NVSHMEM project README：<https://github.com/NVIDIA/nvshmem>
- NVSHMEM Best Practice Guide APIs：<https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html>
- NVSHMEM CUDA interoperability：<https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/cuda-nvshmem-interop.html>
- CUTLASS：<https://github.com/NVIDIA/cutlass>
- cuBLASLt documentation：<https://docs.nvidia.com/cuda/cublas/index.html>
- rocBLAS documentation：<https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>
- hipBLASLt documentation：<https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/>

### 来源与结论边界

- NCCL 官方 stream 文档支持：NCCL 调用关联到传入 stream；调用返回表示操作已入队，操作随后异步执行；状态可通过 CUDA stream 或 event 查询。
- NVSHMEM 官方最佳实践和本机 3.8.0 头文件支持：存在 device-initiated、host/on-stream 和 stream-ordered 的多类 API；`flush_on_stream` 与 `quiet_on_stream` 的保证不同。
- CUTLASS 官方 README 支持其作为 CUDA C++ GEMM 和数据移动的可组合抽象；这不意味着 CUTLASS 能直接统一海光端，也不意味着采用 CUTLASS 本身就是论文贡献。
- rocBLAS/hipBLASLt 官方资料支持其作为海光侧 GEMM 库候选；具体 dtype、shape、算法和版本能力仍需目标机器实测。
- “release curve 能解释端到端排序”“NVSHMEM/DUSHMEM 融合会带来收益”“selector 能泛化”均是待实验验证的假设，当前不能当作结果。

---

## 19. 给当前执行者的最短操作顺序

不要一开始同时编译 CUTLASS、写 NVSHMEM kernel、修改 RCCL 和运行完整模型。实际顺序是：

```text
1. 核验 NVIDIA/A100 与海光/K100_ai 的平台事实
2. 用 CUDA/HIP C++ 写 whole AG-GEMM
3. 加入 same-partition serial baseline
4. 加入 NCCL/RCCL slice + event overlap
5. 记录 T_done、R_i、T_e2e 和 GEMM TFLOPS
6. 判断是否出现排序反转
7. 独立验证 NVSHMEM/DUSHMEM put/fence/quiet/signal/wait
8. 只有正确且有必要时加入 PGAS 路径
9. 最后才考虑 CUTLASS/DUMMA 自定义 GEMM 和 selector
```

本阶段的核心不是“把更多库都用一遍”，而是建立一条可以解释、复现和证伪的证据链：

```text
通信配置
  -> 合法完成/释放时间
  -> GEMM 分片启动
  -> 资源竞争与计算效率
  -> 端到端时间
  -> 跨 NVIDIA/海光的策略边界
```
