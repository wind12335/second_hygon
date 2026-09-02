# RCCL 四卡实验教学与结果解读

本文档用于解释当前海光 `K500SM_AI × 4`、`gfx928`、PCIe 环境下已经完成的 RCCL 实验。

目标读者假设为：了解大模型分布式训练，但还不熟悉 RCCL/NCCL collective、算法/协议选择、channel 调优和实验日志分析。

本文档先解释实验在测什么，再解释每类实验为什么重要，最后说明如何读取日志和 CSV，并总结目前已经观察到的研究线索。

---

## 1. 当前实验平台和结果位置

本批实验固定使用以下平台，所有实验没有混用其他 GPU 架构：

```text
GPU/DCU       K500SM_AI × 4
架构          gfx928
互联          PCIe
通信库        RCCL
数据类型      float32
warmup        10 次
计时迭代      50 次
正确性检查    开启
计时汇总      取 max rank time
```

固定总日志：

```text
/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/formal_master.log
```

修正后的便于阅读的汇总 CSV：

```text
/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/formal_summary.csv
```

需要注意：原始 RCCL-tests CSV 的表头和数据行存在一个额外的 size unit 字段，直接用普通表格软件打开时列可能看起来错位。第一次阅读建议优先使用 `formal_summary.csv`，原始 CSV 用于复核。

主要结果目录如下：

```text
4 卡平台记录：
  results/k500sm_ai_gfx928_4gpu/20260825T100716Z/platform/

4 卡正确性预检：
  results/k500sm_ai_gfx928_4gpu/20260825T100731Z/

4 卡代表性策略矩阵：
  results/k500sm_ai_gfx928_4gpu/20260825T101556Z/

4 卡 channel 扫描：
  results/k500sm_ai_gfx928_4gpu/20260825T102643Z/

4 卡 rank mapping：
  results/k500sm_ai_gfx928_4gpu/20260825T104241Z/

2 卡聚焦矩阵：
  results/k500sm_ai_gfx928_2gpu/20260825T104341Z/

4 卡 1 GiB pilot：
  results/k500sm_ai_gfx928_4gpu/1g_pilot/

LL128 探测：
  results/k500sm_ai_gfx928_4gpu/ll128_probe/

AG-GEMM overlap 探索：
  results/k500sm_ai_gfx928_4gpu/overlap_dushmem_rccl/
```

---

## 2. 为什么要测 collective

在分布式大模型中，GPU 不仅执行 GEMM，还需要不断交换激活、梯度或部分结果。RCCL/NCCL 的 collective API 就是用于多 GPU 集体通信的接口。

### 2.1 AllGather

每张 GPU 把自己的数据发送给所有其他 GPU，最终每张 GPU 都拥有完整数据。

4 张 GPU 的抽象过程：

```text
开始：
GPU0: A0
GPU1: A1
GPU2: A2
GPU3: A3

结束：
GPU0: A0 A1 A2 A3
GPU1: A0 A1 A2 A3
GPU2: A0 A1 A2 A3
GPU3: A0 A1 A2 A3
```

AllGather 常见于：

```text
Tensor Parallel
激活值交换
部分 MoE token exchange
通信后需要继续执行 GEMM 的流水线
```

AllGather 重点回答：

```text
数据分发带宽有多高？
不同协议的延迟和吞吐有什么差异？
某一段数据什么时候能够被后续计算消费？
```

### 2.2 AllReduce

所有 GPU 的数据进行规约，通常是求和，然后每张 GPU 都得到最终结果。

```text
GPU0: X0
GPU1: X1
GPU2: X2
GPU3: X3

结束：
所有 GPU 都得到 X0 + X1 + X2 + X3
```

AllReduce 常见于：

```text
数据并行训练的梯度同步
参数同步
分布式优化器
```

AllReduce 同时包含数据搬运和规约计算，因此通常不能简单等同于一次 memcpy。

### 2.3 ReduceScatter

ReduceScatter 先对所有 GPU 的数据进行规约，然后每张 GPU 只保留结果的一部分。

它可以理解为 AllReduce 的“规约后分片”版本，常见于：

```text
Tensor Parallel
Sequence Parallel
分布式优化器
ReduceScatter + GEMM 流水线
```

这三个 collective 必须分别测试，因为：

```text
AllGather 的最佳策略
不一定是 AllReduce 的最佳策略
也不一定是 ReduceScatter 的最佳策略
```

---

## 3. 如何理解算法、协议和 channel

### 3.1 DEFAULT

DEFAULT 表示不通过环境变量强制选择算法或协议，由 RCCL 根据消息大小、GPU 数量和拓扑自行选择。

它是所有调优实验的基准线。

如果某种强制配置只比 DEFAULT 快 1% 左右，通常不足以单独支撑一个研究问题；如果强制配置在多个场景下稳定优于 DEFAULT，才值得进一步分析。

### 3.2 Ring 算法

Ring 把 GPU 组织成一个环：

```text
GPU0 -> GPU1 -> GPU2 -> GPU3 -> GPU0
```

数据被切成多个块，在环上逐步转发。Ring 通常适合大消息和吞吐导向的通信，但实际效果取决于 PCIe、XGMI、NVLink 或网络拓扑。

### 3.3 Tree 算法

Tree 把 GPU 组织成树形结构。它可能减少某些小消息场景的通信步骤，通常需要观察其延迟和规约行为是否优于 Ring。

Ring 和 Tree 不能只从理论结构判断谁一定更快，必须在实际平台上测量。

### 3.4 Simple 协议

Simple 可以粗略理解为吞吐导向的常规协议。在当前 K500SM_AI/gfx928/PCIe 平台的实验中，Simple 是最稳定、最有效的协议之一。

### 3.5 LL 协议

LL 是 Low Latency 的缩写，设计目标通常是降低小消息延迟。

重要的是：

```text
LL 不是任何消息大小下都更快。
```

在当前海光平台上，强制 LL 在中大消息上经常显著变慢。它可能适合某些小消息场景，但不能默认用于大消息。

### 3.6 LL128 协议

LL128 是另一种低延迟协议变体。当前本机 RCCL 对 Ring/LL128 和 Tree/LL128 的探测均返回状态 3，并出现：

```text
no algorithm/protocol available
internal error
```

因此当前平台应把 LL128 记录为：

```text
UNSUPPORTED 或 NO_VALID_ALGORITHM
```

不能把它记成“带宽为 0”，也不能把失败配置和有效性能配置混合排名。

### 3.7 Channel

channel 可以理解为 RCCL 同时推进通信的并行通道数。

```text
channel=1：并行通信度较低
channel=4：同时推进更多通信块
channel=8：并行度更高，但可能消耗更多 GPU 资源
```

channel 太少可能导致通信链路没有被充分利用；channel 太多可能与 GEMM 争抢：

```text
SM/CU
显存带宽
缓存
调度资源
```

所以 channel 不是越大越好，而是需要结合消息大小、collective、拓扑和计算负载选择。

---

## 4. 已经完成的实验及意义

### 4.1 平台事实记录

记录了：

```text
GPU/DCU 型号
gfx928 架构
DTK/HIP/RCCL 版本
PCIe 拓扑
CPU 和 NUMA
MPI 版本
实际加载的 librccl.so
rank 到物理 GPU 的映射
```

意义：确定实验的可复现条件。没有平台事实，后续无法判断性能差异来自算法、库版本还是硬件拓扑变化。

### 4.2 正确性预检

测试了三个 collective，消息大小为 4 KiB、1 MiB 和 64 MiB。

所有配置都通过：

```text
exit_status = 0
wrong_count = 0
Out of bounds values : 0 OK
```

意义：只有正确的配置才能进入性能分析。一个很快但结果错误的通信配置没有研究价值。

### 4.3 四卡代表性矩阵

测试内容：

```text
collective:
  AllGather
  AllReduce
  ReduceScatter

message size:
  4 KiB
  64 KiB
  1 MiB
  8 MiB
  64 MiB
  256 MiB

strategy:
  DEFAULT
  Ring/Simple
  Ring/LL
  Tree/Simple
  Tree/LL

每个 case：3 次独立运行
```

总计 270 个 case，全部逐 case 通过。

意义：建立四卡孤立通信的主基线，并观察 DEFAULT 与强制策略之间的差异。

### 4.4 四卡 channel 扫描

测试：

```text
collective: AllGather、AllReduce、ReduceScatter
size: 1 MiB、8 MiB、64 MiB
strategy: Ring/Simple、Ring/LL、Tree/Simple、Tree/LL
channels: 1、2、4、8
每个 case：3 次
```

总计 432 个 case，全部逐 case 通过。

意义：判断 channel 是否是重要调优维度，以及最优 channel 是否随着消息大小和 collective 改变。

### 4.5 Rank mapping

测试了：

```text
0123
0213
0321
```

即不同的 rank 到物理 GPU 映射方式。

测试了 AllGather 和 AllReduce 的 1 MiB、64 MiB 配置，每种映射重复 3 次。

意义：判断 PCIe root complex、NUMA 和 GPU 物理拓扑是否需要进入策略选择器。

### 4.6 两卡聚焦矩阵

两卡使用：

```text
HIP_VISIBLE_DEVICES=0,1
mpirun -np 2
```

测试三个 collective、64 KiB/1 MiB/8 MiB/64 MiB，以及 DEFAULT、Ring/Simple、Ring/LL、Tree/Simple、Tree/LL。

意义：验证四卡上观察到的规律是否只属于四卡，并研究 rank 数改变后策略边界是否移动。

### 4.7 1 GiB pilot

四卡分别测试了：

```text
AllGather 1 GiB
AllReduce 1 GiB
ReduceScatter 1 GiB
```

三个配置全部正确通过：

```text
AllGather busbw       ≈ 12.8045 GB/s
AllReduce busbw       ≈ 13.1464 GB/s
ReduceScatter busbw   ≈ 12.2217 GB/s
```

意义：确认当前海光四卡能够稳定覆盖到 1 GiB 消息范围。

注意 AllGather 中的 1 GiB 是每个 rank 的输入大小，不是整个 collective 的总输出大小。

### 4.8 LL128 探测

探测了 Ring/LL128 和 Tree/LL128，在 1 MiB、8 MiB、64 MiB 上各运行两次。

36 个 case 全部返回状态 3，并出现：

```text
no algorithm/protocol available
```

意义：明确记录当前 RCCL/gfx928 平台的协议能力边界。跨 NVIDIA 和海光比较时，不能强行使用完全相同的策略空间。

### 4.9 AG-GEMM overlap 探索

已有程序比较：

```text
RCCL 完整 AllGather + rocBLAS 整块 GEMM
```

与：

```text
DUSHMEM 分块通信 + 分块 rocBLAS GEMM
```

测试 chunk 数 1、2、4、8。

结果：

```text
chunks=1：RCCL baseline ≈ 184.0 ms，DUSHMEM ≈ 217.2 ms
chunks=2：RCCL baseline ≈ 184.0 ms，DUSHMEM ≈ 1292.1 ms
chunks=4：RCCL baseline ≈ 190.1 ms，DUSHMEM ≈ 1293.6 ms
chunks=8：RCCL baseline ≈ 184.3 ms，DUSHMEM ≈ 1291.2 ms
```

另外：

```text
整块 GEMM ≈ 159 ms
分块 GEMM ≈ 1247～1248 ms
```

意义：说明增加 chunk 数并不自动带来端到端收益，分块 GEMM 的效率损失可能抵消通信重叠收益。

但这仍是探索性结果，不是最终严格的 NCCL/RCCL overlap 对比，因为两边使用的通信和计算执行形态还不完全对称。

---

## 5. 当前实测结果如何解读

### 5.1 DEFAULT 在大消息上通常已经接近最优

四卡代表性矩阵中的典型结果：

```text
AllGather，1 MiB：
  DEFAULT ≈ 6.441 GB/s
  最好强制配置 ≈ 6.421 GB/s

AllGather，64 MiB：
  DEFAULT ≈ 12.240 GB/s
  最好强制配置 ≈ 12.256 GB/s

AllReduce，1 MiB：
  DEFAULT ≈ 7.942 GB/s
  Ring/Simple ≈ 7.949 GB/s

AllReduce，64 MiB：
  DEFAULT ≈ 12.832 GB/s
  Ring/Simple ≈ 12.859 GB/s
```

结论：当前数据不支持“固定 Ring/Simple 可以全面超过 RCCL 默认策略”。更准确的结论是：默认启发式在大消息上整体较好，但策略空间中存在平台相关的无效或低效配置。

### 5.2 LL 在当前平台的中大消息上明显退化

四卡 AllGather：

```text
1 MiB：
  DEFAULT ≈ 6.441 GB/s
  Ring/LL ≈ 1.879 GB/s

64 MiB：
  DEFAULT ≈ 12.240 GB/s
  Ring/LL ≈ 2.016 GB/s
```

四卡 AllReduce：

```text
1 MiB：
  DEFAULT ≈ 7.942 GB/s
  Ring/LL ≈ 3.392 GB/s

64 MiB：
  DEFAULT ≈ 12.832 GB/s
  Ring/LL ≈ 3.479 GB/s
```

结论：不能把 NVIDIA 上常见的 LL 配置不加判断地迁移到当前海光平台。协议选择必须感知后端能力和消息大小。

### 5.3 channel 数确实影响性能

四卡 AllGather、Ring/Simple：

```text
1 MiB：
  ch1 ≈ 3.557 GB/s
  ch2 ≈ 5.290 GB/s
  ch4 ≈ 6.426 GB/s
  ch8 ≈ 6.522 GB/s

64 MiB：
  ch1 ≈ 4.956 GB/s
  ch2 ≈ 9.886 GB/s
  ch4 ≈ 12.123 GB/s
  ch8 ≈ 12.688 GB/s
```

结论：从 1 增加到 2/4 通常有明显收益，从 4 增加到 8 的收益开始变小。channel 应作为策略选择器的输入之一。

### 5.4 Rank mapping 影响目前较小

四卡 AllGather、1 MiB：

```text
0123 ≈ 6.390 GB/s
0213 ≈ 6.424 GB/s
0321 ≈ 6.443 GB/s
```

四卡 AllGather、64 MiB：

```text
0123 ≈ 12.253 GB/s
0213 ≈ 12.129 GB/s
0321 ≈ 12.103 GB/s
```

当前差异多数没有超过 5%，暂时不是最强研究信号。但在多 NUMA、多 root complex 或跨节点环境下，mapping 可能更加重要。

---

## 6. 如何看懂一个具体 log

例如一个 RCCL-tests log 中可能出现：

```text
# Collective test starting: all_gather_perf
# nThread 1 nGpus 1 minBytes 1048576 maxBytes 1048576
# warmup iters: 10 iters: 50 validation: 1
#       size         count      type ... time   algbw   busbw #wrong
    1048576       262144     float ... 121.855 8.605 6.454 0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 6.453818
# EXIT_STATUS 0
```

应这样理解：

```text
测试对象：AllGather
每 rank 输入大小：1 MiB
数据类型：float32
warmup：10
计时迭代：50
平均时间：约 121.855 微秒
算法带宽：约 8.605 GB/s
总线带宽：约 6.454 GB/s
错误元素：0
程序退出：成功
```

读取一个配置时，推荐顺序是：

```text
1. 先看 wrong_count 是否为 0
2. 再看 status 是否为 0
3. 再看 time_us
4. 再看 busbw_gbps
5. 对同一个配置的多次重复运行计算均值和波动
6. 最后与 DEFAULT 比较
```

不要只看 `GB/s`，也不要把 `algbw` 和 `busbw` 混为同一个指标。

---

## 7. 推荐的查看命令

查看总阶段日志：

```bash
less /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/formal_master.log
```

查看代表性矩阵的 case 清单：

```bash
column -t -s $'\t' \
  /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/20260825T101556Z/manifest.tsv \
  | less -S
```

查看最终汇总：

```bash
column -s, -t \
  /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/formal_summary.csv \
  | less -S
```

查看正确性：

```bash
rg -n '# Out of bounds values|#wrong|# EXIT_STATUS' \
  /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/20260825T101556Z/logs \
  | head -30
```

查看 LL128 失败原因：

```bash
rg -n 'no algorithm/protocol available|internal error|EXIT_STATUS' \
  /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/ll128_probe/logs
```

查看 overlap 结果：

```bash
rg -n 'base=|DUSHMEM|EXIT_STATUS' \
  /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/overlap_dushmem_rccl/logs
```

---

## 8. 如何判断某个策略是否值得研究

假设要比较四卡 AllGather、1 MiB 的五种策略。

先筛掉：

```text
wrong_count != 0
status != 0
UNSUPPORTED
TIMEOUT
```

然后比较三次重复运行的平均 `busbw_gbps`。

例如：

```text
DEFAULT     = 6.40 GB/s
Ring/Simple = 6.42 GB/s
Ring/LL     = 1.88 GB/s
Tree/Simple = 6.41 GB/s
Tree/LL     = 1.86 GB/s
```

则可以得出：

```text
Ring/Simple、Tree/Simple 和 DEFAULT 几乎一样；
LL 在这个场景显著低效；
默认策略没有明显性能缺口；
真正有价值的研究方向更可能是“如何识别并避开低效策略”，而不是固定使用某个策略。
```

如果要计算相对 DEFAULT 的加速比：

```text
speedup = candidate_busbw / default_busbw
```

例如：

```text
candidate = 6.60 GB/s
default   = 6.00 GB/s
speedup   = 1.10
```

说明 candidate 比 DEFAULT 快 10%。

如果只有 1%～2% 的单点收益，通常不足以作为独立论文问题；如果收益在多个 collective、多个消息大小和多个 rank 数上稳定存在，才更有研究意义。

---

## 9. 当前实验对第二篇论文的启示

当前数据不支持以下简单论断：

```text
固定 Ring/Simple 就能全面超过 RCCL 默认策略。
```

当前数据支持更有价值的三个研究假设。

### 9.1 跨通信基座的有效策略集合不同

当前海光平台实测：

```text
Simple：有效
LL：中大消息明显退化
LL128：不可用
```

因此策略选择器的输入不能只有消息大小和 GPU 数量，还应该包括：

```text
backend
transport
P2P capability
protocol availability
channel range
GPU/DCU 计算能力
内存带宽
通信与计算资源竞争特征
```

### 9.2 channel 应和消息大小、collective 联合选择

channel=1、2、4、8 的性能差异很明显，而且不同 collective 的趋势并不完全相同。

策略选择不能只做：

```text
消息大小 -> protocol
```

更合理的是考虑：

```text
消息大小 + collective + rank 数 + transport + channel
```

### 9.3 孤立通信最优不等于端到端重叠最优

overlap 探索显示：

```text
增加 chunk 数没有自动带来端到端收益；
分块 GEMM 可能严重变慢；
通信和计算必须联合建模。
```

后续严格论文实验需要统一实现：

```text
T_comm
T_gemm
T_serial
T_overlap
T_fill
T_steady_state
T_drain
T_e2e
```

核心问题是：

```text
孤立通信最快的配置，是否仍然是 AG-GEMM 端到端最快的配置？
```

如果在多个消息大小、chunk 数、channel 和平台上都观察到：

```text
配置 A：孤立通信最快
配置 B：端到端 overlap 最快
```

那么“跨通信基座能力感知的端到端重叠策略选择”就具有比较强的研究动机。

---

## 10. 最后应记住的四句话

```text
1. 当前 K500SM_AI/gfx928 四卡 RCCL 默认策略在大消息上整体已经较好。
2. 当前平台的 LL 在中大消息上明显低效，LL128 不可用。
3. channel 数对带宽有明显影响，1/2/4/8 之间存在稳定差异。
4. 分块通信和分块 GEMM 的端到端效果不能由孤立通信带宽直接推断。
```

当前这批实验完成的是：

```text
RCCL 通信基座和策略空间摸底。
```

尚未完成、但最值得继续做的是：

```text
严格统一的 NCCL/RCCL AG-GEMM overlap harness；
按通信 slice 或 tile 记录 release time；
比较 isolated optimum 与 end-to-end optimum 是否一致。
```

