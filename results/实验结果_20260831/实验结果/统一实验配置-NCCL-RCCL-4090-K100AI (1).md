# NCCL/RCCL 统一实验配置规范

## 1. 文档目的

本文档用于在 NVIDIA RTX 4090 与海光 K100AI 平台上，使用尽可能一致的实验口径测试 NCCL/RCCL，并为后续协议、算法、channel、拓扑和通信计算重叠研究提供统一原始数据。

本文档不要求两种硬件的物理能力相同，而是要求所有实验条件和平台限制都被明确记录。目标是保证：

- collective 语义一致；
- 数据类型、消息大小和 rank 数一致；
- warmup、重复次数、同步位置和带宽公式一致；
- 默认策略与强制策略的测试矩阵一致；
- 正确性、transport、拓扑和平台限制都有原始证据；
- 所有结果可由 CSV 和原始日志重新生成。

关联记录：

- NVIDIA 双 RTX 4090 首轮记录：/root/comm-study/NCCL实验结果-双RTX4090.md；
- 海光 RCCL 首轮记录：/root/L0-L1_平台事实与RCCL首轮结果.md。

## 2. 总原则

### 2.1 必须统一的项目

以下项目必须在 NVIDIA 与海光之间保持一致：

| 项目 | 统一要求 |
|---|---|
| collective | 同一 collective，例如两端都测 AllGather |
| datatype | 第一轮统一为 float32 |
| per-rank message size | 使用相同的每 rank 输入字节数 |
| number of ranks | 2 卡对 2 卡，4 卡对 4 卡 |
| warmup | 10 次 |
| timed iterations | 50 次 |
| stream synchronization | 同一位置、同一类型的设备同步 |
| correctness check | 每个配置都启用 |
| bandwidth formula | 使用同一分子和单位 |
| statistical reporting | 至少三次独立运行，报告均值和波动 |

### 2.2 不能人为统一的项目

以下项目不能为了对比而强行设置成相同，必须如实记录：

- GPU/DCU 型号；
- 显存容量；
- 驱动和 CUDA/HIP/DTK 版本；
- PCIe、NVLink、XGMI、HYLink 等拓扑；
- P2P 是否可用；
- SHM、P2P、IB/RoCE 等实际 transport；
- 协议和算法的可用集合；
- NUMA 与 CPU 亲和性；
- 是否需要平台特定环境变量。

如果两端实际 transport 不同，结果可以作为不同平台案例或不同回退路径案例，但不能直接解释为“库 A 比库 B 快”。

### 2.3 双卡和四卡的作用

| GPU 数量 | 主要目的 |
|---:|---|
| 1 | 单卡内存和 GEMM 基线，不作为 collective 性能结论 |
| 2 | API、正确性、基础延迟/带宽、P2P/SHM 和协议初步规律 |
| 4 | Ring/Tree、channel 扩展、rank mapping、拓扑和多卡 scaling |
| 8 或更多 | 条件允许时验证结论能否继续扩展 |

如果论文要研究自适应通信策略，建议 NVIDIA 和海光最终都完成 1/2/4 卡测试。只有双卡时，不应把 Ring/Tree、拓扑感知或多卡扩展结论推广到四卡以上。

如果当前平台只有两张卡，继续完成双卡基线没有问题，但不要使用两台双卡机器模拟四卡单机实验。多节点实验会引入网络路径，应作为单独的 multi-node 条件记录。

## 3. 实验对象与版本记录

### 3.1 NVIDIA 侧

| 项目 | 实际值 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 4090 |
| GPU 数量 | 2 卡或 4 卡，按实际机器填写 |
| CUDA Toolkit | 填写，例如 12.6.85 |
| NVIDIA Driver | 填写 |
| NCCL 版本 | 填写 |
| NCCL commit | 填写 |
| nccl-tests 版本/commit | 填写 |
| 编译架构 | sm_89 |
| 主机名 | 填写 |
| 操作系统和内核 | 填写 |

当前已有双 4090 基线使用独立安装目录 /root/comm-study/install/nccl。不要用系统库、PyTorch 自带 NCCL 或 AutoCCL 的 NCCL 库替代。每次实验应使用 ldd 或 LD_DEBUG=libs 确认实际加载的 libnccl.so。

### 3.2 海光侧

| 项目 | 实际值 |
|---|---|
| DCU 型号 | 海光 K100AI |
| DCU 数量 | 2 卡或 4 卡，按实际机器填写 |
| GCN/ISA 架构 | 填写，必须以 rocminfo 实测为准 |
| DTK 版本 | 填写 |
| HIP/ROCm 版本 | 填写 |
| RCCL 版本 | 填写 |
| RCCL commit | 填写 |
| rccl-tests 或统一 harness 版本 | 填写 |
| 编译架构 | 填写，例如 --offload-arch=gfx928 |
| 主机名 | 填写 |
| 操作系统和内核 | 填写 |

K100AI 的架构不能根据旧记录猜测。每次换机器或换卡都应重新执行 rocminfo，并以实际架构编译。

### 3.3 版本和库路径检查

NVIDIA 命令：

~~~bash
nvidia-smi
nvidia-smi -L
nvcc --version
ldd ./build/all_reduce_perf | grep -E 'libnccl|libcudart'
~~~

海光命令：

~~~bash
rocm-smi --showproductname --showtopotype
rocminfo
hipcc --version
ldd ./build/rccl_test_or_unified_harness | grep -E 'librccl|libamdhip64'
~~~

## 4. L0 平台能力记录

在任何性能测试前，必须先保存平台事实。建议目录：

~~~text
results/
├── 4090/
│   ├── 2gpu/platform.txt
│   └── 4gpu/platform.txt
└── k100ai/
    ├── 2gpu/platform.txt
    └── 4gpu/platform.txt
~~~

### 4.1 NVIDIA 平台检查

~~~bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
nvidia-smi -q -d P2P
lscpu
numactl --hardware
~~~

还应使用小型 CUDA 程序逐方向检查：

~~~cpp
cudaDeviceCanAccessPeer(&canAccess, deviceA, deviceB);
~~~

nvidia-smi topo -p2p 和 CUDA API 的结果都要保留。特别注意 CNS、GNS、TNS 和 OK 的含义不能混用。

当前双 RTX 4090 已确认 P2P 读写为 CNS，NCCL 实际使用 SHM/direct。因此当前结果应标记为 P2P unavailable / SHM fallback，而不是 PCIe P2P 性能。

### 4.2 海光平台检查

~~~bash
rocm-smi --showproductname
rocm-smi --showtopotype
rocminfo
lscpu
numactl --hardware
~~~

还应使用小型 HIP 程序逐方向检查：

~~~cpp
hipDeviceCanAccessPeer(&canAccess, deviceA, deviceB);
~~~

如果 RCCL 日志出现以下警告，必须作为平台条件写入结果：

~~~text
NUMA auto balancing enabled
Missing iommu=pt
Missing HSA_FORCE_FINE_GRAIN_PCIE=1
No IB NIC found
~~~

如果某环境变量是运行 LL 协议的必要条件，应在所有 RCCL 配置中统一设置，并在 CSV 的 env_profile 字段记录。

### 4.3 transport 证据

每个平台至少保存一次详细初始化日志。

NVIDIA：

~~~bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,TUNING \
./run-one-small-test.sh 2>&1 | tee results/4090/transport-info.log
~~~

海光：

~~~bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,TUNING \
./run-one-small-test.sh 2>&1 | tee results/k100ai/transport-info.log
~~~

性能正式跑使用 NCCL_DEBUG=WARN 或关闭 INFO，避免大量调试输出扰乱性能表。详细 INFO 只作为独立的 transport 和选路证据。

需要从日志确认：

~~~text
actual algorithm
actual protocol
actual channel count
actual transport
rank-to-device mapping
P2P or SHM or network path
~~~

不能只根据环境变量是否设置来推断实际执行配置。

## 5. 统一测试程序和运行模型

### 5.1 推荐：同一份 C++ harness 条件编译

为了消除 nccl-tests 与自定义 rccl_smoke.cpp 之间的计时差异，推荐使用同一份测试程序源代码：

~~~text
common collective harness
├── CUDA + NCCL build
└── HIP + RCCL build
~~~

通过编译宏替换 runtime 和 library 调用：

| 功能 | CUDA/NCCL | HIP/RCCL |
|---|---|---|
| 分配 | cudaMalloc | hipMalloc |
| 释放 | cudaFree | hipFree |
| stream | cudaStream_t | hipStream_t |
| event | cudaEvent_t | hipEvent_t |
| 同步 | cudaEventSynchronize | hipEventSynchronize |
| collective | ncclAllGather | RCCL ncclAllGather |
| communicator | NCCL API | RCCL API |

如果暂时不能使用同一份 harness，允许先分别使用 nccl-tests 和 rccl-tests 完成平台摸底，但这类结果必须标记为：

~~~text
platform baseline, not strict cross-vendor comparison
~~~

只有当两边的自定义计时程序经过交叉验证后，才可以将结果用于严格的 NVIDIA/RCCL 性能图。

### 5.2 进程模型

单机多卡统一采用：

~~~text
one process per GPU/rank
one communicator containing all local ranks
one dedicated stream for collective
~~~

两边都要明确记录：

~~~text
rank 0 -> physical device 0
rank 1 -> physical device 1
...
~~~

不允许一边使用单进程多 GPU，另一边使用 MPI 多进程，然后不说明差异。

如果使用 MPI 启动海光侧：

~~~bash
mpirun --allow-run-as-root -np NRANKS -mca coll ^hcoll ./unified_harness
~~~

NVIDIA 侧可以使用统一 harness 的 MPI 启动方式，或者使用官方 nccl-tests，但必须把进程模型写入结果。

## 6. 统一测试参数

### 6.1 数据类型

第一轮统一只使用：

~~~text
datatype = float32
reduction = sum（对 AllReduce/ReduceScatter）
~~~

第二轮再扩展：

~~~text
float16
bfloat16（两端都支持时）
~~~

不同数据类型必须分开画图，不允许把 FP32、FP16 和 BF16 放在同一条带宽曲线中。

### 6.2 消息大小

第一轮两端统一使用每 rank 消息大小：

~~~text
4 KiB
64 KiB
1 MiB
8 MiB
64 MiB
256 MiB
1 GiB
~~~

如果 1 GiB 因显存或运行时间无法完成，应两端同时降到 256 MiB，并记录 max_tested_bytes。不能 NVIDIA 测到 1 GiB、海光只测到 64 MiB 后直接比较平台峰值。

正式扩展曲线可以使用 2 倍递增：

~~~text
4 KiB 到 1 GiB，step factor = 2
~~~

RCCL 文档中的 per-rank message size 必须与 NCCL 侧的 -b/-e 语义对应。AllGather 不能把总输出大小误写成输入 per-rank 大小。

### 6.3 warmup 和计时

统一设置：

~~~text
warmup = 10
timed iterations = 50
~~~

计时流程：

~~~text
communicator initialization
device buffer initialization
correctness warmup
MPI/process barrier（如使用 MPI）
warmup collective x 10
barrier
timed collective x 50
每次操作后在对应 collective stream 上完成 device synchronization
记录所有 rank 的时间并取 max rank time
~~~

初始化时间不计入 collective 时间。显存分配、communicator 创建和首次 kernel 加载时间也不计入性能结果，但必须另行记录。

推荐每次计时使用 CUDA/HIP device event，而不是只使用 host wall clock：

~~~text
start_event -> collective -> stop_event -> event synchronize
~~~

如果程序使用 MPI_Wtime，必须用 CUDA/HIP event 交叉验证一次，确认 host 时间没有把进程调度或 enqueue 开销误计入 GPU 通信时间。

### 6.4 重复运行和统计

每个配置至少独立运行三次：

~~~text
run 1
run 2
run 3
~~~

报告：

~~~text
mean
median
minimum
standard deviation
coefficient of variation
~~~

如果大消息带宽的三次运行变异系数超过 5%，应检查：

~~~text
GPU clock
power limit
NUMA affinity
其他进程
NUMA auto balancing
网络插件
系统负载
~~~

## 7. 第一轮统一 collective 矩阵

### 7.1 必做 collective

两端必须共同完成：

~~~text
AllGather
AllReduce
ReduceScatter
~~~

可选：

~~~text
Broadcast
AllToAll（两端实现和测试定义都一致时）
~~~

第一轮不要把标准 RCCL AllGather 与自定义 duNcclAllgather 混在一组结果里。两者必须分别标记：

~~~text
collective_api = ncclAllGather
collective_api = duNcclAllgather
~~~

### 7.2 正确性要求

每个配置必须启用正确性检查。结果字段至少包括：

~~~text
correctness = PASS/FAIL
wrong_count
first_error_rank
first_error_index
~~~

只有 wrong_count = 0 的配置才能进入性能汇总。挂起、超时、异常退出和结果错误必须保留原始日志，不能从最终表中删除。

## 8. 默认策略与强制策略矩阵

### 8.1 默认策略

每个平台先运行完全不设置 algorithm、protocol、channel 的默认配置：

~~~text
algorithm = DEFAULT
protocol = DEFAULT
channels = DEFAULT
~~~

默认配置必须作为每张图的基准线。

### 8.2 algorithm 和 protocol

两端根据实际支持情况测试：

~~~text
algorithm: Ring, Tree
protocol: LL, LL128, Simple
~~~

如果某个平台不支持某个 protocol，记录：

~~~text
status = UNSUPPORTED
~~~

不要将不支持配置记为 0 GB/s，也不要把不支持配置和真正性能很差的配置混在一起。

常用环境变量模板：

~~~bash
NCCL_ALGO=Ring
NCCL_PROTO=LL
~~~

或：

~~~bash
NCCL_ALGO=Tree
NCCL_PROTO=Simple
~~~

NVIDIA 和 RCCL 版本对环境变量支持范围可能不同。每次强制配置都必须用 INFO 日志确认实际 algorithm 和 protocol。

### 8.3 channel 数

两端优先扫描：

~~~text
1, 2, 4, 8 channels
~~~

如果后端通过最小和最大 channel 变量控制固定 channel 数，可以使用：

~~~bash
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
~~~

如果实际版本支持其他 channel 控制变量，应以该版本文档和 INFO 日志为准。目标是同时记录：

~~~text
requested_channels
actual_channels
~~~

### 8.4 首轮减少组合数量

不建议一开始测试全部笛卡尔积。第一轮可采用：

| 消息大小 | 默认 | Ring/Simple | Ring/LL | Tree/Simple | Tree/LL |
|---:|---:|---:|---:|---:|---:|
| 4 KiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 64 KiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1 MiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 MiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 64 MiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 256 MiB | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1 GiB | ✓ | ✓ | ✓ | ✓ | ✓ |

对每个 algorithm/protocol 组合再扫描 1/2/4/8 channels。如果时间过长，先在 1 MiB、8 MiB、64 MiB 三个代表点扫描 channel，再扩展完整消息曲线。

## 9. 2 卡与 4 卡测试矩阵

### 9.1 2 卡必做矩阵

~~~text
nranks: 2
collective: AllGather, AllReduce, ReduceScatter
datatype: float32
sizes: 4 KiB, 64 KiB, 1 MiB, 8 MiB, 64 MiB, 256 MiB, 1 GiB
algorithm: DEFAULT, Ring, Tree
protocol: DEFAULT, LL, LL128, Simple（按支持情况）
channels: DEFAULT；代表点扫描 1, 2, 4, 8
repetitions: warmup 10, timed 50
runs: 3
~~~

2 卡重点观察：

~~~text
P2P 是否可用
SHM/direct 是否被启用
小消息延迟
大消息带宽平台
protocol 切换点
channel 是否影响性能
~~~

### 9.2 4 卡必做矩阵

~~~text
nranks: 4
collective: AllGather, AllReduce, ReduceScatter
datatype: float32
sizes: 4 KiB, 64 KiB, 1 MiB, 8 MiB, 64 MiB, 256 MiB
algorithm: DEFAULT, Ring, Tree
protocol: DEFAULT, LL, Simple（LL128 按支持情况）
channels: DEFAULT；1, 2, 4, 8
repetitions: warmup 10, timed 50
runs: 3
~~~

4 卡新增观察项：

~~~text
1/2/4 rank scaling
Ring/Tree 差异
rank mapping 差异
channel 扩展效率
P2P 矩阵是否对称
PCIe root complex 和 NUMA 影响
实际 Ring 顺序和 Tree 结构
~~~

### 9.3 Rank mapping

至少测试：

~~~text
natural mapping: 0,1,2,3
topology-aware mapping: 根据实际拓扑设计
~~~

每种 mapping 都应记录：

~~~text
rank_to_device
GPU/DCU BDF
NUMA node
actual ring order
actual tree order
~~~

如果不同 mapping 的差异超过 5%，应单独进行拓扑分析，不要将结果合并成一个平均值。

## 10. 启动命令模板

以下命令是参数模板，路径和 GPU 数量必须按实际平台修改。

### 10.1 NVIDIA 默认配置

使用 nccl-tests 时：

~~~bash
cd /path/to/nccl-tests

env \
  CUDA_VISIBLE_DEVICES=0,1 \
  LD_LIBRARY_PATH=/path/to/nccl/lib \
  NCCL_DEBUG=WARN \
  ./build/all_gather_perf \
    -b 4K \
    -e 256M \
    -f 2 \
    -g 2 \
    -w 10 \
    -n 50 \
    -c 1 \
  2>&1 | tee /path/to/results/4090/allgather-2gpu-default.log
~~~

AllReduce 和 ReduceScatter 只替换可执行文件：

~~~bash
./build/all_reduce_perf ...
./build/reduce_scatter_perf ...
~~~

### 10.2 NVIDIA 强制策略

示例：

~~~bash
env \
  CUDA_VISIBLE_DEVICES=0,1 \
  LD_LIBRARY_PATH=/path/to/nccl/lib \
  NCCL_DEBUG=WARN \
  NCCL_ALGO=Ring \
  NCCL_PROTO=Simple \
  NCCL_MIN_NCHANNELS=4 \
  NCCL_MAX_NCHANNELS=4 \
  ./build/all_gather_perf \
    -b 4K -e 256M -f 2 -g 2 -w 10 -n 50 -c 1 \
  2>&1 | tee /path/to/results/4090/allgather-2gpu-ring-simple-ch4.log
~~~

正式性能结果中如果希望一次覆盖 1 GiB，将 -e 256M 改为 -e 1G，并在 NVIDIA 与海光两侧保持一致。

### 10.3 海光 RCCL 默认配置

如果使用统一 harness：

~~~bash
cd /path/to/unified-harness

env \
  HIP_VISIBLE_DEVICES=0,1 \
  LD_LIBRARY_PATH=/path/to/rccl/lib:/path/to/hip/lib \
  NCCL_DEBUG=WARN \
  mpirun --allow-run-as-root -np 2 \
    -mca coll ^hcoll \
    ./build/unified_harness \
      --collective allgather \
      --dtype float32 \
      --min-bytes 4096 \
      --max-bytes 268435456 \
      --step-factor 2 \
      --warmup 10 \
      --iters 50 \
      --check 1 \
  2>&1 | tee /path/to/results/k100ai/allgather-2gpu-default.log
~~~

RCCL 的具体启动命令应以当前 DTK/OpenMPI 环境为准。重要的是保持 collective、消息大小、warmup、iters、检查方式和带宽公式与 NVIDIA 侧一致。

如果使用自定义 duNcclAllgather，必须单独命名日志和 CSV：

~~~text
api=standard_ncclAllGather
api=duNcclAllgather
~~~

不能将两种 API 的结果混在同一列中。

### 10.4 海光强制策略

仅在平台确实需要时设置：

~~~bash
HSA_FORCE_FINE_GRAIN_PCIE=1
~~~

示例：

~~~bash
env \
  HIP_VISIBLE_DEVICES=0,1 \
  HSA_FORCE_FINE_GRAIN_PCIE=1 \
  NCCL_DEBUG=WARN \
  NCCL_ALGO=Ring \
  NCCL_PROTO=Simple \
  NCCL_MIN_NCHANNELS=4 \
  NCCL_MAX_NCHANNELS=4 \
  mpirun --allow-run-as-root -np 2 \
    -mca coll ^hcoll \
    ./build/unified_harness \
      --collective allgather \
      --dtype float32 \
      --min-bytes 4096 \
      --max-bytes 268435456 \
      --step-factor 2 \
      --warmup 10 \
      --iters 50 \
      --check 1 \
  2>&1 | tee /path/to/results/k100ai/allgather-2gpu-ring-simple-ch4.log
~~~

如果该环境变量对某些协议不是必需的，也建议在所有 RCCL 配置中保持一致，并将其记录为 env_profile，不要一部分配置设置、一部分配置不设置。

## 11. 带宽和单位规范

### 11.1 单位

原始数据统一保存：

~~~text
size_bytes：整数，使用 bytes
time_us：浮点数，使用 microseconds
带宽：统一使用 decimal GB/s，1 GB = 10^9 bytes
~~~

如果工具原始输出是 GiB/s 或 MB/s，必须在导出 CSV 时转换，并保留原始值和原始单位：

~~~text
raw_bandwidth
raw_bandwidth_unit
normalized_bandwidth_gbps
~~~

### 11.2 AllReduce

对单 rank 输入大小为 S、rank 数为 N、时间为 T 的 AllReduce，必须同时记录：

~~~text
logical_payload_bytes = S
~~~

以及工具使用的 bus bytes 定义。若采用 nccl-tests 的双卡/多卡 bus bandwidth 口径，应明确写出公式，不要把 algbw 和 busbw 混写。

在双卡 AllReduce 中，当前 nccl-tests 的 algbw 与 busbw 数值可能相同；这只表示该工具的公式关系，不代表所有 collective 或自定义程序都相同。

### 11.3 AllGather

AllGather 必须明确区分：

~~~text
per_rank_input_bytes = S
per_rank_output_bytes = N * S
logical_collective_bytes = 程序定义的有效负载
bus_bytes = 带宽公式定义的实际通信量
~~~

所有论文图表应在图注中说明带宽分子。不能仅写“GB/s”而不说明是：

~~~text
per-rank input / time
total output / time
logical payload / time
bus bytes / time
~~~

### 11.4 ReduceScatter

同样记录：

~~~text
per_rank_input_bytes
per_rank_output_bytes
logical_payload_bytes
bus_bytes
~~~

NCCL/RCCL 工具的默认公式可能不同，自定义 harness 必须采用同一套公式。

## 12. CSV 数据格式

建议每个平台、每个 collective 生成一个 CSV，字段至少如下：

~~~text
timestamp
host
backend
gpu_model
gpu_arch
nranks
rank_to_device
collective
api_variant
datatype
reduction
size_bytes
warmup
iterations
run_id
requested_algo
actual_algo
requested_proto
actual_proto
requested_channels
actual_channels
transport
p2p_status
env_profile
time_us_mean
time_us_median
time_us_min
time_us_std
algbw_gbps
busbw_gbps
raw_bandwidth
raw_bandwidth_unit
wrong_count
status
log_path
~~~

字段说明：

- status 只能取 PASS、FAIL、UNSUPPORTED、TIMEOUT、ERROR；
- wrong_count 必须为整数；
- actual_algo、actual_proto 和 actual_channels 以 INFO 日志为证据；
- transport 例如 P2P、SHM/direct、IB/RoCE、unknown；
- p2p_status 记录平台能力，例如 OK、CNS、false、not_checked；
- log_path 指向完整原始日志，而不是只保存摘录。

## 13. 推荐结果图表

两边应生成同样的图表，不要只画最终最大带宽。

### 图 1：默认策略性能曲线

~~~text
x = per-rank message size
y = normalized bus bandwidth or latency
series = NVIDIA NCCL, Hygon RCCL
~~~

前提是 collective、计时和 transport 条件可比；否则必须分图。

### 图 2：协议/算法对比

~~~text
x = message size
series = Ring/LL, Ring/Simple, Tree/LL, Tree/Simple, DEFAULT
~~~

### 图 3：channel heatmap

~~~text
x = channel count
y = message size
color = speedup over default or bandwidth
~~~

### 图 4：1/2/4 卡 scaling

~~~text
x = number of ranks
y = normalized throughput or scaling efficiency
~~~

### 图 5：rank mapping 对比

~~~text
x = rank mapping
y = time or bus bandwidth
~~~

### 图 6：默认策略与 oracle 策略差距

~~~text
oracle_speedup = best_valid_configuration / default_configuration
~~~

该图用于判断是否存在稳定且有意义的自适应优化空间。

## 14. 如何从结果中寻找研究问题

每次实验不要只问“谁的 GB/s 更高”，而要检查以下问题。

### 14.1 最优策略是否随消息大小切换

例如：

~~~text
small message: LL
medium message: LL128
large message: Simple
~~~

记录切换点，并检查它是否随 rank 数和 collective 变化。

### 14.2 默认策略和 oracle 策略是否存在稳定差距

如果多种消息大小、collective 和 rank 数上都有明显差距，说明后端默认启发式可能没有利用全部上下文。

但如果只在单个配置上提升 2% 到 3%，通常不足以形成独立研究问题。

### 14.3 单独通信最优是否等于重叠执行最优

在第二阶段加入 GEMM，至少测试：

~~~text
communication only
GEMM only
AllGather -> GEMM（串行）
AllGather || GEMM（双 stream）
chunked AllGather + chunked GEMM（pipeline）
~~~

记录：

~~~text
T_comm
T_compute
T_serial
T_overlap
communication exposed time
GEMM slowdown
end-to-end step time
~~~

如果通信单独最优和重叠场景最优不同，才有必要进一步研究面向端到端的协议、算法和 channel 联合选择。

### 14.4 实际 transport 是否改变最优策略

比较：

~~~text
P2P available
P2P unavailable + SHM fallback
network transport
~~~

如果 transport 改变后最优策略变化，策略输入就不能只有 GPU 型号和消息大小，还必须包含实际 transport capability。

### 14.5 rank mapping 是否影响性能

如果自然映射与拓扑感知映射存在稳定差异，应进一步研究：

~~~text
topology-aware rank mapping
topology-aware Ring construction
topology-aware channel assignment
~~~

## 15. 通信计算重叠第二阶段配置

这一部分不作为第一轮 NCCL/RCCL 基础带宽表的替代，而是在基础口径统一后单独进行。

### 15.1 GEMM 配置

两边先使用逻辑上相同的矩阵规模：

~~~text
M = 8192
N = 8192
K = 8192
dtype = FP16 或 BF16
accumulation = FP32（两端支持时）
~~~

实际矩阵规模要根据显存和库支持能力调整，并记录：

~~~text
M/N/K
dtype
tile shape
CTA/block shape
stream count
workspace size
~~~

NVIDIA 侧可使用 CUDA C++/CUTLASS；海光侧使用 HIP C++ 及平台对应矩阵乘法实现。第一轮重叠实验的重点是计时结构，而不是追求两端 GEMM 峰值完全相同。

### 15.2 分块参数

通信消息切分为：

~~~text
1 chunk
2 chunks
4 chunks
8 chunks
~~~

记录：

~~~text
chunk_bytes
communication_stream
compute_stream
event dependency
pipeline depth
~~~

重点观察：

~~~text
通信单独最优配置是否仍然最优
channel 数是否影响 GEMM
LL/Simple 是否产生不同资源争用
chunk size 是否改变重叠效率
~~~

## 16. 实验通过标准

一组结果可以进入对比报告，至少需要满足：

~~~text
所有有效配置 correctness = PASS
wrong_count = 0
没有未解释的 TIMEOUT 或 ERROR
两端 collective、dtype、size、warmup、iters 一致
带宽单位和分子已经确认
实际 transport 已记录
实际 algo/proto/channel 已记录或标记 unknown
每个配置至少三次独立运行
大消息结果变异系数最好 <= 5%
原始日志和 CSV 均保留
~~~

以下情况不得进入严格跨平台排名图：

~~~text
NCCL 使用 SHM，RCCL 使用 P2P，但图中没有分层说明
一侧是 AllReduce，另一侧是 AllGather
一侧的 19357 是 MB/s，另一侧是 GB/s
一侧使用 10 warmup/50 reps，另一侧使用 3 warmup/10 iters
一侧是标准 API，另一侧是 duNccl 扩展 API
一侧的实际 channels 与请求 channels 不一致但未记录
~~~

## 17. 推荐执行顺序

### 第一步：修正首轮数据

~~~text
核对 RCCL 带宽单位
核对 RCCL 配置点数量
保存 NCCL/RCCL 版本和 commit
保存双方平台拓扑和 P2P 矩阵
~~~

### 第二步：双卡统一 AllGather

~~~text
NVIDIA NCCL AllGather
海光 RCCL AllGather
同一 dtype、size、计时和带宽公式
~~~

### 第三步：双卡统一 AllReduce/ReduceScatter

~~~text
默认策略
Ring/Tree
LL/Simple/LL128（按支持情况）
channel 代表点
~~~

### 第四步：四卡扩展

~~~text
1/2/4 rank scaling
Ring/Tree
channel scaling
rank mapping
topology/transport
~~~

### 第五步：重叠实验

~~~text
通信单独
GEMM 单独
串行
双 stream 重叠
chunked pipeline
~~~

### 第六步：形成研究假设

只在观察到以下稳定现象后形成研究假设：

~~~text
默认策略存在稳定性能缺口
最优配置随消息大小或 collective 切换
P2P/SHM/拓扑改变策略边界
通信单独最优与重叠最优不一致
自然 rank mapping 不是最优 mapping
同一抽象策略在 NCCL/RCCL 上表现出系统性差异
~~~

## 18. 每次实验的最终产物

每个平台、每个 rank 数、每个 collective 至少保留：

~~~text
platform.txt
build.txt
command.txt
default.log
forced-config-*.log
transport-info.log
results.csv
summary.md
~~~

推荐目录结构：

~~~text
results/
├── 4090/
│   ├── 2gpu/
│   │   ├── platform.txt
│   │   ├── transport-info.log
│   │   ├── allgather.csv
│   │   ├── allreduce.csv
│   │   └── reducescatter.csv
│   └── 4gpu/
└── k100ai/
    ├── 2gpu/
    │   ├── platform.txt
    │   ├── transport-info.log
    │   ├── allgather.csv
    │   ├── allreduce.csv
    │   └── reducescatter.csv
    └── 4gpu/
~~~

## 19. 当前平台的特别说明

当前双 RTX 4090 记录已经确认 CUDA P2P 为 CNS，NCCL 实际使用 SHM/direct。因此重新跑统一实验时必须继续记录这一事实，不要为了与 K100AI 的 PCIe P2P 结果看起来一致而修改结论。

如果后续 4 卡 RTX 4090 机器仍然报告 P2P 不可用，则应将 NVIDIA 结果拆成：

~~~text
NCCL P2P-capable condition
NCCL SHM-fallback condition
~~~

只有 transport 条件相同或被明确分层时，才可以进行严格的跨厂商性能比较。

本规范的首轮目标不是立即证明某个库优于另一个库，而是建立一套可以回答以下问题的实验数据：

~~~text
哪种 collective 在什么消息大小下切换协议？
最优 channel 是否依赖 rank 数和拓扑？
默认启发式与 oracle 最优之间有多大差距？
P2P/SHM/网络路径是否改变最优策略？
通信单独最优是否等于通信计算重叠最优？
这些规律是否能在 NCCL 与 RCCL 之间抽象和迁移？
~~~
