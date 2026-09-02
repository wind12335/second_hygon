# 4 卡 RTX 4090 NCCL 实验执行方案

本文档将海光 `K500SM_AI ×4 / gfx928 / RCCL` 的实验体系迁移到 4 卡 NVIDIA RTX 4090，并使用 `nccl-tests` 完成统一的孤立通信、策略、channel、拓扑映射和 1 GiB 能力测试。

本文档的目标不是让 NVIDIA 和海光的硬件条件完全相同，而是让以下实验口径尽量一致：

```text
collective
datatype
per-rank message size
warmup 次数
计时迭代次数
正确性检查
重复运行次数
带宽字段
策略矩阵
```

NVIDIA 和海光的物理拓扑、P2P 能力、协议支持集合和实际 transport 必须如实记录，不能为了形式上的对称而强行设置成相同。

### 当前 `nccl-tests` 版本兼容性修正

当前实际使用的 `nccl-tests` commit `a0b82b2260cf5152b9f8c061bbf7eaf0ba096432`（版本 2.19.6）中，`all_gather_perf --help` 没有 `-Z csv`、`-O 1` 这两个选项；其中 `-x` 是 CTA policy 选项，也不是 CSV 输出选项。因此本机不能直接执行本文后续带有 `-Z csv`、`-x <csv path>` 或 `-O 1` 的命令。

本机采用以下方式保存结果：

```text
每个 case 的 stdout/stderr -> 独立 .log
所有 case 拼接 -> formal_master.log
从 nccl-tests 文本结果提取 -> summary.csv
所有 case 的状态和路径 -> manifest.tsv
```

`-J` 虽然可以生成 JSON，但当前版本会把完整进程环境写入 JSON，可能包含容器注入的敏感变量；正式实验不建议使用 `-J`，应保留日志并自行解析 CSV。当前实际运行命令使用 `-a 3`（max rank time）、`-d float`、`-w 10`、`-n 50` 和 `-c 1`。

### 当前 MPI 路径修正

本机 Open MPI 4.1.2 的实际路径为：

```text
mpirun: /usr/bin/mpirun
MPI headers: /usr/lib/x86_64-linux-gnu/openmpi/include
MPI library: /usr/lib/x86_64-linux-gnu/openmpi/lib
```

因此 MPI 版 `nccl-tests` 应使用：

```bash
MPI=1 \
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi \
NAME_SUFFIX=_mpi
```

不要默认使用本文早先模板中的 `/opt/mpi`，除非换机器后确认该目录确实存在。

---

## 1. 实验平台固定值

本实验假设当前机器是 4 卡 RTX 4090，RTX 4090 的 CUDA 架构为 `sm_89`。

```text
GPU                 NVIDIA GeForce RTX 4090 ×4
CUDA architecture   sm_89
backend             CUDA + NCCL
test suite          nccl-tests
process model       one MPI process per GPU/rank
rank count          4
datatype            float32
reduction           sum（AllReduce/ReduceScatter）
warmup              10
timed iterations    50
correctness         enabled
average             max rank time
repetitions         3（主矩阵）
```

建议预先固定这些路径变量。下面路径必须根据实际机器修改，不能直接假设：

```bash
export NCCL_TESTS_DIR=/root/comm-study/nccl-tests
export NCCL_BUILD_DIR=/root/comm-study/nccl-tests/build
export NCCL_INSTALL_DIR=/root/comm-study/install/nccl
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/opt/mpi
export RESULT_ROOT=/root/private_data/lyc/2ndpaper/results
```

如果当前 nccl-tests 使用的是其他目录，只修改上面的变量，不要把不同 NCCL 安装混在一起。

---

## 2. 结果目录和固定日志

NVIDIA 4 卡结果建议独立保存，不能与海光结果混在同一目录：

```text
/root/private_data/lyc/2ndpaper/results/rtx4090_sm89_4gpu/
```

固定主日志：

```text
/root/private_data/lyc/2ndpaper/results/rtx4090_sm89_4gpu/formal_master.log
```

每一个阶段还应保留自己的时间戳目录：

```text
rtx4090_sm89_4gpu/
├── formal_master.log
├── platform/
├── preflight/
├── representative/
├── channels/
├── mapping/
├── 1g_pilot/
├── ll128_probe/
└── overlap/
```

每个 case 至少保留：

```text
完整 stdout/stderr log
nccl-tests CSV
可重放 command 文件
退出状态文件
```

不要只保留复制出来的带宽数字。原始 log 用于确认实际 algorithm、protocol、channel、transport 和 rank/device 映射。

---

## 3. 编译 nccl-tests

### 3.1 确认 CUDA 架构

先确认 4090：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version
```

RTX 4090 应该使用：

```text
sm_89
```

### 3.2 编译命令

如果 nccl-tests 使用独立安装的 NCCL：

```bash
cd "$NCCL_TESTS_DIR"

make clean
make -j"$(nproc)" \
     CUDA_HOME="$CUDA_HOME" \
     NCCL_HOME="$NCCL_INSTALL_DIR" \
     MPI=1 \
     MPI_HOME="$MPI_HOME" \
     NVCC_GENCODE='-gencode=arch=compute_89,code=sm_89'
```

如果 MPI 安装位置不是 `/opt/mpi`，必须替换 `MPI_HOME`。

确认链接到正确的 NCCL：

```bash
ldd "$NCCL_BUILD_DIR/all_gather_perf" | \
  grep -E 'libnccl|libcudart|libmpi'
```

结果中应该能看到你指定的：

```text
libnccl.so
libcudart.so
libmpi.so
```

不要使用：

```text
PyTorch 自带 NCCL
系统中无法确认来源的 libnccl.so
其他项目私自修改的 NCCL
```

除非这些库就是你明确要测试的版本。

---

## 4. 平台事实记录

在任何性能实验之前先执行以下命令，并将全部输出写入固定日志：

```bash
BASE=/root/private_data/lyc/2ndpaper/results/rtx4090_sm89_4gpu
mkdir -p "$BASE/platform"

{
  echo '# UTC'
  date -u
  echo '# HOST'
  hostname
  echo '# NVIDIA SMI'
  nvidia-smi
  echo '# GPU LIST'
  nvidia-smi -L
  echo '# GPU QUERY'
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,driver_version \
             --format=csv
  echo '# TOPOLOGY'
  nvidia-smi topo -m
  echo '# P2P READ'
  nvidia-smi topo -p2p r
  echo '# P2P WRITE'
  nvidia-smi topo -p2p w
  echo '# P2P NVLINK'
  nvidia-smi topo -p2p n
  echo '# P2P DETAILS'
  nvidia-smi -q -d P2P
  echo '# CUDA'
  nvcc --version
  echo '# CPU'
  lscpu
  echo '# NUMA'
  numactl --hardware
  echo '# MPI'
  mpirun --version
  echo '# NCCL TESTS HELP'
  "$NCCL_BUILD_DIR/all_gather_perf" --help
  echo '# NCCL LIBRARY'
  ldd "$NCCL_BUILD_DIR/all_gather_perf"
} > "$BASE/platform/platform.txt" 2>&1
```

特别要记录：

```text
GPU 的 PCI bus ID
GPU 到 NUMA 节点的关系
P2P 是否可用
NCCL 实际使用 P2P、SHM、NVLink 还是其他路径
```

当前文档中的 `P2P` 不能仅根据 `nvidia-smi topo` 推测最终性能。实际 transport 必须从 NCCL INFO 日志确认。

---

## 5. 公共运行环境

先设置统一环境：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

# 性能 case 使用 ERROR，避免 rank 日志刷屏。
export NCCL_DEBUG=ERROR
```

如果 NCCL 库安装目录不是 `lib/`，按实际目录修改 `LD_LIBRARY_PATH`。

MPI 单机四卡启动模板：

```bash
mpirun --allow-run-as-root \
  -np 4 \
  -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf"
```

如果不需要 root 权限，不要添加 `--allow-run-as-root`。

NVIDIA 侧使用的是：

```text
CUDA_VISIBLE_DEVICES
```

不要在 NVIDIA 命令中使用海光侧的：

```text
HIP_VISIBLE_DEVICES
HSA_FORCE_FINE_GRAIN_PCIE
```

---

## 6. 单点 transport 和实际策略探测

正式性能跑使用 `NCCL_DEBUG=ERROR`，但需要单独做 INFO 探测，确认 NCCL 实际用了什么。

例如四卡 AllGather、1 MiB：

```bash
BASE=/root/private_data/lyc/2ndpaper/results/rtx4090_sm89_4gpu
mkdir -p "$BASE/platform"

env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,TUNING \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 1M -e 1M \
    -g 1 \
    -w 10 -n 50 \
    -c 1 \
    -a 3 \
    -d float \
    -O 1 \
  > "$BASE/platform/transport-allgather-1M-info.log" 2>&1
```

从这个 INFO 日志中确认：

```text
actual algorithm
actual protocol
actual channel count
rank-to-device mapping
P2P / SHM / NVLink / network transport
```

不能只根据环境变量判断实际执行策略。例如设置了 `NCCL_PROTO=LL128`，不代表 NCCL 一定真的用了 LL128。

---

## 7. 正确性预检

先做少量预检，不要直接启动完整矩阵。

测试：

```text
AllGather:      4 KiB、1 MiB、64 MiB
AllReduce:      4 KiB、1 MiB、64 MiB
ReduceScatter:  4 KiB、1 MiB、64 MiB
```

单个示例：

```bash
BASE=/root/private_data/lyc/2ndpaper/results/rtx4090_sm89_4gpu/preflight
mkdir -p "$BASE/logs" "$BASE/csv"

env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=ERROR \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 1M -e 1M \
    -g 1 \
    -w 10 -n 50 \
    -c 1 \
    -a 3 \
    -d float \
    -Z csv \
    -x "$BASE/csv/allgather-1M.csv" \
    -O 1 \
  > "$BASE/logs/allgather-1M.log" 2>&1
echo $? > "$BASE/logs/allgather-1M.status"
```

AllReduce 和 ReduceScatter 只需替换可执行文件：

```text
all_reduce_perf
reduce_scatter_perf
```

只有满足以下条件的 case 才能进入性能汇总：

```text
exit status = 0
wrong_count = 0
Out of bounds values : 0 OK
```

---

## 8. 四卡代表性策略矩阵

这是正式的第一批性能数据。

### 8.1 配置

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

repetitions:
  3
```

总数：

```text
3 collective × 6 sizes × 5 strategies × 3 repetitions = 270 cases
```

### 8.2 默认配置

```bash
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=ERROR \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 1M -e 1M -g 1 -w 10 -n 50 -c 1 -a 3 -d float \
    -Z csv -x "$BASE/csv/rep_r1_allgather_1M_default.csv" -O 1 \
  > "$BASE/logs/rep_r1_allgather_1M_default.log" 2>&1
```

### 8.3 Ring/Simple

```bash
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=ERROR \
  NCCL_ALGO=Ring \
  NCCL_PROTO=Simple \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 1M -e 1M -g 1 -w 10 -n 50 -c 1 -a 3 -d float \
    -Z csv -x "$BASE/csv/rep_r1_allgather_1M_ring_simple.csv" -O 1 \
  > "$BASE/logs/rep_r1_allgather_1M_ring_simple.log" 2>&1
```

### 8.4 其他强制策略

只替换：

```bash
NCCL_ALGO=Ring  NCCL_PROTO=LL
NCCL_ALGO=Tree  NCCL_PROTO=Simple
NCCL_ALGO=Tree  NCCL_PROTO=LL
```

每个配置都要保存完整命令和输出。强制配置如果失败，必须标记为：

```text
UNSUPPORTED
ERROR
TIMEOUT
```

不能把失败配置写成 0 GB/s。

---

## 9. Channel 扫描

### 9.1 配置

```text
collective:
  AllGather
  AllReduce
  ReduceScatter

size:
  1 MiB
  8 MiB
  64 MiB

strategy:
  Ring/Simple
  Ring/LL
  Tree/Simple
  Tree/LL

channels:
  1
  2
  4
  8

repetitions:
  3
```

总数：

```text
3 × 3 × 4 × 4 × 3 = 432 cases
```

### 9.2 固定 channel 的环境变量

```bash
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
```

例如 Ring/Simple、4 channels：

```bash
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=ERROR \
  NCCL_ALGO=Ring \
  NCCL_PROTO=Simple \
  NCCL_MIN_NCHANNELS=4 \
  NCCL_MAX_NCHANNELS=4 \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 8M -e 8M -g 1 -w 10 -n 50 -c 1 -a 3 -d float \
    -Z csv -x "$BASE/csv/channels_allgather_8M_ring_simple_ch4.csv" -O 1 \
  > "$BASE/logs/channels_allgather_8M_ring_simple_ch4.log" 2>&1
```

`requested_channels` 不等于 `actual_channels`。必须通过单独的 INFO 日志确认 NCCL 是否真的运行了请求的 channel 数。

### 9.3 这个实验回答什么

```text
channel=1 是否并行度不足？
channel=4 到 channel=8 是否仍有明显收益？
最优 channel 是否随消息大小改变？
最优 channel 是否随 collective 改变？
channel 增加是否可能在 GEMM overlap 场景中产生资源竞争？
```

---

## 10. Rank mapping 和拓扑实验

### 10.1 映射方式

建议测试：

```text
0123
0213
0321
```

NVIDIA 侧通过 `CUDA_VISIBLE_DEVICES` 改变 rank 到物理 GPU 的映射。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

表示自然映射；

```bash
CUDA_VISIBLE_DEVICES=0,2,1,3
```

表示交错映射。

### 10.2 配置

```text
collective:
  AllGather
  AllReduce

size:
  1 MiB
  64 MiB

repetitions:
  3
```

每个 mapping 都要记录：

```text
rank_to_device
GPU BDF
NUMA node
NCCL actual ring
NCCL actual tree
实际 transport
```

### 10.3 判断方法

如果不同 mapping 差异稳定超过约 5%，应进一步分析：

```text
PCIe root complex
NUMA affinity
P2P 状态
NCCL ring 构造
NCCL tree 构造
```

如果差异只有 1%～2%，通常只能作为平台记录，不宜单独形成论文结论。

---

## 11. 1 GiB pilot

先单独测试：

```text
AllGather 1 GiB
AllReduce 1 GiB
ReduceScatter 1 GiB
```

命令模板：

```bash
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH="$NCCL_INSTALL_DIR/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=ERROR \
  timeout 300s \
  mpirun --allow-run-as-root -np 4 -mca coll ^hcoll \
  "$NCCL_BUILD_DIR/all_gather_perf" \
    -b 1G -e 1G -g 1 -w 10 -n 50 -c 1 -a 3 -d float \
    -Z csv -x "$BASE/csv/allgather_1G.csv" -O 1 \
  > "$BASE/logs/allgather_1G.log" 2>&1
```

如果 1 GiB 因显存、驱动或运行时间无法完成，需要在 NVIDIA 和海光两侧统一把最大消息大小降到 256 MiB，并记录原因。

AllGather 的 `-b 1G` 表示每个 rank 的输入大小，不是整个 collective 的总输出大小。

---

## 12. LL128 探测

NVIDIA 侧需要单独测试 LL128，因为 NVIDIA NCCL 的支持情况可能与海光 RCCL 不同。

测试：

```text
Ring/LL128
Tree/LL128
```

消息大小：

```text
1 MiB
8 MiB
64 MiB
```

使用：

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,TUNING
NCCL_ALGO=Ring
NCCL_PROTO=LL128
```

必须从 INFO 日志确认：

```text
实际是否执行 LL128
是否发生回退
实际 algorithm
实际 protocol
实际 channel
```

如果某个配置报：

```text
no algorithm/protocol available
```

则记录：

```text
status = UNSUPPORTED
```

不能参与有效性能排序。

---

## 13. 建议的日志和 CSV 字段

建议为每个 case 建立一个 manifest.tsv：

```text
case
collective
nranks
size_bytes
datatype
requested_algo
requested_proto
requested_channels
actual_algo
actual_proto
actual_channels
transport
cuda_visible_devices
warmup
iterations
repeat
time_us
algbw_gbps
busbw_gbps
wrong_count
status
log_path
csv_path
```

如果性能日志使用 `NCCL_DEBUG=ERROR`，则 `actual_algo`、`actual_proto`、`actual_channels` 和 `transport` 应由对应的 INFO 探测日志填入，而不是从请求环境变量直接推断。

---

## 14. 如何读一个 NCCL-tests 结果

原始 CSV 通常类似：

```text
num_elements,size_bytes,time,algbw,busbw,#wrong
```

不同 nccl-tests 版本的列名可能略有变化，必须先看第一行 header。

核心字段的含义：

```text
size_bytes    每个 rank 的输入消息大小
time          collective 平均时间，通常是 us
algbw         算法带宽
busbw         总线带宽
#wrong        错误元素数量
```

如果看到：

```text
#wrong = 0
```

表示 correctness 通过。

如果看到：

```text
#wrong > 0
```

则不能把该行用于性能结论。

比较策略时，建议：

```text
1. 先排除错误、超时和不支持配置；
2. 对相同 collective、相同 size、相同 rank 数分组；
3. 比较 time_us 和 busbw_gbps；
4. 对 3 次重复计算均值、最小值、最大值、标准差和 CV；
5. 与 DEFAULT 比较相对加速比。
```

加速比定义：

```text
speedup = candidate_busbw / default_busbw
```

例如：

```text
candidate = 6.60 GB/s
default   = 6.00 GB/s
speedup   = 1.10
```

表示 candidate 比 DEFAULT 快 10%。

---

## 15. 推荐执行顺序

不要一开始直接执行全部矩阵。建议按以下顺序：

```text
第 1 步：编译并检查 ldd
第 2 步：平台事实记录
第 3 步：单点 INFO transport 探测
第 4 步：四卡 correctness preflight
第 5 步：四卡 DEFAULT/算法协议代表性矩阵
第 6 步：四卡 channel 扫描
第 7 步：rank mapping
第 8 步：1 GiB pilot
第 9 步：LL128 探测
第 10 步：后续统一 overlap harness
```

每一步完成后都检查：

```text
是否有错误 case
wrong_count 是否为 0
CSV 是否非空
实际库路径是否正确
实际 algorithm/protocol/channel 是否有证据
```

---

## 16. 与海光 RCCL 结果对比时的注意事项

只有满足以下条件，NCCL 和 RCCL 的结果才适合放在同一张严格对比图中：

```text
collective 相同
datatype 相同
per-rank message size 相同
warmup 相同
iterations 相同
正确性检查方式相同
带宽分子和单位相同
重复运行次数相同
```

以下条件不能强行统一，但必须记录：

```text
GPU 型号
显存容量
PCIe/NVLink/XGMI/HYLink 拓扑
P2P 是否可用
SHM/P2P/NVLink/网络 transport
协议支持集合
NUMA 关系
驱动和库版本
```

特别注意：

```text
如果 NVIDIA 使用 SHM，而海光使用 PCIe P2P，不能在图中不加说明地比较谁更快。
```

正确的做法是：

```text
先分别报告各平台的能力和性能；
再在 transport 条件相同或明确分层的情况下做跨平台比较。
```

---

## 17. 这套实验能回答什么研究问题

完成四卡 RTX 4090 后，可以和海光数据一起回答：

```text
1. DEFAULT 是否在两个通信基座上都接近最优？
2. Ring/Tree 的切换点是否随后端改变？
3. Simple/LL/LL128 的有效集合是否不同？
4. channel 最优值是否依赖 backend、消息大小和 transport？
5. 2 卡到 4 卡的 scaling 是否一致？
6. rank mapping 对两种平台的影响是否相同？
7. NCCL 和 RCCL 是否存在相同的抽象规律？
8. 孤立通信最优策略是否仍然适合 GEMM overlap？
```

当前海光实验已经显示：

```text
Simple 在本机有效；
LL 在中大消息上明显退化；
LL128 不可用；
channel 数有明显影响；
分块通信和分块 GEMM 的端到端效果不能由孤立通信带宽推断。
```

在 4 卡 RTX 4090 上执行相同矩阵，可以判断这些现象是：

```text
NCCL/RCCL 共有规律
还是某个平台独有的能力边界
```

这正是“跨通信基座能力感知的端到端重叠策略选择”需要的第一批实验基础。

---

## 18. 最终检查清单

### 平台

```text
[ ] nvidia-smi -L 已保存
[ ] compute capability 已确认是 sm_89
[ ] nvidia-smi topo -m 已保存
[ ] P2P read/write 结果已保存
[ ] ldd 已确认实际 libnccl.so
[ ] CUDA/MPI/driver/NCCL 版本已保存
```

### 正确性

```text
[ ] AllGather 正确
[ ] AllReduce 正确
[ ] ReduceScatter 正确
[ ] 所有有效配置 wrong_count = 0
```

### 性能

```text
[ ] DEFAULT 已完成
[ ] Ring/Simple 已完成
[ ] Ring/LL 已完成
[ ] Tree/Simple 已完成
[ ] Tree/LL 已完成
[ ] channel 1/2/4/8 已完成
[ ] 每个配置至少 3 次重复
[ ] 1 GiB pilot 已完成或记录失败原因
[ ] LL128 已探测并记录支持状态
```

### 结果保存

```text
[ ] 每个 case 有完整 log
[ ] 每个 case 有 CSV
[ ] 每个 case 有 command 文件
[ ] 每个 case 有退出状态
[ ] manifest.tsv 已生成
[ ] actual algo/protocol/channel 有 INFO 证据
[ ] transport 有 INFO 证据
```

---

## 19. 最后需要记住的四句话

```text
1. 4 卡 RTX 4090 的实验口径要和海光保持一致，但不能假设物理能力相同。
2. DEFAULT 是基准，强制策略只在正确性通过后才有资格比较。
3. requested algorithm/protocol/channel 不等于 actual algorithm/protocol/channel，实际值必须从 INFO 日志确认。
4. 单独通信性能最优，不等于通信计算重叠后的端到端性能最优。
```
