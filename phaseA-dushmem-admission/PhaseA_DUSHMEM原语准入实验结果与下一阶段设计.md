# Phase A: DUSHMEM 原语准入实验结果与下一阶段设计

## 1. 这份实验回答了什么问题

本阶段不是在证明 DUSHMEM 已经超过 RCCL，也不是在证明 AG-GEMM 已经加速。

它只先回答一个必要问题：在本机的 4 卡海光环境中，能否使用 DUSHMEM 的 stream 原语构造一个**正确、可测量、带有明确数据就绪信号**的 AllGather 等价数据通路？如果这一步不成立，后面讨论“按 slice 提前触发 GEMM”没有可信的基础。

结论是：**可以进入下一阶段。**四卡 14 个正式案例全部通过，包含数万次通信、接收端 GPU 全 payload 校验和 slot 重用。这里的“可以”仅针对本机版本与路径：`K500SM_AI / gfx928 / 4 GPUs / PCIe / DTK 26.04 / DUSHMEM 3.2.5`。

## 2. 实验位置与可复现入口

实验工程位于：

```text
/root/private_data/lyc/2ndpaper/dushmem_phase4_admission
```

正式结果位于：

```text
/root/private_data/lyc/2ndpaper/dushmem_phase4_admission/results/phase4_dushmem_admission_formal_20260831T222900Z
```

重新运行时使用：

```bash
cd /root/private_data/lyc/2ndpaper/dushmem_phase4_admission
./scripts/run_admission.sh --smoke
./scripts/run_admission.sh --formal
```

每次运行会拒绝覆盖旧目录。一个结果目录中包含：

```text
manifest.csv                              每个案例的参数、退出状态、目录
build.log                                 编译日志
platform/                                 GPU、DTK、DUSHMEM、MPI 和拓扑快照
cases/<case>/command.txt                  精确重放命令
cases/<case>/stdout_stderr.log            完整 MPI/DUSHMEM stdout 和 stderr
cases/<case>/raw/rank_<rank>.csv          每 rank、每 epoch 的原始时间和校验结果
cases/<case>/raw/capability_rank_*.csv    P2P 与 dushmem_ptr 能力记录
analysis/case_summary.csv                 正式统计表
analysis/global_iteration_max.csv         每 epoch 的四 rank 最大值
analysis/paired_strategy_comparison.csv   同负载策略对照
analysis/admission_report.md              自动生成的简明报告
```

正式结果中的 `case_summary_v1.csv`、`global_iteration_max_v1.csv` 和
`admission_report_v1.md` 是第一版自动汇总的保留副本。v2 分析器增加了 release 相关字段；
其副本和 SHA-256 写在 `analysis/analyze_admission_v2.py` 与
`analysis/analysis_provenance.txt` 中。原始 rank CSV 没有被修改。

## 3. 测的到底是什么

每个 MPI rank 绑定一张 GPU。每个 epoch 中，rank `r` 生成带有 `(producer_rank, epoch,
word_index)` 的确定性 payload，然后把该 payload 发送给其余三个 rank，同时将自己的 payload
复制到本地 AllGather 槽位。每个接收方等待来自另外三个生产方的单调 epoch signal，随后在 GPU
上逐字验证四份 payload。校验能发现旧数据、写错 rank、写入未完成、slot 被提前重用等错误。

这实现的是 AllGather 的数据布局等价物，而不是调用 RCCL 的 `AllGather` API：

```text
rank r:
  source[r]
    -> local D2D copy 到 recv[slot][r]
    -> dushmemx_putmem_signal_on_stream 到每个远端 recv[slot][r]
    -> 等待 recv[slot][producer] 对应的三个 epoch signal
    -> GPU 验证 recv[slot][0..3]
```

因此，该通路恰好适合研究后续的“某个 slice 什么时候可以安全交给 GEMM”。但 Phase A 每个
payload 仍是一个完整块，尚未拆成多个 GEMM 消费 slice。

### 3.1 三种时间不能混为一谈

每个 epoch 记录三种 GPU 时间。它们从同一个 `issue` 事件开始，单位为微秒。

| 字段 | 含义 | 研究价值 |
|---|---|---|
| `issue_to_comm_stream_complete_us` | 本 rank 的通信 stream 中已排队的本地发送工作完成 | 近似“发送方本地工作完成”，不代表所有远端都可读 |
| `issue_to_release_us` | 接收方 wait stream 已观察到全部远端 epoch signal | 此实验中“该完整块已经可合法消费”的边界 |
| `issue_to_checked_us` | release 后又完成了全 payload GPU 校验 | 只用于本阶段正确性证明；包含校验开销，不能当作产品级通信时间 |

汇总时对同一 epoch 的四个 rank 先取最大值，再对所有 epoch 求 p50/p05/p95。这避免把某一个
较快 rank 当作四卡流水线的整体时间。

注意：`max_rank_release - max_rank_comm_stream_complete` 是一个四 rank 聚合后的差值，不是单个
网络操作的规范化 API 延迟。它的作用是显示“本地发送工作完成”和“全远端数据可消费”并非同一事件。

### 3.2 为什么要加 credit / slot 测试

若只使用二值 `ready=0/1` 标志，生产方可能在接收方还没消费旧数据时覆盖同一个 buffer slot，
这就是 ABA / buffer reuse 错误。

本实验让 epoch 从 1 单调递增。接收方在 GPU 验证成功后，向每个生产方写回该 epoch 的 credit；
生产方只有看到对应 credit 后，才可复用环形 buffer 的 slot。`A3_*` 案例验证的就是这个协议，
不是为了宣称它比没有 credit 的路径更快。

### 3.3 四类测试路径

| 案例前缀 | 路径 | 测试目的 |
|---|---|---|
| `A2_put_signal_*` | `dushmemx_putmem_signal_on_stream` + 接收端 `signal_wait_until_on_stream` | signal 可以作为数据释放边界的基本路径 |
| `A2_put_signal_quiet_*` | 上述路径再调用 `dushmemx_quiet_on_stream` | 判断额外 quiet 是否改变正确性或时间 |
| `A3_credit_*` | signal 路径 + 单调 credit + 1/2/4 slot | 验证安全复用，排除二值 flag 的 ABA 问题 |
| `A4_fcollect_*` | `dushmemx_fcollectmem_on_stream` | DUSHMEM collective 路径的 AllGather 等价对照 |

## 4. 平台能力事实

来自正式结果中任一 `capability_rank_*.csv` 以及 `platform/` 快照：

```text
GPU:              4 x K500SM_AI
reported arch:    gfx928:sramecc+:xnack-
topology:         全对全 PCIe，所有对均可达，3 hops
DUSHMEM:          3.2.5
HIP P2P:          每个异 rank 对均为 1
dushmem_ptr:      每个异 rank 对均为 non-null / available
```

这很重要：本机不是“海光没有 device/peer 通信能力”的简单情形。后续工作应该研究的是不同路径的
完成语义、release 时刻、资源竞争和适用区间，而不是把平台差异简化为“有或没有”。

## 5. 正式结果

所有正式案例均为 `PASS`：进程退出码为 0、每个预期 epoch 都收到四个 rank 记录、四个 rank 的
GPU 全 payload 校验均通过。

| 路径 | payload | epoch | p50 release us | p50 comm-stream us | p50 checked us | 结果 |
|---|---:|---:|---:|---:|---:|---|
| put-signal | 4 KiB | 10000 | 139.672 | 116.314 | 170.071 | PASS |
| put-signal | 64 KiB | 10000 | 151.512 | 131.673 | 196.469 | PASS |
| put-signal | 1 MiB | 10000 | 587.648 | 568.449 | 648.763 | PASS |
| put-signal | 8 MiB | 1000 | 3887.629 | 3866.510 | 4023.141 | PASS |
| put-signal | 64 MiB | 100 | 29937.010 | 29911.688 | 30905.249 | PASS |
| put-signal + quiet | 4 KiB | 10000 | 128.153 | 107.194 | 155.192 | PASS |
| put-signal + quiet | 1 MiB | 10000 | 588.448 | 569.728 | 646.205 | PASS |
| put-signal + quiet | 64 MiB | 100 | 29938.967 | 29914.885 | 30904.837 | PASS |
| credit, 1 slot | 4 KiB | 10000 | 138.072 | 115.834 | 167.190 | PASS |
| credit, 2 slots | 64 KiB | 10000 | 149.592 | 128.313 | 197.109 | PASS |
| credit, 4 slots | 1 MiB | 5000 | 569.089 | 550.690 | 616.686 | PASS |
| fcollect | 4 KiB | 10000 | 43.837 | 29.118 | 68.797 | PASS |
| fcollect | 1 MiB | 5000 | 757.397 | 747.318 | 814.834 | PASS |
| fcollect | 64 MiB | 100 | 45923.909 | 45913.667 | 46888.533 | PASS |

同负载、相同的“可消费并已验证”定义下，自动对照表给出：

| 对照 | 4 KiB | 1 MiB | 64 MiB |
|---|---:|---:|---:|
| `put-signal + quiet` 相对 `put-signal` 的 checked 改变 | -8.749% | -0.394% | -0.001% |
| `fcollect` 相对 `put-signal` 的 checked 改变 | -59.548% | +25.598% | +51.717% |

这些数字的直接解释是：在本机、该实现与该正确性验证协议下，**小消息与大消息的较优 DUSHMEM
路径不同**。4 KiB 选择 `fcollect` 明显更好，而 1 MiB 和 64 MiB 选择 signal-push 更好。

## 6. 当前能说什么，不能说什么

### 能说的

1. 不加 `quiet` 的 `putmem_signal_on_stream -> signal_wait_until_on_stream` 在此平台的 4 KiB、
   64 KiB、1 MiB、8 MiB、64 MiB 测试范围内均完成了规定次数的 payload 正确性验证。它是一个可用的
   后续分片 release 原语候选。
2. 单调 epoch + credit 的安全 slot 复用协议通过了 5000 至 10000 个 epoch。后续分片流水线可以
   使用该协议，而不是不安全的 0/1 flag。
3. `release` 比本 rank 的 `comm-stream complete` 晚约 18 到 23 us（fcollect 约 10 到 15 us），
   因而“本地通信工作已完成”不能直接替代“消费者现在可以读取”。
4. 不同 payload 的最佳路径发生翻转，说明静态地只选择一个 collective / protocol 不合理。这为
   “通信基座能力感知的策略选择”提供了第一块实证材料。

### 还不能说的

1. **没有 AG-GEMM 端到端加速结论。**本阶段没有 GEMM 消费者，也没有测量 `T_done`。
2. 不能说 DUSHMEM 已经优于 RCCL。RCCL 基线尚未在同一 tile 形状、同一数学工作量、同一正确性标准下
   对照。
3. 不能把 `checked` 当作产品性能。它包含本实验故意加入的完整 GPU payload verifier；其价值是证明
   数据正确，而不是模拟真实 GEMM 的成本。
4. 不能从“本次不加 quiet 也正确”推出所有 DUSHMEM API、所有传输路径、所有 DTK 版本都不需要 quiet。
   结论范围严格限于这里实际使用的 on-stream put-signal/wait 组合与实测平台。

## 7. 对论文创新点的实际帮助

本阶段把论文从“猜测 DUSHMEM 也许有用”推进到可检验的假设：

```text
同一通信基座并不存在对所有消息大小都最优的路径；
本地发送完成、远端数据 release、完整 collective 完成也不是同一时刻。
```

这还不是论文主结论，但为主问题提供了可行的实验入口：

```text
能否利用每个 slice 的 release 时刻，选择 backend / API / slice 数 / tile 形状，
使 AG-GEMM 的 T_done 小于仅按 collective 带宽或完整 collective 时间选择的方案？
```

需要强调：当前的结果只显示了“路径与消息大小有关”，尚未显示“孤立通信最快的路径在 AG-GEMM 中不是
最快”。后者必须由下一阶段的端到端交叉排名实验来证明。

## 8. 下一阶段：Phase B 的精确实验设计

Phase B 应新建独立目录，不污染本目录，也不直接修改 RCCL。目标是在**相同矩阵问题、相同数据布局、
相同 tile 数和相同正确性检查**下比较下面三类实现：

| ID | 通信/消费方式 | 作用 |
|---|---|---|
| H0 | RCCL `AllGather` 完整完成后再执行全部 GEMM | RCCL 非重叠基线 |
| D0 | DUSHMEM `fcollect` 完整完成后再执行全部 GEMM | 去除“提前消费”因素的 DUSHMEM collective 基线 |
| D1 | DUSHMEM signal-push，按 `q` 个 slice 逐个 wait/release，并立刻启动消费该 slice 的 GEMM tile | 要验证的 release-aware 重叠路径 |

在 D1 已经严格正确后，再加入 H1：RCCL 的可行分片/stream 设计。H1 不能为了看起来可重叠而错误地把
一次 collective 未完成的数据当作可读；必须先定义它的合法 release 语义。

### 8.1 每个 Phase B case 必须记录的时间

对第 `i` 个 slice / GEMM tile 记录：

```text
t_issue                 通信提交时刻
t_release(i)            第 i 个 slice 数据满足消费者依赖的时刻
t_gemm_start(i)         消费该 slice 的 GEMM tile 实际开始时刻
t_gemm_end(i)           该 tile 结束时刻
T_collective_done        整个通信操作完成时刻
T_done                   最后一个 GEMM tile 结束时刻
```

从而可以定义：

```text
R_i       = t_release(i) - t_issue
Delta_R_i = R_(i+1) - R_i
E_0       = T_done(H0) - T_done(D1)
```

并在每个 case 保存逐 rank、逐 iteration 的 CSV，而不只保存平均带宽。

### 8.2 第一轮参数矩阵

先使用一个能在约 1 至 1.5 小时内完成的受控矩阵，而不是立刻扫描所有参数：

```text
world size:       4
AllGather 总量:   4 MiB, 16 MiB, 64 MiB
slice 数 q:       1, 2, 4, 8
GEMM 强度:        light / balanced / compute-heavy 三档
策略:             H0, D0, D1
warmup:           50
measured:         500 (小中消息), 100 (64 MiB)
repetitions:      5 个独立进程运行
```

这得到 `3 x 4 x 3 x 3 x 5 = 540` 个 case-run。为控制实际时间，可先固定 `q=4` 做预筛，只有出现
至少一个端到端交叉排名后再跑完整矩阵。每个 case 仍要有超时、command、stdout/stderr、原始 CSV 和汇总。

### 8.3 论文所需的判据

真正有价值的创新证据不是“DUSHMEM 的带宽更高”，而是同一 workload 出现以下现象之一：

1. `isolated collective` 最快的策略 A，不是 `T_done` 最小的策略 B。
2. 完整 collective 时间较慢的策略，因较早的 `R_0` 或更小的 `Delta_R_i`，在 GEMM 重叠后反而更快。
3. 最优 `q` 和 GEMM tile 形状随 payload、GEMM 强度或基座能力而改变，固定 heuristic 会错选。
4. `fcollect` 在低延迟小片时获益，signal-push 在较大片时获益，策略选择器能根据实测 capability 和少量
   profiling 选择正确路径。

只有当这些交叉排名在 RCCL / DUSHMEM 且未来 NVIDIA / NCCL / NVSHMEM 的对应测试中稳定出现时，才能把
“跨通信基座能力感知的端到端重叠策略选择”写成论文核心主张。

## 9. 现在的实际下一步

下一步不是魔改 RCCL 内核。应该先完成独立的 Phase B harness：H0、D0、D1 使用统一的 tile 数据布局和
GEMM，先建立无歧义的 release 与 `T_done` 记录。它会告诉我们 DUSHMEM 的 early-release 能否真正抵消
PCIe 通信和 GPU 资源竞争，再决定是否值得深入 RCCL 内部修改或探索 device-initiated 路径。
