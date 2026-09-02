# Phase 3: Decision-Boundary and Compute-Intensity Mapping

## 固定平台

```text
K500SM_AI / gfx928 / 4 GPUs / PCIe
```

本实验不使用 `gfx936`、`K100AI` 或任何其他设备名称。每个 MPI rank 固定映射到一张 DCU，
并统一设置 `HIP_VISIBLE_DEVICES=0,1,2,3` 和 `HSA_FORCE_FINE_GRAIN_PCIE=1`。

## 已知事实与本阶段目标

Phase 2 已经在 `m_local=N=K=2048, q=8` 上确认：

1. `C2_RING_SIMPLE_CH8` 的 isolated AllGather `T_done` 最小；
2. `C0_DEFAULT` 的 H0 AllGather-GEMM `T_e2e` 最小；
3. 五次独立 MPI 进程运行中，C0 均优于 C2，H0 p50 差距为 7.165%。

同时，`q=2` 是负对照：C2 在 isolated 和 H0 中均获胜。因此，Phase 3 不再重复回答
“q=8 是否存在反转”，而回答两个可证伪的问题：

1. 反转从哪个分片粒度开始出现，是否在 `q=4`、`q=8`、`q=16` 上形成稳定边界？
2. 当通信 payload 不变而每字节对应的 GEMM 工作量变化时，端到端最优策略是否变化？

## 唯一改变的自变量

固定：

```text
P=4, m_local=2048, K=2048, FP32,
rank-major layout, RCCL AllGather, rocBLAS SGEMM,
相同输出 scatter、warmup、验证、rank mapping 和 device-event 计时。
```

改变：

```text
N in {512, 2048, 4096}
q in {2, 4, 8}
```

AllGather 的每 rank 输入 payload 为 `m_local * K * sizeof(float) = 16 MiB`，不随 N 改变；
因此 N 只改变每个已通信字节后续需要执行的 GEMM 工作量和输出写入量。N=2048 另加入
`q=16`，以定位更激进的细分片是否因 fragmentation 或资源竞争再次改变选择结果。

## 候选策略与路径

通信候选：

```text
C0_DEFAULT             RCCL 默认 algo/proto/channel
C1_RING_SIMPLE_CH4     Ring + Simple + 4 channels
C2_RING_SIMPLE_CH8     Ring + Simple + 8 channels
```

执行路径：

```text
COMM_ONLY       仅分片 AllGather，记录 T_done 和每个 slice 的 release。
H0_EVENT_OVERLAP
                release(i) 后 compute stream 可消费 slice i；
                comm stream 继续发起后续 slice。
B1_SLICE_SERIAL release(i) -> GEMM(i) -> release(i+1)，禁止相邻 slice 重叠。
GEMM_ONLY       使用相同 q、相同 gathered layout 和相同 scatter，测分片 GEMM 本身。
B0_FULL_SERIAL  完整 AllGather -> 完整 GEMM 的未分片串行参考。
```

对任一固定的 `(N,q,candidate)`，B1 和 H0 唯一差异为 stream 依赖图。每个非 COMM_ONLY 路径
在每个 timed iteration 都与预先生成的完整 B0 参考输出做逐元素比较。

## 完整矩阵

主矩阵（270 cases）：

```text
N={512,2048,4096} x q={2,4,8} x C0/C1/C2 x {COMM_ONLY,H0} x 5 process repetitions
```

q=16 边界加密（30 cases）：

```text
N=2048 x q=16 x C0/C1/C2 x {COMM_ONLY,H0} x 5 repetitions
```

串行对照（40 cases）：

```text
N=2048 x q={2,4,8,16} x {C0,C2} x B1 x 5 repetitions
```

GEMM fragmentation 对照（30 cases）：

```text
N={512,2048,4096} x q={2,4,8} x GEMM_ONLY x 3 repetitions
N=2048 x q=16 x GEMM_ONLY x 3 repetitions
```

完整串行参考（15 cases）：

```text
N={512,2048,4096} x B0(q=1) x 5 repetitions
```

总计为 385 个 case、30,800 个四卡 timed distributed iterations。每个 case 使用 20 warmup 和
80 个 timed iterations；根据 Phase 2 实测，完整运行预计约 75--85 分钟，不含异常重试。

## 判定规则

对每个 `(N,q)`：

1. `COMM_ONLY` 按 5 个 process-level `T_done p50` 的中位数选择 isolated winner；
2. H0 按 5 个 process-level `T_e2e p50` 的中位数选择 end-to-end winner；
3. 仅当所有相关 case 退出码为 0、所有输出正确性为 PASS、两侧均有 5 次独立重复时才进入比较；
4. 若两个 winner 不同，且 H0 winner 在 5 个同序 repetition 中均快于 isolated winner，则记录为
   `REPEAT_STABLE_REVERSAL`；H0 差距达到 5% 时记录为 `STRONG_REVERSAL`；
5. per-slice 结果必须检查 release、consumer wait 和 GEMM duration，不能仅凭总时间归因。

本阶段可支持“isolated T_done 无法独立选择依赖流水线策略”的结论，不能单独证明硬件级因果机制，
也不能作为 DUSHMEM 或跨基座结果。DUSHMEM 必须先通过独立的 payload-publish-signal-wait-checksum
正确性准入后，才可以进入公平性能比较。

## 输出与审计

每个 case 保留：可重放命令、stdout/stderr、退出状态、RCCL 日志、manifest、四份 rank CSV、
一份 max-rank CSV、四份 per-slice CSV、max-rank per-slice CSV 和文件清单。根目录保留平台快照、
source snapshot、SHA-256、case manifest、master log 和自动生成的 CSV/Markdown 分析。结果目录创建后
不可覆盖；失败 case 也必须原样保留。
