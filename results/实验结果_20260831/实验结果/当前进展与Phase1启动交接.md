# 当前进展与 Phase 1 启动交接

更新：2026-08-31

## 1. 研究问题的当前状态

第二篇尚未定题，当前被验证的假设是：

    在依赖型 AllGather-GEMM 中，孤立 busbw 或完整 collective 完成时间
    未必能预测端到端最优策略；合法的分片 release 时间、通信-计算争用和
    平台 capability 可能改变策略排序。

这不是已有结论。只有在后续多卡 AG-GEMM sweep 中观察到稳定排序反转，且 release timeline 和 GEMM 效率能解释时，才可成为论文主线。

不能把以下内容当作第二篇创新：

- 调整 NCCL/RCCL 的 algorithm、protocol 或 channel；
- 将 AutoCCL 移植到 RCCL；
- 更细 chunk、one-sided put、ready signal 或 window；
- 将第一篇的 Triton/NVSHMEM 实现重写成 CUDA/HIP C++；
- 使用 NVSHMEM/DUSHMEM、CUTLASS/DUMMA 或“通信-GEMM 融合”本身。

第一篇已经包含 one-sided、ready 单元、细粒度 data-dependent overlap、staging/window 和约束感知短测量搜索；PGAS 是候选实现路径和控制变量，而不是预设贡献。

## 2. 已完成的微基准证据

| 平台/数据 | 已知直接测量事实 | 解释边界 |
|---|---|---|
| 四卡 RTX 4090/NCCL | 原始导出中 547 个 PASS case；另有 222 个 NCCL invalid-usage 组合，应进入 capability mask | 不是所有请求配置都可用；无 correctness error 的通过项才能做性能分析 |
| 四卡 K500SM_AI/gfx928/RCCL | 原始 CSV 959 行、750 个唯一 case_id；channel 阶段有 209 条额外重复；6 个 case 的相对波动不小于 10%，需要重跑 | 不能静默丢弃重复项或用均值掩盖不稳定性 |
| 跨平台对照 | NCCL/RTX 4090 观测为 SHM/direct，RCCL/DCU 为 PCIe P2P；软件、拓扑、GPU 都不同 | 不能写成“NCCL 比 RCCL 快”或通信库源码质量排名 |
| 海光 DUSHMEM 分块探索 | chunks=2/4/8 伴随约 7.8--7.9 倍的分块 GEMM 自身退化 | 不能归因为 PGAS 或重叠策略优劣 |

需要在论文中统一的设备标签是 K500SM_AI/gfx928 还是 K100_ai；现有 RCCL 原始路径和 CSV 写的是 K500SM_AI/gfx928，未核验前不可替换标签。

相关原始/派生材料：

- 四卡总分析：[四卡NCCL-RCCL结果综合分析与第二篇论文定题建议.md](./四卡NCCL-RCCL结果综合分析与第二篇论文定题建议.md)
- 相关工作排重：[第二篇论文相关工作排重与创新证据图谱.md](./第二篇论文相关工作排重与创新证据图谱.md)
- 下一阶段完整协议：[下一阶段实验计划-统一Collective-GEMM基线与NCCL-NVSHMEM融合路线.md](./下一阶段实验计划-统一Collective-GEMM基线与NCCL-NVSHMEM融合路线.md)
- RTX 4090 四卡原始导出：[nccl_rtx4090_4gpu_formal_summary.csv](./nccl_rtx4090_4gpu_formal_summary.csv)
- RCCL 四卡原始导出：[rccl_k500sm_ai_4gpu_formal_summary.csv](./rccl_k500sm_ai_4gpu_formal_summary.csv)

## 3. 这次新完成的 Phase 1 基线

新工程位于 /root/comm-study/ag-gemm-bench，独立于 NCCL/RCCL/NVSHMEM 源码和 PyTorch。它由 CUDA C++、MPI、NCCL 和 cuBLAS 组成，并且仅实现 FP32 correctness-first baseline。

| ID | 实现 | 作用 |
|---|---|---|
| B0_FULL_SERIAL | 一次完整 AllGather，随后一次完整 GEMM | 全块正确性/时间锚点 |
| B1_SLICE_SERIAL | q 个 slice；slice i+1 的通信需等待 slice i 的 GEMM 完成 | 保留同一切分和 GEMM，量化分块代价，禁止 overlap |
| B2_SLICE_EVENT_OVERLAP | comm stream 的 NCCL slice event 触发 compute stream 对应 GEMM | 唯一允许的 CCL event overlap 路径 |

B1 和 B2 的 slice、GEMM、rank-major 输出布局和 CUDA scatter 一致，差异只在 stream 依赖关系。因而 B2 必须首先与相同 q 的 B1 比较，不能只拿 B0 比较。

每个 timed sample 写入：

    run_id, rank_count, M/N/K, q, slice_bytes, strategy,
    t_release_first_us, t_release_last_us, t_done_us,
    gemm_first_start_us, gemm_last_end_us, e2e_us,
    correctness, max_abs_error, max_rel_error, log_path

每次 run 同时产生：

- raw_rank{rank}.csv：每 rank 的原始样本；
- raw_global_samples.csv：跨 rank 最大时间，用于端到端比较；
- summary.csv：均值、p50、p95 派生表；
- manifest.csv 和 rank{rank}.log：配置和 rank/device 映射。

工程入口：

- [主源码](/root/comm-study/ag-gemm-bench/src/ag_gemm_bench.cu)
- [构建配置](/root/comm-study/ag-gemm-bench/CMakeLists.txt)
- [正式矩阵脚本](/root/comm-study/ag-gemm-bench/scripts/run_phase1_matrix.sh)
- [平台事实采集](/root/comm-study/ag-gemm-bench/scripts/collect_platform_facts.sh)
- [使用说明](/root/comm-study/ag-gemm-bench/README.md)

## 4. 本次验证的范围

当前会话只暴露了一张 NVIDIA A16（sm_86，15 GiB），不是此前的 4 卡 RTX 4090/A100 主机。因此只能验证代码、事件依赖、CSV 契约和单 rank 数值正确性，不能产生任何跨 GPU、NCCL transport 或论文性能结论。

已通过：

| run_id | rank | shape | q | warmup/iters | 结果 |
|---|---:|---|---:|---:|---|
| single_gpu_preflight_20260831_final | 1 | M=N=K=128 | 2 | 1/2 | 三策略均 PASS，最大绝对/相对误差均为 0 |
| single_gpu_q1_anchor_20260831 | 1 | M=N=K=64 | 1 | 0/1 | 三策略均 PASS，最大绝对/相对误差均为 0 |

产物：

- [q=2 原始全局样本](/root/comm-study/ag-gemm-bench/results/single_gpu_preflight_20260831_final/raw_global_samples.csv)
- [q=2 汇总](/root/comm-study/ag-gemm-bench/results/single_gpu_preflight_20260831_final/summary.csv)
- [q=1 锚点汇总](/root/comm-study/ag-gemm-bench/results/single_gpu_q1_anchor_20260831/summary.csv)
- [当前 A16 平台事实](/root/comm-study/ag-gemm-bench/results/platform_facts_a16_20260831_clean/platform_facts.txt)

这些单卡耗时只能用于验证程序内部的 event/输出关系，不能与历史四卡数据比较。

## 5. 下一次多卡执行顺序

在目标 2/4 卡 RTX 4090 或 A100 主机：

1. 保存 GPU 型号、拓扑、P2P、driver、CUDA、NCCL、MPI 和实际库路径；先核对 CUDA_VISIBLE_DEVICES 与 MPI local rank 映射。
2. 构建基准：

       cd /root/comm-study/ag-gemm-bench
       cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
       cmake --build build -j

3. 先进行 2 rank、小 shape、q=1/2 的 preflight，确认三个策略的每 rank 和 global CSV 均 PASS。
4. 从第一篇真实 LLM trace 中确定至少 10 个 AG-GEMM shape，覆盖通信主导、平衡、计算主导以及 prefill/decode。不要直接将脚本中的 3 个合成 shape 作为论文正式 workload。
5. 每个 shape 用 q={1,2,4,8,16}（q=16 若已严重降低 GEMM 效率可停止），对稳定合法的 DEFAULT 和 Ring/Simple 候选跑 20 warmup、50 timed samples、5 个独立 process run。
6. 对每个 shape/strategy 依次比较 T_comm_only、T_gemm_only、B0、B1、B2、release curve 和 GEMM TFLOPS；保存 Nsight Systems 时间线。
7. 仅当 B1/B2 的正确性稳定，且出现超过重复噪声的“busbw 或 T_done 排名与 T_e2e 排名不一致”时，才进入 NVSHMEM/DUSHMEM 原语验证和候选 selector。

PGAS 阶段的正确门槛仍是：

    payload write/put -> 文档和运行均核验过的 fence 或 quiet
    -> epoch signal -> consumer wait -> checksum

不能将 signal 本身当作 payload 已可读的证据。

