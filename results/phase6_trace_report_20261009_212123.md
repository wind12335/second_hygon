# Phase6 ③ torch profiler 分析报告 · Llama-3-70B TP16 双机 16 卡 vLLM(自然负载窗口)

**数据源**:`phase6_torch_prof_rdma_20261009_212123/`(16 份 per-rank trace,30.3s 采集窗口)
**负载**:`loadgen natural`,conc=16、maxtok=128(32 请求全部成功,p50=10.6s,吞吐 1.1 req/s)
**运行形态**:vllm 0.11 das / ray 后端 / TP=16 / `--enforce-eager` / 34× `via NET/IBext_v8`(全 RDMA,零回退)
**分析工具**:`analyze_traces.py`(流式解析,断点续跑)+ `summarize_analysis.py`(GEMM 名修正 + skew 标签修正)

## ① 通信 vs 计算:AllReduce 占 85%,通信/计算 ≈ 10×,重叠 = 0

| 类别(busy 占比,节点均值) | worker-0 | worker-1 |
|---|---|---|
| **RCCL AllReduce** | **84.9%** | **85.7%** |
| GEMM(hipBLASLt `Cijk_*`,3 种 tile) | 8.7% | 8.2% |
| attention(paged_attention) | 1.3% | 1.3% |
| elementwise + norm/act/rope | 2.3% | 2.2% |
| 其他 | 2.7% | 2.6% |
| GPU busy 占窗口 | 85.2% | 89.8% |

- 16-rank 均值:AllReduce **22.60s** vs GEMM **2.24s** → **通信/计算 ≈ 10.1×**
- **comm∩compute 重叠 = 0.000s(16/16 rank 全部为 0)**:eager TP 下通信与计算完全串行,无任何重叠
- AllReduce 计数 41,538/rank(各 rank 完全一致,锁步执行)= 160 AR/step × ~260 step;
  另 AllGather 仅 258 次(权重加载期以外偶发)
- 对照 phase5 微基准:M=8192 时 AG 占 37%;本窗口自然 decode(M≈32)→ 85%。
  M 变小、单 step 通信量不变(2×80×8192×bf16),计算塌缩 → 通信占比暴涨,方向与理论一致。

## ② 到达偏斜(phase4 的框架级版本):AR 时长的 2/3 是"等人",不是走线

同节点 8 rank 按集合序号对齐(共 41,538 个 AllReduce),`dur_r − dur_min` = 该 rank 在 kernel 内部自旋等待的时间:

| 节点 | 各 rank 等待占窗口 | intrinsic(逐集合最小 dur) |
|---|---|---|
| worker-0(head) | **30.5% – 51.8%**(均值 ≈45%) | p50=80µs,均值 193µs |
| worker-1 | **10.0% – 18.2%**(均值 ≈16%) | p50=360µs,均值 437µs |

- worker-0 的 rank 普遍**早到、长等**;worker-1 的 rank 晚到、少等 → 跨节点到达不对称,
  head 节点 ~45% 墙钟时间耗在 AR kernel 内自旋等对端。
- 节点内也有稳定异质性:worker-0/pid931 最常"最后到"(14,233/41,538 = 34%),
  worker-1/pid832 最常最后到(25%)——固定 straggler 存在,与 phase4 的 rank 级现象呼应。
- **窗口时间分解(worker-0 视角,30.3s)**:计算 ≈4.6s(15%)‖ AR 走线 ≈8.0s(26%)‖
  **AR 等待 ≈13.8s(45%)** ‖ GPU 空闲 ≈4.5s(15%)。
  即 AR 名义占墙钟 75%,其中约 2/3 是到达等待而非数据搬运——**这部分正是论文要用
  到达异质性去调度消掉的**。

## 口径与局限

- 同节点 rank 共享主机时钟,节点内对齐与时长比较可信;跨节点只做定性比较(时钟独立)。
- `dur_min` 是传输时间上界(全体最晚者在本节点时才等于纯走线)。
- 本轮 trace 用旧分类落盘(GEMM 落在 other);修正值由 top10_kernels 无损换算
  (top10 覆盖 96.4% kernel 总时长),脚本已改好,后续 trace 直接出正确分类。
- 自然负载窗口 prefill/decode 混合,decode 为主导(32 并发、每请求 ≤128 token);
  定长 prefill(M≈8000)对照窗口待跑(脚本 PTOK=8000 默认开启,用户暂缓)。

## 产物清单

- `analysis/rank_*.json` × 16(每 rank:分类占比、重叠、top10、RCCL 分型、AR 序列)
- `analysis/skew.json`、`analysis/summary_corrected.json`(修正口径)
- 脚本:`analyze_traces.py`、`summarize_analysis.py`(可复用于后续 trace 目录)
