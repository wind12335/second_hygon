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

## ④ 修正(2026-10-10,依据 41,538 个 AR 的逐事件 [ts,dur] 序列全量复核)

用逐事件序列重算发现 **§② 的方向解读有误**;数字与证据(复算脚本
`docs/interactive_html/build_07_arrival_data.py`,可视化 `07_arrival_timeline.html`):

1. **16 rank 的 AR 结束时刻高度同步**(跨节点以"结束同步"锚定:offset=7.6µs,
   残差 p50=3.0 / p90=19.7µs)⇒ dur = 结束 − 到达,**时长越长 = 到得越早**。
2. **修正后的结构**:worker-1 八 rank **整队到达**(节点内到达散布中位 4µs);
   worker-0 内部散布中位 295µs,且 **75% 的 AR 全局最后到者在 worker-0**
   (pid931 独占 29.1%、pid927 12.9%、pid936 12.4%)。
   原"worker-0 早到长等 / worker-1 晚到少等"**方向反了**。
3. **修正后的分解**(worker-0/pid927 视角,30.3s):
   真走线 = 每集合**全局**最小时长,均值 91.9µs → 总量 **3.82s(12.6% 墙钟)**(原 26% 高估);
   真等待 = dur − 真走线 → **16.78s(55.4% 墙钟)**(原 45% 低估);16 rank 各自 44.5–65.8%
   (16-rank 平均为 452µs/AR = 18.8s = 62.0% 墙钟,E1 对账口径;55.4% 是 pid927 单 rank 视角)。
4. 原因:原口径"节点内 dur_min≈走线"只在该节点含全局最晚者时成立;
   worker-1 的 dur_min(均值 437µs)大头是在等 worker-0 的迟到者,不是走线。
5. δ(每次 AR 的全局到达散布):**p50=534µs / p90=687 / p99=1176µs**——每次 AR 半毫秒级的等人。
6. 自旋模型互验:(dur − dur_min_node) ≡ (节点内最晚 ts − ts),p50 差 2.4µs ✓。
   §① 的 busy 占比(AR 85%、GEMM 8.5%、重叠 0)不受影响,仍然成立。

## ⑤ A·straggler 可预测性交叉验证(2026-10-11,analyze_straggler_predict.py mode=dur)

目的:④' 单机(custom AR,同钟 ts 直读)得出的"迟到身份短程强可预测"是否在
**双机 NCCL Tree_LL** 场景复现——M2 可行性不能只靠单一部署形态。

口径:跨节点无公共钟 → straggler = 每集合 **argmin dur**(AR 结束同步 ⇒ dur 越长到得越早,
见 §④-1)。自检:此口径复算出 pid931=29.1%、w0 合计 75.3%,与 §④-2 已发布数字吻合。
**注意:argmax dur(最早到达者)是错误方向**——首次跑出 h=1=0.034、hazard 崩塌,
即取反的信号(且 16 rank 中最早者多为 w1 整队块内 4µs 级噪声,近随机)。
N=41,538,skip 前 10%,测试 37,385。

| 量 | ④' 单机 TP8 | ③ 双机 TP16(本次) |
|---|---|---|
| repeat-last h=1 | 0.989 | **0.971** |
| h=8(提前量 ~5.5ms) | 0.933 | **0.875** |
| h=64 | 0.780 | 0.542 |
| 随机 / 全局众数上界 | 0.125 / 0.313 | 0.062 / 0.285 |
| hazard L1 → L100 | 0.886 → 1.000 | **0.607 → 1.000**(L2 即 0.920) |
| 滑动窗多数 vs repeat-last | 更差 | 更差(W=2:0.955 < h=1:0.971,宽窗单调劣化) |

- **结论:可预测性跨部署形态稳健**——双机下 5.6ms 提前量仍 87.5% 集合可正确预知
  迟到者(vs 单机 93.3%);hazard 反向(准吸收长 run)与"结构优于统计平滑"均复现。
- 差异点:双机 L1 stay 0.607(单机 0.886)——跨节点噪声制造更多单点断裂,但 run 一旦
  到 L2 立即 0.92+;边界圈 {931,832,927,929,936} 跨两节点(832 是 w1 唯一常客,10.3%),
  与 §④-2 的 w0 主导 + 个别 w1 参与一致。
