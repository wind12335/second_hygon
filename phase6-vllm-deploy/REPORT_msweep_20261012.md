# M 轴扫描分析报告 — 单次 vLLM 启动 7 相位(wait/wire 分解,论文 #2)

运行:`phase6_torch_prof_rdma_20261012_003342`(vcjob,单机 8×BW1000,TP=8)
模型:LLAMA3-70B(80 层,hidden 8192,maxpos 8192 → MAX_LEN 上限,ptok 封顶 8000)
负载:vLLM v1 + 自定义 allreduce(--enforce-eager),torch profiler CPU+GPU,8 rank 全捕获
相位:`d:4:15, d:16:15, d:64:15, d:256:15, p:512:10, p:2048:10, p:8000:10`
(d=自然 decode 相位 M≈conc;p=定长 prefill 相位,PTOK_CONC=4,seed 逐相位错开防 prefix cache)
产物:PHASES.json manifest → split_phases.py → 56/56 rank 抽取(extract_custom_ar.py)→
analyze_arrival_8rank.py + msweep_aggregate.py(本文数字以 aggregator 为准,见 §6 修正说明)

---

## 1. 载体切换:custom AR 全覆盖,NCCL AR 缺席

| 相位 | 每 forward 的 M(tok) | AR 消息(B=bytes, bf16×8192) | 载体 | N(lockstep) |
|---|---|---|---|---|
| ph1 d4 | 4 | 64 KB | 1stage | 42,665 |
| ph2 d16 | 16 | 256 KB | 1stage | 19,320 |
| ph3 d64 | 64 | 1 MB | 2stage | 20,608 |
| ph4 d256 | 256 | 4 MB | 2stage | 20,608 |
| ph5-7 的 decode | 4 | 64 KB | 1stage | 20,447 |
| ph5-7 的 prefill | 2048(chunk) | 33.6 MB | 2stage | 322(=2 个 chunk-forward) |

- **ncclDevKernel_AllReduce 在所有相位都是 0**(个别 161/483 为相位边界的启动残留):
  vLLM custom AR(1stage/2stage)覆盖了 serving 在单机上发出的**全部** TP allreduce。
- 每 forward 161 = 80 层×2 + 1:42665/161=265 个 decode forward(ph1,15s→56ms/步)等,计数全部自洽。
- logits 采样走 `ncclDevKernel_AllGather_Ring_LL`(129 次/相位,p50 432µs)+ scatter/argmax,与 AR 无关。

## 2. 主表:wait/wire 分解(8 rank 同钟,到达=ts;窗口=名义时长,绝对 µs 与窗口无关)

| 相位 | M | 载体 | wire p50 | δ p50/p90 | meanWait | wait/wire | wait% 窗口 | p_stay |
|---|---|---|---|---|---|---|---|---|
| ph1 d4 | 4 tok | 1stage | 13 µs | 463/618 | 345 µs | **27×** | 98%* | 0.985 |
| ph2 d16 | 16 tok | 1stage | 15 µs | 510/627 | 457 µs | **30×** | 59% | 0.986 |
| ph3 d64 | 64 tok | 2stage | 24 µs | 368/611 | 456 µs | **19×** | 63% | 0.984 |
| ph4 d256 | 256 tok | 2stage | 48 µs | 360/528 | 408 µs | **8×** | 56% | 0.984 |
| ph5 p512 decode | 4 tok | 1stage | 15 µs | 458/636 | 470 µs | 32× | 96%* | 0.986 |
| ph6 p2048 decode | 4 tok | 1stage | 16 µs | 371/621 | 495 µs | 32× | 101%* | 0.984 |
| ph7 p8000 decode | 4 tok | 1stage | 15 µs | 437/642 | 526 µs | 34× | 108%* | 0.986 |
| ph5 prefill | 2048 chunk | 2stage | 52 µs | 312/677 | 378 µs | 7× | — | 0.994 |
| ph6 prefill | 2048 chunk | 2stage | 53 µs | 352/617 | 311 µs | 6× | — | 0.994 |
| ph7 prefill | 2048 chunk | 2stage | 52 µs | 72/519 | 581 µs | 11× | — | **0.268** |

*带 * 的 >100% 或偏高项:单 rank 的 wait 是逐事件累加,窗口占比>100% 表示该 rank 的等待时间之和超过采集窗(等待与计算/别的事件并行计数)。绝对 µs 列不受影响。

互验:结束同步残差 p50 2.4-3.1µs ≪ δ(锚定有效);反演 vs 同钟 p50 <1µs。

## 3. 发现

**F1 δ 与 M 无关(宿主起源)。** decode 相位 M 从 4→256(消息 64KB→4MB,64 倍),
δ p50 始终 360-510µs;prefill chunk(33.6MB)的 δ 312-352µs 同量级。
δ 是宿主/调度侧抖动,不是消息尺寸的函数。

**F2 wait 全程主导 wire,单机上 M* 不在 serving 包络内。** wait/wire 从 27×(M=4)
单调降到 6-8×(M=4MB-33.6MB)但**从不反超**。wire 出现两个"地板":1stage ~13µs
(小消息)与 2stage ~50µs(≥4MB 后 48→52µs 几乎不涨,疑似两段式协议/自旋握手地板,
33.6MB@52µs → busbw≈1.1TB/s,待 D 微基准确认)。即使按 33.6MB 的 chunk AR,
等待仍是走线的 6-11 倍 —— **在真实 serving 发出的每一个 M 上,collective 时间
86-97% 是 wait**。跨机 ③ 对照同向:wire 92µs vs δ 534µs(5.8×)。

**F3 chunked prefill 把 AR 尺寸归一化——"prefill M 轴"在 AR 层面不存在。**
ph5/6/7 的 prompt 512→2048→8000 tok/请求,但 prefill AR 的 wire p50 完全一致
(52/53/52µs):vLLM 按 chunk 预算(≈2048 tok)切分,大 prompt 只增加 chunk-forward
数量(M=33.6MB 不变)。任何"大 M 才需要带宽优化"的论证在 serving 里要重新对准
**chunk 预算**这个自变量,而不是 prompt 长度。

**F4 粘性双态:一个 forward 内"1 拖 7",长 prompt 去相关。**
簇分解(2stage,间隔>5ms 分簇,每大簇≈一个 chunk-forward 的 152-160 个 AR):
- ph5:簇1 stayIn=0.994(pid933 迟到 146/160),簇2 stayIn=**1.000**(pid936 迟到 159/159);
- ph6:簇1 stayIn=**1.000**(pid938 包揽 160/160),簇2 stayIn=0.992;
- ph7:两簇 stayIn=0.331/0.146 —— straggler 逐 AR 换人,近随机。
解读:δ 中有两类成分——**forward 起点的宿主延迟**(一次注入,层间管道传递,
整个 forward 粘在同一 rank 上:stayIn≈1)与**逐 AR 的计算抖动**(短/中 prompt 时被
前者淹没,8000 tok 时显性:δ p50 缩到 72µs 但 p99=11ms 重尾)。decode(连续数万
forward)p_stay=0.98:同一 rank 连续迟到上千 forward。
与 ③ 跨机 dur-反演口径互验:repeat-last h=1 0.971 / h=16 0.795,hazard L10 0.979。

**F5 尾巴税:top 1% 的 AR 事件占 26-39% 的等待总量。**
单事件聚合等待(8 rank 求和)最大 2.2-2.7 秒 —— 即一次 AR 里早到者各等 ~300ms,
是 GC/分配器/调度停顿级别的离群,不是通信本身。top10% 占 38-49%。
ph1(conc=4)例外:top1% 仅 4.9%,并发低时尾部轻。

**F6 测量方法教训(写给实验记录)。**
(a) 多相位跑批脚本里 `case "$d" in *_p*` 误配全路径(含 `_prof_`),decode 相位被按
10s 窗口算,占比虚高 1.5×;本文已用名义窗口(15/10s)重算,绝对 µs 不受影响。
(b) PHASES.json 的 t1-t0 含 stop_profile 的同步落盘等待(ph1 达 373s),不能当采集窗。
(c) 2stage 的 dur=等待+走线混在早到者一侧,分离必须用 min-dur(最晚到者)取 wire。

## 4. 对创新点的启发(结合 tri-dist-hygon 新增的 all-reduce)

**核心定量主张成型:在 decode/短中 prefill 的全部 M 与两种拓扑下,AR 时间 85-97%
是 wait;wire 优化(bandwidth/融合)的理论上限是那 3-14%。**

1. **E1(k*=1)与 S6(解耦面)的分工被数据定死。** E1 打的是集体语义 AR:每个 chunk
   重付一次 δ(k*=1 是最优);S6 的 gemm_allreduce(C=Σaᵣ@bᵣᵀ,逐 chunk 推送 +
   乱序 spin 消费)没有集体屏障,δ 被 chunk 管道摊销。既然 wait:wire = 6-27×,
   v3 的科学问题唯一化:**在 δ≈400µs 的 decode skew 下,解耦面每 chunk 能收回多少
   wait**(而不是省多少 wire)。tri-dist S6 现有骨架(chunk GEMM→对称槽,
   host-engine 逐 (chunk,dest) 流式推,chunk c 的推送与 chunk c+1 的 GEMM 重叠,
   单 host 旗标行 + 混合自旋 fp32 归约消费者)正是这个实验的载体;
   S6.5 当前 1.07ms vs RCCL-seq 0.24ms 的差距(GEMM 3.1×+屏障+残留)在注入 δ 后
   有望反超——RCCL-seq 每次全_reduce 重付 ~400µs 等待,解耦面只付一次。
2. **粘性=可预测性=调度杠杆。** decode p_stay 0.98、hazard@L10≈0.98:迟到者下一个
   AR 大概率还是它 → paper #1 的 anchor/步幅重排(先和已知快者通信)可直接迁移;
   短/中 prefill 的 stayIn≈1.0 更强:**看到 forward 前几个 AR 就知道整个 forward
   的 straggler**,160 个 AR 的重排窗口。长 prompt(8000tok)去相关但代价小
   (δ 72µs)——预测收益天然衰减的地方损失也小,方案不需要为它特化。
3. **尾部是第三条战线。** top1% 事件占 ~1/3 等待、单事件秒级聚合等待:任何只做
   "平均 δ"的方案摸不到这部分。尾部感知(检测停顿 rank → 摘出关键路径/降级备份)
   是独立于重叠的正交创新点,且 ③ 的 hazard 曲线(少数 rank 长期高频迟到)说明
   跨机同样成立。
4. **D v1 的角色更清晰。** serving 侧已给出:两个 wire 地板(13/50µs)+δ 不随 M 变。
   D 微基准补:干净 wire(M) 曲线(隔离协议地板)、δ 注入的可控源、以及 k*>1
   chunk 化在无 skew 时的代价基线。命令就绪未提交。

## 5. 开问题
- ph7 去相关的机制:chunk-forward 背靠背(纯 prefill 批)vs 混合调度(prefill+decode
  交错)——可用 ph6/ph7 的 forward 间隔结构再挖一层;
- 2stage ≥4MB 的 wire 平台是否协议地板(D v1 可直接回答);
- 300ms 级停顿的身份(GC?分配器?调度?):trace 的 CPU 侧 python 栈可定位,
  ph1 的 189MB trace 里应该有;
- torch2.9.0+das wheel 是否可放共享存储(v3 要在 vcjob 上跑 tri-dist)。

## 6. 复现
```bash
python3 msweep_aggregate.py                # 主表+簇分解 → msweep_summary.json/msweep_table.md
for d in phase6_torch_prof_rdma_20261012_003342/ph*/; do \
  python3 analyze_arrival_8rank.py "$d" 15; done   # 旧口径(decode 15s/prefill 10s 分别传)
```
数据:`msweep_summary.json`(权威)、`msweep_table.md`、各 ph*/analysis/。
