# 全部工作成果总览与阅读指南(2026-10-12)

> 目的:暂停实验、消化成果时的**单一入口**。三条线一份地图:哪个文档讲什么、
> 哪个数据文件对应哪个结论、哪些知识值得内化。清理记录见 §5。

## 0. 三条线索一张图

| 线 | 时段 | 平台 | 仓库位置 | 性质 | 状态 |
|---|---|---|---|---|---|
| **A. K500SM 平台线** | 8月底-9月初 | K500SM_AI/gfx928/4GPU/PCIe | `second_hygon/`(phase1-3、phaseA、phaseb、bw1000-port) | Collective-GEMM 重叠基准、双轴反转、边界定律、selector | **已收工**(A800 对照侧同) |
| **B. serving 测量线** | 10月8-12 | 2×8 BW1000 vcjob(RDMA) | `second_hygon/phase4~6`、`results/` | 论文#2 的测量支柱:到达异质性 → e2e 瓶颈 → vLLM wait/wire 分解 | **M-sweep 已完** |
| **C. 迁移+skew 实验** | 10月10-12 | 2×BW1000 迁移实例 | `tri-dist-hygon/` | 论文#2 的系统载体:四融合算子移植 + S4-S9 + 歧斜注入 | **S8 两轮已完,S9/tune2 已落** |

主线一句话(创新板):**把通信优化的目标函数从 wire 换成 wait**——
serving 全部 M、两种拓扑下 AR 时间 85-97% 是 wait,wire 侧优化理论上限就是那 3-14%。

## 1. 必读清单(按优先级)

### ★★★ 核心五篇(理解当前局面)
1. **`second_hygon/INNOVATION_BOARD_20261012.md`** — 主线 thesis、三方向判断
   (编译器不做主线/DUMMA 做子系统/多机扩展=主弧线)、创新点 I1-I6 全表、节奏。
2. **`second_hygon/phase6-vllm-deploy/REPORT_msweep_20261012.md`** — serving 测量主报告:
   载体切换表(custom AR 全覆盖,NCCL AR=0)、wait/wire 主表(27×→6× 从不反超)、
   F1-F6(δ 与 M 无关、chunked prefill 归一化、粘性双态、尾部税 26-39%)、
   对创新点的定量启发。**论文#2 Measurement 章的直接底稿。**
3. **`tri-dist-hygon/docs/S8_SKEW_FINDINGS_zh.md`** — 两轮受控歪斜注入:
   k·δ 定律(斜率=k:1.94/3.91)、fused 吸收窗 w(270→468→827,模型一位不差
   2115=2115)、赢 mono 判据 deficit<w(缺口 153-255µs 恒定→钥匙=S5 GEMM)、
   fill 首轮阴性+测量缺陷诊断。
4. **`tri-dist-hygon/docs/DCU_PORT_FINDINGS_zh.md`** — 移植问题全记录(P1-P10)+
   S5(GEMM 调优/RMA 信号根治尾延迟)+ S6(gemm_allreduce 移植)。
   含 §3"八卡能打赢 NCCL 吗"量化判断。
5. **`tri-dist-hygon/docs/OPS_GUIDE_zh.md`** — 四融合算子可用性与性能现状:
   2 卡 MoE a2a **快 RCCL 2.3-2.9×**;三个 GEMM 算子落后(主因 triton GEMM 1.7-3.1×);
   **8 卡全输**(单播扇出 17-25GB/s/rank + 编排开销随 R 放大)。

### ★★ 按知识域深入(消化"隐藏知识"的路径)
- **推理 serving**:①上列 #2;②`results/phase6_trace_report_20261009_212123.md`
  (双机 16 rank dur-反演:δ p50 534µs/wire 92µs,hazard L10 0.979,straggler 跨节点交替);
  ③`phase5`(e2e 瓶颈:跨节点到达 236ms vs GEMM 2ms=117× → SCHEDULING_VALUABLE);
  ④`paper/JM锚点小节_draft_v3` 等(paper#1 衔接)。
- **并发/流/自旋(SM 级)**:DCU_PORT_FINDINGS 的 P4(缓存一致性)、P5(设备发起传输
  19MB/s 悬崖)、P8(自旋病理三阶段)、P9(SM 预算与提交顺序)、S5.3(引擎信号写
  停顿与 RMA 根治)、S6.2(微型设备原子并发性);S8 fill 阴性(t0 发的 fill 与 GEMM 相
  时间分片,窗口未被探测——流并发≠免费 SM)。
- **通信 collective**:①`second_hygon/docs/B_集合通信调优与策略选择.md`(调优知识);
  ②`C_GPU单边通信与完成语义.md`(单边语义);③phaseb 终判表(`results/INDEX.md` 索引):
  双轴反转、同二进制 d 族终判、边界定律闭式、P15/P16 外推、三基座 selector;
  ④`A_计算通信重叠.md`。
- **论文写作**:①`second_hygon/docs/论文骨架/`(雏形+初稿 1-5 章);
  ②`相关工作盲区详解_细粒度重叠四系统与三盲区_20261010.md`(43K,定位文献用);
  ③`组会汇报-20260901`(讲稿底子);④`零基础概念讲解`(术语)。
- **编年史(可选)**:`NVIDIA与海光的交流窗/海光的进展.md`(倒序,每批工作一条,
  跨平台口径);`docs/interactive_html/`(7 个教学 HTML:tokenizer→全 step→AR skew→
  到达时间线,给人讲解用)。

### 实验数据文件地图(结论 ↔ 数据)
| 结论 | 数据文件 |
|---|---|
| serving wait/wire 全表、粘性、尾部税 | `phase6-vllm-deploy/phase6_torch_prof_rdma_20261012_003342/`(5.4G 原始)+ `msweep_summary.json`/`msweep_table.md`(权威聚合) |
| 跨机 δ/wire/hazard | `phase6_torch_prof_rdma_20261009_212123/analysis/straggler_predict.json` + `results/phase6_trace_report_*.md` |
| wait 内容/pre-AR 间隙 | `phase6_torch_prof_rdma_20261011_153613/`(2.0G)+ `_analyze.log`/`_waitB.log` |
| 到达异质性(同节点 1.8ms vs 跨节点 236ms) | `results/phase4_arrival_full*json`(20261008 两批)+ `phase4-arrival-heterogeneity/` |
| e2e 瓶颈 | `results/phase5_e2e_full_rdma_20261008_165129.json` |
| k·δ/吸收窗/fill | `tri-dist-hygon/bench/results/s8_skew_2card{,_fill,_k4}.json` |
| 四算子 2卡/8卡 | `s4~s7_2card_bw1000.json`、`s9_8card.json` |
| GEMM 调优(逆差攻坚,S5 线) | `tune_gemm.json`、`tune_gemm2.json`(20261012 14:42 最新) |
| phaseb 全系 | `second_hygon/results/INDEX.md`(自带索引,勿重复) |

## 2. 值得内化的知识点(三个域的"干货")

**推理/serving 域**
- AR 时间 86-97% 是 wait;wire 出现两个地板(1stage ~13µs / 2stage ≥4MB ~50µs)。
- δ≈360-510µs 与 M 无关(宿主起源,不是消息尺寸的函数);跨机 534µs。
- chunked prefill 把 AR 归一化到 chunk 预算(33.6MB@52µs)——"prefill M 轴"不存在,
  系统变量是 chunk 预算;LLAMA3-70B 每 forward 161 个 AR。
- 粘性双态:decode p_stay=.98(同一 rank 连续迟到上千 forward);短/中 prefill
  stayIn≈1.0(见前几个 AR 即知全 forward);8000tok 去相关但 δ 缩到 72µs。
- 尾部税:top1% 事件=26-39% 等待总量,单事件聚合 2.7s(~300ms×8 rank,GC/分配器级)。

**并发/SM 域**
- DCU 缓存一致性实测语义(P4);设备发起 dushmem 传输整条路 ~19MB/s(P5)→
  信号必须走 host-engine/RMA。
- 自旋病理(P8):生产者-消费者自旋互饿;S6 消费者满格自旋=S8 fill 融不进窗口的
  头号嫌疑(待 PROBE 裁决)。
- 流并发 ≠ 免费 SM:两个满设备 kernel 时间分片(t0 发的 fill 与 op 精确加性);
  窗口(名义 GPU 空)要靠**调度进去**,不是自动溢出。
- 提交顺序与 SM 预算(P9/P10):基准方法学的两个坑(自坑记录)。

**通信域**
- k·δ 定律:collective 入口数 × δ 是乘法惩罚(k=4 时 3.8× 于 mono)。
  skew 下想 chunk 化只剩解耦面一条路。
- fused 时间模型封闭:t = δ0 + max(0, δ−w),w=自家推送管线时长,窗随 k 线性长
  但汇率 1:1(每 chunk w+~190µs、δ0+~210-350µs)→ 窗工程不是翻盘钥匙,
  GEMM 逆差(~550µs)才是。
- 吸收窗的本质="我还在忙自己的事":吸收别人 δ 与排斥外来 kernel 是同一性质。
- 集合语义(老线):双轴反转(孤立最优≠端到端最优)、协议串行化被计算掩盖的
  N 边界(N*(q8)=2581)、credit 深度是二阶载体、一阶=逐片协议税。

## 3. 各阶段一句话结论速查
- **phase4**:到达异质性——同节点 1.8ms/跨节点 236ms(RCCL TCP 回退量级),
  节点级零翻转+簇内重排=调度可利用形态。
- **phase5**:e2e 瓶颈——一个跨节点 8MB 到达 ≈117 个 GEMM;RDMA(shca IBext)修复后
  进入 phase6。
- **phase6**(三次正式跑):092123 双机 dur-反演(③)→ 153613 wait 内容/PreAR 间隙 →
  **003342 M-sweep 七相位**(④',报告见 #2)。
- **S4-S7**:四算子移植+基线(a2a 2 卡 2.3× 赢;GEMM 三算子输在本地 GEMM 质量)。
- **S8**:两轮歪斜注入(见 #3)。**S9**:8 卡验证(全线输 RCCL 2.6-7×,扇出不扩展)。
- **tune_gemm2**(10-12 晚):GEMM 调优第二轮,4096×2048×2048 已到 0.296ms 档
  (对照 vendor ~0.09ms,仍有 3× 缺口,最新一轮未见文档总结)。

## 4. 悬而未决(恢复实验时的入口)
1. S8 fill2(DELAY+PROBE)——I2 去留裁决(s8 已改好,命令在 S8_SKEW_FINDINGS 尾部)。
2. GEMM 逆差攻坚(tune_gemm2 刚跑,需对照 S8 判据 deficit<w 复算赢区)。
3. ph1 189MB×8 trace 的 300ms 停顿身份(零卡时可做,喂 I3)。
4. ph7 去相关机制(chunk 背靠背 vs 混合调度,定 I1 边界)。
5. D v1+I4 合并批次(集群卡时);③×2 重复;I5 跨机双平面;I6 MoE(S7 骨架已在)。
6. 开问题:DUMMA 身份确认;torch2.9 wheel 已在 `/root/private_data/` 顶层(vcjob 可用)。

## 5. 清理记录(2026-10-12 执行)
**已删(~4.7GB,逐项核实过内容)**:
- `dcu_port_work/core.4791/4792`(4.6G,python ELF core,OPS_GUIDE 记录的双 rank 崩溃残留)
- `dcu_port_work/tcache_s4_rank0/1`(82M,triton 缓存,可再生)
- phase6 `20261012_002838` 失败跑全部痕迹(vLLM max_model_len 配置错,
  下一跑 003342 已修正并成功;phase6-vllm-deploy 与 results/ 两处副本)
- phase5 首次失败尝试日志 `*_163423/163425.log`(torch elastic exitcode 1,正式批 165129 留存)
- `__pycache__` ×12(second_hygon/tri-dist-hygon/dcu_port_work)
- `rdma-shca/topo_lib.tar.gz`(已解压副本,.deb 源保留);顶层 `untitled.txt`
  (A800 退租通知草稿,事项已送达)

**保留决策(未删,说明理由)**:
- 三个 phase6 正式跑原始 trace(003342 5.4G / 092123 2.7G / 153613 2.0G):
  数据纪律"原始数据留盘";003342 是 M-sweep 证据+ph1 待分析;092123 是唯一跨机捕获。
  若空间紧张,153613 的 2.0G 分析已完结,可经确认后删原始 trace 保留 analysis/。
- `Triton-dist-dcu/`(4.2M):第一代移植工作区(kernels/dcu 旧版,无 gemm_allreduce),
  已被 tri-dist-hygon 取代;体积极小,留作历史。
- `worker0*.md`×2 + `报错.md`(共 2MB):vcjob 执行记录+RDMA 诊断报告(含"可转发
  平台管理员"段落),是 shca 修复过程唯一的现场记录。
- `thousand_model/`(269G)、`LJ/`(38G)、`tblogs/`、`job_example/`:非本线工作,未动。
- `*.whl`(triton 3.3+das torch290):vcjob 侧要用,保留在顶层。
