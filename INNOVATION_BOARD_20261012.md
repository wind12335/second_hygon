# 论文 #2 创新点板 — v3 决策记录(2026-10-12)

数据基础(全部实测,可复查):
- ④' M-sweep 单机 8×BW1000:`phase6-vllm-deploy/REPORT_msweep_20261012.md`
  (wait/wire = 27×→6× 不反超;δ 360-510µs 与 M 无关;chunked prefill 把 AR 尺寸归一化;
  粘性双态 stayIn≈1.0 / ph7 去相关 0.15-0.33;尾部税 top1% 事件=26-39% 等待)
- ③ 双机 16 rank dur-反演:`phase6-vllm-deploy/phase6_torch_prof_rdma_20261009_212123/analysis/straggler_predict.json`
  (δ p50 534µs / wire 92µs;hazard L10 0.979;straggler 931↔832 跨节点交替)
- tri-dist-hygon S5/S6:`tri-dist-hygon/docs/DCU_PORT_FINDINGS_zh.md`
  (解耦面原型;RMA 信号无 33ms 尾;P9 自旋消费者饿死生产者=B1 的 kernel 内镜像;
  设备发起信号 19.5MB/s 全局上限 → 信号必须 host-engine/RMA)

**主线一句话:把通信优化的目标函数从 wire 换成 wait。**
依据:serving 全部 M、两种拓扑下,AR 时间 85-97% 是 wait → 一切 wire 侧优化
(带宽/融合/协议)的理论上限是那 3-14%;未被占领的系统位置是"测量-预测-回收 δ"。

## 三个方向判断

### ① 编译器(MLIR/LLVM)——不做主线
- 硬约束:海光 triton 3.3+das 闭源 wheel 不可重编;上游 triton-dist 的编译器件
  (async_task 降级、mori 调度)正是 S1-S6 绕开的部分。真做 MLIR pass = 先补
  gfx936+DTK 适配,以月计。
- 软理由:编译器层位置已被占(FlashComm=融合 kernel,上游=overlap pass);
  "wait 主导"没人做成系统 → 创新落在 runtime/调度层(Lina/Optimus/FlowER 先例)。
- 保留的编译器味道(两层可控):(a) torch/FX 图级 pattern rewrite——识别 AR→GEMM 链
  注入解耦面;(b) kernel 库级**到达剖面引导生成**(Arrival-Profile-Guided
  Scheduling)——把实测 δ/预测器输出烘进 launch 配置(MPC、spin 预算、消费顺序)。
  真 MLIR 留作 future work / 海光开源后。

### ② 分配层(DUMMA 式)——做,作为子系统
候选对标:MLSys'25 DUMMA(统一多粒度内存分配,GNN 训练;身份待确认)。
三个真实落点(目标函数不同:DUMMA 打容量碎片,我们打 skew 下暴露延迟):
1. **k\* 深度=内存预算**:集体语义 k\*=1(E1);解耦面 k>1 chunk 在飞,每 chunk 一份
   DUSHMEM 对称槽。serving 三路并发(decode AR+logits allgather+prefill chunk)共享
   槽池,按实测 δ/wire 定 k → 画"延迟-足迹 Pareto"。
2. **尾部停顿根因位**:top1% 事件 26-39% 等待、单事件 ~300ms×8;若根因是分配器/GC
   (ph1 189MB trace 零卡时可定位)→ rank 交错 GC/预注册 arena 即尾部章节的 fix。
3. **注册成本**:DUSHMEM malloc 贵,S5/S6 靠预注册;统一池免逐 op 注册。

### ③ 基于第一篇扩展到多机——是,这就是第二篇的主弧线,不拆篇
paper#1 = 单机分解+预测;paper#2 = 跨拓扑成立性(③ 已证)+ 第一个系统载体
(tri-dist-hygon)+ 下表 I1-I4 的能力(单机重排做不到)。多机新科学:无公共时钟
方法论(dur 反演已建)、双 δ 尺度(机内 400µs 宿主源 vs 机间?)、
**anchor 按拓扑分层(跨节点最长杆先走,反朴素 nearest-first)**、
DUSHMEM ibgda/ibrc 传输让解耦面天然跨机。

## 创新点板

| # | 创新点 | 数据依据 | 机制(tri-dist-hygon 上) | 主要风险 |
|---|---|---|---|---|
| I1 | 预测器进环 | decode p_stay=.98;短/中 prefill stayIn≈1.0(见前几个 AR 即知全 forward);hazard L10=.98 | 宿主侧 hazard 预测器(每 forward 一次,近零开销)→ 驱动消费顺序、spin 预算、MPC;跨机按方向分预测 | ph7 类去相关段要自动降级;预测开销 |
| I2 | 等待窗口算力回收 | 每 rank 等待占窗口 56-108%;decode 步时间 60-98% 就是 AR 等待(56ms/步里 ~55ms=2×80×345µs) | 解耦面把 push 移到 host-engine/RMA,等待窗口 GPU 可调度:填另一 microbatch/独立工作 | 纯填窗口与 Sarathi 类调度同质;差异=δ 预测定预算+先解 P9(spin 有预算地让出) |
| I3 | 尾部容忍 | top1% 事件=26-39% 等待;单事件 2.7s 聚合 | 宿主看门狗(旗标超时→宣告 stall)→ 单侧跳过该 rank 的 chunk 推送/降级消费;叠加 ② 的分配干预 | 降级正确性(和的结合律)+恢复时延;先零卡时定位 300ms 根因 |
| I4 | wait-aware 协议 | 2stage ≥4MB wire 平台 ~50µs(疑两段握手地板) | 旗标写入 payload 尾部的单相 RMA put;机间(wire 92µs)收益更大 | 被 wait 主导封顶 → 补全性贡献,与 D v1 一起验 |

次级(第二幕):
- I5 跨机双平面:机内 xGMI 单侧 + 机间 ibgda;anchor 按拓扑分层;③×2 重复加固统计。
- I6 MoE all2all 解耦面:迁移仓库自带 FlashComm → `bench/s7_bench_moe_a2a.py` 已在
  (第二幕已开工)。MoE 的"δ"=专家负载不均,比 decode straggler 更可预测
  (专家热度 Zipf 且粘,待验证),预测器故事平移后更强;对上 128 卡路线 M3 缺口。

## 节奏
1. **v3 核心实验**(S6+δ 注入,三架构×两注入模式×δ 网格,测 δ 响应斜率与反超点 δ*)
   —— ✅ 两轮已跑(2026-10-12 双卡):k·δ 定律**闭合**(chunked 斜率 1.94@k2、
   **3.91@k4**,δ1600 下 chunked=mono 的 3.8×);fused 时间模型
   δ0+max(0,δ−w) **精确命中**(k4:预测 2115=实测 2115,越窗斜率 0.93);
   窗工程确认为维度(w=270→468→827 随 k)但**汇率 1:1**(每 chunk w+~190µs 同时
   δ0+~210-350µs),赢区空缺 153→217→255µs 恒定 → 钥匙仍是 S5 GEMM 调优;
   fused 对 chunked 优势随 k 放大(δ800:1372 vs 3602)。
   fill 首轮**阴性且方向反**(hidden 均值 mono+18%/chunked+13%/fused−3%,全加性)
   ——诊断为测量设计缺陷:t0 发的 fill 与 op 的 GEMM 相时间分片,等待窗未被探测;
   已修 s8(S8_FILL_DELAY_US 把 fill 排进窗内 + S8_FILL_PROBE 流并发探针),待跑。
   详见 `tri-dist-hygon/docs/S8_SKEW_FINDINGS_zh.md`(两轮全记)。
2. 零卡时:ph1 trace 定位 300ms 停顿身份(喂 I3);ph7 去相关机制(定 I1 边界)。
3. I1 进环 → I2(先解 P9)→ I3;fill 修正轮定 I2 的"窗口回收"表述去留。
4. D v1+I4 合并一批集群卡时;③×2+I5 第二批;I6 收尾。

## 开问题
- DUMMA 身份确认(MLSys'25 内存分配?还是 MoE 通信重叠类——后者打 wire 侧,
  恰是我们数据的反击对象);
- 迁移实例双卡已确认(S6 结果在);torch2.9 wheel 是否上共享存储(vcloak 走 vcjob 用);
- ph7 去相关机制(chunk 背靠背 vs 混合调度);300ms 停顿身份。
