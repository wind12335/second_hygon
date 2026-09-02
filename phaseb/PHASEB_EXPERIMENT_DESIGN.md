# Phase B 实验设计：跨基座（RCCL vs DUSHMEM）AG-GEMM 统一基准

日期：2026-09-02
平台：`K500SM_AI / gfx928 / 4 GPUs / 全对全 PCIe / DTK 26.04 / RCCL 2.22.3 / DUSHMEM 3.2.5`

## 1. 目的（要证明/证伪什么）

Phase 1-3 已在 RCCL 内部证明：**isolated collective 最优（T_done / busbw）≠ 有数据依赖的 AG-GEMM 端到端最优**（C2 在 q=8 端到端反而慢 ~7%，9 格中 6 格反转）。
Phase A 已证明 DUSHMEM on-stream put-signal / fcollect 路径正确，且 isolated 层面存在尺寸翻转（4KiB fcollect 好，1/64MiB put-signal 好）。

**Phase B 回答的问题**：
1. 把 RCCL 路径与 DUSHMEM 路径放进**同一二进制、同一 GEMM、同一正确性口径**后，跨基座的 isolated 排序与端到端排序是否一致？
2. D1（DUSHMEM put-signal 分片重叠）相对其公平基线 DS（同分块串行）有没有净收益？相对 D0？
3. 每条路径的 per-slice t_release / t_gemm_start / t_gemm_end 曲线能否解释端到端排序（即"释放语义"是否是必要的选型维度）？

## 2. 统一命名（一次定死，后续论文沿用）

| 新名 | 含义 | 旧名（Phase 1-3 / Phase A 文档） | 家族 |
|---|---|---|---|
| `comm` | 分片 RCCL AllGather 仅通信（isolated 参照） | COMM_ONLY | RCCL |
| `gemm` | 分片 GEMM 仅计算（碎片化代价参照） | GEMM_ONLY | - |
| `r0` | RCCL 整传整算：全量 AG → 全量 GEMM | B0 | RCCL |
| `rs` | RCCL 分片串行：AG(i)→GEMM(i)→AG(i+1) | B1 | RCCL |
| `r1` | RCCL 分片事件重叠：AG(i) → event → GEMM(i) 并行 | **H0**（Phase1-3 语义） | RCCL |
| `fc` | DUSHMEM fcollect 仅通信（isolated 参照） | - | DUSHMEM |
| `dc` | DUSHMEM 分片 put-signal 仅通信，wait 流测 release | - | DUSHMEM |
| `d0` | DUSHMEM fcollect 整传整算 | D0（PhaseA 文档） | DUSHMEM |
| `ds` | DUSHMEM 分片 put-signal 串行 | - | DUSHMEM |
| `d1` | DUSHMEM 分片 put-signal 重叠（ready-wait 驱动） | D1（PhaseA 文档） | DUSHMEM |

注意：组会汇报（20260901）里的 "H0=串行基线" 是术语漂移，**作废**；以本表为准。

## 3. 关键实现决策

- **对称堆统一**：`x_local / full_a / gathered[q] / ready / credit` 全部 `dushmem_malloc`，RCCL 与 DUSHMEM 路径用**完全相同地址布局**喂同一个 rocBLAS sgemm —— 消除"缓冲区来源不同"的混杂。
- **D1 协议**（Phase A 验证过的单调 epoch + credit）：producer 每 slice 先等 `credit[consumer][slot] ≥ 前次 epoch` 与自身 `gemm_end[i]`（自 WAR），再本地 D2D 拷贝 + 对 3 peer `putmem_signal(SET epoch)`；consumer 每 slice 等 `ready[producer][slot] ≥ epoch` → record release → GEMM+scatter → 发 credit。slots = q × window_mult（默认 1）。
- **测量口径**：每 case 独立 MPI 进程组，warmup 后逐迭代 `MPI_Barrier` 对齐；t_issue 在 comm 流、release 在消费流、done 在 comm 流、e2e 在计算流；聚合取 max-rank（与 Phase 2/3 一致）。每迭代元素级校验（abs 1e-2 / rel 1e-4，参考 = 同进程 RCCL 全量 AG+GEMM）。
- **公平性**：D1 只与 DS/D0 比（同分块/同传输），绝不再与 r0 的整块 GEMM 直接比（7 月旧机器的教训）。

## 4. 矩阵

- 形状：m_local = K = 2048（每 rank 16MiB payload，与 Phase 2/3 可比），N ∈ {512, 2048, 4096}
- q ∈ {2, 4, 8}（+ formal 附 N=2048, q=16 边界）
- 路径：上表 10 条全跑；`comm` 与 `r1` 额外跑 C2（Ring/Simple/ch8）以携带 Phase 2/3 的 RCCL 内部反转
- 重复：formal 5 reps × 80 iters（warmup 20）；smoke 1 rep × 40 iters
- 预计 formal ≈ 590 case，~1.5-2h

## 5. 判据

- **跨基座反转**（新）：某 (N,q) 格内 isolated 排序赢家家族 ≠ e2e 排序赢家家族 → `REVERSAL`
- **RCCL 内部配置反转**（2026-09-02 补充）：跨 candidate 边界表把 isolated 池 {comm_C0, comm_C2, fc, dc} 与 e2e 池 {r1_C0, r1_C2, r0, rs, d0, ds, d1} 放进同一 (N,q) 格：C2 isolated 赢但 e2e 输（符号相反且两侧都 ≥1%）→ `RCCL_CONFIG_REVERSAL`。smoke 首格 N2048/q8 已复现 Phase 2/3 的 STRONG_REVERSAL（iso +1.6% / e2e −5.4%）
- **基座能力向量**：comm/fc/dc 的 T_done 差距量化传输能力；r1-rs 与 d1-ds 的差距量化"重叠可挽回多少传输劣势"
- **净收益**：d1 vs ds > 0 且稳定，否则 D1 只作为测量载体不作为推荐策略
- **释放语义代价**（2026-09-02 补充）：控制表 `d1_vs_dc_done_stretch_pct` = d1 的 t_done 相对纯 dc 传输的拉伸率（协议轮询 + 干扰），smoke N2048/q8 = +103.4%；这是"分片释放的价格"首次在同一二进制内被分离测量

## 6. 冒烟发现（2026-09-02，N=2048 q=8 C0，p50 us，非正式数据）

| 路径 | e2e | done | 备注 |
|---|---|---|---|
| r1 | 5401 | 4681 | 与 Phase2/3 一致性 ✓（重叠收益 21% vs r0=6847） |
| rs | 7962 | 7516 | 串行对照 ✓ |
| dc | 9879 | 9879 | DUSHMEM 分片传输慢于 RCCL ~2.4× |
| fc | 11545 | 11545 | Phase A 尺寸翻转在 16MiB 复现：fcollect 已差于 put-signal |
| d0 | 14364 | 11574 | fcollect 整传整算 |
| ds | 21304 | 21303 | ⚠️ 见 §8：当时 DS 串行门控缺失，此数据实为 D1 行为 |
| d1 | 20072 | 20070 | **e2e≈done：GEMM 完全藏进通信阴影**；与"ds"差 5.8% 系噪声 |

初步解读：本机上 DUSHMEM 传输先天慢 2.4-2.8×，16MiB 处 d1 不可能赢 r1 —— 论文叙事**不是**"D 赢 R"，而是：
1. isolated 排序 ≠ e2e 排序在跨基座尺度同样成立（传输快的基座 ≠ 端到端快的策略族）；
2. 传输劣势可以被重叠部分挽回（e2e≈done 现象；净收益待 §8 修复后重测）；
3. 选择器必须同时看 release 语义、GEMM 分块效率与基座能力向量 B。

## 8. 已知缺陷与修复（2026-09-02 晚发现）

**缺陷**：DS 的两处 `if (serial...)` 块只有注释没有代码（producer 侧与 consumer 侧各一处），DS 与 D1 发出**完全相同的流操作序列**——DS 并非串行基线。formal 5 个格的 d1_vs_ds = −0.17%~+0.005% 反向证实了两路径零差异（smoke 的 5.8% 差为噪声）。

**修复**（`ag_gemm_phaseb.cpp` 与 NVIDIA 移植版同步改）：把 producer/consumer 两个循环合并为单 slice 循环，串行门 `hipStreamWaitEvent(comm_stream, gemm_end[i], 0)`（= RS 同款）放在本 slice GEMM 入队之后。**必须合并循环的原因**：`hipStreamWaitEvent` 在入队时刻快照事件最近一次 record，分离的双循环会把门指向上一迭代的旧事件。D1（serial=false）合并前后每流操作序列不变。

**影响面**：
- 作废：`d1_vs_ds_gain_pct`、`d0_vs_ds_gain_pct` 列；冒烟解读中"d1/ds 净收益 5.8%"
- 不受影响：其余 9 条路径（RS 的串行门在 550 行真实存在）、`d1_vs_d0`、`d1_vs_dc_done_stretch_pct`、全部 isolated/reversal 结论
- formal（`phaseb_formal_20260902_160115`）**不中断**：runner 调旧二进制，provenance 完整；其 ds case 保留作历史记录，分析时标记为无效

**补测计划**：formal 结束后 `make` 重编译 → 冒烟验证 ds>d1 且 ds≈串行和 → `bash run_phaseb_dsfix.sh`（跑 ds/d1/d1w × 全 (N,q) 矩阵 × 5 reps，**外加 N4096/q16 配置轴四件套 comm/r1×C0/C2**——若 formal 的 N4096/q8 不反转，Track B 只剩一个正例，深一档切片是嵌套 LOO 可学习性的保险；新时间戳根 `phaseb_dsfix_*`，与 formal 同参数）→ `merge_phaseb_dsfix.py --formal-root <formal> --dsfix-root <dsfix>` 出跨根合并视图（ds/d1/d1w 用修复后二进制数据，其余沿用 formal；含 d1w_vs_ds/d0/d1 与双 stretch 的显著性）。

## 9. 机理发现：等待放置（wait placement）决定 release 语义能否兑现（2026-09-02，F6 提取）

`extract_release_curves.py` 从 formal 部分数据发现的**新机理**（绝对时刻，p50，max-rank）：

| 格 | 路径 | 首个 release | 末个 release | 节奏 | GEMM/片 |
|---|---|---|---|---|---|
| N512/q2 | dc（等待在 wait_stream） | 5443 | 7446 | 2000（=传输） | - |
| N512/q2 | d1（等待在 compute_stream） | **7469** | 8127 | 656（=GEMM 579+ε） | 579 |
| N512/q8 | dc | 1373 | 11382 | 622 | - |
| N512/q8 | d1 | **14443**（比 dc 全传完还晚 3ms） | 17818 | 304 | 241 |
| N2048/q8 | d1 | 14555（dc 全传完=10896） | 19624 | 543 | 465 |

结论：**d1 的 consumer 等待放在承载数据的 compute_stream 上时，首个 release 被推迟到全部传输完成之后**，切片按 GEMM 节奏串行放行——d1 实际上"没有在重叠"，这从机理上解释了 d1≈ds 与 stretch 随 q 恶化。dc 用独立 wait_stream 放置，首个 release 能在 1373us 解析。**等待放置本身是释放语义的一个设计变量**（放进论文的语义轴叙事）。

对策：新增 `d1w` 路径（`D1W_WAITSTREAM_OVERLAP`，等待+release 放 wait_stream，compute_stream 用事件桥接入），随 dsfix 批次测量——若 GEMM0 能提前到 ~dc 首个 release 时刻，DUSHMEM 侧将首次出现真实重叠收益。release→GEMM 延迟：d1=5.4us、d0/r0=16.2us、r1=10-47us——语义档位梯子数据（F6）。

## 10. 选择器双轨评估的早期形态（2026-09-02 晚，部分数据 7/10 格，待 formal 完成后复核）

F5（selector_phaseb.py）拆成两个选择问题，理由是本基座上策略轴是退化的（C0 池内 r1 全胜）：

- **Track A（策略轴）**：P0 majority = 天花板；按"传输主导选 bulk"直觉写的规则（P2）反而 mean regret ~14%——带宽直觉用在策略轴会被惩罚，这本身是论文论据。
- **Track B（配置轴，判别性所在）**：r1 载体下选 C0 还是 C2。B0（谁 isolated 快选谁，带宽思维基线）与 B2（q≥8 且 iso_gap≤2% → C0）top-1 同为 6/7，但 worst regret **5.49% → 0.22%**：规则把灾难误判（N2048/q8 选 C2，损 5.5%）换成近中性误判（N512/q8 选 C0，损 0.2%）。机理解释：C2 的 isolated 优势随 q 衰减（2.9%→2.2%→1.6%），通道建立成本不再被摊薄，iso_gap 本身就是 q 的传感器。
- **跨基座预期**：策略轴的判别性应在 NVIDIA 侧显现（若 r1 不全胜，选择器真正做策略选择）；本基座的选择器价值主张落在配置轴 + 边界死区设计（M1 滞回思想的落地）。

## 11. 两个新发现（2026-09-02 深夜，formal 部分数据 7/10 格，待全矩阵复核）

**A. 策略轴反转：d1 在 q=8 连自家串行基线 d0 都打不过**（control 表 `d1_vs_d0_gain_pct`，C0 载体）：
- N512：q2 +30.3% → q4 +28.8% → **q8 −45.4%**
- N2048：q2 +26.5% → q4 +23.9% → **q8 −41.1%**
- 机理链闭合：wait placement 病理（§9）→ stretch 随 q 爆炸（N512 17→18→64%，N2048 41→44→93%）→ 分片重叠的传输完成被拖到比串行 fcollect 还慢。**"重叠策略"在深切片下退化为反模式**——这不只是"选错配置"（配置轴），是"选错策略"（策略轴），论文叙事从单一反转升级为**双轴反转**。
- 验证闭环留给 dsfix：d1w（wait_stream 放置）若在 q8 把 d1_vs_d0 拉回正值，即"机理→修复→验证"完整故事。

**B. 嵌套 LOO 的诚实性：单正例学不出边界**（`refit_b2_thresholds.py`，已挂入 finalize 第 5 步）：
- 手写规则 (q≥8, gap≤2%) 全样本 top1=6/7、worst=0.22%；但**嵌套 LOO**（留一后训练集重拟合阈值）top1=5/7、worst=5.49%——留一掉唯一反转格（N2048/q8）后训练集无正例，规则退化为"永不触发"，被留出格吃满 regret。
- 论文含义：B2 的价值主张**不是**"从本矩阵数据学出规则"，而是"机先验（iso_gap 随 q 衰减）+ 单点标定"，且必须如实如此措辞；always-C2 基线 mean 0.78%/worst 5.49% 就是退化 LOO 的水平。
- **formal 完成后第一检查项**：N4096/q4、q8 若也是反转（正例 ≥2），嵌套 LOO 才可能诚实学出边界，worst 显著下降——这决定 Track B 段落写"可学习"还是"先验+标定"。

## 12. 预注册：数据落地前的定量预测（2026-09-02 深夜写死，落地后对照）

写下时点：formal 501/590，N4096/q8 的 r1 尚未落地、q16 批未跑、dsfix 未编译。已有衰减链：N512 +5.2→+1.8→+0.2；N2048 +5.7→+1.7→−5.5；N4096 q2=+3.8。

| # | 预测 | 依据 | 判定 |
|---|---|---|---|
| P1 | N4096/q4 r1 e2e(C2 vs C0) ∈ [−2.5%, +0.5%] | N 越大衰减越快；N2048 同位置 +1.7，N4096 起点低 1.9pt | ✅ **HIT**（实测 +0.127，死平） |
| P2 | N4096/q8 ∈ [−9%, −3%]（若为负则 Track B 获第 2 个正例，嵌套 LOO 可学习） | 衰减斜率随 N 增大；外推 N2048 的 −5.5 | ❌ **MISS**（实测 **+1.768**，回弹！）→ 引出 §13 平衡律 |
| P3 | N2048/q16 e2e delta < −5.5%（更负，估 −8~−12%） | 单调衰减未见饱和 | ❌ **MISS**（实测 **−5.371**，反转持续但不加深——q8 阈值后饱和） |
| P4 | N2048/q16 stretch(d1 vs dc) > 90%（smoke 曾 103.4%） | stretch 随 q 单调涨 | ❌ **MISS**（实测 **+53.3**，见顶回落：dc 自身在 q16 恶化 +42% 比 d1 快） |
| P5 | dsfix 后 ds > d1（e2e，全部格，幅度≈GEMM 串行化代价） | DS 真串行化后按 GEMM 节奏放行 | 待验（冒烟预演：N2048/q8 仅 +1.6%——d1 因 wait placement 本已"意外串行"，真串行与之几乎同速；预期小正而非大差） |
| P6 | d1w 在 q8 的首个 release 时刻从 ~14.5ms 量级降到 ~dc 首释放量级（N512/q8≈1.25ms），d1w_vs_d0 回正 | wait placement 是 binding constraint 的假设 | 待验 |
| P7 | d1_vs_d0 @N4096/q8 符号不定（stretch 78-93% 但 d0 的 fcollect 也差 36%+，两个慢源对冲）——这是真空点不是预测 | N4096/q4 仍 +26.7 | ✅ **已观测：−19.4（d0 也赢）→ 结构轴反转在 q8 成全列**（N512 −45.4 / N2048 −41.1 / N4096 −19.4） |

P2 MISS 的处理：**这正是预注册的价值**——"N 越大衰减越快"的单调外推是错的，反转在 (N,q) 平面是**局部盆地**（见 §13），论文若按单调边界写会被 N4096/q8 直接打脸。~~Track B 仍只有 1 个正例~~ **q16 落地后 Track B 有 2 个正例（N2048/q8、q16），嵌套 LOO 仍 7/10**（q8 折的阈值被 q16 拉到 q≥16 → 留出格选错，worst 仍 5.49%）；N4096/q16 四件套（dsfix）成为 ratio≈0.59 固定下 q 独立效应的判别实验。

**记分牌元结论（P1-P7 全判定后）**：HIT 1（P1）/ MISS 3（P2 回弹、P3 饱和、P4 见顶）/ 观测 1（P7 全列负）/ 待验 3（P5/P6/DX 在 dsfix）。三个 MISS 全是"单调外推"方向——**反转与代价曲线都是有界/非单调的，任何单坐标单调模型都会错**。判定脚本：`check_prereg_phaseb.py --summary-dir <root>/summary [--dsfix-summary ...]`。

## 13. ★平衡律（2026-09-02 深夜，formal 九格全齐后发现的核心选择律）

**反转不是单调曲面，是"重叠区共振"**。q8 行（comm/gemm isolated 比值 vs e2e delta）：

| cell | ratio=comm/gemm | e2e Δ(C2 vs C0) |
|---|---|---|
| N512/q8 | 2.32（通信主导） | +0.22 |
| N2048/q8 | **1.12（平衡）** | **−5.49** |
| N4096/q8 | 0.59（计算主导） | +1.77 |

q4 对照行：ratio 1.27（N2048/q4）不反转（+1.69）→ **ratio≈1 不充分，还需 q≥8**。

机理（三段论）：
1. 通信主导（ratio≫1）：comm 是关键路径，C2 的 isolated 带宽优势传导到 e2e（N512/q8 已近耗尽 +0.22）；
2. **平衡（ratio≈1）：双资源同时饱和——C2 的多通道竞争从 GEMM 关键路径偷时间 → isolated 优势反转为 e2e 劣势**。反转只活在真重叠区，而这恰是重叠策略最重要的工作点；
3. 计算主导（ratio≪1）：GEMM 阴影淹没配置差异（+1.77，C2 的 iso 优势部分存活）。

**iso_gap 与 N 无关**（q8 三格：1.64/1.65/1.72）→ 单 gap 传感器原理上分不开这三个格；必须加 ratio（N 传感器）。两者都是 isolated 微基准可测量 → **B3 规则：`c0 if (q≥8 and 0.9≤ratio≤1.35 and gap≤2.0) else c2`**，九格全对（top1 9/9，worst regret ≈0）。

**q16 落地后的加固（2026-09-02 深夜）**：
- N2048/q16：ratio **1.229（仍在带内）**、gap 0.73%（全矩阵最薄）、delta **−5.371** → 带内出现第二负点，B3 预测正确（10/10，worst 0.00%）。
- **带内全列**：固定 N2048（ratio 1.30/1.27/1.12/1.23 ≈ 恒定），delta 随 q 走 **+5.65 → +1.69 → −5.49 → −5.37**——q 阈值在 8、**16 处饱和**（P3 MISS 的正面价值：反转深度由 ratio（双资源饱和度）决定，gap 变薄/加深切片不再加深反转）。
- gap 与 q 强相关（N2048：2.98/2.12/1.65/0.73 单调变薄）→ **gap=q 传感器、ratio=N 传感器**的分工说法成立。
- 诚实口径微调：正例 2 格（N2048/q8、q16），嵌套 LOO 仍 7/10/worst 5.49%（阈值漂移）——B3 依旧是机先验+标定，不改口径。
- 判别实验收敛为 **N4096/q16 四件套**（ratio 恒 0.59、q 加倍）：B3 预测 delta 仍正（q 无独立效应）；N2048/q16 已验（B3 ✓）。

论文落点：§5.2 从"单调衰减"改写为"平衡律"（F2b 衰减线保留但叙事反转点在 N 非单调）；C1 主张升级——**isolated 排名失效的模式本身可预测**（双特征平衡带），这是选择器必要性的最硬论据。

## 7. 文件

- `ag_gemm_phaseb.cpp` — 统一基准（10 路径）
- `Makefile` — 两阶段 hipcc（gfx928, rdc, dushmem+rccl+rocblas 同链）
- `run_phaseb.sh {smoke|formal}` — runner（时间戳结果根目录、防覆盖、platform/source snapshot）
- `analyze_phaseb.py` — case_summary / cell_matrix / selection_boundary / boundary_xcand（跨 candidate）/ control_table（含 d1-vs-dc stretch）/ analysis.md
- 结果：`results/k500sm_ai_gfx928_4gpu/phaseb_{mode}_{timestamp}/`

## 14. NVIDIA 侧预注册（P8-P13，2026-09-02 深夜封存——先于任何 4090 数据）

> 写死时点：4090 包已交付但 env_check_report.txt 未回传，零 NVIDIA 数据在手。
> 平台假设：RTX 4090×4 / sm_89 / PCIe（无 NVLink）/ NCCL + NVSHMEM。
> 判定脚本：check_prereg_phaseb.py 待扩展 N 项。同口径：e2e p50、MW U α=1e-4、5/5 方向一致。

| # | 预测 | 依据（机理，非拟合） | 判定 |
|---|---|---|---|
| P8 | **带移位**：同格 ratio_4090 < ratio_海光（4090 FP32 时钟/吞吐更高、PCIe 逐卡带宽相当→计算相对更快）；若 9 格内出现反转，反转格的 N **小于** 2048（预计 512 或无反转） | 平衡点=双资源饱和，资源速度比决定带的位置；带的 (N,q) 坐标是基座属性 | 待验 |
| P9 | **策略轴非退化**：4090 上至少一格 e2e 赢家 ≠ r1（d 族或 rs 在某格获胜） | r1 在海光全胜部分依赖 RCCL event 重叠的特定实现；NCCL/NVSHMEM 的释放语义不同，B 向量不同则边界不同 | 待验 |
| P10 | **wait placement 病理跨基座可移植**：4090 上 d1（wait 在 compute stream）在 q≥8 仍劣化，d1w（wait_stream+事件桥）优于 d1；首个 release 时刻 d1w ≈ dc 量级、d1 ≫ dc | 病根是流指派代码结构（consumer 等待占用计算流），不是 DUSHMEM 特有；NVSHMEM 的 on-stream wait 同样占流 | 待验 |
| P11 | **gap 更薄**：4090 上 C2 类（NCCL_ALGO/PROTO/NCHANNELS 调参）的 iso_gap 中位数 < 1.5%（NCCL 自动调参比 RCCL 强，可买空间小）→ B3 的 gap≤2 条件很少触发；若 9 格无反转，config 轴在 4090 判别性弱本身即结果 | 自动调参收敛→env 微调收益小；无 gap 即无可传导优势、也无干扰可藏 | 待验 |
| P12 | **B3 阈值直移**：海光标定的 (q≥8, 0.9≤ratio≤1.35, gap≤2) 直接用于 4090 传感器，top-1 ≥ 8/10（带外默认 C2 在带外仍对）；单格重标定后 10/10 | 规则结构=机先验（带外安全、带内保守），阈值=基座标定；这正是"能力向量+标定"主张 | 待验 |
| P13 | **能力向量 B' 可区分**：4090 的 fc/dc 尺寸翻转点与 release→GEMM 延迟梯子与海光可测量地不同（交叉点位置偏移） | 原语实现不同（NVSHMEM fcollect vs DUSHMEM fcollect 的聚合实现路径不同） | 待验 |

预注册纪律：NVIDIA 数据落地后只允许对照判定+写 MISS 分析，不允许改预测文本；预测本身若被证伪，按 P2/P3/P4 先例提炼机理再入论文。
