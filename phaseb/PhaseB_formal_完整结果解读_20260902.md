# Phase B formal 完整结果解读（2026-09-02 深夜版）

> 结果根：`results/k500sm_ai_gfx928_4gpu/phaseb_formal_20260902_160115/`
> 585 case 全 PASS（0 fail）。矩阵 = 10 路径 × 9 格 × 5 rep（C0）+ comm/r1 的 C2 × 9 格
> + N2048/q16 批 9 组（C0: comm/gemm/rs/r1/dc/ds/d1 + C2: comm/r1）。
> **注意**：本根的 ds 数据无效（DS 串行门缺失 = 与 D1 同序列），dsfix 根
> `phaseb_dsfix_20260902_184341/` 补测中；本解读凡涉 ds/d1w 均标注。
> 另：q16 批未跑 d0/r0/fc，control 表对应格为**空串**（缺数据，非 0）。

## 1. 一页结论

1. **资源轴反转 + 平衡律加固**：q16 点落在平衡带内且仍负（−5.37%），反转**持续但不加深**
   （q8 −5.49%）。带内全列（N2048，ratio≈1.2 恒定）：
   **+5.65 → +1.69 → −5.49 → −5.37**（q=2/4/8/16）——q 阈值在 8，16 处饱和。
2. **B3 平衡带选择器 10/10、worst regret 0.00%**（带宽基线 B0 8/10、worst 5.49%）。
   第二个正例（q16）出现后嵌套 LOO 仍只有 7/10——"机先验+标定"的诚实口径不变。
3. **结构轴反转成为全列**：d1_vs_d0 @q8 三个 N 全负（N512 −45.4 / N2048 −41.1 /
   **N4096 −19.4（P7 真空点解出，d0 也赢）**）。q2/q4 全正（+21~30%）。
4. **预注册记分牌：HIT 1 / MISS 3（P2/P3/P4）/ 观测 1（P7）/ 待验 3（P5/P6/DX）**。
   三个 MISS 全是同一类型：**单调外推失败**——效果是有界共振带，不是随 q/N 单调
   增长的曲面。预注册连续三次防住论文写错方向。
5. **stretch 非单调见顶回落**：N2048 stretch(d1 vs dc) 41→44→**93（q8 峰）**→53（q16）。
   原因不是 d1 变好，是分母 dc 在 q16 恶化 +42%（10.5→14.9ms）比 d1（20.3→22.9ms）
   更快。"语义代价随 q 无限增长"的叙事必须改为"协议代价与传输代价双涨、比值见顶"。

## 2. 资源轴：十格全表（C2 vs C0，r1 e2e，+ : C2 快）

| 格 | ratio | iso_gap | e2e delta | 赢家 | B3 预测 |
|---|---|---|---|---|---|
| N512/q2 | 3.520 | +2.88% | +5.17% | C2 | C2 ✓ |
| N512/q4 | 3.513 | +2.14% | +1.68% | C2 | C2 ✓ |
| N512/q8 | 2.325 | +1.64% | +0.22% | C2 | C2 ✓ |
| N2048/q2 | 1.298 | +2.98% | +5.65% | C2 | C2 ✓ |
| N2048/q4 | 1.266 | +2.12% | +1.69% | C2 | C2 ✓ |
| N2048/q8 | **1.123** | +1.65% | **−5.49%** | **C0** | **C0 ✓** |
| N2048/q16 | **1.229** | +0.73% | **−5.37%** | **C0** | **C0 ✓** |
| N4096/q2 | 0.644 | +3.15% | +4.10% | C2 | C2 ✓ |
| N4096/q4 | 0.669 | +2.13% | +0.13% | C2 | C2 ✓ |
| N4096/q8 | 0.594 | +1.72% | +1.77% | C2 | C2 ✓ |

- isolated 层 C2 十格全快（+0.73% ~ +3.15%，全部 SIG 5/5）——带宽思维十个信号全指向 C2。
- e2e 层两格翻负，都在带内（0.9≤ratio≤1.35）、都是深切片（q≥8）、gap 都薄（≤1.65%）。
- **iso_gap 仍与 N 无关但与 q 强相关**（N2048: 2.98/2.12/1.65/0.73 随 q 单调变薄）——
  gap 是 q 传感器，ratio 是 N 传感器，二者不可互相替代（B2 只用 gap 在 N4096/q8 误开 C0，
  regret 1.80%；B0 无条件 C2 在 N2048/q8 regret 5.49%）。
- q16 的 gap 0.73% 是全矩阵最薄：C2 的 isolated 优势随切片数继续衰减，但 e2e 反转深度
  不再增加——**"gap 变薄 ⇒ 反转加深"不成立**，反转深度由双资源饱和度（ratio）决定，
  gap 只决定"是否有优势可传导"。

## 3. 结构轴（q8 全列 + q16 侧写）

| 对比 | N512 | N2048 | N4096 |
|---|---|---|---|
| d1_vs_d0 @q2 | +30.3% | +26.5% | +21.4% |
| d1_vs_d0 @q4 | +28.8% | +23.9% | +26.7% |
| d1_vs_d0 @q8 | **−45.4%** | **−41.1%** | **−19.4%** |

- 三行三列全部 SIG 5/5（significance_phaseb.py）。q8 列全负 = **"重叠"策略在深切片下
  全面退化为反模式**，且与大 N 无关——wait placement 病理（d1 首释放被推迟到全部传输后）
  是根因，d1w 验证排队中（P6）。
- N4096/q8 的 −19.4 解答了 P7 真空点：d0 的 fcollect 惩罚（大消息下 fcollect 很差）
  依然小于 d1 的释放病理。
- q16 侧写（无 d0）：d1 e2e 23122us，比 rs（RCCL 逐片串行，8339us）慢 2.8×；
  r1 比 rs 快 +30.1%。DUSHMEM 分片路径在 q16 全面不可用，RCCL r1 一枝独秀。

## 4. 语义代价（stretch）与 q16 见顶

| N | q2 | q4 | q8 | q16 |
|---|---|---|---|---|
| N512 | +17.4% | +18.1% | +64.4% | — |
| N2048 | +41% | +44% | **+92.9%** | **+53.3%** |
| N4096 | — | +77.8% | **+97.3%** | — |

- N2048 见顶回落；N4096/q8 97.3% 继续走高但无 q16 点。
- 机理改述：q16 处 dc（纯传输）自身代价爆炸（+42%），d1 只 +13%——分片释放协议的
  每片开销与传输每片开销都在涨，**比值不是协议代价的单调函数**。论文 C2 主张措辞：
  "语义代价非线性、不可从 isolated 排名外推，且与传输代价纠缠"。
- release→GEMM 延迟梯子（F6）不变：d1 5.4us / r1 10-47us / d0 16.2us。

## 5. 选择器（Track A/B，全 10 格）

- **Track A（策略）**：P0 majority = P1 nearest = 10/10（r1 全胜，包括 q16）。
  带宽直觉规则 P2_feature 3/10、worst 33.7%——策略轴在本基座退化，判别性留 NVIDIA。
- **Track B（配置）**：
  - B0 isolated（=带宽思维）：8/10，worst **5.49%**
  - B1 majority：8/10，worst 5.49%
  - B2 (q≥8 ∧ gap≤2%)：8/10，worst 1.80%（gap 是 N 盲，N4096/q8 误开）
  - **B3 (q≥8 ∧ 0.9≤ratio≤1.35 ∧ gap≤2%)：10/10，worst 0.00%**
- **嵌套 LOO（诚实口径）**：7/10，worst 5.49%。q16 成为第二正例后，LOO N2048/q8 折
  拟合出 q≥16/gap≤1（阈值被 q16 拉走）→ 留出格选错。两个正例仍不够数据驱动学习；
  B3 = 机先验（双资源饱和才有干扰）+ 单点标定。**DX 判别实验（N4096/q16 四件套，
  dsfix 批内）将直接检验"q 无独立效应"假设。**

## 6. 预注册记分牌（check_prereg_phaseb.py）

```
P1   HIT      N4096/q4 delta ∈ [-2.5,+0.5]        observed +0.127
P2   MISS     N4096/q8 delta ∈ [-9,-3]            observed +1.768  → 平衡律
P3   MISS     N2048/q16 delta < -5.5 (est -8~-12) observed -5.371  → 饱和不加深
P4   MISS     N2048/q16 stretch > 90              observed +53.3   → 见顶回落
P5   PENDING  dsfix: ds 慢于 d1 全格
P6   PENDING  dsfix: d1w 首释放 ~dc 量级 + d1w_vs_d0@q8 回正
P7   OBSERVED N4096/q8 d1_vs_d0 无预测            observed -19.4   → d0 也赢
DX   PENDING  N4096/q16 delta > 0（B3 判别）
```

**元结论（论文可用）**：三次预注册 MISS 全是"单调外推"方向——衰减曲线外推（P2）、
深度外推（P3）、代价增长外推（P4）。反转是 (q, ratio, gap) 空间的**有界共振带**，
任何单坐标单调模型都会错。这正是"释放感知选择器"必要性的最硬论据。

## 7. dsfix 批（进行中）与两个新注意事项

1. **d1 的 A/B 异常（重要 provenance 事实）**：DS 修复合并了 producer/consumer 双循环
   （为支持 d1w），d1 的每流操作序列不变，但**宿主 enqueue 顺序**从"先全部 producer
   后全部 consumer"变为逐片交错。同机同参 A/B：旧二进制 d1@N2048/q8 = 21639us，
   新二进制 = 18761us（**−13.3%**，本机正常波动 <2%）。⇒ formal 的 d1 与 dsfix 的 d1
   **不可跨根直接比较**；dsfix 内部 ds/d1/d1w 同批同二进制配对不受影响（merge 脚本
   FIXED_PATHS 优先 dsfix 值）。这个 enqueue 顺序效应本身是可写的观察。
2. **ds 串行门已验证生效**：修复后 N2048/q8 冒烟 ds=19054 vs d1=18761（+1.6%），且
   ds 在 N512/q8 冒烟 ≈ d1（+0.1%）——与 wait placement 病理一致（d1 本身已"意外串行"，
   真串行 ds 与之几乎同速；P5 的"幅度≈GEMM 串行化代价"预计是小正值而非大差异）。
3. dsfix 根：`phaseb_dsfix_20260902_184341/`，内容 = ds/d1/d1w × 9 格 × 5 rep +
   N2048/q16 三件 + **N4096/q16 四件套（comm/r1 × C0/C2，DX 判别）**。完成后
   `finalize_phaseb.sh` 的 bonus 步自动 merge 出跨根视图。

## 8. 论文直接可用的三句话

1. *"The isolated winner (C2) is the e2e loser exactly where both resources are
   saturated and slicing is deep — a bounded resonance band in (q, comm/gemm,
   gap) space, not a monotone surface: three pre-registered monotone
   extrapolations all failed (P2/P3/P4), twice by rebounding."*
2. *"A three-sensor rule from isolated microbenchmarks alone (B3) picks the e2e
   winner in 10/10 cells with zero worst-case regret, where the
   bandwidth-naive rule pays 5.5% in the reversal cell; nested LOO with two
   positive cells still cannot learn the rule — it is a mechanism prior,
   honestly stated."*
3. *"The 'overlap' strategy D1 is an anti-pattern at q≥8 on every N (up to
   −45% vs its serial sibling D0): consumer waits placed on the compute stream
   defer the first release until all transfers complete — overlap exists only
   in the release structure, not in the stream assignment."*

## 9. 图表状态（summary/figures/，已含 q16）

F1_money / F2a_winner_map（10 格）/ F2b_decay（N2048 线 4 点）/ **F2c_balance（10 点，
带内双子星 q8+q16，刻度已修）** / F3_stretch（含 q16 回落点）/ F4_crossover（q8 全列）/
F5b_capability / F6_release。全部经程序化内省验证数据完整。
