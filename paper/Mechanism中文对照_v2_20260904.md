# §Mechanism 机理节 — 中文对照 v2（2026-09-04）

> **用途**：英文骨架（`Mechanism英文骨架_v2_20260904.md`）的中文对照，供组内审读；内容逐段对应，不是逐句直译。锁定术语首次出现时给英文原词，后文直接用中文。
> **v2 = v1 + 对方【19】【20】内审采纳**（与英文骨架 v2 同步）：①A800 trace 数字全部换为对方配位稿 v2.1 的 device-0 口径并标 provisional（corrected table 前只作描述层）；②H3 A800 行降 proxy/pending；③H8 下界升 P21、H5 换 P24 终判（w\*=4）；④D3 改名 rank 缩放分化、D1 共同决定、D2 局部仿射边际项、M.6 删"必须在线探"直推。
> **证据分级**：direct（直接测量）/ derived（推导）/ proxy（代理）/ pending（暂缺，显式标注不硬凑）。

---

## M.0 术语（合并稿锁定）

五阶段链：

**发送方受理（S1 sender accepted）→ 对端可见（S2 remote visible）→ 通知满足（S3 notification satisfied）→ 消费合法释放 R_i（S4 consumer-legal release）→ 实际消费（S5 actual consume）**

- **消费合法释放 R_i**：消费者被允许消费分片 i 的最早时刻；由流依赖、credit 门控、等待位置（wait placement）共同决定。
- 本节三条规律（substrate-conditioned mechanism regularities，受基座制约的机制规律，**不称定律**）：**D1 释放语义（release semantics）· D2 协议税（protocol tax）· D3 rank 缩放分化（rank-scaling divergence）**。
- 因果动词纪律：trace 单独只配"显示/与…一致"（shows / is consistent with）；"造成/解释"（causes / explains）只用于 API contract + 干预实验 + e2e 多源闭环。
- 路径代号沿用基准：**d0**（fcollect 串行）、**d1**（逐片 put–signal 重叠）、**d1w**（d1 把等待挪到独立流）、**r0 / r1**（RCCL 串行 / 事件重叠）。
- 深度以"一轮 = q 个分片"计量：A800 轴 **w** 是在途分片上限，海光轴 **wm** 是槽位跨轮复用深度。两轴并排报告，**永不合并**。
- P(X vs Y) = (t_X − t_Y)/t_Y，正 = X 更慢。
- 基座：**K500SM_AI**（4 卡）、**BW1000**（8 卡）、**A800**（4 卡）。

## M.1 五阶段链：带宽模型看不见的那几步

带宽模型对 all-gather–GEMM 一段只装得下两个事件：集合通信把数据搬完，GEMM 把数据吃掉。中间至少还隔着三步，模型对它们没有表示。我们把这五步显式列出：

- **S1 发送方受理**：通信 API 接受分片 i 的写请求（putmem_signal_on_stream 入队 / NCCL group 入队）。
- **S2 对端可见**：分片 i 在对端对称堆可读。
- **S3 通知满足**：信号谓词成立——signal_wait_until 返回，或 NCCL 事件置位。
- **S4 消费合法释放 R_i**：消费者可以消费分片 i 的最早时刻。
- **S5 实际消费**：该片的 GEMM 开始执行。

S1→S2 由 X-Stage 建立 [引文]。本节承担 S3→S5——它们决定一次"已完成的传输"到底能不能变成实际到手的重叠。以下每条主张只由持有直接证据的一方作出；Table M1 的每个锚点行都带证据等级、预注册编号和结果文件。

证据归拢成三条规律：释放语义（D1，§M.2）、协议税（D2，§M.3）、rank 缩放分化（D3，§M.4）。它们主张的是符号与结构；幅度是基座各自的。这就是论文元结论在机理层的形态——**规律形式可迁移，参数必须逐基座标定**。本节由此推出的只是"参数不能跨基座搬"；"应该在线探而不是离线标"这更远的一步由 selector 章论证 [引 P25]，§M.6 只回指、不重复。

## M.2 D1 释放语义：本地完成不等于可以消费

**完成 ≠ 可消费（JM-H1，direct）**。K500SM_AI 的 DUSHMEM 准入实验（14/14 case，全量 payload 校验）里，消费侧 release 落后通信流完成 18–23µs（put–signal）/ 10–15µs（fcollect）。on-stream 原语保证的是流序，不是对端可见。本地流报告"集合通信做完"的那一刻，不是可以开始消费的那一刻。

**等待位置决定 S3→S4（JM-H2；语义 direct，效应来自配对 e2e）**。消费者的等待放在哪条流上，决定 R_i 最早能早到什么程度。d1 把等待放在计算流上，首个 release 被推迟到全部传输完成之后——重叠路径实际串行化。把等待挪到独立流加事件桥（d1w）恢复了释放位置，但事件桥要花钱：K500SM_AI 与 A800 上 d1w–d1 差都在 ±3% 内（预注册 P6 与 P10 在两个基座各自封存，双双 MISS）。BW1000 八卡下同格（N2048/q8）冲出了这个带：d1w 比 d1 慢 11.05%（五次运行中位 17421.7 vs 15688.2µs；四卡时慢 4.1%）。等待位置是必要条件，代价随 rank 数增长显形，不是免费修理。

> **[NSYS-SLOT-1 — NVIDIA 段；v2.1 数字，provisional]** 04 号 trace（d1，N4096/q8）：barrier_on_stream 把"消费流被阻塞"变成 kernel 可见——**每 rank 34 次**（四卡合计口径，每格一张代表性封存 trace）、p50 7.8–15.3µs、逐 rank max 0.34/9.0/8.2/13.1ms——重尾且 rank 不对称；与 wait 阻塞的 API/源码 contract 一致，trace 单独不主张证明 legal release。加 d1 首个 release 被推迟的可视化。

**S4→S5 语义梯子（ladder；JM-H3；K500SM_AI 为 derived，A800 行 proxy/待重提取）**。K500SM_AI 的逐分片 release 曲线把 release→GEMM 延迟分成族级档位（rung）：d1 5.4µs、r1 10–47µs、d0/r0 16.2µs。A800 的 kernel 时间戳行**停在 proxy**：初版提取（两档 direct）被对方内审撤回——配对语义未证（无法确认哪个 AG kernel 对应哪片 GEMM；稳态阈值系事后设定）、v1 数字跨 device 污染。device-0 审计数（d0 p50 14.2µs/15 对；r1 快端 p10 16.8/14.0µs）只是描述性审计数、不是档位值；时间线只支持 regime 级结构——粗粒度串行与细粒度流水两种形态都存在。d1 档在 device-0 只有 1 个稳态对，提不出来——保留 derived 的 5.4µs，不硬凑。主张的是 K500SM_AI 上的档位**序**——由各族同步原语的语义决定；幅度是标定问题，不主张跨基座同量级。

> **[NSYS-SLOT-2 — 可选，NVIDIA 自定位置；v2.1 数字，provisional]** r1 的 p50 514.7µs 是流水占位（pipeline occupancy）不是档位值；若使用，归入 §M.3 串行化证据。

## M.3 D2 协议税：切 q 片就是 q 次握手的协议工作量

**对 q 一阶、与深度无关（JM-H4）**。把一次集合通信切成 q 片，等于用 q 次握手换掉一次同步——这是**逐片协议工作总量（aggregate per-slice protocol work）**，不是每片一个常数税。K500SM_AI 的两个结果把税钉死为对 q 一阶。其一，槽位复用深度 wm∈{1,2,4} 轮扫过去，四个锚点格没有一个动超过 3.6pt；预注册救回预测最大的爆炸格（N2048/q16）在 K500SM_AI 纹丝不动（+0.0pt），BW1000 上 +2.9pt 在 3-rep 噪声带内（这是两基座唯一的判决分歧，锚点表有记录）；唯一有响应的是过渡格（主格），也只 −2.7pt / −3.6pt。预注册的救回排序失败。其二，RCCL allgather 控制量**局部仿射拟合**为 3938 + 38.1·q µs、与 N 无关——已测 q 域内边际项 ≈38.1µs/片；拟合支持的是"实测域内每片边际"，不是普适常数。**深度不是 d 族惩罚的一阶载体**。

A800 上 r1 族的 kernel trace 在 kernel 粒度上给出同一条规律：

> **[NSYS-SLOT-3 — NVIDIA 段；v2.1 数字，provisional]** 02/03 号 trace 同构输赢对照（每卡 121 片 GEMM + 129 个 AG_LL，每格一张代表性封存 trace）：N2048/q8 每片 AG p50 108µs（尖刺至 1,211µs）；AG 结束到下一片 GEMM 的 p50 514.7µs，恰好等于每片 GEMM 时长（495µs）——实质交替，并发覆盖（concurrent coverage）仅 10.7%。N512/q8 同结构覆盖 37.5%（3.5×）。**税压过每片计算，通信与计算就交替；压不过，才真重叠。**

**深度是阈值变量（JM-H5）**。深度只在"在途不足一轮"时起作用。A800 的 w 轴给出害区（harm zone），口径换成 **P24 e2e 终判（93/93，随机区组）**：在途压到 w1 时，B2/B1 的重叠收益被压到与串行基线 ±0.2% 以内；w4 恢复到 22.9–32.2%——**w\*=4 是当前 shape/基座的 operational threshold（工作阈值），不是普适常数**（旧口径"w1/w2 慢 30–40%"作废）。半轮处（q16 的 w8，16 片中 8 片在途）已见恢复；无界 w0 是各曲线回归的平台线。

> **[NSYS-SLOT-4 — NVIDIA 段；v2.1 数字，provisional]** kernel 级三连（00 / 01-w1 / 08-w8，**每卡 496×496**，每格一张代表性封存 trace）：并发覆盖 18.74%（w0）→ **0.00%（w1——并发完全归零，严格锁步）** → 18.65%（w8，0.1pt 内回到平台），而每次 AG p50 三者持平 ~45–47µs——**深度改变流水结构，不改变单次成本**。

海光的 wm 轴补上"一轮及以上平直"段（1/2/4 轮；主格 K500SM_AI −2.7pt、BW1000-np8 −3.6pt，K500SM_AI 其余格 ≤1pt），并且格序精确复现：boom > sub > main > win 从 K500SM_AI 到 BW1000 完全守恒，绝对税级平移 ×2.6。**结构可迁移，标定不可**。两轴语义不同——在途上限 vs 槽位复用——并排报告。

**可证伪出口**。税住在 signal 次数、流切换还是远端写里，可以测：grouped-signaling 变体把逐片 signal 聚合摊销，同时插桩计量每片 signal 次数与字节。税变薄→税随 signal 次数；不变→随另外两项。摊销想法属于 Perseus [引文]；我们把这个干预（DUSHMEM 准入分支）连插桩一起预注册后再发射，两种结果都报告。

## M.4 D3 rank 缩放分化：两条缩放律推着边界走

**两条缩放律（JM-H6，direct）**。两个路径族付 rank 数的方式不同。d0 走一次 fcollect，地板随 np 涨：4→8 卡 3.35→7.09ms（2.12×）。d1 每片推给 np−1 个对端，成本随 q·(np−1) 涨——但 rank 翻倍时的倍率**随 workload 而异**：N512/q8 ×5.16（超线性崩塌）、N2048/q8 ×1.40、N4096/q8 ×1.55（次线性）。两条成本曲线的交叉点随加卡移动，且**非单向**：小格把交叉点推向更大 N，大格拉回来。在 (N4096, q8) 上符号直接翻转：np4 时 d1 比 d0 慢 14.2%，np8 时快 16.1%——两次运行同一二进制。跨 np 绝对比值的保留意见：GEMM_ONLY 控制量在同 (N,q) 下 np4→np8 也 ~2× 膨胀、无遥测解释——批内配对可信，跨 np 绝对倍率含系统状态风险；边界随 np 继续移动的方向不作预测（np 只有两水平）。

**一个机制、两种投影（JM-H8）**。分化投影到 N8192/q16 的边界谱系上——**三个实测同格锚点 + 一个缺格**：K500SM_AI +4.6（贴界，其实证过界点在 N≈8192）；BW1000-np8 +26.2（同号放大；np4 时 q 轴交叉点就在已测网格内——BW1000-np4 的 N8192 格即缺格）；A800 +16.5（**早期 matched/formal 证据、非 P19/P21 随机区组 cell，图注须注明 provenance**），随机区组 bracketing 经 P21（45/45）扩至 N=32768（q8）/N=65536（q16）仍无过零——因此下界 **N\*(q8)>32768、N\*(q16)>65536**（右删失，幅度非单调收敛）[引 P21]。**域内有过零的基座与没有的基座，是同一套成本几何的两种投影，不是两种现象**。相应地，我们逐基座拟合，不做全局闭式。

*诚实声明*。A800 是 np4 单点；D3 的直接证据是 BW1000 的 np4→np8 翻号，A800 贡献的是远侧投影不是缩放数据。（A800 卡数迁移复现系列——163 次启动、8/8 串行胜、regret ≤0.078%——强化的是该行稳定性，不是 np 证据。）

## M.5 我们不主张什么

S2→S3（对端可见→通知满足）：双方都没有直接 trace 证据。现有的是代理链：on-stream 原语的 API contract、A800 准入 47µs 地板（derived）、§M.2 的完成→release 间隙（测的是整段跨度，不是中间内部）。S2→S3 标 pending，作为本文接在 X-Stage 之后留下的开放问题。另两条边界：档位值是族级中位不是逐格常数；w 与 wm 两轴永不合成一条曲线。

## M.6 为什么这逼出了"探针"（probe）

三条规律给出的是"任何纯带宽模型的能耐上限"。(i) 孤立计时看不见 S3→S5：释放语义、门控、联合争用只存在于依赖路径内部——BW1000 上孤立赢家在每一格都是 COMM_ONLY、领先 70–90%，而端到端差距只有 2–38%。(ii) 结构可迁移、标定不可：三个基座三个最优零知识默认（zero-knowledge default：A800→always-d0；K500SM_AI→always-r1；BW1000→无，最好也要 9–22% p95 regret）。由此推出的是**参数不能跨基座搬**——不多推：单基座内离线标定甚至可以短路（K500SM_AI always-r1 9/9）；"该在线探还是离线标"由 selector 章的 hidden residual 与探针价值论证裁决 [引 P25]，此处不重复。(iii) 边界随 rank 数移动，再好的标定也会随部署形态变旧。通用解不是更好的先验，而是便宜的鲁棒探针：k≥3 时，三基座 p95 regret 全部压在 0.00–0.06%。

---

## Table M1（锚点注册表压缩版，照 JM 锚点小节 v2 排版）

| 锚点 | 阶段 | 等级 | 一行证据 |
|---|---|---|---|
| JM-H1 | S1→S3（跨度） | direct | release 落后 comm-complete +18–23µs（put–signal）/ +10–15µs（fcollect），14/14 |
| JM-H2 | S3→S4 | direct+derived | d1 等待在计算流→串行化；d1w ±3%（K500SM_AI、A800），BW1000-np8 +11.05% |
| JM-H3 | S4→S5 | derived（A800 proxy/待重提取） | 档位 d1 5.4 / r1 10–47 / d0,r0 16.2µs（K500SM_AI derived）；A800 只有 device-0 审计数，regime 级结构 |
| JM-H4 | S3→S4 成本 | direct | wm 钝感；allgather 局部仿射 3938+38.1·q µs（已测 q 域）；A800 覆盖 10.7% vs 37.5%（provisional） |
| JM-H5 | credit 门控 | direct | 害区<1 轮、≥1 轮平直；P24 w\*=4 operational threshold；格序守恒、税级 ×2.6 |
| JM-H6 | 传输结构 | direct | d0∝np vs d1∝q·(np−1)、倍率随 workload；(4096,q8) 随 np 翻号；交叉点双向移动 |
| JM-H7 | S5 | proxy（可升 direct） | release_slices 与 kernel 时间戳配对 |
| JM-H8 | e2e 投影 | direct | N8192/q16：三锚点 +4.6 / +16.5 / +26.2（+BW1000-np4 缺格）；P21 下界 N\*(q8)>32768、N\*(q16)>65536 |

## 图位

- **F-JM1**（海光）：wm 平直曲线，K500SM_AI vs BW1000 四格归一。
- **F-JM2**（海光）：两条缩放律，np4→np8，(4096,q8) 翻号标注。
- **F-JM3**（NVIDIA）：梯子 + 首个 release 推迟 + barrier 尖刺——**暂缓，等 corrected table**（per-device/迭代感知重提取）后再画，只用 paper 级数字。
- **F-JM4**（海光出图，A800 覆盖域句 NVIDIA 审校）：谱系主格 N8192/q16 三个实测柱 + BW1000-np4 缺格标注 + P21 下界箭头（至 65536/q16、32768/q8）；A800 柱图注注明 provenance（早期 matched/formal 证据、非 P21 随机区组 cell）。

## 槽位清单（合并稿用）

NSYS-SLOT-1（M.2，04 号 barrier）· NSYS-SLOT-2（M.2，可选 r1 澄清）· NSYS-SLOT-3（M.3，输赢对照）· NSYS-SLOT-4（M.3，w 三连）。槽内数字全部引自封存 trace 重提取 schema v2（second_nvidia `docs/nsys锚点提取_20260904/anchor_extraction_summary.json`；device-0、每卡审计数），**corrected table（NVTX/迭代对齐）发布前一律 provisional**——正文只到描述层，µs 数字只进脚注。
