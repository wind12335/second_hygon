# results 索引（海光 K500SM_AI/gfx928/4GPU）

**入仓策略**：正式批与大型根只入 `summary/` + `run_metadata.txt` + `platform/` + manifest；
逐迭代原始数据（`cases/`）体量大，不入 git——完整原始数据在本机
`~/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/` 同名时间戳根下。
早期小型探针根（2026-08-25 一批、1g_pilot 等）体积小，整根入仓。
**跨根比较必须核对 `run_metadata.txt` 的 `binary_sha256`**（见 README 数据纪律）。

## 阶段 B（Phase B，主战场）

| 根 | 内容 | case 数 | binary_sha256（前 8 位） |
| --- | --- | --- | --- |
| `phaseb_formal_20260902_160115/` | formal 主批：11 路径 × 9 格 × 5 rep + C2 对照 + q16 + DX 四件套 | 585，0 失败 | （早于 dsfix 修复） |
| `phaseb_dsfix_20260902_184341/` | DS bug 修复后补批：d1/ds/d1w | 170，0 失败 | `f17aae1d` |
| `phaseb_d0dc_20260902_195753/` | 同二进制 d0/dc 参照批（与 dsfix 哈希一致 → d 族终判有效） | 100，0 失败 | `f17aae1d` |
| `phaseb_q16fill_20260902_212009/` | P14 预注册补批：N4096/q16 × d 族五路径（与上同根哈希） | 25，0 失败 | `f17aae1d` |
| `phaseb_smoke_20260902_1553*` | 冒烟 ×2 | 小 | — |
| `d_family_same_binary_20260902.csv` | **同二进制 d 族终判表**（d0/dc 来自 d0dc 根，d1/ds/d1w 来自 dsfix 根；每 case 逐迭代 e2e 中位 → rep 中位） | — | f17aae1d ×2 |
| `d_family_same_binary_mw_20260902.csv` | 终判表的 MW 精确 p 值（C(10,5)=252 全排列，星标格） | — | — |
| `family_axis_dushmem_vs_rccl_20260902.csv` | **家族轴对比**（dc/d1/d0 干净值 vs r1/rs/r0@formal 根；N4096/q2、q4 出现 dc 反超 r1 的家族级反转；⚠r1 侧跨二进制，论文级使用前需 P15 同根补批） | — | 混根，见 `docs/D2_跨领域机制借鉴_边界定律版_20260902.md` §5 警示 |
| `unified_decomposition_fit_20260902.csv`(+`_meta.json`) | **统一分解拟合**：边界定律闭式 P≈−44.3+9.22q+(0.032−0.0154q)·cols（R²=0.907，符号 11/11；脚本 `phaseb/fit_unified_decomposition.py`）+ 分块损失/流水线份额分解 + 边界轨迹 N*(q8)=2581、N*(q16)=7700 | — | 控制量取 formal 根；G(4096,16) 插补已标注 |

## 阶段 1–3 与阶段 A（RCCL 释放时刻 → 释放曲线 → 边界；DUSHMEM 准入）

| 根 | 内容 |
| --- | --- |
| `phase1_release_20260831_09*` ×3 | 阶段 1 释放时刻基准（三次发射） |
| `phase2_release_curve_20260831_105000/` | 阶段 2 逐分片释放曲线（平衡律发现数据） |
| `phase3_boundary_20260831_112437/` | 阶段 3 重叠边界扫描（双轴反转发现数据） |
| `实验结果_20260831/` | 阶段 1 时期整理结果 |

## 早期平台探针（2026-08-25，RCCL/NCCL 早期摸底）

`20260825T*` ×7（NCCL/RCCL kernel 探针）、`1g_pilot`（对称堆 1G 试点）、
`manual_preflight`、`overlap_dushmem_rccl`（首次 DUSHMEM×RCCL 重叠）、`ll128_probe`（LL128 协议探针）。
结论记录见 `platform/L0-L1_平台事实与RCCL首轮结果.md` 与 `docs/RCCL实验教学与结果解读.md`。

## 2026-09-03 下午（CPU 侧三件，回 NVIDIA【9】/【13】）
- `a800_boundary_refit_20260903.csv` — A800 版边界律重拟合（`phaseb/fit_a800_boundary.py`）：P=3.29+1.13q+(0.0004−0.0008q)·cols，R²=0.78、幅度 LOO 4.7pt、locked 10rep 对照 Δ≤0.4pt；截距 −44.3→+3.3 变号、N*(q8)≈15.6k/N*(q16)≈26k（q8 惩罚凸收敛→线性外推保守）
- `selector_v03_k500_20260903/` — selector v0.3 K500SM_AI 复算（`phaseb/selector_phaseb_v03_k500.py`）：always-r1 9/9 regret 0%（vs A800 always-d0 5/9 基座翻转）；两项式 7/9 p95 12.0%（vs A800 2/9——特征充分性是基座变量）；probe-1iter 8/9 p95 12.51%（vs A800 0.08%——k=1 探针在慢基座有真实代价）
- `r0_quantiles_20260903.csv` — R_0 样本 P50+P95 代算（7 组重尾格）：mean/p50 最大 4.09×（S7/q8）、p95/p50 2–4.7×——B3 需分位数的定量实锤
- `probe_k3_k500_20260903.csv` — probe-k 次小值模拟（`phaseb/probe_k3_sim_k500.py`）：k=1 首值 8/9 p95 12.51% → k=3 次小值 **9/9 regret 0%**（唯一错格 (2048,2) r0/r1 近平局被首迭代噪声翻错，次小值即修复）——回答【13】v0.4"探 2-3 次取次小值"：够，且与 B3 median-of-means 方向一致

## 2026-09-03 晚（P15 终判，根 phaseb_p15_wmr1_20260903_163530，135/135 PASS）
- `p15_wm_curve_20260903.csv` + `p15_r1_family_20260903.csv` — P15 window_mult 曲线 + r1 家族轴 11/11（终判 `phaseb/P15_终判_20260903.md`）：D1 方向 HIT 4/4（主格 wm4/wm1=−2.75% 部分载体）、D2/D3 MISS——**credit 深度二阶载体、一阶=逐片协议税**（爆炸/副格对 wm1→4 钝感）；R2 HIT 反超带不扩 q16；11/11 同二进制配对（旧跨根家族轴 CSV 作废）；分层第 3 层 K500 阈值=cols 三区（≤128 全显 / 128-512 过渡 / ≥512 掩蔽）

## 2026-09-03 晚（P16 终判，根 phaseb_p16_extrap_20260903_171133，30/30 PASS）
- `p16_extrap_verdict_20260903.csv` — 边界律包络外三格（终判 `phaseb/P16_终判_20260903.md`）：**符号外推 3/3 全对**（E1 N8192/q8 −12.2pt d1 赢 / E2 N8192/q16 +4.6pt d0 微赢 HIT 带内 / E3 N4096/q32 +122.4pt q 爆炸延续）；E1/E4 幅度 MISS（残差 51.7pt）→ **cols 项凸饱和**（与 A800 q8 斜率凸收敛两侧同构）；E2 双基座同格对表定版：K500 +4.6（贴界 N*(q16)=7700）vs A800 +16.5（深 d0 区 N*=26221）
