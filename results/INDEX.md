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
