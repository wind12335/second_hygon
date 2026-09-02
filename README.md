# second_hygon — 第二篇论文 海光(K500SM_AI / gfx928) 侧实验套件

异构 GPU 通信基座上的**依赖释放感知 Collective-GEMM 重叠**（H1：孤立带宽最优 ≠ 端到端最优）。
本仓库是海光分区（**K500SM_AI / gfx928 / 4 GPUs / PCIe**，DTK 26.04，RCCL 2.22.3，DUSHMEM 3.2.5）上跑的完整实验线。
NVIDIA(RTX 4090) 侧对照仓库：[second_nvidia](https://github.com/wind12335/second_nvidia)，CSV/CLI 口径与本仓库一致，可合并分析。

## 目录

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `phaseb/` | 阶段 B 主基准：10+1 路径统一 Collective-GEMM（`comm/gemm/r0/rs/r1/fc/dc/d0/ds/d1/d1w` × 候选 `C0/C2` × `N×q` 格子）。含三轮：formal（585 case）→ dsfix（DS bug 修复，170 case）→ d0dc（同二进制 d0/dc 参照，100 case），全部零失败；附预注册判定器（P1–P7+DX）、B3 选择器、显著性配对、八图脚本 | 主战场，三轮完成 |
| `phaseA-dushmem-admission/` | DUSHMEM 消费语义准入微基准（epoch signal + credit 槽位 + 全 payload 校验） | 完成 |
| `phase1-ag-gemm/` | 阶段 1：RCCL AG-GEMM 释放时刻基准（发现"释放时刻≠完成时刻"） | 完成 |
| `phase2-release-curve/` | 阶段 2：逐分片释放曲线（平衡律 R_i/G_i 的发现地） | 完成 |
| `phase3-boundary/` | 阶段 3：重叠边界扫描（双轴反转的发现地） | 完成 |
| `nvidia-port/` | 交付 4090 侧的移植源（v2 含 d1w）+ 两份往来汇报文档 | v2 已交付 |
| `bw1000-port/` | 8 卡 bw1000 手动执行包（体检→编译→冒烟→formal np=4/np=8→打包回传，按步骤复制粘贴手册） | 待上机 |
| `NVIDIA与海光的交流窗/` | **跨平台异步交流通道**：`海光的进展.md` 每批工作追加一条（时间戳精确到秒+更新内容+想问 NVIDIA 的问题），最新在最上；NVIDIA 侧读此文件了解海光动向 | 持续更新 |
| `platform/` | 平台事实快照（L0-L1 平台事实与 RCCL 首轮结果） | 完整 |
| `results/` | 汇总级结果：各时间戳根的 `summary/`、run_metadata、manifest、预注册记分牌、同二进制 d 族终判表。**逐迭代原始数据（`cases/`）不入 git** | 持续更新 |
| `docs/` | 实验设计、结果解读、论文初稿、组会汇报、零基础讲解、文献调研 | 持续更新 |

## 快速复现（阶段 B 四步闸门）

```bash
cd phaseb
make                                    # 两阶段 hipcc + -fgpu-rdc，默认 ARCH=gfx928
bash run_phaseb.sh smoke                # 冒烟（N2048/q8 全路径 ×1 rep）
bash run_phaseb.sh formal               # 正式批（11 路径 × 9 格 × 5 rep + C2 配置轴对照 + q16 边界 + DX 四件套）
python3 analyze_phaseb.py --result-root <时间戳根>          # cell matrix / 控制表
python3 significance_phaseb.py --result-root <时间戳根>     # Mann-Whitney 配对
python3 check_prereg_phaseb.py ...      # 预注册记分牌（P1–P7+DX）
python3 selector_phaseb.py ...          # B3 选择器 + LOO 评估
python3 plot_phaseb_figures.py ...      # F1/F4/F6/B 向量八图
```

结果落独立时间戳根（`results/k500sm_ai_gfx928_4gpu/phaseb_<轮>_<STAMP>/`），绝不覆盖。

## 平台事实速览（2026-09-02）

- **K500SM_AI / gfx928 / 4 GPUs / PCIe**（严禁写成 K100AI/gfx936——那是另一台旧机）。
- DTK 26.04（`/opt/dtk` 是软链，find 需 `-L`）；RCCL 2.22.3；DUSHMEM 3.2.5（`libdushmem_device` 为静态库 `.a`）。
- `rocm-smi` 的 GPU 行前缀是 `HCU[` 不是 `GPU[`；设备自分配读 `OMPI_COMM_WORLD_LOCAL_RANK`。
- C2 候选 = `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN/MAX_NCHANNELS=8`；launcher 统一 `mpirun --allow-run-as-root -np 4 -mca coll ^hcoll`。
- 详见 `platform/` 与各结果根 `platform/platform_facts.txt`。

## 核心发现速览（2026-09-02，详见 docs/）

1. **双轴反转**：结构轴（q）与配置轴（C0/C2）上都存在"孤立带宽最优 ≠ 端到端最优"的反转格。
2. **q8 反模式有 N 边界（同二进制终判，d1_vs_d0 时间差，d0/dc×d1/ds/d1w 两根同为 `f17aae1d`）**：N512 **+40.0%** / N2048 **+29.5%**（d1 更慢，均 p=0.004）→ **N4096 反转为 d1 快 21.0%**（反向 p=0.004）。机理：每片 GEMM 列数 N/q=64/256/512，切片计算变大后协议串行化被计算掩盖，边界落在 N2048–N4096 之间。N2048/q16（128 列 ×16 同步点）惩罚爆炸至 **+69.9%**；q2/q4 则 d1 快 24–30%。
3. **DS bug 修复本身在 N4096/q8 造成 40 点符号翻转**（d1：21929→14487 µs；d0 跨根稳定 ±0.1%）——旧二进制（formal 轮）的 d 族数字不可用于终判；d1w 修复无效（全 10 格 d1w≈d1≈ds，|差|≤3.5%）→ 病理不是 wait placement，是**协议族结构性串行化**（P6 MISS，机理升级）。
4. **B3 选择器**：`q≥8 ∧ 0.9≤ratio≤1.35 ∧ gap≤2% → C0`，10/10 无漏检（worst regret 0.66%，DX 格暴露 ratio 窗口需扩）。

## 数据纪律

- 每个 run 落独立时间戳目录，绝不覆盖；失败 case 原样保留。
- **跨结果根比较必须同二进制**：`run_metadata.txt` 里的 `binary_sha256` 是对账依据（dsfix 与 d0dc 两轮同为 `f17aae1d…`，d 族终判表才有效；formal 轮跨根比较有 enqueue 顺序偏移 3–34%，不可直接引用）。
- 逐迭代原始 CSV（`cases/`）体积大，不入 git——留在本机 `results/` 树与时间戳 tar 包；git 内保留 summary、记分牌、manifest 与平台事实。
- 新文档、新脚本、新发现**当天入仓**；跨天的工作在文件名上标日期（`*_YYYYMMDD.md`）。

## 关联

- **NVIDIA 侧读这里了解海光动向：[`NVIDIA与海光的交流窗/海光的进展.md`](NVIDIA与海光的交流窗/海光的进展.md)**（倒序，最新在最上）
- NVIDIA 侧仓库与预注册预测（P8–P13）：[second_nvidia](https://github.com/wind12335/second_nvidia)
- 发往 4090 侧的执行说明：`nvidia-port/NVIDIA侧汇报_海光发现与执行说明_20260902.md`
- 论文骨架与创新点主张：`docs/论文骨架/`
