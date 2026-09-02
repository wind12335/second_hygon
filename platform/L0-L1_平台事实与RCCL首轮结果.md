# L0 平台事实表 + L1 RCCL 首轮结果（2026-08-12 实测）

文档性质：第二篇论文首轮 profiling 的原始记录（实验前分析见 `../当前研究方向与首轮Profiling分析.md`）。

## L0 平台事实表（本次实测确认）

| 项目 | 值 | 来源 |
|---|---|---|
| 卡片型号 | 海光 K500SM_AI ×2（C-3000 设计） | `rocm-smi --showproductname` |
| GCN 架构 | **gfx928**（注意：记忆/旧记录为 gfx936，本机实测 gfx928，编译必须用 `--offload-arch=gfx928`） | `rocminfo` |
| 显存 | 每卡 63GB | HIP 属性 |
| 卡间互联 | **PCIe（Link Type=PCIE，非 XGMI/HYLink）** | `rocm-smi --showtopotype` |
| DTK | 26.04（dcc 25.10.0, clang 17.0.0） | `hipcc --version` |
| HIP/ROCm 版本 | 6.3.26102 / 6.3.3.0 | RCCL 日志 |
| RCCL | **2.22.3-HEAD:19c80f7**（`/opt/dtk/lib/librccl.so.1.0`） | RCCL 日志 |
| DUSHMEM | host 3.2.5（`libdushmem_host.so.3.2.5`）+ device `.a` | `ls /opt/dtk/dushmem/lib` |
| MPI | OpenMPI 5.0.3（`/opt/mpi`） | `mpirun --version` |
| 启停 | `mpirun --allow-run-as-root -np 2 -mca coll ^hcoll` | 实测成功 |

### RCCL 初始化关键警告（影响性能/正确性）
1. `NUMA auto balancing enabled` → `sudo sysctl kernel.numa_balancing=0`（否则性能波动）
2. `Missing "iommu=pt" from kernel command line` → 需内核启动参数（系统级，需管理员）
3. `Missing "HSA_FORCE_FINE_GRAIN_PCIE=1"` → **实测必需**：LL 协议在 PCIe 上无此变量会**挂起**（首次跑 2×99% CPU 空转）。设 `HSA_FORCE_FINE_GRAIN_PCIE=1` 后正常。
4. `No IB NIC found` → 本机无 IB/ROCE，RCCL 只能走 PCIe P2P（与记忆一致：PCIe P2P DMA 可用的前提下）。

### 对第二篇 L0 层的含义
- 本机两卡是 **PCIe 互联 + gfx928**。这与论文需要 NVIDIA 对照（A100-40GB-PCIE）在"PCIe 互联"维度是**对称**的，对照干净。
- gfx928 与记忆中的 gfx936 不符：**所有旧编译产物（dushmem_test 下 gfx936 二进制）在本机很可能不可用，必须重编 gfx928**。已有旧数据若要复现需重跑。

## L1 RCCL 首轮结果

### 测试方法
- `rccl_smoke.cpp`：双卡 AllGather，标准 `ncclAllGather` + 细粒度 `duNcclAllgather`。
- 扫描：消息大小 4KB/64KB/1MB/4MB/16MB/64MB/128MB(per rank)，proto ∈ {LL, Simple}，algo ∈ {Ring, Tree}，nChannels ∈ {0(默认), 2, 4, 8}。
- 计时：`MPI_Wtime` 墙钟，warmup=10，reps=50，每次 `hipStreamSynchronize` 后取平均。
- 正确性：recv 逐元素校验（第 r 段应为 rank r 发送值），全部 OK。
- 环境：`HSA_FORCE_FINE_GRAIN_PCIE=1`，`NCCL_DEBUG=WARN`。

### 完整数据（GB/s，per-rank 消息大小 → 总数据量=2×per-rank）

干净跑（`NCCL_DEBUG=WARN`，warmup=10 reps=50，102 个配置点全部 OK）。原始数据 `/tmp/rccl_clean.log`，关键点摘要：

| per-rank | proto | algo | nch | ms | GB/s |
|---|---|---|---|---|---|
| 4KB | (std default) | - | - | 0.091 | 85 |
| 4KB | LL | Ring | - | 0.079 | 99 |
| 4KB | Simple | Ring | - | 0.089 | 87 |
| 1MB | Simple | Ring | - | 0.115 | 1087 |
| 2MB | LL | Ring | - | 0.872 | 2295 |
| 2MB | Simple | Ring | - | 0.210 | 9529 |
| 8MB | LL | Ring | - | 4.861 | 1646 |
| 8MB | Simple | Ring | - | 0.551 | 14523 |
| 32MB | LL | Ring | - | 12.078 | 2649 |
| 32MB | Simple | Ring | - | 1.820 | 17583 |
| 128MB | LL | Ring | - | 48.163 | 2658 |
| 128MB | Simple | Ring | - | 6.613 | 19357 |
| 128MB | Simple | Tree | - | 6.573 | 19472 |

### 已确认的关键现象（首轮、去重后）

**P1. Simple vs LL 的带宽差距随消息大小急剧放大**
| per-rank | LL Ring(默认nch) | Simple Ring(默认nch) | 比值 |
|---|---|---|---|
| 4KB | ~99 GB/s | ~87 GB/s | LL≈Simple（噪声级） |
| 1MB | ~986 GB/s | ~1087 GB/s | Simple 略胜 |
| 2MB | ~2295 GB/s | ~9529 GB/s | **Simple 胜 4.1×** |
| 8MB | ~1646 GB/s | ~14523 GB/s | **Simple 胜 8.8×** |
| 32MB | ~2649 GB/s | ~17583 GB/s | **Simple 胜 6.6×** |

> 注：此处 "GB/s" 数值本身仅反映总数据量/耗时，绝对值意义有限（PCIe 理论 ~16GB/s/方向？待核），**重要的是 LL/Simple 的比值**。LL 在大消息上掉到 Simple 的 1/4~1/8，与手册"Simple 有效带宽更高"一致但幅度远大于预期。**这直接支撑"孤立协议带宽与端到端最优策略可能不一致"的假设**（LL 的延迟优势在重叠场景可能另有价值，需在 AG–GEMM 端到端验证）。

**P2. nChannels 的影响随协议/大小不同**
- 小消息（4KB/64KB）：nChannels 差异小（±20%）。
- 2MB+ LL：nChannels=2 明显恶化（2.3→1.2 GB/s），=4/8 恢复。
- Simple 大消息：nChannels 影响小（<10%），默认值已接近最优。
- 含义：**nChannels 的最优值不是全局常数**，依赖消息大小与协议 → 支持"策略选择必须感知配置上下文"。

**P3. Ring vs Tree 差异很小（PCIe 双卡）**
- 双卡 Ring 和 Tree 在所有大小上差异 <5%。两卡拓扑下 Ring/Tree 结构等价（一条直连），符合预期。**4 卡以上才可能显现差异**。

**P4. 正确性全部 OK**：所有 87 个配置点逐元素校验通过 → `duNcclAllgather` 细粒度接口在 gfx928 上可用、可复现。

### 附注与待办
- 完整数据表在 `/tmp/rccl_clean.log`（87 行数据）。首轮含 DEBUG 刷屏的版本在 `/tmp/rccl_full.log`（128MB 组被 timeout 截断）。
- **未完成**：128MB 组在干净版本中是否跑完需确认。
- **下一步（L1 延伸）**：① 补 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TASK` 跑一个代表性配置，确认实际采用的 algo/proto/nchannels 与参数一致；② 导出 `NCCL_TOPO_DUMP_FILE`/`NCCL_GRAPH_DUMP_FILE` 拓扑与通信图（论文 L0 需要）；③ 转入 DUSHMEM 原语测试（put/get/fence/quiet/wait + 完成时间线）。
