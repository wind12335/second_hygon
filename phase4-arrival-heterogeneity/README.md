# Phase 4: 多机到达异质性测量（BW1000 双机 16 卡）

> **任务来源**：NVIDIA 侧 ZCode，2026-10-07 发起
> **目的**：为第二篇论文的核心创新点（到达异质性感知调度）采集 motivation 数据
> **需要**：SCNet 模型训练模块，双机 × 8 卡 BW1000，约 10 分钟机时

---

## 一、我们（NVIDIA 侧）在做什么

第二篇论文的核心创新点是：**把多机环境中不同节点数据到达时间的差异（异质性）从问题变成调度资源**。

我们已经完成了：
1. ✅ 源码级审计：确认 FLUX、FlashOverlap、AnchorOverlap 都不做到达时间调度
2. ✅ 单卡 A800 motivation 实验：均匀到达=调度无价值（0%），异质到达=调度有价值（最高 22%）
3. ⏳ **缺：真实多机环境的到达时间分布数据 ← 这批实验补的就是这个**

## 二、要测什么（三个问题）

| 问题 | 测什么 | 为什么重要 |
|---|---|---|
| Q1: 跨节点延迟有多大？ | rank 0 (节点 0) 收到来自节点 0 vs 节点 1 各 rank 的数据，各花多少 ms | 决定"异质性是否存在" |
| Q2: 异质性是系统性的还是随机的？ | 每对 (源, 目标) 的延迟方差——如果跨节点的 std 远大于同节点，说明是拓扑驱动 | 决定"可不可以预测"（能预测才能调度） |
| Q3: 通信/计算比值？ | 到达延迟 vs 同形状 GEMM 时间 | 决定"调度有没有肉吃"（比值 > 20% 才值得） |

## 三、执行方式

### 方式 A：自动作业（推荐）

```bash
# 在 SCNet 上提交
sbatch scnet_bw1000_job.sh
```

脚本会自动完成：环境检查 → 逐源到达时间测量（100 轮）→ GEMM 时间测量 → 输出 JSON。

### 方式 B：手动分步

如果 SCNet 不支持 sbatch，按以下顺序手动执行：

```bash
# 1. 环境检查
bash env_check.sh

# 2. 到达时间测量（需要在 2 节点 16 卡上运行）
python3 measure_arrival.py --output arrival_results.json

# 3. 查看结果
python3 analyze_arrival.py --input arrival_results.json
```

## 四、输出格式

JSON 文件包含：

```json
{
  "topology": {
    "nodes": 2, "gpus_per_node": 8,
    "interconnect": "需填写：IB/RoCE/以太网，带宽多少 Gbps"
  },
  "arrival_times": {
    "rank_0_to_rank_0": {"median_ms": 0.05, "std_ms": 0.01, "label": "same-node"},
    "rank_8_to_rank_0": {"median_ms": 2.3,  "std_ms": 0.8,  "label": "cross-node"},
    ...
  },
  "gemm": {
    "shape": [8192, 4096, 8192],
    "median_ms": 3.5
  },
  "heterogeneity_summary": {
    "same_node_median_ms": 0.05,
    "cross_node_median_ms": 2.3,
    "ratio": 46.0,
    "vs_gemm_pct": 65.7,
    "conclusion": "SCHEDULING_VALUABLE"
  }
}
```

## 五、给海光侧 AI 的说明

- 这批实验**不涉及你们已有的 phase1-3 代码**，是独立的新测量
- 跑完后请把 JSON 文件放到 `results/` 目录下
- 在 `NVIDIA与海光的交流窗/海光的进展.md` 里追加一条，注明"Phase 4 到达异质性数据已采集"
- 如果遇到环境问题（DUSHMEM 初始化失败、跨节点通信不通等），请在交流窗里标【急】
