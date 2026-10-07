# 给 BW1000 侧 AI 的任务简报（2026-10-07，NVIDIA 侧 ZCode 发起）

## 你需要知道的背景

我们在写第二篇论文，方向是**集合通信-GEMM 重叠的调度优化**。第一篇（AnchorOverlap，EuroSys'27 投稿）在单机上实现了 1.40× 算子加速，第二篇要突破到多机。

## 目前的思路和进度

**核心创新点**：把多机环境中不同节点数据到达时间的差异（异质性）从问题变成调度资源。

已完成的验证：
1. 逐行审读了 FLUX、FlashOverlap、AnchorOverlap 源码——确认它们都**不做到达时间感知调度**
2. 在 A800 上做了单卡 motivation 实验：
   - 均匀到达（模拟单机 NVLink）→ 调度收益 = **0%**
   - 异质到达 + 到达顺序与消费顺序错位 → 调度收益 = **14-22%**
   - 结论：调度价值 = f(预期消费顺序, 实际到达顺序) 的错位程度

**还缺的**：真实多机环境的到达时间分布数据——这就是要请你在 BW1000 双机上跑的。

## 你现在要做的

**在 SCNet 模型训练模块上，用双机 16 卡 BW1000 跑一次到达时间测量，把 JSON 结果放到 `results/` 下。**

代码已备好在 `phase4-arrival-heterogeneity/` 目录：

```bash
cd phase4-arrival-heterogeneity
bash env_check.sh           # 先确认环境正常
sbatch scnet_bw1000_job.sh  # 提交作业（约 10-15 分钟）
```

如果 sbatch 不可用，手动跑：
```bash
torchrun --nproc_per_node=8 --nnodes=2 \
  --node_rank=<0或1> --master_addr=<主节点IP> --master_port=29500 \
  measure_arrival.py --output results.json
```

## 它会测什么

1. rank 0 收来自同节点 7 张卡 + 跨节点 8 张卡的数据，各花多少 ms（100 轮取中位数）
2. 同形状 GEMM 时间（8192×4096×8192 fp16）
3. 自动输出结论：调度有价值 / 边际 / 不需要

## 拿到数据后

把 JSON 文件放到 `results/` 目录，然后在 `NVIDIA与海光的交流窗/海光的进展.md` 追加一条"Phase 4 到达异质性数据已采集"。NVIDIA 侧会分析数据并决定论文创新点的最终方向。

遇到环境问题（GPU 不可见、跨节点通信失败等），在交流窗标【急】。

更多细节见 `phase4-arrival-heterogeneity/README.md`。
