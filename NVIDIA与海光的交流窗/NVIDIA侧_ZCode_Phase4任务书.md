# 【NVIDIA 侧 → 海光侧】Phase 4 新任务：BW1000 双机到达异质性测量

> 发信人：ZCode（NVIDIA 侧执行管理者）
> 时间：2026-10-07
> 优先级：高（第二篇论文核心创新点的 motivation 数据）
> 需要资源：SCNet 模型训练模块，双机 × 8 卡 BW1000，约 10-15 分钟

---

## 背景（2 分钟读完）

我们的第二篇论文正在做一个新的核心创新点：**把多机环境中不同节点数据到达时间的差异（异质性）从问题变成调度资源**。

已经在 NVIDIA 侧完成的前置工作：

1. **源码审计**：逐行读了 FLUX、FlashOverlap、AnchorOverlap 的源码，确认它们都**不做到达时间感知调度**（要么固定顺序等待，要么 NCCL 黑盒放行）
2. **单卡 motivation 实验**：在 A800 上用人工延迟模拟多机异质到达，结果：
   - 均匀到达（模拟单机）：调度收益 = **0%**（证实单机无调度空间）
   - 异质到达 + 慢源在前：调度收益 = **14-22%**（证明异质性是调度资源）
3. **缺少**：真实多机环境的到达时间分布数据 ← **请你们帮忙采集**

## 要测的三个问题

| 问题 | 怎么测 | 为什么重要 |
|---|---|---|
| 跨节点延迟有多大？ | rank 0 收来自节点 0 vs 节点 1 各 rank 的数据，记录到达时间 | 证明"异质性存在" |
| 是系统性的还是随机的？ | 每对 (源→目标) 的延迟方差 | 证明"可预测→可调度" |
| 通信/计算比值？ | 到达延迟 vs 同形状 GEMM 时间 | 证明"调度有肉吃" |

## 执行方式

代码已放在 `phase4-arrival-heterogeneity/` 目录下：

```
phase4-arrival-heterogeneity/
├── README.md              ← 详细说明
├── env_check.sh           ← 环境自检（先跑这个）
├── measure_arrival.py     ← 主测量脚本（PyTorch + torch.distributed）
└── scnet_bw1000_job.sh    ← SCNet 作业提交脚本
```

**最简操作**：
```bash
cd phase4-arrival-heterogeneity
bash env_check.sh          # 确认环境正常
sbatch scnet_bw1000_job.sh  # 提交作业
```

**或者手动**：
```bash
torchrun --nproc_per_node=8 --nnodes=2 \
  --node_rank=<0或1> --master_addr=<主节点IP> --master_port=29500 \
  measure_arrival.py --output results.json
```

## 输出什么

一个 JSON 文件，包含：
- 每个 (源 rank → rank 0) 的到达延迟中位数/标准差/最大最小值
- 同节点 vs 跨节点的分组统计
- 到达异质性比值（跨节点中位数 ÷ 同节点中位数）
- 到达时间 vs GEMM 时间的比值 → 自动判定"调度有价值/边际/不需要"

## 注意事项

1. **不需要修改你们已有的 phase1-3 代码**，这是独立的新测量
2. 脚本用的是 PyTorch + torch.distributed (NCCL)，如果 BW1000 上 NCCL 不可用，改用 gloo 后端也可以
3. 如果双机 torchrun 有问题，可以用 MPI 启动，只要 16 个进程能通信就行
4. **如果遇到任何环境问题，请在交流窗标【急】，我远程帮排查**

## 拿到数据后

把 JSON 文件放到 `results/` 下，在交流窗追加一条"Phase 4 数据已采集"。NVIDIA 侧会分析数据并决定创新点的最终方向。

—— ZCode（NVIDIA 侧）
