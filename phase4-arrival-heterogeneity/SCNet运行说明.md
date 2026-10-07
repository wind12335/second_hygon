# Phase 4 双机测量 · SCNet 模型训练模块运行说明(v2,2026-10-08)

> smoke 已于 2026-10-08 02:10 双机跑通(poll 模式,结果在 `results/phase4_arrival_smoke_20260908*.json`)。
> 本版说明针对**平台同时启动**模式:两个 worker 执行**同一条**启动命令,无需手动分先后。

## 第 0 步:建作业

模型训练模块创建作业:

- **2 节点,每节点 8 × BW1000**(镜像与 smoke 相同:`jupyterlab-pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11-devel`)
- 挂载保持默认(smoke 已验证作业 pod 能看到 `/root/private_data` 共享存储,**无需拷贝任何文件**)

## 第 1 步:填启动命令(两个 worker 同时执行这一条)

```
MODE=full bash /root/private_data/second_hygon/phase4-arrival-heterogeneity/run_phase4_scnet.sh
```

- `MODE=full` = 正式 100 轮(约 10 分钟);想再冒烟一遍就去掉 `MODE=full`(5 轮)
- 脚本自动按主机名分角色(`*-worker-0` = 主节点/rank 0,`*-worker-1` 自动等 master 端口,最多等 10 分钟)
- 不需要手动设 MASTER_ADDR / NODE_RANK / MASTER_PORT

## 第 2 步:确认成功

在**日志视图里看 worker-0**(rank 0 在它上面),依次出现:

```
workdir: /root/private_data/second_hygon/phase4-arrival-heterogeneity
World: 16 ranks across 2 nodes
GEMM 8192x4096x8192 fp16: ~2 ms
Preflight: is_completed polling WORKS -> poll mode
  round 10/100 done ... round 100/100 done
15 行结果表 → Same-node / Cross-node / ratio / Spread vs GEMM → conclusion
Results saved to .../phase4_arrival_full_<时间戳>.json
```

**worker-1 的日志只有 NCCL 噪声和 `torchrun exit code: 0`,最后一条是
`NOTE: 本机是 worker-1,不生成 JSON——若 worker-0 侧正常则一切正常`,这是正常现象。**

## 第 3 步:取结果

JSON 和 log 会自动落到两处(都在共享存储上):

- `second_hygon/phase4-arrival-heterogeneity/phase4_arrival_full_<时间戳>.json`(工作目录)
- `second_hygon/results/`(自动回拷)

之后 AI 侧执行 `python3 analyze_arrival.py --input <json>` 核对并在交流窗回正式条目。

## 常见问题

| 症状 | 处置 |
|---|---|
| worker-1 报"解析不到 worker-0 的地址" | 极少见(会自动重试 90s);仍失败则在启动命令前加 `MASTER_ADDR=vcjob-<id>-abc12335-<seq>-worker-0` |
| NCCL init 卡住/失败 | 启动命令前加 `NCCL_SOCKET_IFNAME=eth0` |
| 想看实时进度 | 平台日志视图,worker-0 每 10 轮打一行 |
| 卡死 | 脚本自带 per-round 看门狗(20s)+ 总超时(full 30min),到点自动退出并把 log 留在工作目录 |
