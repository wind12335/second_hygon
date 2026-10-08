# Phase 4 双机测量 · SCNet 模型训练模块运行说明（v4 RDMA 全通版，2026-10-08）

> TCP 版（socket 回退）与 **shca IB 版（400G NDR，全通道 RDMA）** 均已双机跑通并入库；
> 启动脚本 **v7 会自动安装 shca 用户态 + 配好 RCCL 外置插件**（包在共享存储 `rdma-shca/`，
> 免网络、幂等，每 pod 约 10 秒），默认 `NET=rdma`。跑完自动判定是否真走 RDMA。
> 旧 TCP 数据保留作对照组（`MODE=full NET=tcp` 可复现）。

## 第 0 步：建作业

模型训练模块创建作业（与之前完全相同）：

- **2 节点，每节点 8 × BW1000**（镜像：`jupyterlab-pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11-devel`）
- 挂载保持默认（作业 pod 能看到 `/root/private_data` 共享存储，**无需拷贝任何文件**）
- 前提：必须由**模型训练模块**建作业（notebook 不挂 `/dev/infiniband`）

## 第 1 步：填启动命令（两个 worker 同时执行这一条）

```
bash /root/private_data/second_hygon/phase4-arrival-heterogeneity/run_phase4_scnet.sh
```

先跑这条 **smoke（5 轮）**确认 RDMA 生效，再跑正式：

```
MODE=full bash /root/private_data/second_hygon/phase4-arrival-heterogeneity/run_phase4_scnet.sh
```

- 默认 `NET=rdma`；`MODE=full NET=tcp ...` 复现 TCP 对照；**`NET=check` 只输出可转发管理员的
  RDMA 分层诊断报告，10 秒退出，不启动训练**
- 脚本自动按主机名分角色（`*-worker-0` = 主节点/rank 0，`*-worker-1` 自动等 master 端口，最多等 10 分钟）
- 不需要手动设 MASTER_ADDR / NODE_RANK / MASTER_PORT

## 第 2 步：确认成功

在**日志视图里看 worker-0**（rank 0 在它上面），依次出现：

```
== 安装 shca RDMA 用户态(来源 .../rdma-shca/shca-tools_*.deb) ==
   安装完成: /usr/lib/x86_64-linux-gnu/libshca-rdmav34.so      <-- 自动装好
IB vars(平台配方): NET_PLUGIN=shca HCA=shca_0:1,shca_1:1,...
NET/Plugin: Plugin name set by env to librccl-net-shca.so        <-- 插件被认领
768 via NET/IBext_v8                                             <-- 全通道 RDMA
transport: ✓✓ 全部通道走 RDMA                                    <-- 关键行
World: 16 ranks across 2 nodes
GEMM 8192x4096x8192 fp16: ~1.9 ms
Preflight: is_completed polling WORKS -> poll mode  (15/15)
  round 5/5 done (smoke) 或 100/100 done (full)
15 行结果表 → Same-node / Cross-node / ratio / Spread vs GEMM → conclusion
Results saved to .../phase4_arrival_<mode>_rdma_<时间戳>.json
```

（IB 版参考值：same ~1.7ms / cross ~2.1ms / vs GEMM ~22% → SCHEDULING_VALUABLE）

**worker-1 的日志只有 NCCL 噪声和 `torchrun exit code: 0`，最后一条是
`NOTE: 本机是 worker-1,不生成 JSON——若 worker-0 侧正常则一切正常`，这是正常现象。**

## 第 3 步：取结果

JSON 和 log 会自动落到两处（都在共享存储上）：

- `second_hygon/phase4-arrival-heterogeneity/phase4_arrival_<mode>_<net>_<时间戳>.json`（工作目录）
- `second_hygon/results/`（自动回拷）

之后 AI 侧执行 `python3 analyze_arrival.py --input <json>` 核对并在交流窗回正式条目。

## 常见问题

| 症状 | 处置 |
|---|---|
| `transport: ✗ 仍含 NET/Socket` | 先看日志开头安装段是否 `安装完成: ...libshca-rdmav34.so`；再跑一次 `NET=check bash .../run_phase4_scnet.sh`，把"RDMA 诊断报告"整段转发平台管理员 |
| 想确认环境缺什么 | `NET=check`（分层报告：内核/设备/用户态/官方检测命令/结论，10 秒退出） |
| RCCL 报 GID index 相关错误 | 启动命令前加 `NCCL_IB_GID_INDEX=1` 重试（仍不行再试 `=0`） |
| RCCL 报找不到 HCA | `ibv_devinfo -l` 看到的名字若不是 shca_*，启动命令前加 `NCCL_IB_HCA=<实际名>:1,...` |
| worker-1 报"解析不到 worker-0 的地址" | 极少见（会自动重试 90s）；仍失败则在启动命令前加 `MASTER_ADDR=vcjob-<id>-abc12335-<seq>-worker-0` |
| NCCL init 卡住/失败 | 启动命令前加 `NCCL_SOCKET_IFNAME=eth0` |
| 想看实时进度 | 平台日志视图，worker-0 每 10 轮打一行 |
| 卡死 | 脚本自带 per-round 看门狗（20s）+ 总超时（full 30min），到点自动退出并把 log 留在工作目录 |
