#!/bin/bash
# ============================================================================
# BW1000 双机 Phase 4 到达异质性测量 · SCNet 模型训练模块启动脚本 (v7 shca 适配版:自动装 shca 用户态+RCCL 插件)
#
# 【本文件自包含】measure_arrival_v2.py 已内嵌,只需拷贝这一个文件到两个
# worker(放任何目录都行),首次运行会自动把测量脚本生成到 ~/phase4/ 并在那里执行。
#
# v5: 默认 NET=rdma —— 按海光官方《模型训练模块使用 RDMA》教程设置 IB 变量,
#     但 NCCL_IB_HCA 按本分区实际网卡改为 shca(教程默认 mlx5 不适配,节点上是
#     shca_0..3,400G NDR);跑完自动判定 RCCL 是否真走了 NET/IB。
#     NET=tcp 可复现 v4 行为(RCCL socket 回退,对照组)。
#
# 用法 A【平台同时启动,推荐】:把下面这条填进模型训练模块的"启动命令",
#         两个 worker 会同时执行同一条命令,角色自动按主机名区分,无需任何手动先后:
#     MODE=full bash /root/private_data/second_hygon/phase4-arrival-heterogeneity/run_phase4_scnet.sh
#       (MODE=full 正式 100 轮;去掉 MODE=full 则为 smoke 5 轮)
#
# 用法 B【手动两个终端】:worker-0 先、worker-1 后,各执行一次:
#     bash run_phase4_scnet.sh                 # 默认 MODE=smoke + NET=rdma
#     MODE=full bash run_phase4_scnet.sh       # 正式测量(100 轮)
#     MODE=full NET=tcp bash run_phase4_scnet.sh   # TCP 对照(复现 v4 的 socket 回退)
#
# 可选环境变量:
#   MODE=smoke|full      smoke=5 轮快速验证 / full=100 轮正式(默认 smoke)
#   NET=rdma|tcp|check   rdma=按教程开 IB(默认) / tcp=socket 对照 / check=纯诊断 10 秒退出
#   PHASE4_HOME=dir      工作目录(默认:脚本旁边有测量脚本则用同目录,否则 ~/phase4)
#   MASTER_ADDR=...      手动指定 master 地址(自动推导失败时才需要)
#   MASTER_PORT=23456    默认 29500
# ============================================================================
set -uo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORK_DIR="${PHASE4_HOME:-}"
if [ -z "$WORK_DIR" ]; then
  if [ -f "$LAUNCH_DIR/measure_arrival_v2.py" ]; then
    WORK_DIR="$LAUNCH_DIR"                          # 测量脚本就在旁边 → 直接用
  else
    WORK_DIR="${HOME:-/root}/phase4"                # 单文件分发 → 生成到 ~/phase4
  fi
fi
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

MODE="${MODE:-smoke}"
NET="${NET:-rdma}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"

# ---- 首次运行:写出内嵌的测量脚本 -------------------------------------------
if [ ! -f "$WORK_DIR/measure_arrival_v2.py" ]; then
  echo "(首次运行:在 $WORK_DIR 生成 measure_arrival_v2.py ...)"
  cat > "$WORK_DIR/measure_arrival_v2.py" <<'EMBED_V2_PYEOF'
#!/usr/bin/env python3
"""
[v2] BW1000 dual-node arrival heterogeneity measurement — hardened variant.

Methodology and JSON schema identical to measure_arrival.py (v1, NVIDIA 侧),
with three hardening changes for the BW1000/RCCL environment:

1. PREFLIGHT: before the measured rounds, one trial round checks whether
   Work.is_completed() polling actually fires on this backend. (On this
   torch build's gloo backend it never fires -> v1 would hang forever; RCCL
   could not be verified locally because RCCL refuses 2 ranks on 1 GPU.)
   If polling works -> "poll" mode (exactly v1 behaviour). If not ->
   "wait_fallback" mode: irecv all sources, then wait() each in posting
   order (order-biased, clearly flagged in JSON), so the job always
   terminates and still yields per-source latencies + GEMM ratio.
2. WATCHDOG: every round is bounded by --poll-timeout-s; leftover requests
   are drained with wait() and the round is flagged degraded.
3. PROGRESS + ORDER STATS: rank 0 prints one heartbeat line per round
   (liveness in pod logs), and the mean arrival POSITION of each source is
   recorded — the scheduling-value metric is precisely arrival order vs
   consumption order (rank order) mismatch, which v1 discarded.

Output JSON is a strict superset of v1's: every v1 field keeps its
name/meaning; extras live under "measurement", "platform", per-source
"mean_arrival_position", plus a "heterogeneity_summary" alias matching the
field names in README.md's example.
"""
import argparse
import json
import os
import socket
import statistics
import time
from collections import defaultdict

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-mb", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gemm-shape", type=int, nargs=3, default=[8192, 4096, 8192])
    parser.add_argument("--output", type=str, default="arrival_results_v2.json")
    parser.add_argument("--poll-timeout-s", type=float, default=20.0,
                        help="per-round watchdog; leftover irecvs drained via wait()")
    parser.add_argument("--preflight-timeout-s", type=float, default=5.0,
                        help="trial-round budget for deciding poll vs wait_fallback")
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl")  # RCCL on BW1000
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    hostname = socket.gethostname()
    node_id = rank // 8

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"World: {world} ranks across {world // 8} nodes")
        print(f"Chunk: {args.chunk_mb} MB per producer")
        print(f"Iterations: {args.iterations} (+{args.warmup} warmup)")
        print()

    # ---- platform facts (all nodes) --------------------------------------
    hosts = [None for _ in range(world)]
    dist.all_gather_object(hosts, hostname)
    node_names = []
    for h in hosts:
        if h not in node_names:
            node_names.append(h)
    if rank == 0:
        platform_info = {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpus_per_node_visible": torch.cuda.device_count(),
            "nodes_actual": node_names,
            "hostname_rank0": hostname,
        }

    # ================================================================
    # Part 1: GEMM time (every rank measures independently)
    # ================================================================
    M, N, K = args.gemm_shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    gemm_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        gemm_times.append((time.perf_counter() - t0) * 1000)
    del a, b, c
    torch.cuda.empty_cache()

    # Gather GEMM times to rank 0
    gemm_tensor = torch.tensor([statistics.median(gemm_times)], device="cuda")
    gemm_all = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(gemm_all, gemm_tensor)
    if rank == 0:
        gemm_result = {
            "shape": args.gemm_shape,
            "median_ms": gemm_all[0].item(),
            "all_ranks_ms": [g.item() for g in gemm_all],
        }
        print(f"GEMM {M}x{N}x{K} fp16: {gemm_result['median_ms']:.2f} ms (rank 0)")
    dist.barrier()

    # ================================================================
    # Part 2: Arrival time measurement
    # ================================================================
    chunk_elems = int(args.chunk_mb * 1024 * 1024) // 2  # fp16 elements
    data = torch.ones(chunk_elems, dtype=torch.float16, device="cuda")

    def post_requests():
        reqs = []
        for src in range(1, world):
            buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
            reqs.append((src, dist.irecv(buf, src=src), buf))
        return reqs

    # Warmup rounds (blocking receives, as v1)
    for _ in range(args.warmup):
        if rank == 0:
            for src in range(1, world):
                buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
                dist.recv(buf, src=src)
        else:
            dist.send(data, dst=0)
        dist.barrier()

    # ---- PREFLIGHT: does is_completed() polling work on this backend? ----
    degraded_rounds = 0
    if rank == 0:
        round_start = time.perf_counter()
        requests = post_requests()
        completed = set()
        while len(completed) < world - 1 and \
                (time.perf_counter() - round_start) < args.preflight_timeout_s:
            for src, req, buf in requests:
                if src not in completed and req.is_completed():
                    completed.add(src)
            time.sleep(0.0001)
        mode = "poll" if len(completed) == world - 1 else "wait_fallback"
        if len(completed) < world - 1:
            for src, req, buf in requests:
                if src not in completed:
                    req.wait()
        print(f"Preflight: is_completed polling "
              f"{'WORKS -> poll mode (v1 behaviour)' if mode == 'poll' else 'DOES NOT FIRE -> wait_fallback mode'}"
              f" ({len(completed)}/{world - 1} completions in {args.preflight_timeout_s}s)",
              flush=True)
    else:
        dist.send(data, dst=0)
    dist.barrier()
    mode_holder = [mode] if rank == 0 else [None]
    dist.broadcast_object_list(mode_holder, src=0)
    mode = mode_holder[0]
    dist.barrier()

    # Measurement rounds
    arrival_data = defaultdict(list)   # src_rank -> [latency_ms, ...]
    position_data = defaultdict(list)  # src_rank -> [1-based completion position, ...]

    for it in range(args.iterations):
        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            round_start = time.perf_counter()
            requests = post_requests()
            completed = set()
            degraded = False

            if mode == "poll":
                while len(completed) < world - 1:
                    if (time.perf_counter() - round_start) > args.poll_timeout_s:
                        degraded = True
                        break
                    for src, req, buf in requests:
                        if src not in completed and req.is_completed():
                            lat = (time.perf_counter() - round_start) * 1000
                            arrival_data[src].append(lat)
                            position_data[src].append(len(completed) + 1)
                            completed.add(src)
                    time.sleep(0.0001)  # 0.1ms polling granularity (v1 spec)
                if degraded:
                    for src, req, buf in requests:
                        if src not in completed:
                            req.wait()
                            lat = (time.perf_counter() - round_start) * 1000
                            arrival_data[src].append(lat)
                            position_data[src].append(len(completed) + 1)
                            completed.add(src)
                    degraded_rounds += 1
            else:  # wait_fallback
                for src, req, buf in requests:
                    req.wait()
                    lat = (time.perf_counter() - round_start) * 1000
                    arrival_data[src].append(lat)
                    position_data[src].append(len(completed) + 1)
                    completed.add(src)

            if (it + 1) % 10 == 0 or it == 0 or args.iterations <= 10:
                lats = [arrival_data[s][-1] for s in completed]
                print(f"  round {it + 1}/{args.iterations} done, "
                      f"last-round arrivals min/med/max = "
                      f"{min(lats):.2f}/{statistics.median(lats):.2f}/{max(lats):.2f} ms",
                      flush=True)
        else:
            dist.send(data, dst=0)

        dist.barrier()

    # ================================================================
    # Part 3: Analyze and output
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 75)
        print("ARRIVAL TIME MEASUREMENT RESULTS")
        print("=" * 75)
        print(f"{'Rank':>5s} {'Node':>5s} {'Label':>12s} {'Median(ms)':>10s} "
              f"{'Std(ms)':>8s} {'Min(ms)':>8s} {'Max(ms)':>8s} {'AvgPos':>6s}")
        print("-" * 82)

        arrival_stats = {}
        for src in sorted(arrival_data.keys()):
            times = arrival_data[src]
            src_node = src // 8
            label = "SAME-NODE" if src_node == 0 else "CROSS-NODE"
            med = statistics.median(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            mn, mx = min(times), max(times)
            pos = statistics.mean(position_data[src]) if position_data[src] else -1
            print(f"{src:5d} {src_node:5d} {label:>12s} {med:10.3f} "
                  f"{std:8.3f} {mn:8.3f} {mx:8.3f} {pos:6.1f}")
            arrival_stats[f"rank_{src}"] = {
                "node": src_node,
                "label": label,
                "median_ms": round(med, 3),
                "std_ms": round(std, 3),
                "min_ms": round(mn, 3),
                "max_ms": round(mx, 3),
                "mean_arrival_position": round(pos, 2),
                "n": len(times),
            }

        # Summary metrics (v1 definitions)
        same_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "SAME-NODE"]
        cross_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "CROSS-NODE"]

        same_med = statistics.median(same_node) if same_node else 0
        cross_med = statistics.median(cross_node) if cross_node else 0
        spread = cross_med - same_med
        ratio = cross_med / same_med if same_med > 0.001 else float("inf")
        gemm_ms = gemm_result["median_ms"]
        vs_gemm = spread / gemm_ms * 100 if gemm_ms > 0 else 0

        print("-" * 82)
        print(f"Same-node median:  {same_med:.3f} ms")
        print(f"Cross-node median: {cross_med:.3f} ms")
        print(f"Spread:            {spread:.3f} ms")
        print(f"Heterogeneity ratio: {ratio:.1f}×")
        print(f"GEMM time:         {gemm_ms:.2f} ms")
        print(f"Spread vs GEMM:    {vs_gemm:.1f}%")
        print()

        if vs_gemm > 20:
            conclusion = "SCHEDULING_VALUABLE"
            print("→ ✓ Scheduling is VALUABLE (spread > 20% of GEMM time)")
        elif vs_gemm > 5:
            conclusion = "SCHEDULING_MARGINAL"
            print("→ ~ Scheduling is MARGINAL (5-20% of GEMM time)")
        else:
            conclusion = "SCHEDULING_NOT_NEEDED"
            print("→ ✗ Scheduling NOT needed (spread < 5% of GEMM time)")

        summary = {
            "same_node_median_ms": round(same_med, 3),
            "cross_node_median_ms": round(cross_med, 3),
            "spread_ms": round(spread, 3),
            "heterogeneity_ratio": round(ratio, 1),
            "spread_vs_gemm_pct": round(vs_gemm, 1),
            "conclusion": conclusion,
        }

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": hostname,
            "script": "measure_arrival_v2.py",
            "topology": {
                "nodes": world // 8,
                "gpus_per_node": 8,
                "world_size": world,
                "note": "Please fill in interconnect type and bandwidth",
            },
            "config": {
                "chunk_mb": args.chunk_mb,
                "iterations": args.iterations,
                "gemm_shape": args.gemm_shape,
            },
            "measurement": {
                "method": mode,
                "method_note": (
                    "identical to v1 polling"
                    if mode == "poll" else
                    "is_completed() never fired on this backend; per-source "
                    "latency recorded at wait() completion in posting order — "
                    "ORDER BIASED, see mean_arrival_position with care"
                ),
                "poll_granularity_ms": 0.1,
                "degraded_rounds": degraded_rounds,
                "poll_timeout_s": args.poll_timeout_s,
            },
            "platform": platform_info,
            "gemm": gemm_result,
            "arrival_times": arrival_stats,
            "summary": summary,
            # alias matching README.md example field names (same values)
            "heterogeneity_summary": dict(summary),
        }
        if not cross_node:
            results["measurement"]["warning"] = (
                "no cross-node sources in this run (world=%d) — run with "
                "--nnodes=2 x 8 for the dual-node measurement" % world)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
EMBED_V2_PYEOF
fi

SCRIPT="$WORK_DIR/measure_arrival_v2.py"
if ! python3 -m py_compile "$SCRIPT" 2>/dev/null; then
  echo "ERROR: 内嵌测量脚本写出后校验失败($SCRIPT)。请把本文件完整拷贝后再试。"; exit 1
fi

echo "=============================================="
echo " Phase4 arrival-heterogeneity launcher ($MODE, net=$NET)"
echo " Host: $(hostname)   Date: $(date)"
echo " workdir: $WORK_DIR"
echo "=============================================="

# ---- 1. 节点角色识别 --------------------------------------------------------
HOSTSHORT=$(hostname -s)
case "$HOSTSHORT" in
  *-worker-0) NODE_RANK=0 ;;
  *-worker-1) NODE_RANK=1 ;;
  *) if [ -n "${SLURM_NODEID:-}" ]; then NODE_RANK="$SLURM_NODEID";
     elif [ -n "${NODE_RANK:-}" ]; then NODE_RANK="$NODE_RANK";
     else echo "无法从主机名识别节点角色(期望 *-worker-0 / *-worker-1)。"; echo "请手动指定: NODE_RANK=0|1 bash $0"; exit 1; fi ;;
esac
echo "node_rank = $NODE_RANK"

# ---- 2. GPU 数量检查 --------------------------------------------------------
NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "visible GPUs = $NGPU (expect $NPROC)"
if [ "$NGPU" -lt "$NPROC" ]; then
  echo "ERROR: GPU 数量不足($NGPU < $NPROC),检查作业是否申请了 8 卡。"; exit 1
fi

# ---- 3. 平台必需环境变量 ----------------------------------------------------
export HSA_FORCE_FINE_GRAIN_PCIE=1     # DTK/RCCL 在 PCIe 上必需,否则可能挂起
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  DEF_IF=$(ip route 2>/dev/null | awk '/^default/{print $5; exit}')
  [ -n "$DEF_IF" ] && export NCCL_SOCKET_IFNAME="$DEF_IF" && echo "NCCL_SOCKET_IFNAME=$DEF_IF (auto)"
fi

# ---- 3a. shca RDMA 用户态安装(包在共享存储,免网络,幂等;两个 worker 各自装) --
SHCA_HOME="${SHCA_HOME:-$LAUNCH_DIR/../rdma-shca}"
if [ "$NET" != "tcp" ] && [ ! -e /usr/lib/x86_64-linux-gnu/libshca-rdmav34.so ]; then
  DEB=$(ls "$SHCA_HOME"/shca-tools_*.deb 2>/dev/null | head -1)
  if [ -n "$DEB" ]; then
    echo "== 安装 shca RDMA 用户态(来源 $DEB) =="
    rm -rf /tmp/shca-install && mkdir -p /tmp/shca-install
    if dpkg -x "$DEB" /tmp/shca-install 2>/dev/null; then
      LIBD=/usr/lib/x86_64-linux-gnu
      cp -a /tmp/shca-install$LIBD/. $LIBD/
      rm -f $LIBD/libibverbs.so.1.14.47.0 $LIBD/libmlx5.so.1.24.47.0 \
            $LIBD/librdmacm.so.1.3.47.0 $LIBD/libibverbs.so.1.*47* 2>/dev/null
      ln -sf libibverbs.so.1.14.44.0 $LIBD/libibverbs.so.1
      ln -sf librdmacm.so.1.3.44.0   $LIBD/librdmacm.so.1
      ln -sf libmlx5.so.1.24.44.0    $LIBD/libmlx5.so.1
      mkdir -p /etc/libibverbs.d /usr/etc/libibverbs.d /etc/cfgFile
      cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /etc/libibverbs.d/ 2>/dev/null
      cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /usr/etc/libibverbs.d/ 2>/dev/null
      cp -a /tmp/shca-install/etc/cfgFile/. /etc/cfgFile/ 2>/dev/null
      mkdir -p /usr/local/bin && cp -a /tmp/shca-install/usr/bin/. /usr/local/bin/ 2>/dev/null
      ldconfig
      echo "   安装完成: $(ls $LIBD/libshca-rdmav34.so 2>/dev/null || echo '异常:未找到 libshca-rdmav34.so')"
    else
      echo "   解包失败,继续用镜像自带(Mellanox)用户态"
    fi
  else
    echo "== 未找到 $SHCA_HOME/shca-tools_*.deb,跳过 shca 用户态安装(将只有 Mellanox) =="
  fi
fi

# ---- 3b. 传输模式:rdma(默认,海光教程变量) / tcp(对照) / check(纯诊断报告) ----
if [ "$NET" != "tcp" ]; then
  IBV_DIR=/usr/lib/x86_64-linux-gnu/libibverbs
  echo "================================================================"
  echo " RDMA 诊断报告(NET=$NET) —— 本段可整段转发给平台管理员"
  echo " Host: $(hostname)    Date: $(date '+%F %T')    Kernel: $(uname -r)"
  echo "================================================================"

  echo "[1] 内核/硬件层  /sys/class/infiniband:"
  SYS_IB=$(ls /sys/class/infiniband 2>/dev/null)
  if [ -n "$SYS_IB" ]; then
    echo "    设备: $(echo $SYS_IB)"
    for d in /sys/class/infiniband/*; do
      [ -d "$d/ports" ] || continue
      for p in "$d"/ports/*; do
        echo "    $(basename "$d")/$(basename "$p"): $(cat "$p/state" 2>/dev/null)  $(cat "$p/rate" 2>/dev/null)  link_layer=$(cat "$p/link_layer" 2>/dev/null)"
      done
    done
    echo "    => PASS: 网卡内核可见、链路 ACTIVE"
  else
    echo "    (空) => FAIL: 容器看不到 sysfs RDMA 设备"
  fi

  echo "[2] 容器设备层  /dev/infiniband:"
  if [ -d /dev/infiniband ]; then
    echo "    $(ls /dev/infiniband | tr '\n' ' ')"
    echo "    => PASS: RDMA 设备已挂载进 pod(平台 RDMA 特性对训练模块已生效)"
  else
    echo "    (不存在) => FAIL: pod 未挂载 RDMA 设备,需平台侧开通"
  fi

  echo "[3] 用户态层  ibverbs providers($IBV_DIR):"
  echo "    $(ls "$IBV_DIR" 2>/dev/null | tr '\n' ' ')"
  echo "    rdma 包版本: $(dpkg -l 2>/dev/null | awk '/^ii.*(rdma-core|libibverbs1|ibverbs-providers|librdmacm1)/{printf "%s=%s  ", $2, $3}')"
  if [ -e /usr/lib/x86_64-linux-gnu/libshca-rdmav34.so ] || ls "$IBV_DIR" 2>/dev/null | grep -q shca; then
    echo "    => PASS: 含 shca provider;RCCL shca 插件: $(ls "$SHCA_HOME"/topo_lib/lib/librccl-net-shca.so 2>/dev/null || echo 未找到)"
  else
    echo "    => FAIL: 无 shca 的 provider(如 libshca-rdmav34.so)—— 现有插件只认 Mellanox,"
    echo "       shca 网卡在用户态不可见,任何 ibv_* 程序(含 RCCL)枚举结果为 0"
  fi

  echo "[4] 官方文档《RDMA 使用案例》步骤 1 检测命令原样输出:"
  echo "    \$ ibv_devices"
  ibv_devices 2>&1 | sed 's/^/        /'
  echo "    \$ ibstatus"
  ibstatus 2>&1 | sed 's/^/        /' | head -8
  echo "    \$ ibv_devinfo -l"
  ibv_devinfo -l 2>&1 | sed 's/^/        /'

  echo "[5] 结论与请求:"
  if [ -e /usr/lib/x86_64-linux-gnu/libshca-rdmav34.so ] || ls "$IBV_DIR" 2>/dev/null | grep -q shca; then
    echo "    用户态齐备(shca provider 已就位) —— 若上述 ibv_* 输出仍为 0,请把本报告发回进一步定位"
  else
    if [ -n "$SYS_IB" ] && [ -d /dev/infiniband ]; then
      echo "    内核有驱动([1] PASS)、设备已挂载([2] PASS),但镜像用户态只有 Mellanox OFED"
      echo "    的 provider([3] FAIL),故步骤 1 检测为 0 HCAs,NCCL/RCCL 只能回退 TCP socket"
      echo "    (双机实测跨节点仅 ~0.034 GB/s)。环境变量无法解决——缺的是厂商二进制库文件。"
    else
      echo "    平台侧设备挂载未生效([1]或[2] FAIL),请先补齐;若补齐后仍 0 HCAs,则同下:"
      echo "    镜像用户态只有 Mellanox OFED provider([3] FAIL),需 shca 用户态库。"
    fi
    echo "    请求: ① 告知 BW1000 分区上 ibv_devices 能列出 shca 设备的镜像名;或"
    echo "          ② 提供宿主机 shca_ib 驱动包中的用户态部分(libshca-rdmav34.so 及配套库)。"
  fi
  echo "================================================================"
fi
if [ "$NET" = "check" ]; then
  echo "(NET=check: 只诊断,不启动训练)"; exit 0
fi
if [ "$NET" = "rdma" ]; then
  # 平台管理员配方(vllm/sglang 同款):RCCL 走 librccl-net-shca 插件 + 平台拓扑文件
  export NCCL_IB_DISABLE=0
  export NCCL_NET_PLUGIN=shca
  export NCCL_IB_HCA="${NCCL_IB_HCA:-shca_0:1,shca_1:1,shca_2:1,shca_3:1}"
  if [ -f "$SHCA_HOME/topo_lib/built-in-508-topo-input-tj-default.xml" ]; then
    export NCCL_TOPO_FILE="$SHCA_HOME/topo_lib/built-in-508-topo-input-tj-default.xml"
  fi
  export LD_LIBRARY_PATH="$SHCA_HOME/topo_lib/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  echo "IB vars(平台配方): NET_PLUGIN=$NCCL_NET_PLUGIN HCA=$NCCL_IB_HCA"
  echo "                   TOPO=${NCCL_TOPO_FILE:-none} LD_LIBRARY_PATH+=$SHCA_HOME/topo_lib/lib"
else
  export NCCL_IB_DISABLE=1             # 对照组:复现 v4 的 NET/Socket 回退
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# ---- 4. MASTER_ADDR 解析 ----------------------------------------------------
resolve_master() {
  getent hosts "$1" 2>/dev/null | awk '{print $1; exit}'
}
if [ -n "${MASTER_ADDR:-}" ]; then
  echo "MASTER_ADDR = $MASTER_ADDR (manual)"
elif [ "$NODE_RANK" = "0" ]; then
  M=$(resolve_master "$HOSTSHORT" || true)
  [ -z "$M" ] && M=$(hostname -I 2>/dev/null | awk '{print $1}')
  if [ -z "$M" ]; then echo "ERROR: 无法确定本机 IP,请手动 MASTER_ADDR=<worker-0 IP> bash $0"; exit 1; fi
  MASTER_ADDR="$M"
else
  BASE="${HOSTSHORT%-worker-*}"
  MASTER_ADDR=""
  # 同时启动场景:worker-0 的 DNS 记录可能晚几秒就绪,重试等待而不是立刻失败
  for i in $(seq 1 45); do
    MASTER_ADDR=$(resolve_master "$BASE-worker-0" || true)
    [ -n "$MASTER_ADDR" ] && break
    sleep 2
  done
  [ -z "$MASTER_ADDR" ] && MASTER_ADDR=$(grep -E "worker-0\b" /etc/hosts 2>/dev/null | awk '{print $1; exit}')
  if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: 90s 内解析不到 ${BASE}-worker-0 的地址。"
    echo "  请在启动命令前加 MASTER_ADDR=<worker-0 主机名或IP> 重试"
    exit 1
  fi
fi
echo "MASTER_ADDR = $MASTER_ADDR   MASTER_PORT = $MASTER_PORT"

# ---- 5. worker-1 等 master 端口就绪 -----------------------------------------
if [ "$NODE_RANK" = "1" ]; then
  echo "waiting for master ${MASTER_ADDR}:${MASTER_PORT} to accept connections ..."
  ok=0
  for i in $(seq 1 300); do
    if timeout 2 bash -c "exec 3<>/dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then ok=1; break; fi
    sleep 2
  done
  if [ "$ok" != "1" ]; then echo "ERROR: 600s 内 master 端口未就绪,请确认两个 worker 用的是同一条启动命令。"; exit 1; fi
  echo "master port is up."
fi

# ---- 6. 运行 ----------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
if [ "$MODE" = "full" ]; then
  ITERS=100; WARM=10; TIMEOUT_S=1800
else
  ITERS=5;   WARM=2;  TIMEOUT_S=600
fi
OUT="$WORK_DIR/phase4_arrival_${MODE}_${NET}_${TS}.json"
LOG="$WORK_DIR/phase4_arrival_${MODE}_${NET}_${TS}.log"

echo "--- torchrun: nnodes=$NNODES nproc=$NPROC node_rank=$NODE_RANK ---"
timeout "$TIMEOUT_S" torchrun \
  --nproc_per_node="$NPROC" --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --max_restarts=0 \
  "$SCRIPT" \
  --chunk-mb 8 --iterations "$ITERS" --warmup "$WARM" \
  --gemm-shape 8192 4096 8192 \
  --output "$OUT" ${EXTRA_ARGS:-} 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "torchrun exit code: $RC (log: $LOG)"

# ---- 6b. 传输判定(打印实际使用的通道类型) ----------------------------------
echo "--- transport check ---"
TRANS=$(grep -oE 'via NET/[A-Za-z0-9_]+' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -5)
if [ -n "$TRANS" ]; then
  echo "$TRANS" | sed 's/^/  /'
  if echo "$TRANS" | grep -q 'NET/Socket'; then
    echo "transport: ✗ 仍含 NET/Socket(TCP 回退未消除)"
  else
    echo "transport: ✓✓ 全部通道走 RDMA"
  fi
else
  echo "transport: ? 日志无通道信息(可能初始化即失败)"
fi
grep -m3 -E "No IB NIC|NET/IB :|NET.*[Ss]hca|Failed to open|Invalid GID|plugin" "$LOG" 2>/dev/null | sed 's/^/  /'

# ---- 7. 结果汇总与回拷 -------------------------------------------------------
if [ -f "$OUT" ]; then
  echo ""
  echo "--- Results summary ---"
  python3 - "$OUT" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
s = r.get("summary", {})
m = r.get("measurement", {})
print("method:            {}".format(m.get("method", "v1(no flag)")))
print("same-node median:  {:.3f} ms".format(s.get("same_node_median_ms", 0)))
print("cross-node median: {:.3f} ms".format(s.get("cross_node_median_ms", 0)))
print("spread:            {:.3f} ms".format(s.get("spread_ms", 0)))
print("vs GEMM:           {:.1f}%".format(s.get("spread_vs_gemm_pct", 0)))
print("conclusion:        {}".format(s.get("conclusion", "N/A")))
PYEOF
  for D in "/root/private_data/second_hygon/results" "$HOME/second_hygon/results" /mnt/public/home/*/second_hygon/results; do
    [ -d "$D" ] || continue
    [ "$D" = "$WORK_DIR" ] && continue
    cp -v "$OUT" "$LOG" "$D"/ && break
  done
  echo ""
  echo ">>> 结果 JSON: $OUT (worker-1 不生成 JSON,属正常)"
else
  if [ "$NODE_RANK" = "1" ]; then
    echo "NOTE: 本机是 worker-1,不生成 JSON(worker-0 才写)——若 worker-0 侧正常则一切正常。"
  else
    echo "WARNING: worker-0 未生成结果文件 $OUT —— 把 $LOG 全文带回给 AI 侧排查。"
  fi
  [ "$RC" = "124" ] && echo "(exit 124 = 触发总超时;如果是卡死,日志里最后几行能看出卡在哪)"
fi
exit "$RC"
