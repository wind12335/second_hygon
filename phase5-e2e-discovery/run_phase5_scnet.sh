#!/bin/bash
# ============================================================================
# Phase 5 端到端 AG-GEMM 瓶颈发现 · SCNet 模型训练模块启动脚本 (v1 平台适配版)
#
# 在 NVIDIA 侧 run_phase5.sh(手动两终端模型)基础上适配本平台:
#   - 模型训练模块是两个 worker 同时执行同一条命令 → 主机名自动分角色,
#     master 自动解析 + worker-1 等待,无需手动先后、无需传 IP;
#   - 接入 shca RDMA(v7 机制):自动安装 shca 用户态 + librccl-net-shca 插件,
#     默认 NET=rdma;NET=tcp 可跑 socket 对照;NET=check 只出诊断报告;
#   - 总超时看门狗 + 传输判定 + 结果自动回拷 results/。
# 测量逻辑完全用 NVIDIA 侧 measure_e2e_bottleneck.py,本脚本不改动它。
#
# 用法(两个 worker 的"启动命令"填同一条):
#   bash /root/private_data/second_hygon/phase5-e2e-discovery/run_phase5_scnet.sh
#     (默认 MODE=smoke, REPS=2 快速验证)
#   MODE=full bash /root/private_data/second_hygon/phase5-e2e-discovery/run_phase5_scnet.sh
#     (REPS=20 正式)
#
# 可选环境变量:
#   MODE=smoke|full   smoke=2 轮验证 / full=20 轮正式(默认 smoke)
#   NET=rdma|tcp|check  rdma=shca IB(默认) / tcp=socket 对照 / check=纯诊断 10 秒退出
#   REPS=N            覆盖轮数
#   EXTRA_ARGS="--skip 3,4"  透传给测量脚本(如跳过 Q3/Q4)
# ============================================================================
set -uo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SCRIPT="$LAUNCH_DIR/measure_e2e_bottleneck.py"
[ -f "$SCRIPT" ] || { echo "ERROR: 找不到 $SCRIPT(NVIDIA 侧测量脚本应与本文件同目录)。"; exit 1; }

MODE="${MODE:-smoke}"
NET="${NET:-rdma}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
if [ "$MODE" = "full" ]; then REPS="${REPS:-20}"; else REPS="${REPS:-2}"; fi

echo "=============================================="
echo " Phase5 E2E Bottleneck Discovery ($MODE, net=$NET, reps=$REPS)"
echo " Host: $(hostname)   Date: $(date)"
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
# fd 上限抬高:Q2 全对全会并发建 120 个 pair comm(proxy 套接字众多);
# 崩溃案例见交流窗 2026-10-08(串行预热补丁为主,此项为保险)
ulimit -n 65535 2>/dev/null || true
echo "ulimit -n (soft) = $(ulimit -n)"
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

# ---- 3b. 传输模式:rdma(默认) / tcp(对照) / check(纯诊断报告) ----------------
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
  else
    echo "    (空) => FAIL: 容器看不到 sysfs RDMA 设备"
  fi
  echo "[2] 容器设备层  /dev/infiniband:"
  if [ -d /dev/infiniband ]; then
    echo "    $(ls /dev/infiniband | tr '\n' ' ')"
  else
    echo "    (不存在) => FAIL: pod 未挂载 RDMA 设备"
  fi
  echo "[3] 用户态层  ibverbs providers($IBV_DIR):"
  echo "    $(ls "$IBV_DIR" 2>/dev/null | tr '\n' ' ')"
  echo "    ibv_devinfo -l:"
  ibv_devinfo -l 2>&1 | sed 's/^/        /'
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
else
  export NCCL_IB_DISABLE=1             # 对照组:socket 回退
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
OUT="$LAUNCH_DIR/phase5_e2e_${MODE}_${NET}_${TS}.json"
LOG="$LAUNCH_DIR/phase5_e2e_${MODE}_${NET}_${TS}.log"
TIMEOUT_S=1800

echo "--- torchrun: nnodes=$NNODES nproc=$NPROC node_rank=$NODE_RANK reps=$REPS ---"
timeout "$TIMEOUT_S" torchrun \
  --nproc_per_node="$NPROC" --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --max_restarts=0 \
  "$SCRIPT" \
  --shape 8192 4096 8192 \
  --reps "$REPS" \
  --output "$OUT" ${EXTRA_ARGS:-} 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "torchrun exit code: $RC (log: $LOG)"

# ---- 6b. 传输判定 -----------------------------------------------------------
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

# ---- 7. 结果汇总与回拷 ------------------------------------------------------
if [ -f "$OUT" ]; then
  echo ""
  echo "--- Results summary (rank 0) ---"
  python3 - "$OUT" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
for q in ("q1", "q2", "q3", "q4"):
    if q not in r: continue
    print(f"[{q}]")
    for k, v in r[q].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        elif isinstance(v, dict) and "median_ms" in v:
            print(f"  {k}: {v['median_ms']:.3f} ms")
        elif isinstance(v, dict) and "per_ag_ms" in v:
            print(f"  {k}: per_ag {v['per_ag_ms']:.3f} ms (degradation {v['degradation_pct']:+.1f}%)")
PYEOF
  # 注入平台/传输上下文,让 JSON 自含(NVIDIA 侧拿到即可离线分析,无需再问环境)
  python3 - "$OUT" "$LOG" "$NET" <<'PYEOF'
import json, os, socket, subprocess, sys
out, log, net = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(out))
trans = subprocess.run(
    "grep -oE 'via NET/[A-Za-z0-9_]+' %s | sort | uniq -c" % log,
    shell=True, capture_output=True, text=True).stdout.strip()
try:
    import torch
    r["platform"] = {
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpus_per_node": 8,
        "nodes": 2,
    }
except Exception as e:
    r["platform"] = {"hostname": socket.gethostname(), "error": str(e)}
r["environment"] = {
    "net_mode": net,
    "transport_channels": trans,
    "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN", ""),
    "nccl_ib_hca": os.environ.get("NCCL_IB_HCA", ""),
    "nccl_topo_file": os.environ.get("NCCL_TOPO_FILE", ""),
    "kernel": os.uname().release,
    "log_file": os.path.basename(log),
    "note": ("rdma: RCCL external plugin librccl-net-shca -> NET/IBext_v8 (GPUDirect RDMA) "
             "over shca 400G NDR IB x4/node; tcp: NCCL_IB_DISABLE=1 socket-fallback control"
             if net == "rdma" else "socket-fallback control batch (NCCL_IB_DISABLE=1)"),
}
json.dump(r, open(out, "w"), indent=2, ensure_ascii=False)
print("(JSON enriched with platform/transport context)")
PYEOF
  for D in "/root/private_data/second_hygon/results" "$HOME/second_hygon/results" /mnt/public/home/*/second_hygon/results; do
    [ -d "$D" ] || continue
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
