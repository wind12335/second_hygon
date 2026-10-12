#!/bin/bash
# ============================================================================
# Phase 6 vLLM 双机 16 卡部署 · SCNet 模型训练模块启动脚本 (v1)
#
# 用法(模型训练模块 2 节点 x 8 卡,两个 worker 的"启动命令"填同一条):
#   MODEL_PATH=/root/private_data/thousand_model/model/Llama-3.1-8B \
#     bash /root/private_data/second_hygon/phase6-vllm-deploy/run_vllm_scnet.sh
#
# 流程:角色自动分角色(SCNet 注入的 RANK 优先) → vllm 自检(缺则离线装 wheel)
#   → shca RDMA 自动装配 → worker-0 起 ray head / worker-1 加入 →
#   worker-0 上 vllm serve(TP=16, ray 后端) → 健康检查 + 一条测试推理 →
#   (SMOKE=1 时)双端干净退出。日志与测试结果回拷 results/。
#
# 可选环境变量:
#   MODEL_PATH=<dir>    必填!HF 格式模型目录(含 config.json/权重文件)
#   TP=16               tensor parallel size(默认 16;单机验证用 8)
#   PP=1                pipeline parallel size
#   MAX_LEN=8192        --max-model-len
#   GPU_UTIL=0.90       --gpu-memory-utilization
#   PORT=8000           API 端口
#   NET=rdma|tcp|check  rdma=shca IB(默认) / tcp=socket 对照 / check=纯诊断 10 秒退出
#   SMOKE=1             1=验证模式:健康检查+一条测试推理后自动退出(默认);
#                       0=服务模式:一直挂着对外服务
#   PROFILE=0           1=健康后进入 torch profiler 采集:API 触发 /start_profile
#                       → 内置负载发生器打窗口 → /stop_profile,16 卡 per-rank
#                       trace(含 RCCL kernel 行)落共享存储(通信-重叠分析主力)
#   LOAD_CONC=16        负载发生器并发数(PROFILE=1 窗口1)
#   LOAD_SECONDS=30     采集窗口秒数(PROFILE=1 窗口1)
#   LOAD_MAXTOK=128     每请求生成 token 上限
#   PTOK=8000           窗口2 定长 prefill 的 token 数(M≈PTOK,与微基准 M 系列对齐;
#                       0=关掉窗口2);PTOK_CONC=4 PTOK_SECONDS=40 可调
#   PHASES="d:4:15,d:16:15,d:64:15,d:256:15,p:2048:10,p:8192:10,p:16384:10"
#                       M 轴扫描:单次 vLLM 启动内多相位采集(设了就替代窗口1/2,
#                       省 N-1 次模型加载)。d:conc:dur = 自然 decode 相位(M≈conc);
#                       p:ptok:dur = 定长 prefill 相位(M≈ptok,conc=PTOK_CONC)。
#                       每相位独立 warmup(不采集)→ start/stop_profile → 等
#                       EXPECT_GPU 个新 trace;新增文件名快照+起止时间增量写入
#                       trace 目录下 PHASES.json(分析按相位分组,防跨相位混窗)。
#                       注意:最大 ptok 必须 < MAX_LEN-16(p:16384 需 MAX_LEN≥16400)。
#   VLLM_WHEEL=<path>   vllm 缺失时的离线安装包(默认 /root/private_data 下自动找 cp311 wheel)
#   EXTRA_ARGS="..."    透传给 vllm serve(如 --dtype half)
# ============================================================================
set -uo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
TP="${TP:-16}"
PP="${PP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"
NET="${NET:-rdma}"
SMOKE="${SMOKE:-1}"
RAY_PORT=6379
# 期望的集群卡数 = TP*PP(双机 TP16=16;单机 TP8=8,ray 等待/trace 文件数都用它)
EXPECT_GPU=$(( TP * PP ))

# 模型上下文上限预检:MAX_LEN 超过 max_position_embeddings 时 vLLM 启动即拒
# (勿用 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 绕过——RoPE 越界位置会 NaN,数据作废)
MAXPOS=$(python3 -c "import json;print(json.load(open('${MODEL_PATH}/config.json')).get('max_position_embeddings',0))" 2>/dev/null || echo 0)
if [ "$MAXPOS" -gt 0 ] && [ "$MAX_LEN" -gt "$MAXPOS" ]; then
  echo "ERROR: MAX_LEN=${MAX_LEN} > 模型 max_position_embeddings=${MAXPOS}(config.json)"
  echo "       请 MAX_LEN≤${MAXPOS} 重跑;PHASES 的 prefill 相位 ptok 上限 = $((MAXPOS-16))"
  exit 1
fi

# ---- 1. 角色识别(SCNet 注入 RANK 优先,主机名兜底)——必须在 runid 握手之前 ----
if [ -n "${RANK:-}" ] && [ "$RANK" -lt 2 ] 2>/dev/null; then
  NODE_RANK=$RANK
elif [ -n "${SLURM_NODEID:-}" ]; then
  NODE_RANK="$SLURM_NODEID"
else
  HOSTSHORT=$(hostname -s)
  case "$HOSTSHORT" in
    *-worker-0) NODE_RANK=0 ;;
    *-worker-1) NODE_RANK=1 ;;
    *) echo "无法识别节点角色(RANK/主机名均失败)。请手动 NODE_RANK=0|1 bash $0"; exit 1 ;;
  esac
fi
echo "node_rank = $NODE_RANK"

# 运行 ID 握手:worker-0 落盘、worker-1 读取,保证两节点的日志/旗标文件名一致
RUNID_FILE="$LAUNCH_DIR/.phase6_runid"
if [ "$NODE_RANK" = "0" ]; then
  TS=$(date +%Y%m%d_%H%M%S); echo "$TS" > "$RUNID_FILE"
else
  TS=""
  for i in $(seq 1 90); do
    AGE=$(( $(date +%s) - $(stat -c %Y "$RUNID_FILE" 2>/dev/null || echo 0) ))
    [ "$AGE" -lt 300 ] && { TS=$(cat "$RUNID_FILE" 2>/dev/null); break; }
    sleep 2
  done
  [ -n "$TS" ] || { TS=$(date +%Y%m%d_%H%M%S); echo "WARN: 未等到 worker-0 的 runid,用自己的 TS(停机旗标可能不同名)"; }
fi
LOG="$LAUNCH_DIR/phase6_vllm_${NET}_${TS}_n${NODE_RANK}.log"   # TS 共享、按节点分日志
STOP_FLAG="$LAUNCH_DIR/.phase6_stop_${TS}"     # 共享存储上的停机旗标(worker-1 轮询)

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo " Phase6 vLLM dual-node serve (TP=$TP PP=$PP net=$NET smoke=$SMOKE rank=$NODE_RANK)"
echo " Host: $(hostname)   Date: $(date)"
echo " model: ${MODEL_PATH:-<未指定!>}"
echo "=============================================="

# ---- 2. 前置检查 ------------------------------------------------------------
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH 未设置或不是 HF 模型目录(缺 config.json): '${MODEL_PATH}'"
  echo "  例: MODEL_PATH=/root/private_data/thousand_model/model/Llama-3.1-8B bash $0"
  exit 1
fi
# 权重完整性预检:index 列出的分片必须齐全(下载未完会在加载 20 分钟后才失败,白烧机时)
python3 - "$MODEL_PATH" <<'PYEOF'
import glob, json, os, sys
mp = sys.argv[1]
idx = glob.glob(os.path.join(mp, "*.safetensors.index.json"))
if idx:
    shards = sorted(set(json.load(open(idx[0]))["weight_map"].values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(mp, s))]
    total = sum(os.path.getsize(os.path.join(mp, s)) for s in shards if os.path.exists(os.path.join(mp, s)))
    if missing:
        print(f"ERROR: 模型分片缺失 {len(missing)}/{len(shards)} 个(下载未完成?),如: {missing[:3]}")
        sys.exit(1)
    print(f"weights OK: {len(shards)} shards, {total/1e9:.1f} GB")
elif glob.glob(os.path.join(mp, "*.safetensors")) or glob.glob(os.path.join(mp, "*.bin")):
    print("weights OK: 单文件权重")
else:
    print("ERROR: 目录里找不到权重文件(.safetensors/.bin),下载还没开始?")
    sys.exit(1)
PYEOF
[ $? -ne 0 ] && exit 1
NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "visible GPUs = $NGPU (expect 8)"
[ "$NGPU" -lt 8 ] && { echo "ERROR: GPU 数量不足($NGPU < 8)。"; exit 1; }

# vllm 自检:缺失则离线装 wheel(--no-deps,镜像已带全套 das 适配依赖,勿让 pip 动它们)
if ! python3 -c "import vllm" >/dev/null 2>&1; then
  W="${VLLM_WHEEL:-$(ls /root/private_data/vllm-*cp311*.whl 2>/dev/null | head -1)}"
  echo "== vllm 缺失,离线安装: $W =="
  pip3 install --no-deps "$W" || { echo "ERROR: vllm 安装失败。"; exit 1; }
fi
echo "vllm $(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | ray $(ray --version 2>/dev/null | awk '{print $3}')"

# ---- 2a. DCU 适配补丁(三合一,幂等,保留 .orig 备份) ----
# 现象链:ray 数不出 DCU 的卡(--num-gpus=8 覆盖解决调度) → ray 给每个 worker 发单卡
# CUDA_VISIBLE_DEVICES(本地有效);但 das 版 vllm 执行器随后会用"全局卡号"掩码再次覆盖
# (worker-1 的 rank8-15 → "8"-"15",节点上不存在) → 进程可见设备被清空 → 任何
# cuda ordinal 都 "HIP error: invalid device ordinal"(两次实测均只挂 worker-1 的 rank)。
# P1: 拦截执行器的掩码覆盖,沿用 ray 的本地掩码(探测已证有效)
# P2: 按掩码实况把 local_rank 钳到 [0, 可见卡数)(单卡掩码 → 节点内 rank=0)
# P3: gpu_worker 选卡取模兜底
python3 - <<'PYEOF'
import os, pathlib
root = pathlib.Path(os.environ.get("VLLM_PATCH_ROOT",
    "/usr/local/lib/python3.11/site-packages/vllm"))
PATCHES = [
    ("worker/worker_base.py",
     "        envs = envs_list[self.rpc_rank]",
     "        envs = dict(envs_list[self.rpc_rank])\n"
     "        # [DCU patch] 执行器下发的 CUDA_VISIBLE_DEVICES 是全局卡号,节点上不存在,沿用 ray 本地掩码\n"
     "        envs.pop('CUDA_VISIBLE_DEVICES', None)"),
    ("worker/worker_base.py",
     "        kwargs = all_kwargs[self.rpc_rank]",
     "        kwargs = all_kwargs[self.rpc_rank]\n"
     "        # [DCU patch] 按进程可见掩码钳制 local_rank(ray 单卡掩码 → 节点内 rank=0)\n"
     "        _m = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
     "        _n = len([x for x in _m.split(',') if x.strip() != '']) if _m is not None else 0\n"
     "        if _n >= 1 and kwargs.get('local_rank') is not None:\n"
     "            kwargs['local_rank'] = int(kwargs['local_rank']) % _n"),
    ("v1/worker/gpu_worker.py",
     'torch.device(f"cuda:{self.local_rank}")',
     'torch.device(f"cuda:{self.local_rank % max(1, torch.cuda.device_count())}")'),
]
for sub, old, new in PATCHES:
    p = root / sub
    if not p.exists():
        print(f"[patch] SKIP: {p} 不存在"); continue
    s = p.read_text()
    if new in s:
        print(f"[patch] 已生效: {sub}"); continue
    if old not in s:
        print(f"[patch] WARN: {sub} 无目标模式(das 版本差异?)"); continue
    if not (root / (sub + ".orig")).exists():
        (root / (sub + ".orig")).write_text(s)
    p.write_text(s.replace(old, new, 1))
    print(f"[patch] 已修补 {sub}")
PYEOF

# ---- 3. 环境变量 ------------------------------------------------------------
ulimit -n 65535 2>/dev/null || true
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 权重走本地目录,禁止外网探测
export VLLM_HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
export NCCL_CUMEM_ENABLE=0                       # ROCm/DCU 上 cumem 与部分 kernel 不兼容,保守关
# torch profiler 输出目录:必须两节点都导出(ray 引擎继承各自节点 raylet 的环境,
# 只在 worker-0 导出会让 worker-1 的 8 个 rank 把 trace 写到 /tmp,pod 销毁即丢)
if [ "${PROFILE:-0}" = "1" ]; then
  export VLLM_TORCH_PROFILER_DIR="$LAUNCH_DIR/phase6_torch_prof_${NET}_${TS}"
  mkdir -p "$VLLM_TORCH_PROFILER_DIR"
fi
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  DEF_IF=$(ip route 2>/dev/null | awk '/^default/{print $5; exit}')
  [ -n "$DEF_IF" ] && export NCCL_SOCKET_IFNAME="$DEF_IF" && echo "NCCL_SOCKET_IFNAME=$DEF_IF (auto)"
fi

# ---- 3a. shca RDMA 用户态安装(共享存储,免网络,幂等) --------------------------
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
      echo "   安装完成: $(ls $LIBD/libshca-rdmav34.so 2>/dev/null || echo '异常')"
    else
      echo "   解包失败,继续用镜像自带(Mellanox)用户态"
    fi
  else
    echo "== 未找到 $SHCA_HOME/shca-tools_*.deb,跳过安装 =="
  fi
fi

# ---- 3b. 传输模式 -----------------------------------------------------------
if [ "$NET" != "tcp" ]; then
  echo "--- RDMA 自检(详细分层报告用 NET=check) ---"
  echo "providers: $(ls /usr/lib/x86_64-linux-gnu/libibverbs 2>/dev/null | tr '\n' ' ')"
  echo "ibv_devinfo -l:"; ibv_devinfo -l 2>&1 | sed 's/^/    /'
fi
[ "$NET" = "check" ] && { echo "(NET=check: 只诊断退出)"; exit 0; }
if [ "$NET" = "rdma" ]; then
  export NCCL_IB_DISABLE=0
  export NCCL_NET_PLUGIN=shca
  export NCCL_IB_HCA="${NCCL_IB_HCA:-shca_0:1,shca_1:1,shca_2:1,shca_3:1}"
  [ -f "$SHCA_HOME/topo_lib/built-in-508-topo-input-tj-default.xml" ] && \
    export NCCL_TOPO_FILE="$SHCA_HOME/topo_lib/built-in-508-topo-input-tj-default.xml"
  export LD_LIBRARY_PATH="$SHCA_HOME/topo_lib/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  echo "IB vars(平台配方): NET_PLUGIN=$NCCL_NET_PLUGIN HCA=$NCCL_IB_HCA"
else
  export NCCL_IB_DISABLE=1
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# ---- 4. ray 集群编排 --------------------------------------------------------
resolve_master() { getent hosts "$1" 2>/dev/null | awk '{print $1; exit}'; }
HOSTSHORT=$(hostname -s)
if [ -n "${MASTER_ADDR:-}" ] && [ "$NODE_RANK" != "0" ]; then
  HEAD_IP=$(resolve_master "$MASTER_ADDR" || echo "$MASTER_ADDR")
elif [ "$NODE_RANK" = "0" ]; then
  HEAD_IP=$(resolve_master "$HOSTSHORT" || hostname -I | awk '{print $1}')
else
  BASE="${HOSTSHORT%-worker-*}"; HEAD_IP=""
  for i in $(seq 1 45); do HEAD_IP=$(resolve_master "$BASE-worker-0" || true); [ -n "$HEAD_IP" ] && break; sleep 2; done
fi
[ -z "${HEAD_IP:-}" ] && { echo "ERROR: 解析不到 head 地址。"; exit 1; }
echo "ray head = $HEAD_IP:$RAY_PORT   VLLM_HOST_IP=$VLLM_HOST_IP"

ray stop -f >/dev/null 2>&1
# --num-gpus=8: 显式声明每节点 8 卡。ray 默认靠 rocm-smi 自动数 DCU 的卡,
# 数不出来就报 0 卡、16 卡永远凑不齐;显式覆盖与物理事实一致,自动探测正常时也无副作用。
if [ "$NODE_RANK" = "0" ]; then
  ray start --head --port="$RAY_PORT" --include-dashboard=false --num-gpus=8 \
    --resources="{\"node\": 1}" >/dev/null 2>&1 || { echo "ERROR: ray head 启动失败"; exit 1; }
  echo "ray head started. (--num-gpus=8 显式声明)"
else
  for i in $(seq 1 120); do
    timeout 2 bash -c "exec 3<>/dev/tcp/${HEAD_IP}/${RAY_PORT}" 2>/dev/null && break
    sleep 2
  done
  ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8 >/dev/null 2>&1 || { echo "ERROR: ray worker 加入失败"; exit 1; }
  echo "ray worker joined head."
fi

if [ "$NODE_RANK" = "0" ]; then
  # 等全部卡上集群(每轮约 20~25s:import ray 本身就慢,属正常;打印进度防"看起来卡住")
  for i in $(seq 1 60); do
    G=$(python3 -c "import ray;ray.init(address='auto',logging_level='ERROR');print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null)
    echo "  [ray-wait ${i}/60] cluster GPUs = ${G:-?}/${EXPECT_GPU}   ($(date '+%T'))"
    [ "${G:-0}" -ge "$EXPECT_GPU" ] && { echo "ray cluster ready: ${G} GPUs."; break; }
    sleep 5
  done
  if [ "${G:-0}" -lt "$EXPECT_GPU" ]; then
    echo "ERROR: ray 集群 GPU 不足(${G:-?}/${EXPECT_GPU})。ray status:"; ray status 2>/dev/null | head -15
    touch "$STOP_FLAG"; ray stop -f >/dev/null 2>&1; exit 1
  fi

  # ---- 4a. ray worker GPU 掩码探测(诊断 invalid device ordinal 类问题) ----
  echo "--- ray worker GPU mask probe ---"
  python3 - <<'PYEOF' 2>&1 | tail -8
import ray
ray.init(address='auto', logging_level='ERROR')

@ray.remote(num_gpus=1)
def probe():
    import os, torch
    return {"ip": ray.util.get_node_ip_address(),
            "CUDA": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "ROCR": os.environ.get("ROCR_VISIBLE_DEVICES", "<unset>"),
            "HIP": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
            "n_dev": torch.cuda.device_count()}

for r in ray.get([probe.remote() for _ in range(2)]):
    print("probe:", r)
PYEOF

  # ---- 5. vllm serve --------------------------------------------------------
  [ -n "${VLLM_TORCH_PROFILER_DIR:-}" ] && echo "torch profiler: traces -> $VLLM_TORCH_PROFILER_DIR"
  export RAY_ADDRESS="auto"
  # DTK torch2.7.1 的 dynamo 编译不了 vllm parameter.__torch_function__(dummy run 即崩),
  # 默认 --enforce-eager 跳过;EAGER=0 可关掉对照
  EAGER_ARGS=""
  [ "${EAGER:-1}" = "1" ] && EAGER_ARGS="--enforce-eager"
  echo "--- vllm serve: TP=$TP PP=$PP port=$PORT max_len=$MAX_LEN util=$GPU_UTIL eager=${EAGER:-1} ---"
  SERVE_LOG="$LAUNCH_DIR/phase6_vllm_serve_${NET}_${TS}.log"
  vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --disable-log-requests \
    ${EAGER_ARGS} ${EXTRA_ARGS:-} > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!
  echo "vllm serve pid=$SERVE_PID (serve log: $SERVE_LOG)"

  # ---- 6. 健康检查(大模型加载慢,最多 40 分钟) -------------------------------
  OK=0
  for i in $(seq 1 240); do
    kill -0 $SERVE_PID 2>/dev/null || { echo "ERROR: vllm serve 提前退出!最后 30 行:"; tail -30 "$SERVE_LOG" | tee -a "$LOG"; break; }
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)
    [ "$CODE" = "200" ] && { OK=1; echo "API ready (round $i)."; break; }
    sleep 10
  done

  if [ "$OK" = "1" ]; then
    echo "--- transport check (serve log + ray session logs) ---"
    grep -hoE 'via NET/[A-Za-z0-9_]+' "$SERVE_LOG" /tmp/ray/session_latest/logs/*.err 2>/dev/null | sort | uniq -c | sort -rn | head -5
    if grep -hq 'NET/Socket' "$SERVE_LOG" /tmp/ray/session_latest/logs/*.err 2>/dev/null; then
      echo "transport: ✗ 含 NET/Socket(TCP 回退)"
    else
      echo "transport: ✓✓ 全部通道走 RDMA"
    fi
    # ---- M 轴扫描:单次启动内多相位采集(PHASES 设了就替代窗口1/2) ----
    if [ "${PROFILE:-0}" = "1" ] && [ -n "${PHASES:-}" ]; then
      echo "--- M-sweep 多相位 profiler 窗口(PHASES=$PHASES) ---"
      # 预检:prefill 相位 ptok 必须 < MAX_LEN-16(prompt+生成 token 放得下)
      for _ph in ${PHASES//,/ }; do
        IFS=: read -r _k _v _d <<< "$_ph"
        if [ "$_k" = "p" ] && [ "${_v:-0}" -ge $(( MAX_LEN - 16 )) ] 2>/dev/null; then
          echo "ERROR: 相位 ${_ph} 的 ptok=${_v} ≥ MAX_LEN-16=$((MAX_LEN-16)),prompt 放不下;"
          echo "       请加大 MAX_LEN(如 MAX_LEN=$(( _v + 256 )))后重跑"
          exit 1
        fi
      done
      _tot=$(awk -F, '{print NF}' <<< "$PHASES")
      MANIFEST="$VLLM_TORCH_PROFILER_DIR/PHASES.json"
      echo "[" > "$MANIFEST"; _first=1; _prevn=0; _idx=0
      for _ph in ${PHASES//,/ }; do
        IFS=: read -r _k _v _dur <<< "$_ph"
        _idx=$((_idx+1))
        [ -z "${_dur:-}" ] && _dur=15
        if [ "$SMOKE" = "1" ]; then _dur=5; _warm=2; else
          if [ "$_k" = "d" ]; then _warm=8; else _warm=4; fi
        fi
        if [ "$_k" = "d" ]; then
          _conc=$_v; _mode=natural; _ptok=0
          echo "--- [phase $_idx/$_tot] decode conc=$_v dur=${_dur}s (M≈$_v) ---"
        else
          _conc="${PTOK_CONC:-4}"; _mode=fixed; _ptok=$_v
          echo "--- [phase $_idx/$_tot] prefill ptok=$_v dur=${_dur}s conc=$_conc (M≈$_v) ---"
        fi
        echo "warmup ${_warm}s(不采集;fill 批/清请求队列) ..."
        python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" \
          "$_conc" "$_warm" "${LOAD_MAXTOK:-128}" "$_mode" "$_ptok" $((20261009+_idx)) 2>/dev/null || true
        _t0=$(date +%s%6N)
        _r=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" \
             -H 'Content-Type: application/json' -d '{"activities":["CPU","GPU"]}')
        [ -z "$_r" ] && _r=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile")
        echo "start_profile: $_r"
        python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" \
          "$_conc" "$_dur" "${LOAD_MAXTOK:-128}" "$_mode" "$_ptok" $((20261009+_idx)) || true
        curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" >/dev/null; echo "stop_profile sent"
        _t1=$(date +%s%6N)
        echo "waiting trace flush(先 15s,再等文件数稳定且较上相位 +${EXPECT_GPU},最多再 2 分钟) ..."
        sleep 15; _p=-1
        for _i in $(seq 1 24); do
          _n=$(ls "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null | grep -v '^PHASES\.json$' | wc -l)
          { [ "$_n" = "$_p" ] && [ "$_n" -ge $((_prevn + EXPECT_GPU)) ]; } && break
          _p=$_n; sleep 5
        done
        _newn=$((_n - _prevn)); _prevn=$_n
        # 本相位新增 trace(mtime 最新优先)记入 manifest;分析按 files 分组防混窗
        _files=$(ls -t "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null | grep -v '^PHASES\.json$' \
                 | head -n "$_newn" | sed 's/^/"/; s/$/"/' | paste -sd, -)
        [ "$_first" = "0" ] && echo "," >> "$MANIFEST"
        _first=0
        printf '  {"idx": %d, "kind": "%s", "conc": %d, "ptok": %d, "dur_s": %d, "t0_us": %s, "t1_us": %s, "n_new_files": %d, "files": [%s]}\n' \
          "$_idx" "$_k" "$_conc" "$_ptok" "$_dur" "$_t0" "$_t1" "$_newn" "$_files" >> "$MANIFEST"
        echo "phase $_idx done: +$_newn trace files(累计 $_n)"
      done
      echo "]" >> "$MANIFEST"
      echo "phase manifest -> $MANIFEST"
    fi

    if [ "${PROFILE:-0}" = "1" ] && [ -z "${PHASES:-}" ]; then
      echo "--- torch profiler window (vLLM /start_profile + loadgen) ---"
      echo "warmup ..."
      python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" 1 4 32 2>/dev/null || true
      R1=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" \
           -H 'Content-Type: application/json' -d '{"activities":["CPU","GPU"]}')
      [ -z "$R1" ] && R1=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile")
      echo "start_profile: $R1"
      python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" \
        "${LOAD_CONC:-16}" "${LOAD_SECONDS:-30}" "${LOAD_MAXTOK:-128}" || true
      curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" >/dev/null; echo "stop_profile sent"
      echo "waiting trace flush(先 30s,再等 16 个 rank 的文件数稳定,最多再 2 分钟) ..."
      sleep 30; PREV=-1
      for i in $(seq 1 24); do
        N=$(ls "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null | wc -l)
        { [ "$N" = "$PREV" ] && [ "$N" -ge "$EXPECT_GPU" ]; } && break
        PREV=$N; sleep 5
      done
      echo "trace files: ${N:-?} (窗口1 自然负载)"

      # ---- 窗口2:定长受控 prefill(token-id prompt,M≈PTOK,与微基准 M 系列可比) ----
      if [ "${PTOK:-8000}" != "0" ]; then
        echo "--- window2: fixed-shape prefill (ptok=${PTOK:-8000}, conc=${PTOK_CONC:-4}) ---"
        R2=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile" \
             -H 'Content-Type: application/json' -d '{"activities":["CPU","GPU"]}')
        [ -z "$R2" ] && R2=$(curl -s -X POST "http://127.0.0.1:${PORT}/start_profile")
        echo "start_profile(2): $R2"
        python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" \
          "${PTOK_CONC:-4}" "${PTOK_SECONDS:-40}" 8 fixed "${PTOK:-8000}" || true
        curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile" >/dev/null; echo "stop_profile(2) sent"
        sleep 30; PREV=-1
        for i in $(seq 1 24); do
          N2=$(ls "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null | wc -l)
          { [ "$N2" = "$PREV" ] && [ "$N2" -ge "$N" ]; } && break
          PREV=$N2; sleep 5
        done
        echo "trace files: ${N2:-?} (窗口1+2)"; ls -la "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null | head -22
      fi
    fi

    echo "--- test completion ---"
    HAS_CHAT=$(python3 -c "
import json
try: print('yes' if json.load(open('$MODEL_PATH/tokenizer_config.json')).get('chat_template') else 'no')
except Exception: print('no')")
    if [ "$HAS_CHAT" = "yes" ]; then
      curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_PATH\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}]}" \
        | tee "$LAUNCH_DIR/phase6_test_completion_${TS}.json"
    else
      echo "(基座模型无 chat template,改用 /v1/completions)"
      curl -s "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_PATH\",\"max_tokens\":64,\"prompt\":\"Introduce yourself in one sentence:\"}" \
        | tee "$LAUNCH_DIR/phase6_test_completion_${TS}.json"
    fi
    echo ""
  fi

  # ---- 7. 收尾 --------------------------------------------------------------
  touch "$STOP_FLAG"    # 通知 worker-1 退出
  if [ "$SMOKE" = "1" ]; then
    echo "(SMOKE=1: 验证完毕,停服退出)"
    kill $SERVE_PID 2>/dev/null; sleep 8; kill -9 $SERVE_PID 2>/dev/null
    ray stop -f >/dev/null 2>&1
  else
    echo "(SMOKE=0: 服务模式,前台等待;平台停止作业即结束)"
    wait $SERVE_PID
  fi
  RC=$?
  for D in "/root/private_data/second_hygon/results" "$HOME/second_hygon/results"; do
    [ -d "$D" ] && { cp -v "$LOG" "$SERVE_LOG" "$LAUNCH_DIR"/phase6_test_completion_${TS}.json "$D"/ 2>/dev/null; break; }
  done
  exit $RC
else
  # worker-1: 引擎进程由 ray 调度,这里只保活并等停机旗标
  echo "(worker-1: ray 已加入,引擎由 head 调度;轮询停机旗标保活)"
  while [ ! -f "$STOP_FLAG" ]; do
    sleep 15
    [ $(( $(date +%s) % 300 )) -lt 15 ] && echo "  [$(date '+%T')] worker-1 alive, ray procs: $(pgrep -c -f raylet || echo 0)"
  done
  echo "(worker-1: 收到停机旗标,退出)"
  ray stop -f >/dev/null 2>&1
  exit 0
fi
