#!/bin/bash
# ============================================================================
# Phase 6-B hipprof 采集 · 单节点 8 卡 vLLM(对标 nsys 时间线 + ncu PMC 计数器)
#
# 【拓扑】模型训练模块提交 1 节点 x 8 卡作业(不是双机!)。vLLM 用 mp 后端,
#         引擎进程是 vllm 的子进程,hipprof --follow-fork 能完整包住。
# 【双机说明】双机 16 卡的 kernel 级数据用 run_vllm_scnet.sh 的 PROFILE=1
#         (vLLM 原生 torch profiler,per-rank trace 含 RCCL kernel 行);
#         本脚本负责 hipprof 独有的系统级视图:RCCL API 统计、kernel 栈、PMC。
#
# 用法(作业启动命令):
#   MODEL_PATH=/root/private_data/thousand_model/model/LLAMA3-70B \
#     bash /root/private_data/second_hygon/phase6-vllm-deploy/run_vllm_hipprof.sh
#
# 可选环境变量:
#   MODEL_PATH   必填(HF 模型目录)
#   TP=8         单节点内 TP
#   MAX_LEN=8192 / GPU_UTIL=0.90 / PORT=8000
#   HIPPROF_MODE=trace|pmc|session
#     trace   全程时间线(默认;含模型加载,trace 文件较大)
#     pmc     时间线+硬件计数器(每 kernel 重放,很慢,窗口自动缩短)
#     session 动态窗口:--trace-off 启动,健康后 --start/--stop 只采负载段(小而准)
#   HIPPROF_KERNEL=<name>    pmc 模式的 kernel 过滤(如 'gemm';空=全部,极慢)
#   LOAD_SECONDS=30           采集窗口秒数(pmc 时默认 10)
#   EXTRA_ARGS="..."          透传 vllm serve
# ============================================================================
set -uo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
TP="${TP:-8}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"
HIPPROF_MODE="${HIPPROF_MODE:-trace}"
HIPPROF_KERNEL="${HIPPROF_KERNEL:-}"
[ "$HIPPROF_MODE" = "pmc" ] && LOAD_SECONDS="${LOAD_SECONDS:-10}" || LOAD_SECONDS="${LOAD_SECONDS:-30}"
# finalize 预算提前定义:外层 timeout 必须覆盖它,否则调大 FINALIZE_WAIT 后 timeout 会在
# finalize 中途杀 hipprof,重演丢数据事故
FW=${FINALIZE_WAIT:-900}

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$LAUNCH_DIR/phase6_hipprof_${HIPPROF_MODE}_${TS}"
LOG="$LAUNCH_DIR/phase6_hipprof_${HIPPROF_MODE}_${TS}.log"
mkdir -p "$OUTDIR"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo " Phase6-B hipprof collection (mode=$HIPPROF_MODE tp=$TP)"
echo " Host: $(hostname)   Date: $(date)"
echo " model: ${MODEL_PATH:-<未指定!>}   out: $OUTDIR"
echo "=============================================="

command -v hipprof >/dev/null || { echo "ERROR: hipprof 不在 PATH(需要 DTK 镜像)。"; exit 1; }
echo "hipprof: $(command -v hipprof)"

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH 未设置或不是 HF 模型目录。"; exit 1
fi
python3 - "$MODEL_PATH" <<'PYEOF'
import glob, json, os, sys
mp = sys.argv[1]
idx = glob.glob(os.path.join(mp, "*.safetensors.index.json"))
if idx:
    shards = sorted(set(json.load(open(idx[0]))["weight_map"].values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(mp, s))]
    if missing:
        print(f"ERROR: 分片缺失 {len(missing)}/{len(shards)}(下载未完成?): {missing[:3]}"); sys.exit(1)
    print(f"weights OK: {len(shards)} shards")
elif glob.glob(os.path.join(mp, "*.safetensors")) or glob.glob(os.path.join(mp, "*.bin")):
    print("weights OK")
else:
    print("ERROR: 无权重文件"); sys.exit(1)
PYEOF
[ $? -ne 0 ] && exit 1

NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "visible GPUs = $NGPU (expect 8)"; [ "$NGPU" -lt 8 ] && { echo "ERROR: GPU 不足。"; exit 1; }

# vllm 自检:缺失则离线装 wheel(镜像一般已预装)
if ! python3 -c "import vllm" >/dev/null 2>&1; then
  W="${VLLM_WHEEL:-$(ls /root/private_data/vllm-*cp311*.whl 2>/dev/null | head -1)}"
  echo "== vllm 缺失,离线安装: $W =="
  pip3 install --no-deps "$W" || { echo "ERROR: vllm 安装失败。"; exit 1; }
fi

ulimit -n 65535 2>/dev/null || true
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO        # 日志里保留通道证据(单机也可能有 NET/P2P 通道行)
export NCCL_CUMEM_ENABLE=0    # 与双机脚本保持一致

# hipprof 参数组装(-o 必须带结尾斜杠表示目录,否则会被当文件名)
# KSTACK=0 可去掉 --kernel-stack(finalize 事务显著变小,导出更快;stacks 只用于归因,不影响时间线)
HPARGS="--follow-fork --rccl-trace --hip-trace --output-type 0 -o $OUTDIR/"
[ "${KSTACK:-1}" = "1" ] && HPARGS="--kernel-stack $HPARGS"
SESSION_NAME=""
if [ "$HIPPROF_MODE" = "pmc" ]; then
  HPARGS="--pmc $HPARGS"
  [ -n "$HIPPROF_KERNEL" ] && HPARGS="--kernel-name $HIPPROF_KERNEL $HPARGS"
elif [ "$HIPPROF_MODE" = "session" ]; then
  SESSION_NAME="prof6_${TS}"
  HPARGS="--session $SESSION_NAME --trace-off $HPARGS"
fi
echo "hipprof args: $HPARGS"

# ---- 启动被包裹的 vllm(mp 后端,引擎为子进程) --------------------------------
SERVE_LOG="$LAUNCH_DIR/phase6_hipprof_serve_${TS}.log"
# 与双机脚本一致:DTK torch 编译路径不可用,默认 eager
EAGER_ARGS=""
[ "${EAGER:-1}" = "1" ] && EAGER_ARGS="--enforce-eager"
echo "--- vllm serve (mp, TP=$TP, eager=${EAGER:-1}) under hipprof ---"
timeout $((FW + 2700)) hipprof $HPARGS \
  vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --distributed-executor-backend mp \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --disable-log-requests \
    ${EAGER_ARGS} ${EXTRA_ARGS:-} > "$SERVE_LOG" 2>&1 &
HP_PID=$!
echo "hipprof pid=$HP_PID (serve log: $SERVE_LOG)"

# ---- 健康检查(最多 40 分钟) --------------------------------------------------
OK=0
for i in $(seq 1 240); do
  kill -0 $HP_PID 2>/dev/null || { echo "ERROR: hipprof/vllm 提前退出!最后 30 行:"; tail -30 "$SERVE_LOG"; break; }
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)
  [ "$CODE" = "200" ] && { OK=1; echo "API ready (round $i)."; break; }
  sleep 10
done
if [ "$OK" != "1" ]; then
  # 同样不能直接杀 hipprof(会丢全部 trace):先 TERM 子进程,给它 5 分钟 finalize
  for p in $(pgrep -P $HP_PID 2>/dev/null); do kill -TERM $p 2>/dev/null; done
  for i in $(seq 1 60); do kill -0 $HP_PID 2>/dev/null || break; sleep 5; done
  kill -KILL $HP_PID 2>/dev/null
  echo "=== 收尾:让 hipprof 导出已有数据 ==="
  ls -la "$OUTDIR"; cp -v "$SERVE_LOG" "$OUTDIR"/ 2>/dev/null
  exit 1
fi

# ---- 采集窗口:负载 ----------------------------------------------------------
if [ -n "$SESSION_NAME" ]; then
  echo "--- session window start ($SESSION_NAME) ---"
  hipprof --session-client "$SESSION_NAME" --start 2>&1 | tail -3
fi
echo "--- loadgen window ${LOAD_SECONDS}s (pmc 模式下进程内 kernel 重放,慢属正常) ---"
python3 "$LAUNCH_DIR/loadgen.py" 127.0.0.1 "$PORT" "$MODEL_PATH" 8 "$LOAD_SECONDS" 128 || true
if [ -n "$SESSION_NAME" ]; then
  hipprof --session-client "$SESSION_NAME" --stop 2>&1 | tail -3
  hipprof --session-client "$SESSION_NAME" --flush 2>&1 | tail -3
  sleep 20
fi

# ---- 停止并导出(v2,2026-10-11 依据 ④⑤ 丢数据事故重写) ----------------------
# 事故根因:hipprof 的事件数据全程在一个未提交的 SQL 大事务里,只有被包裹的 app 退出、
# hipprof 自己走 finalize 才会 commit+导出。旧版对 hipprof 发 INT/TERM/KILL = finalize
# 中途被杀 = 1.4GB 数据全部滞留在未提交事务里,官方读取器(--db)也救不回。
# 新逻辑:只 TERM 子进程(vllm serve),绝不打断 hipprof,耐心等它自然退出。
echo "stopping vllm (只 TERM 子进程,让 hipprof 自然 finalize) ..."
if [ -n "$SESSION_NAME" ]; then
  hipprof --session-client "$SESSION_NAME" --exit 2>&1 | tail -3
fi
APP_PIDS=$(pgrep -P $HP_PID 2>/dev/null)
[ -z "$APP_PIDS" ] && APP_PIDS=$(pgrep -f "vllm serve $MODEL_PATH" 2>/dev/null | head -5)
for p in $APP_PIDS; do kill -TERM $p 2>/dev/null; done
sleep 30
for p in $APP_PIDS; do kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null; done

# 等 hipprof 自己退出:commit 大事务+写 json 可能要几分钟(FW=FINALIZE_WAIT,头部已定义)
echo "waiting hipprof finalize (最多 ${FW}s;journal 消失/导出文件出现即接近完成) ..."
for i in $(seq 1 $((FW / 10))); do
  kill -0 $HP_PID 2>/dev/null || { echo "hipprof exited after ~$((i * 10))s"; break; }
  [ $((i % 6)) -eq 0 ] && echo "  [finalize $((i * 10))s] outdir: $(ls "$OUTDIR" 2>/dev/null | tr '\n' ' ')"
  sleep 10
done
if kill -0 $HP_PID 2>/dev/null; then
  echo "WARN: hipprof ${FW}s 仍未退出——最后手段 KILL(可能丢数据,可考虑调大 FINALIZE_WAIT)"
  kill -KILL $HP_PID 2>/dev/null; sleep 10
fi
echo "=== hipprof outputs ==="
ls -la "$OUTDIR" | head -30
cp -v "$SERVE_LOG" "$OUTDIR"/ 2>/dev/null
JSONS=$(ls "$OUTDIR"/*.json "$OUTDIR"/*.pftrace 2>/dev/null | wc -l)
JRN=$(ls "$OUTDIR"/*.db-journal 2>/dev/null | wc -l)
echo "exported json/pftrace: $JSONS 个;残留 db-journal: $JRN 个"
if [ "$JSONS" -eq 0 ]; then
  echo "WARN: 无导出文件——finalize 未完成,数据大概率不可读(判定依据:残留 journal)。"
  echo "      下次:① FINALIZE_WAIT=1800 调大;② KSTACK=0 去掉 kernel-stack 减小事务;"
  echo "            ③ HIPPROF_MODE=session 动态窗口试 --exit 正常收尾"
fi

# ---- 汇总回拷 ---------------------------------------------------------------
for D in "/root/private_data/second_hygon/results" "$HOME/second_hygon/results"; do
  [ -d "$D" ] && { cp -rv "$OUTDIR" "$D"/ 2>/dev/null; break; }
done
echo "done. outputs: $OUTDIR (json 时间线可在 perfetto.dev / chrome://tracing 打开;"
echo "      *.stack.csv = kernel 调用栈;pmc 模式另有计数器表,--pmc-type 3 可导 csv)"
