#!/bin/bash
# ============================================================================
# D·decode 形状 GEMM-AR 微基准(paper2 机制验证,单机 8 卡 vcjob)
#
# 目的:
#   D1 物理验证 E1 的 k*=1(chunk 化在粘性 straggler 下从第 2 块起净负)
#   D2 标定 M4 分域规则 α(GEMM_chunk_time vs δ 的交点)
#   D3 量化 M3(FILL:等待窗双流注入 filler)与 M2/M3 混合(PARTIAL:预测驱动
#      预规约,等待期先做 7-rank AR,把死等换成 wire)的收益曲线
#
# 机制(网格 M × δ × mech):
#   BASE    k=1 整块 GEMM→AR(基线,含全部等待)
#   CHUNK   k∈{2,4,8} 行分块 GEMM→逐块 AR(paper1 机制在 decode 小 M 下)
#   FILL    k=1,AR 上 comm 流 + filler GEMM 上计算流(M3:等待窗注入)
#   PARTIAL k=1,非 straggler 先做排除 straggler 的 7-rank AR,等 straggler
#           到达后 bcast 其 partial 并相加,再由非 straggler bcast final
#           (M2/M3 混合:把 min(δ, t_AR7) 的等待换成有用 wire)
#   straggler 注入:按 A 实测结构(粘性 RUNLEN=80 + SUBSET 轮换),host 侧
#   spin 延迟(B2 证实真实成因就是 host 侧断粮)
#
# 用法:
#   SMOKE=1 bash /root/private_data/second_hygon/phaseD/run_d_microbench_scnet.sh  # 冒烟
#   bash /root/private_data/second_hygon/phaseD/run_d_microbench_scnet.sh          # 正式(~10min)
# 输出: /root/private_data/second_hygon/results/d_microbench_<ts>/(JSON+日志)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="/root/private_data/second_hygon/phaseD"
mkdir -p "$SCRIPT_DIR"
export HSA_FORCE_FINE_GRAIN_PCIE=1

# 平台双 worker 同命令:仅 worker-0 执行,worker-1 挂起等待(单机实例不会进来)
if [ "${RANK:-0}" != "0" ]; then
  echo "[D] RANK=${RANK} 非 worker-0,挂起 2h"
  sleep 7200
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="/root/private_data/second_hygon/results/d_microbench_${TS}"
mkdir -p "$OUTDIR"
export SMOKE="${SMOKE:-0}"
export D_OUTDIR="$OUTDIR"

# ---- 内嵌测量核心(幂等提取) ----
cat > "$SCRIPT_DIR/d_bench_core.py" <<'PYEOF'
#!/usr/bin/env python3
"""D·decode 形状 GEMM-AR 微基准测量核心(torchrun 8 进程,RCCL)。"""
import os, json, time, sys
import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", RANK))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUTDIR = os.environ.get("D_OUTDIR", "/tmp/d_microbench")

torch.cuda.set_device(LOCAL)
dist.init_process_group("nccl")  # DTK torch: nccl → RCCL
dev = torch.device(f"cuda:{LOCAL}")
torch.manual_seed(1000 + RANK)

MS = [8, 16, 32, 64, 128, 256]
KS = [2, 4, 8]
DELTAS = [0, 250, 500, 1000]
NFILLS = [1, 2, 4]
ITERS, WARM = 128, 8
SUBSET, RUNLEN = [1, 3, 5, 7], 80
if SMOKE:
    MS, KS, DELTAS, NFILLS = [16], [2], [0, 500], [1]
    ITERS, WARM = 16, 4
if WORLD < 8:
    SUBSET = []  # 本机 1 卡冒烟:无注入无 PARTIAL
H, I = 8192, 3584  # Llama3-70B TP8 down_proj:(M,I)x(I,H) → AR(M×H)


def log(msg):
    if RANK == 0:
        print(f"[D] {msg}", flush=True)


def straggler_of(t):
    return SUBSET[(t // RUNLEN) % len(SUBSET)] if SUBSET else -1


def spin(us):
    if us <= 0:
        return
    end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < end:
        pass


G7 = {}
if WORLD >= 8 and SUBSET:
    for s in SUBSET:
        G7[s] = dist.new_group([r for r in range(WORLD) if r != s])
log(f"world={WORLD} SUBSET={SUBSET} RUNLEN={RUNLEN} SMOKE={SMOKE}")

comm_s = torch.cuda.Stream()


def chain_iter(mech, M, k, n_fill, delta_us, t, A, W, out, buf, F1, F2, Fout):
    """单次链,返回本 rank elapsed(µs)。迭代间 barrier 对齐,注入在链前。"""
    dist.barrier()
    s = straggler_of(t)
    if RANK == s:
        spin(delta_us)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    if mech == "BASE":
        torch.mm(A, W, out=out)
        dist.all_reduce(out)
    elif mech == "CHUNK":
        csz = M // k
        for c in range(k):
            a0, a1 = c * csz, (c + 1) * csz
            torch.mm(A[a0:a1], W, out=out[a0:a1])
            dist.all_reduce(out[a0:a1])
    elif mech == "FILL":
        ev_g = torch.cuda.Event()
        ev_ar = torch.cuda.Event()
        torch.mm(A, W, out=out)
        ev_g.record()
        with torch.cuda.stream(comm_s):
            comm_s.wait_event(ev_g)
            dist.all_reduce(out)
            ev_ar.record()
        for _ in range(n_fill):
            torch.mm(F1, F2, out=Fout)  # filler,结果丢弃
        torch.cuda.current_stream().wait_event(ev_ar)
    elif mech == "PARTIAL":
        torch.mm(A, W, out=out)
        q = next(r for r in range(WORLD) if r != s)
        if RANK == s:
            dist.broadcast(out, src=s)        # 发自己的 partial
            dist.broadcast(out, src=q)        # 收 final(覆盖)
        else:
            dist.all_reduce(out, group=G7[s])  # 等待期先做 7-rank 规约
            dist.broadcast(buf, src=s)         # 收 straggler partial
            out += buf
            dist.broadcast(out, src=q)         # q 广播 final(其余覆盖为同值)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0


def timed(fn, iters, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time() * 1000.0 / iters  # µs/次


def verify(mech, M, k, A, W, out, buf, F1, F2, Fout):
    """BASE/CHUNK/FILL/PARTIAL 各验一次(δ=0,t=0,世界 AR fp32 为参考)。"""
    pf = A.float() @ W.float()
    ref = pf.clone()
    dist.all_reduce(ref)
    chain_iter(mech, M, k, 1, 0, 0, A, W, out, buf, F1, F2, Fout)
    ok = torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
    tag = "PASS" if ok else "FAIL"
    if not ok or RANK == 0:
        print(f"[D][verify] M={M} mech={mech} k={k} rank={RANK} {tag}", flush=True)
    return bool(ok)


res = {"meta": {"world": WORLD, "iters": ITERS, "warm": WARM, "subset": SUBSET,
                "runlen": RUNLEN, "smoke": SMOKE, "H": H, "I": I,
                "nccl_algo": os.environ.get("NCCL_ALGO", "(default)")},
       "shapes": {}, "cells": {}}

for M in MS:
    A = torch.randn(M, I, device=dev, dtype=torch.bfloat16)
    W = torch.randn(I, H, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    buf = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    F1 = torch.randn(256, I, device=dev, dtype=torch.bfloat16)
    F2 = torch.randn(I, H, device=dev, dtype=torch.bfloat16)
    Fout = torch.empty(256, H, device=dev, dtype=torch.bfloat16)

    tg = timed(lambda: torch.mm(A, W, out=out), 64, 8)
    ta = timed(lambda: dist.all_reduce(out), 64, 8)
    res["shapes"][M] = {"gemm_us": round(tg, 1), "ar_us": round(ta, 1),
                        "ar_msg_kb": round(M * H * 2 / 1024, 1)}
    log(f"M={M}: GEMM {tg:.1f}µs | AR {ta:.1f}µs | msg {M*H*2//1024}KB")

    for mech, k in [("BASE", 1)] + [("CHUNK", kk) for kk in KS]:
        verify(mech, M, k, A, W, out, buf, F1, F2, Fout)
    if WORLD >= 8 and SUBSET:
        verify("PARTIAL", M, 1, A, W, out, buf, F1, F2, Fout)
    verify("FILL", M, 1, A, W, out, buf, F1, F2, Fout)

    for delta in DELTAS:
        cells = [("BASE", 1, 0)] + [("FILL", 1, n) for n in NFILLS]
        cells += [("CHUNK", kk, 0) for kk in KS]
        if WORLD >= 8 and SUBSET:
            cells.append(("PARTIAL", 1, 0))
        for mech, k, n in cells:
            for t in range(WARM):
                chain_iter(mech, M, k, n, delta, t, A, W, out, buf, F1, F2, Fout)
            el = [chain_iter(mech, M, k, n, delta, WARM + t, A, W, out, buf, F1, F2, Fout)
                  for t in range(ITERS)]
            el_t = torch.tensor(el, device=dev)
            parts = [torch.zeros(ITERS, device=dev) for _ in range(WORLD)]
            dist.all_gather(parts, el_t)
            if RANK == 0:
                emax = torch.stack(parts).max(0).values
                res["cells"][f"M{M}|d{delta}|{mech}|k{k}|n{n}"] = {
                    "p50_us": round(emax.median().item(), 1),
                    "p90_us": round(torch.quantile(emax, 0.9).item(), 1),
                    "mean_us": round(emax.mean().item(), 1)}
            dist.barrier()
    # 注入有效性:BASE(δ=500)−BASE(δ=0) 应 ≈ 500µs
    if 500 in DELTAS and RANK == 0:
        b0 = res["cells"].get(f"M{M}|d0|BASE|k1|n0")
        b5 = res["cells"].get(f"M{M}|d500|BASE|k1|n0")
        if b0 and b5:
            log(f"M={M}: 有效注入 δ_eff ≈ {b5['p50_us']-b0['p50_us']:.0f}µs(目标 500)")

if RANK == 0:
    with open(os.path.join(OUTDIR, "d_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[D] -> {OUTDIR}/d_result.json", flush=True)
dist.barrier()
dist.destroy_process_group()
PYEOF

echo "[D] SMOKE=${SMOKE} OUTDIR=$OUTDIR  启动 torchrun 8 进程..."
torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port=23557 \
  "$SCRIPT_DIR/d_bench_core.py" 2>&1 | tee "$OUTDIR/run.log"
cp "$SCRIPT_DIR/d_bench_core.py" "$OUTDIR/" || true
echo "[D] 完成: $OUTDIR"
