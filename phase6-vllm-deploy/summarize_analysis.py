#!/usr/bin/env python3
"""③ trace 分析后处理:
  1. GEMM 修正:本轮 16 份用旧分类落盘(Cijk_* GEMM 落在 other),用 top10_kernels 无损换算回来
  2. skew 修正:max-dur rank = 等待最久(最早到);min-dur rank = 真正最后到
用法: python3 summarize_analysis.py <analysis_dir>
"""
import glob, json, os, re, sys
from collections import Counter

ADIR = sys.argv[1] if len(sys.argv) > 1 else ""
if not ADIR:
    sys.exit("用法: python3 summarize_analysis.py <analysis_dir>")
if not os.path.isabs(ADIR):
    ADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ADIR)

GEMM_RE = re.compile(r"(?i)(cijk|gemm|rocblas|hipblas|matmul)")
ranks = []
for f in sorted(glob.glob(os.path.join(ADIR, "rank_*.json"))):
    with open(f) as fh: ranks.append(json.load(fh))

print("===== ① 修正后每类 kernel 占 busy 比例(节点均值;GEMM 从 top10 名单换算) =====")
for node in ("worker-0", "worker-1"):
    rs = [r for r in ranks if r["node"] == node]
    agg = Counter(); n = len(rs)
    for r in rs:
        busy_ms = r["busy_s"] * 1e3
        gemm_ms = sum(d for nm, c, d in r["top10_kernels"] if GEMM_RE.search(nm))
        s = r["sum_dur_ms"]
        other_corr = s.get("other", 0) - gemm_ms
        agg["rccl"] += s.get("rccl", 0) / busy_ms / n
        agg["gemm"] += gemm_ms / busy_ms / n
        agg["attn"] += s.get("attention", 0) / busy_ms / n
        agg["elem+norm"] += (s.get("elem", 0) + s.get("norm_act", 0)) / busy_ms / n
        agg["other(修正)"] += other_corr / busy_ms / n
        agg["busy%/窗口"] += r["busy_frac_of_window"] / n
    print(f"[{node}] busy占窗口={agg['busy%/窗口']*100:.1f}%  " +
          "  ".join(f"{k}={v*100:.1f}%" for k, v in sorted(agg.items(), key=lambda x: -x[1]) if k != "busy%/窗口"))

tot_gemm = sum(sum(d for nm, c, d in r["top10_kernels"] if GEMM_RE.search(nm)) for r in ranks) / 16
tot_rccl = sum(r["sum_dur_ms"].get("rccl", 0) for r in ranks) / 16
print(f"\n16-rank 均值: AllReduce={tot_rccl:.0f}ms vs GEMM={tot_gemm:.0f}ms → 通信/计算 ≈ {tot_rccl/tot_gemm:.1f}×")

print("\n===== ② per-collective 到达偏斜(修正标签;同节点 8 rank,41,538 个 AllReduce) =====")
out = {}
for node in ("worker-0", "worker-1"):
    rs = [r for r in ranks if r["node"] == node]
    lists = [sorted(r["rccl_by_kind"]["AllReduce"]["list"]) for r in rs]
    pids = [r["file"].split("_")[-1].split(".")[0] for r in rs]
    n = min(len(l) for l in lists)
    early = Counter(); late = Counter(); excess = [0.0]*len(lists); intr = []
    for i in range(n):
        durs = [l[i][1] for l in lists]
        dmin = min(durs); intr.append(dmin)
        early[durs.index(max(durs))] += 1
        late[durs.index(dmin)] += 1
        for j, d in enumerate(durs): excess[j] += d - dmin
    intr.sort()
    print(f"[{node}] intrinsic(集合最小dur): p50={intr[len(intr)//2]}us 均值={sum(intr)/n:.0f}us")
    print(f"  {'pid':>5} {'等待总计ms':>10} {'占窗口%':>8} {'最早到次数':>9} {'最后到次数':>9}")
    for j, p in enumerate(pids):
        print(f"  {p:>5} {excess[j]/1e3:>10.1f} {excess[j]/1e6/30.3*100:>8.1f} {early[j]:>9} {late[j]:>9}")
    out[node] = {"intrinsic_p50_us": intr[len(intr)//2],
                 "wait_ms": {pids[j]: excess[j]/1e3 for j in range(len(pids))},
                 "earliest_counts": {pids[j]: early[j] for j in range(len(pids))},
                 "latest_counts": {pids[j]: late[j] for j in range(len(pids))}}
with open(os.path.join(ADIR, "summary_corrected.json"), "w") as f:
    json.dump({"gemm_fix_mean_ms": tot_gemm, "rccl_mean_ms": tot_rccl, "skew": out}, f, ensure_ascii=False, indent=1)
print(f"\n-> {os.path.join(ADIR, 'summary_corrected.json')}")
