#!/usr/bin/env python3
"""M 轴扫描权威聚合:修正窗口 bug + 载体表 + prefill 簇分解 → msweep_summary.json + markdown 表。

修正/超越 analyze_arrival_8rank.py 的三点:
  1) 窗口:旧跑批的 case 匹配把所有相位都当 10s(decode 实际 15s,% 虚高 1.5×)。
     这里用 PHASES.json 的 t1-t0(真实采集窗)重算所有占比;绝对 µs 本就不依赖窗口。
  2) 双载体:每个相位 1stage/2stage 都算(JSON 只落了 1stage,ph3/4 连 JSON 都没有)。
  3) prefill 簇分解:ph5-7 的 2stage 实为 chunked-prefill forward(间隔>5ms 分簇,
     每簇≈一个 forward 的 ~152-161 个 AR)。算簇内粘性(straggler 在一个 forward 内
     是否稳定)与簇间切换,回答"prefill 的 straggler 可不可以预测"。

用法: python3 msweep_aggregate.py [run_dir]   (默认 20261012_003342 那次)
"""
import json, os, sys, glob
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "phase6_torch_prof_rdma_20261012_003342")
if not os.path.isabs(RUN):
    RUN = os.path.join(HERE, RUN)


def pct(a, q):
    a = sorted(a)
    k = (len(a) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def key_stats(ranks, pids, KEY, window_s):
    """复刻 analyze_arrival_8rank.py 的口径,返回 dict 或 None。"""
    counts = [ranks[p]["n_" + KEY] for p in pids]
    if min(counts) == 0:
        return None
    N = min(counts)
    TS = {p: [e[0] for e in sorted(ranks[p][KEY])[:N]] for p in pids}
    DUR = {p: [e[1] for e in sorted(ranks[p][KEY])[:N]] for p in pids}
    W = [min(DUR[p][i] for p in pids) for i in range(N)]          # 真走线=最晚到者时长
    WAIT = {p: [DUR[p][i] - W[i] for i in range(N)] for p in pids}
    delta = [max(TS[p][i] for p in pids) - min(TS[p][i] for p in pids) for i in range(N)]
    ends = [[TS[p][i] + DUR[p][i] for p in pids] for i in range(N)]
    resid = [max(e) - min(e) for e in ends]
    mw = st.mean(st.mean(WAIT[p][i] for p in pids) for i in range(N))
    order = [max(pids, key=lambda p: TS[p][i]) for i in range(N)]
    stay = sum(1 for i in range(1, N) if order[i] == order[i - 1]) / max(N - 1, 1)
    lastCnt = {}
    for r in order:
        lastCnt[r] = lastCnt.get(r, 0) + 1
    return {"N": N, "windowS": window_s,
            "wireP50": pct(W, 50), "wireMean": st.mean(W),
            "wireSharePct": st.mean(W) * N / 1e6 / window_s * 100,
            "deltaP50": pct(delta, 50), "deltaP90": pct(delta, 90), "deltaP99": pct(delta, 99),
            "residP50": pct(resid, 50),
            "meanWaitUs": mw, "meanWaitOverDelta": mw / max(st.mean(delta), 1),
            "waitSharePct": st.mean(sum(WAIT[p]) for p in pids) / 1e6 / window_s * 100,
            "waitOverWire": mw / max(st.mean(W), 1e-9),
            "pStay": stay, "lastCnt": lastCnt}


def cluster_split(ev, gap_us=5000):
    """按间隔分簇:返回 [ [ (ts,dur), ... ], ... ]"""
    out = [[ev[0]]]
    for e in ev[1:]:
        if e[0] - out[-1][-1][0] > gap_us:
            out.append([])
        out[-1].append(e)
    return out


def prefill_clusters(ranks, pids, KEY="2stage", min_n=40):
    """prefill 相位:每 rank 分簇,取簇数一致的代表 rank,算簇内/簇间 straggler 结构。"""
    counts = [ranks[p]["n_" + KEY] for p in pids]
    if min(counts) == 0:
        return None
    N = min(counts)
    p0 = pids[0]
    ev0 = sorted(ranks[p0][KEY])[:N]      # 代表 rank 的 (ts,dur) 对
    cl = cluster_split(ev0)
    big = [c for c in cl if len(c) >= min_n]
    res = {"nClusters": len(cl), "nBigClusters": len(big),
           "clusters": [{"n": len(c), "spanMs": (c[-1][0] - c[0][0]) / 1e3} for c in big]}
    if len(big) >= 2:
        # 簇内粘性:每簇(≈一个 prefill forward)内部 straggler 是否集中
        TS = {p: [e[0] for e in sorted(ranks[p][KEY])[:N]] for p in pids}
        order = [max(pids, key=lambda p: TS[p][i]) for i in range(N)]
        ev_ts = [e[0] for e in ev0]
        for ci, c in enumerate(big):
            lo, hi = c[0][0], c[-1][0]
            members = [order[i] for i, t in enumerate(ev_ts) if lo <= t <= hi]
            if len(members) < 10:
                continue
            stayIn = sum(1 for j in range(1, len(members)) if members[j] == members[j - 1]) / (len(members) - 1)
            top = sorted(set(members), key=lambda r: -members.count(r))[:2]
            res["clusters"][ci]["stayIn"] = round(stayIn, 3)
            res["clusters"][ci]["topStraggler"] = {r: members.count(r) for r in top}
    return res


phases = json.load(open(os.path.join(RUN, "PHASES.json")))
summary = []
for ph in phases:
    sub = os.path.join(RUN, f"ph{ph['idx']}_{ph['kind']}{ph['ptok'] or ph['conc']}")
    adir = os.path.join(sub, "analysis")
    files = sorted(glob.glob(os.path.join(adir, "rank_custom_ar_*.json")))
    assert files, sub
    ranks = {}
    for f in files:
        r = json.load(open(f))
        ranks[r["file"].split("_")[-1].split(".")[0]] = r
    pids = sorted(ranks)
    # 注:manifest 的 t1-t0 含 stop_profile 同步落盘等待(ph1 达 373s),不是采集窗;
    # 事件只在 start→stop 间积累,真实窗口=名义时长 dur_s
    win = float(ph["dur_s"])
    row = {"phase": os.path.basename(sub), "kind": ph["kind"],
           "conc": ph["conc"], "ptok": ph["ptok"], "trueWindowS": round(win, 1),
           "carriers": {k: ranks[pids[0]][k] for k in ("n_1stage", "n_2stage", "n_nccl_ar")}}
    for KEY in ("1stage", "2stage"):
        s = key_stats(ranks, pids, KEY, win)
        if s:
            row[KEY] = s
    if ph["kind"] == "p":
        row["prefillClusters"] = prefill_clusters(ranks, pids)
    summary.append(row)
    print(f"[ok] {row['phase']} win={win:.0f}s carriers={row['carriers']}")

out = os.path.join(RUN, "msweep_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("->", out)

# ---- markdown 总表 ----
md = ["| 相位 | M(语义) | 载体(主) | N | wire p50 µs | δ p50/p90 µs | meanWait µs | wait/wire | wait%窗 | p_stay |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in summary:
    key = "1stage" if r.get("1stage", {}).get("N", 0) >= r.get("2stage", {}).get("N", 0) else "2stage"
    s = r[key]
    m = f"{r['ptok']}tok×{r['conc']}req" if r["kind"] == "p" else f"conc={r['conc']}"
    md.append(f"| {r['phase']} | {m} | {key} | {s['N']} | {s['wireP50']:.0f} | "
              f"{s['deltaP50']:.0f}/{s['deltaP90']:.0f} | {s['meanWaitUs']:.0f} | "
              f"{s['waitOverWire']:.0f}× | {s['waitSharePct']:.0f}% | {s['pStay']:.3f} |")
mdp = os.path.join(RUN, "msweep_table.md")
with open(mdp, "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
print("->", mdp)
