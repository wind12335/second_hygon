#!/usr/bin/env python3
"""④' 到达分析:单机 TP8 custom allreduce 的 wait/δ/straggler 三件套。

与 ③(build_07_arrival_data.py)的方法差异:
  - ③ 跨节点无公共时钟 → 用"AR 结束时刻同步"锚定 + 时长反演推到达;
  - ④' 单机 8 rank 共享主机时钟 → ts 即到达,直接算,反演仅作互验;
  - 载体是 vLLM custom AR(cross_device_reduce_1stage 为主,decode 稳态),
    NCCL AR 只有大消息 483 次(另列,不作主口径)。

输出: 终端汇总 + arrival_8rank.json(供与 ③ 对比表引用)
用法: python3 analyze_arrival_8rank.py <analysis_dir> [window_s]
"""
import json, os, sys, glob
import statistics as st

ADIR = sys.argv[1] if len(sys.argv) > 1 else ""
if ADIR and not os.path.isabs(ADIR):
    ADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ADIR)
WINDOW_S = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0


def pct(a, q):
    a = sorted(a)
    k = (len(a) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


files = sorted(glob.glob(os.path.join(ADIR, "rank_custom_ar_*.json")))
assert files, f"未找到 rank_custom_ar_*.json: {ADIR}"
ranks = {}
for f in files:
    r = json.load(open(f))
    pid = r["file"].split("_")[-1].split(".")[0]
    ranks[pid] = r
pids = sorted(ranks)
print(f"{len(pids)} ranks: {pids}")

for KEY in ("1stage", "2stage"):
    counts = [ranks[p]["n_" + KEY] for p in pids]
    if min(counts) == 0:
        continue
    N = min(counts)
    if len(set(counts)) != 1:
        print(f"[{KEY}] 锁步计数不一致 {counts},取前 {N}")
    TS = {p: [e[0] for e in sorted(ranks[p][KEY])[:N]] for p in pids}
    DUR = {p: [e[1] for e in sorted(ranks[p][KEY])[:N]] for p in pids}

    # ---- 真走线 W / 真等待 ----
    W = [min(DUR[p][i] for p in pids) for i in range(N)]
    WAIT = {p: [DUR[p][i] - W[i] for i in range(N)] for p in pids}

    # ---- 到达 = ts(同钟) / 结束同步残差(互验) ----
    ends = [[TS[p][i] + DUR[p][i] for p in pids] for i in range(N)]
    resid = [max(e) - min(e) for e in ends]
    delta = [max(TS[p][i] for p in pids) - min(TS[p][i] for p in pids) for i in range(N)]

    # 互验(反演口径):dur−dur_min ≈ max_ts−ts
    valD = []
    for i in range(min(N, 4000)):
        tmax = max(TS[p][i] for p in pids)
        for p in pids:
            valD.append(abs((DUR[p][i] - W[i]) - (tmax - TS[p][i])))

    # ---- straggler 与粘性 ----
    lastRank = {}
    for i in range(N):
        lp = max(pids, key=lambda p: TS[p][i])
        lastRank[lp] = lastRank.get(lp, 0) + 1
    order = [max(pids, key=lambda p: TS[p][i]) for i in range(N)]
    stay = sum(1 for i in range(1, N) if order[i] == order[i - 1]) / max(N - 1, 1)
    runs, cur = [], 1
    for i in range(1, N):
        if order[i] == order[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)

    # ---- "前 N-1 名到齐等最后 1 名"形态:meanWait/δ ----
    mw = st.mean(st.mean(WAIT[p][i] for p in pids) for i in range(N))

    print(f"\n===== {KEY}: N={N} =====")
    print(f"结束同步残差: p50={pct(resid,50):.1f} p90={pct(resid,90):.1f}µs(应≪δ)")
    print(f"互验(反演vs同钟): p50={pct(valD,50):.1f} p90={pct(valD,90):.1f}µs")
    print(f"真走线 W: mean={st.mean(W):.1f} p50={pct(W,50):.1f}µs 总量={st.mean(W)*N/1e6:.2f}s"
          f"({st.mean(W)*N/1e6/WINDOW_S*100:.1f}% 窗口)")
    print(f"δ(到达散布): p10={pct(delta,10):.0f} p50={pct(delta,50):.0f} p90={pct(delta,90):.0f} p99={pct(delta,99):.0f}µs")
    print(f"meanWait/δ(mean) = {mw:.1f}/{st.mean(delta):.1f} = {mw/max(st.mean(delta),1):.2f}(均匀错开应≈0.5)")
    print(f"straggler 粘性 p_stay={stay:.3f};run 长 mean={st.mean(runs):.1f} p90={pct(runs,90):.0f} max={max(runs)}")
    top = sorted(lastRank.items(), key=lambda x: -x[1])[:5]
    print("最常迟到 top5: " + ", ".join(f"pid{p}:{c}({c/N*100:.1f}%)" for p, c in top))
    for p in pids:
        print(f"  pid{p}: 真等待均值 {st.mean(WAIT[p]):7.1f}µs 占窗口 {sum(WAIT[p])/1e6/WINDOW_S*100:5.1f}%  最迟到 {lastRank.get(p,0)}次")

    if KEY == "1stage":
        out = {"pids": pids, "N": N, "windowS": WINDOW_S,
               "wireMeanUs": st.mean(W), "wireP50Us": pct(W, 50),
               "wireSharePct": st.mean(W) * N / 1e6 / WINDOW_S * 100,
               "deltaP10": pct(delta, 10), "deltaP50": pct(delta, 50),
               "deltaP90": pct(delta, 90), "deltaP99": pct(delta, 99),
               "residP50": pct(resid, 50), "residP90": pct(resid, 90),
               "valP50": pct(valD, 50), "valP90": pct(valD, 90),
               "pStay": stay, "runMean": st.mean(runs), "runP90": pct(runs, 90), "runMax": max(runs),
               "meanWaitOverDelta": mw / max(st.mean(delta), 1),
               "waitMeanUs": {p: st.mean(WAIT[p]) for p in pids},
               "waitSharePct": {p: sum(WAIT[p]) / 1e6 / WINDOW_S * 100 for p in pids},
               "lastCnt": {p: lastRank.get(p, 0) for p in pids}}
        with open(os.path.join(ADIR, "arrival_8rank.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"-> {ADIR}/arrival_8rank.json")

print("\n(③ 双机对照: wait 62% rank-mean / δ p50=534µs / p_stay=0.977 / wire 91.9µs=12.6%)")
