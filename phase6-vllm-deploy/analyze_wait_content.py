#!/usr/bin/env python3
"""B·等待窗口内容盘点 + straggler 迟到归因(④' 单机 TP8)。

三个问题:
  B1 wait 暴露度:custom AR 的等待窗口 [本rank启动, 最晚rank到达] 里,
     非 AR kernel 覆盖率 → wait 是否完全裸露(M3 可藏空间的地板)。
  B2 straggler 归因:迟到 rank 在迟到时段 [最早rank启动, 本rank启动] 里,
     GPU 在算 kernel(compute 侧)还是空转(launch/CPU 侧)。
  B3 间隙账本:全窗 idle gap 总量/分布 + AR 启动前 gap(straggler vs 非对照)。

纪律:单文件流式串行(4MB 缓冲)、numpy/array 存放(峰值<100MB)、断点续跑。
用法: python3 analyze_wait_content.py <trace_dir>
输出: analysis/wait_content_<pid>.json ×8 + wait_content_summary.json
"""
import gzip, json, os, sys, glob, re
from array import array
import numpy as np

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else ""
if TRACE_DIR and not os.path.isabs(TRACE_DIR):
    TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACE_DIR)
ADIR = os.path.join(TRACE_DIR, "analysis")
KEY = "1stage"
SKIP_PCT = 10.0
KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

CLS_ORDER = ["custom_ar", "gemm", "attention", "norm_act", "elem", "memcpy", "memset", "rccl", "other"]


def classify(name, cat):
    if "cross_device_reduce" in name:
        return "custom_ar"
    if cat == "gpu_memcpy":
        return "memcpy"
    if cat == "gpu_memset":
        return "memset"
    if "ncclDevKernel" in name or "ncclKernel" in name:
        return "rccl"
    if re.search(r"(?i)(gemm|matmul|rocblas|hipblas|bgemm|cijk|ailk|bljk|s16816|mfma)", name):
        return "gemm"
    if re.search(r"(?i)(fmha|flash|attention|mha|paged)", name):
        return "attention"
    if re.search(r"(?i)(rmsnorm|layernorm|layer_norm|rotary|rope|silu|gelu|act_and_mul|activation)", name):
        return "norm_act"
    if re.search(r"(?i)(elementwise|vectorized|unrolled|reduce|softmax|copy|fill|index|arange|where|mask|cdiv|triu|scan)", name):
        return "elem"
    return "other"


def iter_events(path, chunk=1 << 22):
    """流式解析 chrome trace(与 analyze_traces.py 同实现)。"""
    dec = json.JSONDecoder()
    with gzip.open(path, "rt", errors="replace") as fh:
        buf = fh.read(chunk)
        p = buf.find('"traceEvents"')
        while p == -1:
            more = fh.read(chunk)
            if not more:
                return
            buf = buf[-1024:] + more
            p = buf.find('"traceEvents"')
        pos = buf.index("[", p) + 1
        buf = buf[pos:]
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                data = fh.read(chunk)
                if not data:
                    return
                buf = buf[pos:] + data
                pos = 0
                continue
            if buf[pos] in "]}":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if len(buf) > (64 << 20):
                    raise
                data = fh.read(chunk)
                if not data:
                    return
                buf = buf[pos:] + data
                pos = 0
                continue
            yield obj
            pos = end
            if pos > (1 << 20):
                buf = buf[pos:]
                pos = 0


def pct(a, q):
    if a is None or not len(a):
        return 0.0
    return float(np.percentile(np.asarray(a), q))


# ---- 前置:跨 rank 对齐(最早/最晚到达、straggler 身份) ----
arfiles = sorted(glob.glob(os.path.join(ADIR, "rank_custom_ar_*.json")))
assert arfiles, f"缺 rank_custom_ar_*.json: {ADIR}"
ranks = {}
for f in arfiles:
    r = json.load(open(f))
    pid = r["file"].split("_")[-1].split(".")[0]
    ranks[pid] = r
pids = sorted(ranks)
N = min(ranks[p]["n_" + KEY] for p in pids)
TS = {p: np.array([e[0] for e in sorted(ranks[p][KEY])[:N]], dtype=np.float64) for p in pids}
tsmin = np.min(np.stack([TS[p] for p in pids]), axis=0)
tsmax = np.max(np.stack([TS[p] for p in pids]), axis=0)
strag = np.array([pids[int(np.argmax([TS[p][i] for p in pids]))] for i in range(N)])
skip = int(N * SKIP_PCT / 100)
print(f"[align] {len(pids)} ranks, N={N}, skip={skip}", flush=True)


def process_rank(path, pid):
    outfile = os.path.join(ADIR, f"wait_content_{pid}.json")
    cid = {c: i for i, c in enumerate(CLS_ORDER)}
    ts_a, dur_a, cls_a = array("q"), array("i"), bytearray()
    n_ev = 0
    for ev in iter_events(path):
        n_ev += 1
        if ev.get("cat", "") not in KERNEL_CATS:
            continue
        ts = ev.get("ts")
        if ts is None:
            continue
        ts_a.append(int(round(ts)))
        dur_a.append(int(round(ev.get("dur", 0))))
        cls_a.append(cid[classify(ev.get("name", "?"), ev["cat"])])
    ts = np.frombuffer(ts_a, dtype=np.int64)
    dur = np.frombuffer(dur_a, dtype=np.int32)
    cls = np.frombuffer(cls_a, dtype=np.uint8)
    del ts_a, dur_a, cls_a
    order = np.argsort(ts, kind="stable")
    ts, dur, cls = ts[order], dur[order], cls[order]
    end = ts + dur
    t0, t1 = int(ts[0]), int(end[-1])
    span = t1 - t0
    steady0 = t0 + int(span * SKIP_PCT / 100)  # 丢 warmup 段(gap 账本口径)

    # ---- merged 区间(全部 kernel,含 AR):busy/idle + merged 端点序列 ----
    m_start_a, m_end_a, gap_a = array("q"), array("q"), array("q")
    cur_s, cur_e = int(ts[0]), int(end[0])
    busy = 0
    for k in range(1, len(ts)):
        s, e = int(ts[k]), int(end[k])
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            m_start_a.append(cur_s)
            m_end_a.append(cur_e)
            busy += cur_e - cur_s
            gap_a.append(s - cur_e)
            cur_s, cur_e = s, e
    m_start_a.append(cur_s)
    m_end_a.append(cur_e)
    busy += cur_e - cur_s
    m_end = np.frombuffer(m_end_a, dtype=np.int64)
    m_start = np.frombuffer(m_start_a, dtype=np.int64)
    gaps = np.frombuffer(gap_a, dtype=np.int64)
    # gap 发生时间 = merged 端点处;仅保留稳态段
    gap_times = m_end[:-1] if len(m_end) > 1 else np.array([t0])
    gaps_steady = np.array([g for g, gt in zip(gaps, gap_times) if gt >= steady0 and g > 0], dtype=np.int64)

    # ---- 非 AR 行(覆盖查询;已按 ts 排序) ----
    nar_mask = cls != cid["custom_ar"]
    nar_ts, nar_end, nar_cls = ts[nar_mask], end[nar_mask], cls[nar_mask]

    def cover_cls(a, b):
        """[a,b) 被非 AR kernel 覆盖量,按类累计(并流双计可能→上限口径)。"""
        acc = [0] * len(CLS_ORDER)
        i = int(np.searchsorted(nar_end, a, side="right"))
        n = len(nar_ts)
        while i < n and nar_ts[i] < b:
            ov = min(int(nar_end[i]), b) - max(int(nar_ts[i]), a)
            if ov > 0:
                acc[int(nar_cls[i])] += ov
            i += 1
        return acc

    # 本 rank 的 AR 序列(与 TS[pid] 同源:sorted 前 N 个)
    ar_ts = TS[pid].astype(np.int64)

    # ---- B1:wait 暴露度;i≥skip ----
    w_tot = w_cov = 0.0
    n_zero = n_w = 0
    cov_list = []
    for i in range(skip, N):
        a, b = int(ar_ts[i]), int(tsmax[i])
        w = b - a
        if w <= 0:
            continue
        acc = cover_cls(a, b)
        c = sum(acc)
        w_tot += w
        w_cov += c
        n_w += 1
        cov_list.append(c / w)
        if c == 0:
            n_zero += 1

    # ---- B2:straggler 迟到归因 + B3:pre-AR gap(修正口径) ----
    # 修正:a 严格落在某 merged 区间内部(AR 启动时有 kernel 在跑,含与前一 AR
    # 无缝融合)→ gap=0;仅当 a 恰为区间起点时,gap = a − 前一区间末端(真实空窗)。
    # 旧口径把融合区间的跨度当 gap,总和会超出 idle 总量近 10×(verify_prear_gap.py 实证)。
    lat_tot = lat_idle = 0.0
    lat_cls = [0] * len(CLS_ORDER)
    prear_strag, prear_norm = [], []
    n_inside = 0
    for i in range(skip, N):
        a = int(ar_ts[i])
        j = int(np.searchsorted(m_end, a, side="left"))
        if j < len(m_end) and m_start[j] < a < m_end[j]:
            g = 0
            n_inside += 1
        else:
            g = a - int(m_end[j - 1]) if j > 0 else 0
        (prear_strag if strag[i] == pid else prear_norm).append(max(g, 0))
        if strag[i] == pid:
            lo = int(tsmin[i])
            L = a - lo
            if L > 0:
                acc = cover_cls(lo, a)
                lat_tot += L
                lat_idle += L - sum(acc)
                for k in range(len(CLS_ORDER)):
                    lat_cls[k] += acc[k]

    res = {
        "pid": pid, "n_events": n_ev, "n_kernels": int(len(ts)),
        "windowS": span / 1e6, "busyPct": busy / span * 100,
        "B3_idle_steady": {"totalMs": float(gaps_steady.sum()) / 1e3,
                           "count": int(len(gaps_steady)),
                           "p50Us": pct(gaps_steady, 50), "p90Us": pct(gaps_steady, 90),
                           "p99Us": pct(gaps_steady, 99), "maxUs": float(gaps_steady.max()) if len(gaps_steady) else 0},
        "B3_preARgap": {"stragglerN": len(prear_strag),
                        "stragglerMeanUs": float(np.mean(prear_strag)) if prear_strag else 0,
                        "stragglerP50Us": pct(prear_strag, 50),
                        "stragglerP90Us": pct(prear_strag, 90),
                        "normalMeanUs": float(np.mean(prear_norm)) if prear_norm else 0,
                        "normalP50Us": pct(prear_norm, 50),
                        "normalP90Us": pct(prear_norm, 90),
                        "launchBusyPct": n_inside / (N - skip) * 100},
        "B1_wait_exposure": {"nWait": n_w, "waitTotalMs": w_tot / 1e3,
                             "covPooledPct": w_cov / w_tot * 100 if w_tot else 0,
                             "covMeanPerEvtPct": float(np.mean(cov_list)) if cov_list else 0,
                             "zeroCovEvtPct": n_zero / n_w * 100 if n_w else 0},
        "B2_straggler": {"nLate": int((strag[skip:] == pid).sum()),
                         "latenessTotalMs": lat_tot / 1e3,
                         "latenessMeanUs": lat_tot / max(int((strag[skip:] == pid).sum()), 1),
                         "idleSharePct": lat_idle / lat_tot * 100 if lat_tot else 0,
                         "clsCoverMs": {CLS_ORDER[k]: lat_cls[k] / 1e3 for k in range(len(CLS_ORDER)) if lat_cls[k] > 0}},
    }
    json.dump(res, open(outfile, "w"), ensure_ascii=False, indent=1)
    print(f"[done] pid{pid}: kernels={len(ts)} busy={res['busyPct']:.1f}% "
          f"waitCov={res['B1_wait_exposure']['covPooledPct']:.1f}% "
          f"stragIdleShare={res['B2_straggler']['idleSharePct']:.1f}%", flush=True)


def aggregate():
    rows = []
    for f in sorted(glob.glob(os.path.join(ADIR, "wait_content_*.json"))):
        if "summary" in f:
            continue
        rows.append(json.load(open(f)))
    agg = {
        "nRank": len(rows), "windowS": rows[0]["windowS"],
        "busyPct": {r["pid"]: r["busyPct"] for r in rows},
        "idleSteadyMs": {r["pid"]: r["B3_idle_steady"]["totalMs"] for r in rows},
        "idleP99Us": {r["pid"]: r["B3_idle_steady"]["p99Us"] for r in rows},
        "waitCovPooledPct": {r["pid"]: r["B1_wait_exposure"]["covPooledPct"] for r in rows},
        "waitZeroCovEvtPct": {r["pid"]: r["B1_wait_exposure"]["zeroCovEvtPct"] for r in rows},
        "stragN": {r["pid"]: r["B2_straggler"]["nLate"] for r in rows},
        "stragIdleSharePct": {r["pid"]: r["B2_straggler"]["idleSharePct"] for r in rows},
        "preARgapStragP50Us": {r["pid"]: r["B3_preARgap"]["stragglerP50Us"] for r in rows},
        "preARgapNormP50Us": {r["pid"]: r["B3_preARgap"]["normalP50Us"] for r in rows},
        "launchBusyPct": {r["pid"]: r["B3_preARgap"]["launchBusyPct"] for r in rows},
        "stragLatenessMeanUs": {r["pid"]: r["B2_straggler"]["latenessMeanUs"] for r in rows},
        "stragClsCoverMs": {r["pid"]: r["B2_straggler"]["clsCoverMs"] for r in rows},
    }
    with open(os.path.join(ADIR, "wait_content_summary.json"), "w") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)
    print("\n===== B·聚合(8 rank) =====")
    print(f"busy%: " + ", ".join(f"{r['pid']}:{r['busyPct']:.1f}" for r in rows))
    print(f"稳态 idle 总量(ms): " + ", ".join(f"{r['pid']}:{r['B3_idle_steady']['totalMs']:.0f}" for r in rows))
    print(f"B1 wait 非AR覆盖率%(pooled): " + ", ".join(f"{r['pid']}:{r['B1_wait_exposure']['covPooledPct']:.2f}" for r in rows))
    print(f"B2 straggler 迟到段 idle 占比%: " + ", ".join(f"{r['pid']}:{r['B2_straggler']['idleSharePct']:.1f}" for r in rows))
    print(f"B3 pre-AR gap µs(修正口径;straggler 实例 p50/mean,普通实例 p50/mean,AR启动时GPU忙占比%):")
    for r in rows:
        g = r["B3_preARgap"]
        print(f"  pid{r['pid']}: {g['stragglerP50Us']:.0f}/{g['stragglerMeanUs']:.0f} · "
              f"{g['normalP50Us']:.0f}/{g['normalMeanUs']:.0f} · {g['launchBusyPct']:.1f}%")
    print(f"-> {ADIR}/wait_content_summary.json")


if __name__ == "__main__":
    os.makedirs(ADIR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(TRACE_DIR, "*.pt.trace.json.gz"))):
        base = os.path.basename(f)
        if "async_llm" in base:
            continue
        pid = base.split("_")[-1].split(".")[0]
        if os.path.exists(os.path.join(ADIR, f"wait_content_{pid}.json")):
            print(f"[skip] pid{pid} 已完成", flush=True)
            continue
        process_rank(f, pid)
    aggregate()
