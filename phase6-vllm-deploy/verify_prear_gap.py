#!/usr/bin/env python3
"""验证 B3 pre-AR gap 的矛盾:921 normal 实例均值 509µs×5万 ≈25s ≫ idle 总量 2.46s。
重算单 rank:分布(中位/p90/p99/max)、gap 总和一致性、AR 启动点是否落在
某个 merged 区间内部(重叠→gap 误计)。
用法: python3 verify_prear_gap.py <trace_dir> <pid>
"""
import json, os, sys, glob, gzip
from array import array
import numpy as np

TRACE_DIR = sys.argv[1]
PID = sys.argv[2]
if not os.path.isabs(TRACE_DIR):
    TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACE_DIR)
ADIR = os.path.join(TRACE_DIR, "analysis")
KEY = "1stage"
KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def iter_events(path, chunk=1 << 22):
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


# 跨 rank 对齐(同主脚本)
ranks = {}
for f in sorted(glob.glob(os.path.join(ADIR, "rank_custom_ar_*.json"))):
    r = json.load(open(f))
    ranks[r["file"].split("_")[-1].split(".")[0]] = r
pids = sorted(ranks)
N = min(ranks[p]["n_" + KEY] for p in pids)
TS = {p: np.array([e[0] for e in sorted(ranks[p][KEY])[:N]], dtype=np.float64) for p in pids}
tsmin = np.min(np.stack([TS[p] for p in pids]), axis=0)
strag = np.array([pids[int(np.argmax([TS[p][i] for p in pids]))] for i in range(N)])
skip = int(N * 0.10)

# 找本 rank 的 trace 文件
tf = [f for f in glob.glob(os.path.join(TRACE_DIR, "*.pt.trace.json.gz"))
      if "async_llm" not in os.path.basename(f)
      and os.path.basename(f).split("_")[-1].split(".")[0] == PID][0]

ts_a, dur_a = array("q"), array("i")
for ev in iter_events(tf):
    if ev.get("cat", "") not in KERNEL_CATS or ev.get("ts") is None:
        continue
    ts_a.append(int(round(ev["ts"])))
    dur_a.append(int(round(ev.get("dur", 0))))
ts = np.frombuffer(ts_a, dtype=np.int64)
dur = np.frombuffer(dur_a, dtype=np.int32)
order = np.argsort(ts, kind="stable")
ts, dur = ts[order], dur[order]
end = ts + dur

# merged 区间(全 kernel)
m_start_a, m_end_a = array("q"), array("q")
cur_s, cur_e = int(ts[0]), int(end[0])
for k in range(1, len(ts)):
    s, e = int(ts[k]), int(end[k])
    if s <= cur_e:
        cur_e = max(cur_e, e)
    else:
        m_start_a.append(cur_s)
        m_end_a.append(cur_e)
        cur_s, cur_e = s, e
m_start_a.append(cur_s)
m_end_a.append(cur_e)
m_start = np.frombuffer(m_start_a, dtype=np.int64)
m_end = np.frombuffer(m_end_a, dtype=np.int64)
idle_total = int((m_end - m_start).sum() - 0)  # busy;idle 用差
span = int(end[-1] - ts[0])
busy = int((m_end - m_start).sum())

ar_ts = TS[PID].astype(np.int64)
gn, gs = [], []  # normal / straggler 实例的 pre-AR gap
inside_cnt = 0  # a 严格在区间内部(启动时有 kernel 在跑,含与前一 AR 融合)→ gap=0
for i in range(skip, N):
    a = int(ar_ts[i])
    j = int(np.searchsorted(m_end, a, side="left"))
    if j < len(m_end) and m_start[j] < a and m_end[j] > a:
        g = 0  # 严格内部:启动时有 kernel 在跑(含前一 AR 融合区间)
        inside_cnt += 1
    else:
        g = (a - int(m_end[j - 1])) if j > 0 else 0
    (gs if strag[i] == PID else gn).append(max(g, 0))

gn, gs = np.array(gn), np.array(gs)
print(f"pid{PID}: kernels={len(ts)} span={span/1e6:.1f}s busy={busy/span*100:.1f}% idle={(span-busy)/1e3:.0f}ms")
print(f"merged 区间数={len(m_end)}, AR启动时GPU正被其他kernel占用(含融合)={inside_cnt}/{N-skip}")
print(f"normal 实例 n={len(gn)}: sum={gn.sum()/1e3:.0f}ms mean={gn.mean():.1f} p50={np.percentile(gn,50):.1f} "
      f"p90={np.percentile(gn,90):.1f} p99={np.percentile(gn,99):.1f} max={gn.max()/1e3:.1f}ms")
print(f"strag  实例 n={len(gs)}: sum={gs.sum()/1e3:.0f}ms mean={gs.mean():.1f} p50={np.percentile(gs,50):.1f} "
      f"p90={np.percentile(gs,90):.1f} p99={np.percentile(gs,99):.1f} max={gs.max()/1e3:.1f}ms")
print(f"一致性: preAR gap 总和 (normal+strag)={(gn.sum()+gs.sum())/1e3:.0f}ms vs 全窗 idle={(span-busy)/1e3:.0f}ms")
top = np.sort(gn)[-10:]
print(f"normal gap top10(µs): {top.tolist()}")
print(f">1ms 的 normal gap 个数: {(gn>1000).sum()}, 占总和 {gn[gn>1000].sum()/max(gn.sum(),1)*100:.1f}%")
