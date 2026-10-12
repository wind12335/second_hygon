#!/usr/bin/env python3
"""④' 第二遍抽取:从单机 TP8 trace 提取 vLLM custom allreduce 逐事件序列。

背景:单机 8 卡启用 vLLM 自研 custom AR(cross_device_reduce_1stage/2stage),
decode 步的 61,824 次集合通信全走它而非 NCCL;analyze_traces.py 的分类把它落进
"elem",rccl_by_kind 只有 483 次大消息 NCCL AR —— 到达分析必须以本脚本产物为准。

每 rank 输出 rank_custom_ar_<pid>.json:
  {"file":..., "1stage": [[ts,dur],...], "2stage": [[ts,dur],...], "nccl_ar": [[ts,dur],...]}

纪律:与 analyze_traces.py 相同 —— 单文件流式串行、4MB 缓冲、断点续跑、不并行。
用法: python3 extract_custom_ar.py <trace_dir> [outdir]
"""
import gzip, json, os, sys, glob

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else ""
if TRACE_DIR and not os.path.isabs(TRACE_DIR):
    TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACE_DIR)
OUTDIR = (sys.argv[2] if len(sys.argv) > 2
          else (os.path.join(TRACE_DIR, "analysis") if TRACE_DIR else ""))

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


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


def out_name(path):
    base = os.path.basename(path)
    pid = base.split("_")[-1].split(".")[0]
    return os.path.join(OUTDIR, "rank_custom_ar_" + pid + ".json")


def extract_file(path):
    s1, s2, nar = [], [], []
    n_ev = 0
    for ev in iter_events(path):
        n_ev += 1
        if ev.get("cat", "") not in KERNEL_CATS:
            continue
        ts = ev.get("ts")
        if ts is None:
            continue
        name = ev.get("name", "?")
        dur = ev.get("dur", 0)
        if "cross_device_reduce_1stage" in name:
            s1.append([ts, dur])
        elif "cross_device_reduce_2stage" in name:
            s2.append([ts, dur])
        elif "ncclDevKernel_AllReduce" in name:
            nar.append([ts, dur])
    s1.sort(); s2.sort(); nar.sort()
    res = {"file": os.path.basename(path), "n_events_total": n_ev,
           "n_1stage": len(s1), "n_2stage": len(s2), "n_nccl_ar": len(nar),
           "1stage": s1, "2stage": s2, "nccl_ar": nar}
    with open(out_name(path), "w") as f:
        json.dump(res, f)
    print(f"[done] {os.path.basename(path)}: 1stage={len(s1)} 2stage={len(s2)} nccl_ar={len(nar)}", flush=True)


def is_done(path):
    o = out_name(path)
    if not os.path.exists(o):
        return False
    try:
        with open(o) as f:
            return "1stage" in json.load(f)
    except Exception:
        return False


if __name__ == "__main__":
    if not TRACE_DIR or not os.path.isdir(TRACE_DIR):
        sys.exit(f"ERROR: trace 目录不存在: {TRACE_DIR}")
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "*.pt.trace.json.gz")))
    files = [f for f in files if "async_llm" not in os.path.basename(f)]
    todo = [f for f in files if not is_done(f)]
    print(f"{len(files)} rank traces, 待抽取 {len(todo)}(断点续跑跳过 {len(files) - len(todo)})", flush=True)
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] 开始: {os.path.basename(f)}", flush=True)
        extract_file(f)
    print("all done -> " + OUTDIR + "/rank_custom_ar_*.json")
