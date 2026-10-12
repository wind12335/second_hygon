#!/usr/bin/env python3
"""Phase6 ③ torch-profiler trace 分析(16 rank Chrome trace,~175MB gz/each).

每 rank 输出:
  - kernel 分类占比(RCCL/GEMM/attention/norm_act/elem/memcpy/other),按 busy-union 与窗口两种口径
  - comm-compute 重叠(merged 区间求交):串行 vs 重叠结构的直接证据
  - RCCL 按集合通信类型(AllReduce/AllGather/...)的次数/时长分布
  - ar_list: 主导集合通信的 (ts,dur) 序列 → 供跨 rank 对齐算到达偏斜(phase4 的框架级版本)

combine 阶段:
  - 16 rank 汇总表;worker-0 vs worker-1 对比
  - 同节点 8 rank 按集合序号对齐:per-collective dur spread = 到达异质性;
    excess = dur_r - dur_min(该 rank 等最慢者的时间);straggler 统计(谁最常最后到)

用法: python3 analyze_traces.py <trace_dir> [outdir]
"""
import gzip, json, os, re, sys, glob
from collections import Counter, defaultdict

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else ""
# 相对路径一律相对脚本自身目录解析(pod 重启后 cwd 不确定,防"找不到目录")
if TRACE_DIR and not os.path.isabs(TRACE_DIR):
    TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACE_DIR)
OUTDIR = (sys.argv[2] if len(sys.argv) > 2
          else (os.path.join(TRACE_DIR, "analysis") if TRACE_DIR else ""))

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

def classify(name, cat):
    if cat == "gpu_memcpy": return "memcpy"
    if cat == "gpu_memset": return "memset"
    if "ncclDevKernel" in name or "ncclKernel" in name: return "rccl"
    if re.search(r"(?i)(gemm|matmul|rocblas|hipblas|bgemm|cijk|ailk|bljk|s16816|mfma)", name): return "gemm"
    if re.search(r"(?i)(fmha|flash|attention|mha|paged)", name): return "attention"
    if re.search(r"(?i)(rmsnorm|layernorm|layer_norm|rotary|rope|silu|gelu|act_and_mul|activation)", name): return "norm_act"
    if re.search(r"(?i)(elementwise|vectorized|unrolled|reduce|softmax|copy|fill|index|arange|where|mask|cdiv|triu|scan)", name): return "elem"
    return "other"

def merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def union_len(iv):
    return sum(e - s for s, e in iv)

def intersect_len(a, b):
    i = j = 0; tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: tot += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot

def pct(sorted_vals, p):
    if not sorted_vals: return 0.0
    return sorted_vals[min(int(len(sorted_vals) * p), len(sorted_vals) - 1)]

RCCL_KIND = re.compile(r"ncclDevKernel_([A-Za-z]+)_")

def iter_events(path, chunk=1 << 22):
    """流式解析 chrome trace 的 traceEvents 数组:不整文件载入、不落盘解压,
    每次只在线解 chunk(4MB)逐条 yield 事件,峰值内存 = 缓冲区 + 单条事件。"""
    dec = json.JSONDecoder()
    with gzip.open(path, "rt", errors="replace") as fh:
        buf = fh.read(chunk)
        p = buf.find('"traceEvents"')
        while p == -1:                      # header 跨 chunk(几乎不会发生)
            more = fh.read(chunk)
            if not more: return
            buf = buf[-1024:] + more
            p = buf.find('"traceEvents"')
        pos = buf.index("[", p) + 1
        buf = buf[pos:]; pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":   # 跳过分隔符
                pos += 1
            if pos >= len(buf):
                data = fh.read(chunk)
                if not data: return
                buf = buf[pos:] + data; pos = 0
                continue
            if buf[pos] in "]}":            # traceEvents 数组结束
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:    # 事件不完整 → 补数据再试
                if len(buf) > (64 << 20):   # 单事件超 64MB 仍解析失败 = 格式异常
                    raise
                data = fh.read(chunk)
                if not data: return
                buf = buf[pos:] + data; pos = 0
                continue
            yield obj
            pos = end
            if pos > (1 << 20):             # 定期回收已消费前缀
                buf = buf[pos:]; pos = 0

def analyze_file(path):
    base = os.path.basename(path)
    node = "worker-0" if "worker-0" in base else ("worker-1" if "worker-1" in base else "api")
    res = {"file": base, "node": node}
    kev = []           # (ts, dur, cls, name)
    by_name = Counter()
    by_name_dur = Counter()
    annot = []         # user_annotation(cpu 侧分段,量少)
    n_ev = 0
    for ev in iter_events(path):
        n_ev += 1
        cat = ev.get("cat", "")
        if cat in KERNEL_CATS:
            ts = ev.get("ts"); dur = ev.get("dur", 0)
            if ts is None: continue
            name = ev.get("name", "?")
            cls = classify(name, cat)
            kev.append((ts, dur, cls, name))
        elif cat in ("user_annotation", "cpu_user_annotation"):
            annot.append((ev.get("ts", 0), ev.get("dur", 0), ev.get("name", "?")))
    res["n_events_total"] = n_ev
    if not kev:
        res["no_kernel"] = True
        return res
    res["n_kernel_events"] = len(kev)

    t0 = min(ts for ts, _, _, _ in kev)
    t1 = max(ts + dur for ts, dur, _, _ in kev)
    res["window_s"] = (t1 - t0) / 1e6

    cls_iv = defaultdict(list)
    rccl_by_kind = defaultdict(lambda: {"n": 0, "durs": [], "list": []})
    for ts, dur, cls, name in kev:
        cls_iv[cls].append((ts, ts + dur))
        short = name[:90]
        by_name[short] += 1
        by_name_dur[short] += dur
        if cls == "rccl":
            m = RCCL_KIND.search(name)
            kind = m.group(1) if m else name[:30]
            k = rccl_by_kind[kind]
            k["n"] += 1; k["durs"].append(dur); k["list"].append([ts, dur])

    merged = {c: merge_intervals(iv) for c, iv in cls_iv.items()}
    busy = union_len(merge_intervals([iv for c in cls_iv.values() for iv in c]))
    res["busy_s"] = busy / 1e6
    res["busy_frac_of_window"] = busy / max(t1 - t0, 1)
    res["sum_dur_ms"] = {c: union_len(m) / 1e3 for c, m in merged.items()}
    res["share_of_busy"] = {c: union_len(m) / max(busy, 1) for c, m in merged.items()}
    res["share_of_window"] = {c: union_len(m) / max(t1 - t0, 1) for c, m in merged.items()}

    # comm-compute 重叠:rccl ∩ (gemm+attention) —— vLLM eager 里主要看这个
    comp = merge_intervals(cls_iv.get("gemm", []) + cls_iv.get("attention", []))
    comm = merged.get("rccl", [])
    ov = intersect_len(comp, comm)
    res["overlap"] = {
        "comm_s": union_len(comm) / 1e6, "comp_s": union_len(comp) / 1e6,
        "overlap_s": ov / 1e6,
        "comm_overlapped_frac": ov / max(union_len(comm), 1),
        "comp_overlapped_frac": ov / max(union_len(comp), 1),
    }
    res["top10_kernels"] = [(n, by_name[n], round(by_name_dur[n] / 1e3, 1)) for n, _ in by_name_dur.most_common(10)]
    rk = {}
    for kind, k in rccl_by_kind.items():
        d = sorted(k["durs"])
        rk[kind] = {"n": k["n"], "tot_ms": sum(d) / 1e3,
                    "p50_us": pct(d, .5), "p90_us": pct(d, .9), "max_us": d[-1] if d else 0,
                    "list": k["list"]}
    res["rccl_by_kind"] = rk
    if annot:
        ac = Counter()
        for ts, dur, name in annot: ac[name[:60]] += 1
        res["annotations_top"] = ac.most_common(8)
    out = os.path.join(OUTDIR, "rank_" + node + "_" + base.split(".")[0].split("_")[-1] + ".json")
    with open(out, "w") as f: json.dump(res, f)
    res["_per_rank_file"] = out
    for kind in rk: rk[kind].pop("list", None)  # 内存副本去掉大 list(磁盘上已保留)
    print(f"[done] {base}: kernel={res['n_kernel_events']} window={res['window_s']:.1f}s "
          f"busy={res['busy_frac_of_window']*100:.1f}% rccl={res['share_of_busy'].get('rccl',0)*100:.1f}% "
          f"gemm={res['share_of_busy'].get('gemm',0)*100:.1f}% attn={res['share_of_busy'].get('attention',0)*100:.1f}%", flush=True)
    return res

def combine(rank_files):
    ranks = []
    for f in rank_files:
        with open(f) as fh: ranks.append(json.load(fh))
    ranks = [r for r in ranks if "n_kernel_events" in r]
    report = {"n_ranks": len(ranks)}
    def node_avg(node, key, sub=None):
        vals = [r[key].get(sub, 0) if sub else r[key] for r in ranks if r["node"] == node and key in r]
        return sum(vals) / len(vals) if vals else 0
    print("\n===== ① 每 rank 总览(share_of_busy = 占 busy 时间的比例) =====")
    print(f"{'rank':<28}{'window_s':>9}{'busy%':>7}{'rccl%':>7}{'gemm%':>7}{'attn%':>7}{'elem%':>7}{'other%':>7}{'重叠:comm被重叠%':>10}")
    for r in sorted(ranks, key=lambda x: (x["node"], x["file"])):
        s = r["share_of_busy"]; o = r["overlap"]
        print(f"{r['node']}/{r['file'].split('_')[-1].split('.')[0]:<10}"
              f"{r['window_s']:>9.1f}{r['busy_frac_of_window']*100:>7.1f}"
              f"{s.get('rccl',0)*100:>7.1f}{s.get('gemm',0)*100:>7.1f}{s.get('attention',0)*100:>7.1f}"
              f"{s.get('elem',0)*100:>7.1f}{s.get('other',0)*100:>7.1f}{o['comm_overlapped_frac']*100:>10.1f}")
    for node in ("worker-0", "worker-1"):
        rs = [r for r in ranks if r["node"] == node]
        if not rs: continue
        print(f"\n[{node} 均值] rccl={node_avg(node,'share_of_busy','rccl')*100:.1f}% "
              f"gemm={node_avg(node,'share_of_busy','gemm')*100:.1f}% "
              f"attn={node_avg(node,'share_of_busy','attention')*100:.1f}% "
              f"elem={node_avg(node,'share_of_busy','elem')*100:.1f}% "
              f"comm被重叠={node_avg(node,'overlap','comm_overlapped_frac')*100:.1f}%")

    # ---- ② 同节点跨 rank:按集合序号对齐的到达偏斜 ----
    print("\n===== ② 同节点 per-collective 到达偏斜(dur_r - dur_min = 该 rank 等最慢者) =====")
    skew = {}
    for node in ("worker-0", "worker-1"):
        rs = [r for r in ranks if r["node"] == node]
        if len(rs) < 2: continue
        # 主导 kind = 总时长最大的 rccl 类型
        kind_tot = Counter()
        for r in rs:
            for k, v in r["rccl_by_kind"].items(): kind_tot[k] += v["tot_ms"]
        if not kind_tot: continue
        kind = kind_tot.most_common(1)[0][0]
        lists = [r["rccl_by_kind"][kind]["list"] for r in rs if kind in r["rccl_by_kind"]]
        n = min(len(l) for l in lists)
        # 只统计对齐可信的前 n 个(按 ts 排序)
        lists = [sorted(l)[:n] for l in lists]
        straggler = Counter(); excess_tot = [0.0] * len(lists); intr = []
        n_multi = 0
        for i in range(n):
            durs = [l[i][1] for l in lists]
            dmin = min(durs)
            intr.append(dmin)
            if max(durs) - dmin > 1:  # >1us 才算有偏斜
                n_multi += 1
                straggler[durs.index(max(durs))] += 1
            for r_i, d in enumerate(durs): excess_tot[r_i] += d - dmin
        intr.sort()
        print(f"[{node}] kind={kind} 对齐集合数={n} 有偏斜(>1us)集合占比={n_multi/max(n,1)*100:.0f}%")
        print(f"  intrinsic(逐集合最小dur) p50={pct(intr,.5):.0f}us p90={pct(intr,.9):.0f}us 均值={sum(intr)/max(n,1):.0f}us")
        ranks_lbl = [r["file"].split("_")[-1].split(".")[0] for r in rs if kind in r["rccl_by_kind"]]
        for i, lbl in enumerate(ranks_lbl):
            print(f"  rank(pid {lbl}): excess(等最慢者)总计={excess_tot[i]/1e3:.1f}ms "
                  f"占窗口={excess_tot[i]/1e6/max(rs[0]['window_s'],1)*100:.2f}%  最常最后到={straggler[i]}次")
        skew[node] = {"kind": kind, "n": n, "excess_ms": {ranks_lbl[i]: excess_tot[i]/1e3 for i in range(len(ranks_lbl))},
                      "straggler_counts": {ranks_lbl[i]: straggler[i] for i in range(len(ranks_lbl))},
                      "intrinsic_us_p50": pct(intr, .5)}
    return skew

def out_name(path):
    base = os.path.basename(path)
    node = "worker-0" if "worker-0" in base else ("worker-1" if "worker-1" in base else "api")
    return os.path.join(OUTDIR, "rank_" + node + "_" + base.split(".")[0].split("_")[-1] + ".json")

def is_done(path):
    """断点续跑:产物存在且能解析出 kernel 统计才算完成(pod 重启不浪费已算完的 rank)"""
    o = out_name(path)
    if not os.path.exists(o): return False
    try:
        with open(o) as f: return "n_kernel_events" in json.load(f)
    except Exception: return False

if __name__ == "__main__":
    if not TRACE_DIR or not os.path.isdir(TRACE_DIR):
        sys.exit(f"ERROR: trace 目录不存在: {TRACE_DIR}")
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "*.pt.trace.json.gz")))
    files = [f for f in files if "async_llm" not in os.path.basename(f)]
    todo = [f for f in files if not is_done(f)]
    print(f"{len(files)} rank traces, 待算 {len(todo)}(断点续跑跳过 {len(files)-len(todo)}) -> {OUTDIR}", flush=True)
    if todo:
        print("单文件流式串行执行(峰值内存≈几十MB,不整文件解压,不并行)", flush=True)
        for i, f in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] 开始: {os.path.basename(f)}", flush=True)
            analyze_file(f)
    rank_files = sorted(glob.glob(os.path.join(OUTDIR, "rank_*.json")))
    skew = combine(rank_files)
    with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
        pass  # 终端已打印;原始数据在 rank_*.json
    with open(os.path.join(OUTDIR, "skew.json"), "w") as f:
        json.dump(skew, f, ensure_ascii=False, indent=1)
    print(f"\nper-rank JSON -> {OUTDIR}/rank_*.json; skew -> {OUTDIR}/skew.json")
