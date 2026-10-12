#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1: k-分块等待放大模拟(数据驱动,零新采集)
==========================================
问题:Flux / TileLink 式"把一次 AllReduce 拆成 k 块与计算交错"在我们实测的高偏斜
环境里,等待(wait)与固定开销会被放大多少?从几块开始亏本?

输入:phase6 ③ 的 16 rank 逐事件 AR 序列(analysis/rank_*.json,41,538 个集合/窗口)
口径:与 docs/interactive_html/07_arrival_timeline.html 完全一致 ——
  · 跨节点锚定:OFF = median(mean_end_w0 − mean_end_w1),worker-1 的 ts += OFF(结束对齐);
  · 每集合 offsets o_r = ts_r − min(ts);δ = max(o);mean_wait = δ − mean(o)(16 rank 平均);
  · cohort = 去掉全局最晚者后的 15 个 rank,其内部散布 = "无 straggler 时的残余抖动";
  · 走线 W = 每集合全局 16 rank 最小时长(最晚到达者的 kernel ≈ 纯协议+搬运)。

三个模型(文档 §3.2):
  a 持续性慢:每个 mini-barrier 重新付全额到达差        Wa(k) = k·MW
  b 一次性迟到:第 1 块付 δ,后续只付 cohort 残余抖动    Wb(k) = MW + (k−1)·CW
  c 实测持续性:用相邻 AR"同一槽位继续当最晚者"的实测概率 p_stay 内插
                Wc(k) = MW + (k−1)·(p_stay·MW + (1−p_stay)·CW)

账本(每集合,µs):
  额外固定开销 = (k−1)·W̄   (每块付一次协议地板;小 M 下 LL 地板与块大小基本无关)
  可藏收益上限 = min(W̄, C_slot),C_slot = 每集合平均计算量(4.6s ÷ 41,538)
  净收益(k) = 可藏收益 − [额外固定开销 + (Wait(k) − Wait(1))]

输出:results/e1_chunk_wait_<ts>/{e1_result.json, e1_curve.csv} + 控制台摘要。
"""
import json, glob, os, random, statistics as st, time

BASE = "/root/private_data/second_hygon"
RUN  = f"{BASE}/phase6-vllm-deploy/phase6_torch_prof_rdma_20261009_212123/analysis"

# —— 论文口径常量(REPORT.md §①/§④ 修正版)——
COMPUTE_TOTAL_S = 4.6        # 30.3s 窗口全部计算 ≈ 15% 墙钟
AR_PER_STEP     = 161
STEP_US         = 117.0 * 1000
KS              = [1, 2, 4, 8, 16, 32, 64]
BOOT_REPS       = 200
SEED            = 20261010


def load_pools():
    files = sorted(glob.glob(f"{RUN}/rank_worker-*.json"))
    assert len(files) == 16, f"expect 16 rank json, got {len(files)}"
    seqs = [json.load(open(f))["rccl_by_kind"]["AllReduce"]["list"] for f in files]
    N = len(seqs[0])
    assert all(len(s) == N for s in seqs), [len(s) for s in seqs]

    # ① 锚定:节点间 offset 用"结束时刻均值差"的中位数
    diffs = []
    for i in range(N):
        e0 = sum(seqs[r][i][0] + seqs[r][i][1] for r in range(8)) / 8
        e1 = sum(seqs[r][i][0] + seqs[r][i][1] for r in range(8, 16)) / 8
        diffs.append(e0 - e1)
    OFF = st.median(diffs)

    # ② 每集合的标量池
    MW, CW, DE, WG, LAST = [], [], [], [], []
    for i in range(N):
        a    = [seqs[r][i][0] + (OFF if r >= 8 else 0.0) for r in range(16)]
        durs = [seqs[r][i][1] for r in range(16)]
        mn  = min(a)
        o   = [x - mn for x in a]
        de  = max(o)
        MW.append(de - sum(o) / 16)          # 16 rank 平均等待
        co  = sorted(o)[:15]                  # cohort:去掉最晚者
        CW.append(co[-1] - sum(co) / 15)      # cohort 内部平均等待
        DE.append(de)
        WG.append(min(durs))                  # 走线 W(全局最小时长)
        LAST.append(max(range(16), key=lambda r: o[r]))
    return N, OFF, MW, CW, DE, WG, LAST


def persistence(LAST, N):
    stay = sum(1 for i in range(N - 1) if LAST[i] == LAST[i + 1]) / (N - 1)
    runs, cur = [], 1
    for i in range(1, N):
        if LAST[i] == LAST[i - 1]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    runs_sorted = sorted(runs)
    p90 = runs_sorted[int(0.9 * (len(runs_sorted) - 1))]
    return stay, st.mean(runs), p90, max(runs)


def main():
    t0 = time.time()
    random.seed(SEED)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = f"{BASE}/results/e1_chunk_wait_{ts}"
    os.makedirs(outdir, exist_ok=True)

    N, OFF, MW, CW, DE, WG, LAST = load_pools()
    p_stay, run_mean, run_p90, run_max = persistence(LAST, N)

    MWm, CWm, Wm = st.mean(MW), st.mean(CW), st.mean(WG)
    C_slot = COMPUTE_TOTAL_S * 1e6 / N
    hide   = min(Wm, C_slot)

    # 模态槽位占比(与 REPORT 的 pid931≈29% 对账)
    from collections import Counter
    modal = Counter(LAST).most_common(3)

    # —— 主曲线 ——
    rows = []
    for k in KS:
        Wa = k * MWm
        Wb = MWm + (k - 1) * CWm
        Wc = MWm + (k - 1) * (p_stay * MWm + (1 - p_stay) * CWm)
        fl = (k - 1) * Wm
        r = {"k": k, "wait_a_us": Wa, "wait_b_us": Wb, "wait_c_us": Wc, "extra_floor_us": fl}
        for name, W in (("a", Wa), ("b", Wb), ("c", Wc)):
            cost = fl + (W - MWm)
            r[f"cost_{name}_us"]   = cost
            r[f"net_{name}_us"]    = hide - cost
            r[f"step_extra_{name}_us"] = AR_PER_STEP * cost
            r[f"step_extra_{name}_pct"] = 100.0 * AR_PER_STEP * cost / STEP_US
        rows.append(r)

    # —— bootstrap 95% CI(对 MW/CW 重采样,报告 k=8 的等待)——
    def boot(pool_fn, k=8):
        vals = []
        for _ in range(BOOT_REPS):
            mw = st.mean(random.choices(MW, k=N))
            cw = st.mean(random.choices(CW, k=N))
            vals.append(pool_fn(mw, cw, k))
        vals.sort()
        return vals[int(0.025 * BOOT_REPS)], vals[int(0.975 * BOOT_REPS)]
    ci = {
        "wait_a_k8": boot(lambda mw, cw, k: k * mw),
        "wait_b_k8": boot(lambda mw, cw, k: mw + (k - 1) * cw),
        "wait_c_k8": boot(lambda mw, cw, k: mw + (k - 1) * (p_stay * mw + (1 - p_stay) * cw)),
    }

    def kstar(col):
        ok = [r["k"] for r in rows if r[col] >= 0]
        return max(ok) if ok else 0

    result = {
        "meta": {"ts": ts, "run": RUN, "n_collectives": N, "anchor_off_us": OFF,
                 "seed": SEED, "boot_reps": BOOT_REPS, "elapsed_s": round(time.time() - t0, 1)},
        "validation": {"wire_mean_us": round(Wm, 1), "mean_wait_us": round(MWm, 1),
                       "cohort_wait_us": round(CWm, 1),
                       "delta_mean_us": round(st.mean(DE), 1),
                       "delta_p50_us": round(st.median(DE), 1),
                       "note": "wire≈92 / mean_wait≈400 / delta_p50≈534 与 REPORT §④ 对账"},
        "persistence": {"p_stay": round(p_stay, 3), "run_len_mean": round(run_mean, 1),
                        "run_len_p90": run_p90, "run_len_max": run_max,
                        "modal_last_slots": [[s, c, round(100 * c / N, 1)] for s, c in modal]},
        "budget": {"C_slot_us": round(C_slot, 1), "hide_us": round(hide, 1),
                   "W_bar_us": round(Wm, 1), "ar_per_step": AR_PER_STEP, "step_us": STEP_US},
        "curves": rows,
        "ci95_k8": {kk: [round(v[0], 1), round(v[1], 1)] for kk, v in ci.items()},
        "headline": {"kstar_a": kstar("net_a_us"), "kstar_b": kstar("net_b_us"),
                     "kstar_c": kstar("net_c_us")},
    }
    json.dump(result, open(f"{outdir}/e1_result.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{outdir}/e1_curve.csv", "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(round(r[c], 2)) for c in cols) + "\n")

    # —— 控制台摘要 ——
    v, p, b = result["validation"], result["persistence"], result["budget"]
    print(f"[E1] N={N}  锚定 OFF={OFF:.1f}µs  用时 {time.time()-t0:.1f}s")
    print(f"[对账] W̄={v['wire_mean_us']}µs  MW={v['mean_wait_us']}µs  CW={v['cohort_wait_us']}µs  "
          f"δ mean/p50={v['delta_mean_us']}/{v['delta_p50_us']}µs")
    print(f"[持续性] p_stay={p['p_stay']}  最晚者连续当值 run 长 mean/p90/max="
          f"{p['run_len_mean']}/{p['run_len_p90']}/{p['run_len_max']} 个 AR")
    print(f"[持续性] 模态最晚槽位(槽位,次数,占比%):{p['modal_last_slots']}")
    print(f"[预算] C_slot={b['C_slot_us']}µs  可藏上限 min(W,C)={b['hide_us']}µs")
    print(f"\n  k |  等待a |  等待b |  等待c | 固定开销 |   净收益a |   净收益b |   净收益c | 步额外%c")
    for r in rows:
        print(f" {r['k']:3d} | {r['wait_a_us']:6.0f} | {r['wait_b_us']:6.0f} | {r['wait_c_us']:6.0f} | "
              f"{r['extra_floor_us']:8.0f} | {r['net_a_us']:8.0f} | {r['net_b_us']:8.0f} | {r['net_c_us']:8.0f} | "
              f"{r['step_extra_c_pct']:6.1f}")
    print(f"\n[CI95@k=8] " + "  ".join(f"{k}={v2}" for k, v2 in result['ci95_k8'].items()))
    print(f"[交叉点] 净收益≥0 的最大 k:模型a={result['headline']['kstar_a']} "
          f"模型b={result['headline']['kstar_b']} 实测c={result['headline']['kstar_c']}")
    print(f"[输出] {outdir}/e1_result.json + e1_curve.csv")


if __name__ == "__main__":
    main()
