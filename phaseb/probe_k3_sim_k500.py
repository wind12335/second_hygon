#!/usr/bin/env python3
# probe-k3 次小值模拟（2026-09-03，回 NVIDIA【13】v0.4 议题：探 2-3 次取次小值是否够）
#
# 背景：selector v0.3 跨基座复算（selector_v03_k500_20260903）显示 probe-1iter 在
# K500SM_AI 为 8/9、p95 12.51%——唯一错格 (2048,2) 是 r0/r1 近平局被首迭代噪声翻错。
# 本脚本验证：把 k=1（首迭代）升级为 k=3 取次小值（≈3 样本中位），能否修复该错格。
#
# 数据源：与 selector_phaseb_v03_k500.py 完全一致（f17aae1d 族终判表口径）
#   r0/r1：formal 根；d0：d0dc 根；d1：dsfix 根；均取 rep1 的 raw_global_samples 前 k 个迭代
#   （ITERS=80、WARMUP=20，首迭代已是 post-warmup 计量迭代）
# 规则：k=1 → 首值；k=3 → sorted(vals)[1]（次小值，抗首迭代尖刺）
# 口径：oracle = 各策略 e2e p50（全 5 rep × 80 iter 池化中位）的最优；regret 对 oracle。
import csv
import glob
import os

BASE = "/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu"
ROOTS = {"r0": (BASE + "/phaseb_formal_20260902_160115", "R0_FULL_SERIAL"),
         "r1": (BASE + "/phaseb_formal_20260902_160115", "R1_EVENT_OVERLAP"),
         "d0": (BASE + "/phaseb_d0dc_20260902_195753", "D0_FCOLLECT_SERIAL"),
         "d1": (BASE + "/phaseb_dsfix_20260902_184341", "D1_PUSHSIG_OVERLAP")}
OUT = "/root/private_data/lyc/second_hygon/results/probe_k3_k500_20260903.csv"
CELLS = [(N, q) for N in (512, 2048, 4096) for q in (2, 4, 8)]


def load_summary(root):
    table = {}
    for r in csv.DictReader(open(os.path.join(root, "summary", "phaseb_case_summary.csv"))):
        if r["candidate"] != "C0_DEFAULT":
            continue
        table.setdefault((r["path"], int(r["N"]), int(r["q"])), []).append(float(r["e2e_max_us_p50"]))
    return table


def median(table, path_full, N, q):
    v = table.get((path_full, N, q))
    return sorted(v)[len(v) // 2] if v else None


def first_k(root, short, path_full, N, q, k):
    # case 目录用短名，manifest.csv 校验全名（同 selector_phaseb_v03_k500.py）
    for d in sorted(glob.glob(os.path.join(root, "cases", f"*_{short}_C0_DEFAULT_N{N}_q{q}_rep1"))):
        sample = os.path.join(d, "raw_global_samples.csv")
        manifest = os.path.join(d, "manifest.csv")
        if not (os.path.exists(sample) and os.path.exists(manifest)):
            continue
        with open(manifest, newline="") as handle:
            head = next(csv.DictReader(handle), {})
        if head.get("path") != path_full:
            continue
        vals = [float(r["e2e_max_us"]) for r in csv.DictReader(open(sample))
                if r.get("e2e_max_us") not in (None, "")][:k]
        if vals:
            return vals
    return None


def main():
    summaries = {s: load_summary(root) for s, (root, _) in ROOTS.items()}
    rows, regs = [], {1: [], 3: []}
    for (N, q) in CELLS:
        e2e = {s: median(summaries[s], full, N, q) for s, (root, full) in ROOTS.items()}
        if None in e2e.values():
            continue
        oracle = min(e2e, key=e2e.get)
        for k in (1, 3):
            scores = {}
            for s, (root, full) in ROOTS.items():
                v = first_k(root, s, full, N, q, k)
                if v is None:
                    scores = None
                    break
                scores[s] = sorted(v)[1] if (k >= 2 and len(v) >= 2) else v[0]
            if scores is None:
                continue
            pick = min(scores, key=scores.get)
            reg = 100 * (e2e[pick] - e2e[oracle]) / e2e[oracle]
            regs[k].append(reg)
            rows.append({"N": N, "q": q, "k": k, "probe_pick": pick, "oracle": oracle,
                         "regret_pct": round(reg, 2)})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "q", "k", "probe_pick", "oracle",
                                                    "regret_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print("probe-k 模拟（K500SM_AI，f17aae1d 族，与 selector_v03_k500 同口径）")
    for k in (1, 3):
        v = sorted(regs[k])
        med = v[len(v) // 2]
        p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
        perfect = sum(1 for x in v if x == 0)
        print(f"  k={k}（{'首值' if k == 1 else '次小值'}）: 完全命中 {perfect}/{len(v)}"
              f"  regret med={med:.2f}% p95={p95:.2f}% max={v[-1]:.2f}%")
    print(f"csv -> {OUT}")


if __name__ == "__main__":
    main()
