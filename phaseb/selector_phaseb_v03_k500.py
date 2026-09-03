#!/usr/bin/env python3
# selector v0.3 K500SM_AI 复算（2026-09-03，回 NVIDIA【13】Q3：跨基座验证表）
# 与 selector_phaseb_v03.py（A800）方法逐条对齐：同两项式模型、同 LOO、同基线、同指标。
#
# 数据源（f17aae1d 族终判表口径）：
#   控制量 COMM/FC/DC/GEMM + r0/r1 e2e：formal 根（phaseb_formal_20260902_160115）
#   d0 e2e：d0dc 根（phaseb_d0dc_20260902_195753，f17aae1d）
#   d1 e2e：dsfix 根（phaseb_dsfix_20260902_184341，f17aae1d）
#   C2 COMM（B3 iso_gap）：formal 根 C2_RING_SIMPLE_CH8
#   probe-1iter：各策略 rep1 的 raw_global_samples.csv iteration_index==0（首轮模拟，同 A800 口径）
# 网格 3×3（N512/2048/4096 × q2/4/8）与 A800 完全同格——q16 因 formal 缺 R0/FC 无法进同口径
# 网格（如实注明，两基座同因）。
import csv, glob, json, os

BASE = "/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu"
FORMAL = os.path.join(BASE, "phaseb_formal_20260902_160115")
D0DC = os.path.join(BASE, "phaseb_d0dc_20260902_195753")
DSFIX = os.path.join(BASE, "phaseb_dsfix_20260902_184341")
OUT = "/root/private_data/lyc/second_hygon/results/selector_v03_k500_20260903"
os.makedirs(OUT, exist_ok=True)

CELLS = [(N, q) for N in (512, 2048, 4096) for q in (2, 4, 8)]
PATH = {"comm": "COMM_ONLY", "fc": "FC_FCOLLECT_ONLY", "dc": "DC_PUSHSIG_ONLY",
        "gemm": "GEMM_ONLY", "r0": "R0_FULL_SERIAL", "r1": "R1_EVENT_OVERLAP",
        "d0": "D0_FCOLLECT_SERIAL", "d1": "D1_PUSHSIG_OVERLAP"}


def load_summary(root):
    table = {}
    path_csv = os.path.join(root, "summary", "phaseb_case_summary.csv")
    for r in csv.DictReader(open(path_csv)):
        if r["candidate"] != "C0_DEFAULT":
            continue
        key = (r["path"], int(r["N"]), int(r["q"]))
        table.setdefault(key, []).append(float(r["e2e_max_us_p50"]))
    return table


def p50_of(table, path, N, q):
    v = table.get((PATH[path], N, q))
    return sorted(v)[len(v) // 2] if v else None


formal, d0dc, dsfix = load_summary(FORMAL), load_summary(D0DC), load_summary(DSFIX)

# C2 comm（iso_gap 用）
c2 = {}
for r in csv.DictReader(open(os.path.join(FORMAL, "summary", "phaseb_case_summary.csv"))):
    if r["path"] == "COMM_ONLY" and r["candidate"] == "C2_RING_SIMPLE_CH8":
        c2.setdefault((int(r["N"]), int(r["q"])), []).append(float(r["e2e_max_us_p50"]))


def first_iter_e2e(root, path, N, q):
    # case 目录名用短名（case001_d1_C0_DEFAULT_...），manifest 里才是全名——按短名 glob + manifest 校验
    dirs = sorted(glob.glob(os.path.join(root, "cases",
                                         f"*_{path}_C0_DEFAULT_N{N}_q{q}_rep1")))
    for d in dirs:
        sample = os.path.join(d, "raw_global_samples.csv")
        manifest = os.path.join(d, "manifest.csv")
        if not (os.path.exists(sample) and os.path.exists(manifest)):
            continue
        with open(manifest, newline="") as handle:
            head = next(csv.DictReader(handle), {})
        if head.get("path") != PATH[path]:
            continue
        for row in csv.DictReader(open(sample)):
            if row.get("e2e_max_us") not in (None, ""):
                return float(row["e2e_max_us"])  # iteration_index==0
    return None


grid = {}
for (N, q) in CELLS:
    comm, fc = p50_of(formal, "comm", N, q), p50_of(formal, "fc", N, q)
    dc, gemm = p50_of(formal, "dc", N, q), p50_of(formal, "gemm", N, q)
    r0, r1 = p50_of(formal, "r0", N, q), p50_of(formal, "r1", N, q)
    d0 = p50_of(d0dc, "d0", N, q)
    d1 = p50_of(dsfix, "d1", N, q)
    if None in (comm, fc, dc, gemm, r0, r1, d0, d1):
        continue
    c2v = sorted(c2[(N, q)])[len(c2[(N, q)]) // 2] if c2.get((N, q)) else None
    iso_gap = abs(c2v - comm) / comm * 100 if c2v else None
    probe = {s: first_iter_e2e({"r0": FORMAL, "r1": FORMAL, "d0": D0DC, "d1": DSFIX}[s], s, N, q)
             for s in ("r0", "r1", "d0", "d1")}
    grid[(N, q)] = dict(comm=comm, fc=fc, dc=dc, gemm=gemm, iso_gap=iso_gap, probe=probe,
                        e2e={"r0": r0, "r1": r1, "d0": d0, "d1": d1})


def predict(cell, q, a1, b1, a2, b2):
    c, f, d, g = cell["comm"], cell["fc"], cell["dc"], cell["gemm"]
    return {"r0": c + g, "d0": f + g, "r1": max(c, g) + a1 * q + b1,
            "d1": max(d, g) + a2 * q + b2}


def fit_pair(cells, key):
    xs, ys = [], []
    for (N, q), cell in cells.items():
        base = {"r1": max(cell["comm"], cell["gemm"]), "d1": max(cell["dc"], cell["gemm"])}[key]
        xs.append(q); ys.append(cell["e2e"][key] - base)
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / den if den else 0.0
    return a, (sy - a * sx) / n


def serial_by_bw(cell):
    return "d0" if cell["fc"] < cell["comm"] else "r0"


def b3_adapt(cell, q):
    ratio = cell["comm"] / cell["gemm"]
    gap_ok = (cell["iso_gap"] is not None and cell["iso_gap"] <= 2.0)
    return "r1" if (q >= 8 and 0.9 <= ratio <= 1.35 and gap_ok) else serial_by_bw(cell)


def evaluate(name, chooser):
    hits, regs, rows = 0, [], []
    for cellkey, cell in grid.items():
        pick = chooser(cellkey, cell)
        oracle = min(cell["e2e"], key=cell["e2e"].get)
        actual = cell["e2e"][pick]; best = cell["e2e"][oracle]
        reg = 100 * (actual - best) / best
        hits += (pick == oracle); regs.append(reg)
        rows.append((cellkey, pick, oracle, round(reg, 2)))
    regs.sort()
    med = regs[len(regs) // 2]; p95 = regs[min(len(regs) - 1, int(0.95 * len(regs)))]
    print(f"  {name:22s} top1={hits}/{len(grid)}  regret: med={med:.2f}% p95={p95:.2f}%")
    return dict(name=name, top1=f"{hits}/{len(grid)}", regret_median=round(med, 2),
                regret_p95=round(p95, 2), rows=rows)


print(f"selector v0.3 K500SM_AI 复算  网格: {len(grid)} 格 × 4 策略（LOO；d0/d1=f17aae1d clean 根）\n")

loo_picks = {}
for hold in grid:
    train = {k: v for k, v in grid.items() if k != hold}
    a1, b1 = fit_pair(train, "r1"); a2, b2 = fit_pair(train, "d1")
    q = hold[1]
    preds = predict(grid[hold], q, a1, b1, a2, b2)
    loo_picks[hold] = min(preds, key=preds.get)
    grid[hold]["_coeffs"] = (a1, b1, a2, b2)

results = [
    evaluate("selector v0.3 (LOO)", lambda k, c: loo_picks[k]),
    evaluate("always-d0", lambda k, c: "d0"),
    evaluate("always-r1", lambda k, c: "r1"),
    evaluate("serial-by-bandwidth", lambda k, c: serial_by_bw(c)),
    evaluate("B3 适配版", lambda k, c: b3_adapt(c, k[1])),
    evaluate("probe-1iter", lambda k, c: min(c["probe"], key=c["probe"].get)),
]

with open(os.path.join(OUT, "selector_v03_k500_summary.csv"), "w") as f:
    f.write("method,top1,regret_median_pct,regret_p95_pct\n")
    for r in results:
        f.write(f"{r['name']},{r['top1']},{r['regret_median']},{r['regret_p95']}\n")
with open(os.path.join(OUT, "selector_v03_k500_choices.csv"), "w") as f:
    f.write("N,q,oracle(e2e_us)," + ",".join(r["name"] for r in results) + ",iso_gap_pct\n")
    idx = {k: i for i, k in enumerate(grid)}
    for k in sorted(grid):
        oracle = min(grid[k]["e2e"], key=grid[k]["e2e"].get)
        picks = [r["rows"][idx[k]][1] for r in results]
        gap = grid[k]["iso_gap"]
        f.write(f"{k[0]},{k[1]},{oracle}({grid[k]['e2e'][oracle]:.0f})," +
                ",".join(picks) + "," + (f"{gap:.2f}" if gap is not None else "NA") + "\n")
json.dump({f"N{N}q{q}": grid[(N, q)]["_coeffs"] for (N, q) in sorted(grid)},
          open(os.path.join(OUT, "loo_coeffs.json"), "w"), indent=1)
print(f"\n产出: {OUT}/")
