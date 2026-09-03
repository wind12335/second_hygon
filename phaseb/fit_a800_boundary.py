#!/usr/bin/env python3
"""A800 版边界定律重拟合（2026-09-03，回 NVIDIA【9】合作提议）.

数据源（全部来自 second_nvidia 只读 clone，A800/4GPU，未锁频，p50 逐迭代池化）：
  - formal 3×3（N512/2048/4096 × q2/4/8）d0/d1 中位（d_family_medians_partial_20260903.csv，n=400/格）
  - matched q16 两格（N2048/q16、N8192/q16）从 mh_d0/mh_d1 raw_global_samples.csv 现算（n=250/格）
  - locked_rerun 10-rep（N2048/q8、N4096/q8）作为敏感性对照（locked=0，实为多样本符号稳定口径）

模型与 K500SM_AI 完全同构（闭式可比性优先）：
  P = (T_d1−T_d0)/T_d0 = a + b·q + (c + d·q)·cols   [pt, cols=N/q, 正=d1 慢]
  K500SM_AI 参照：a=−44.3, b=+9.22, c=+0.032, d=−0.0154（R²=0.907, 符号 11/11, LOO 15pt）

产出：second_hygon/results/a800_boundary_refit_20260903.csv + 控制台判定
"""
import csv
import glob
import os
import statistics

NV = "/root/private_data/lyc/second_nvidia/phaseb-nvidia-port/results"
OUT_CSV = "/root/private_data/lyc/second_hygon/results/a800_boundary_refit_20260903.csv"
HYGON = {"a": -44.3, "b": 9.22, "c": 0.032, "d": -0.0154}


def pooled_median(paths):
    values = []
    for path in paths:
        sample = os.path.join(path, "raw_global_samples.csv")
        if not os.path.exists(sample):
            continue
        with open(sample, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("e2e_max_us") not in (None, ""):
                    values.append(float(row["e2e_max_us"]))
    return statistics.median(values) if values else None, len(values)


def main():
    cells = {}  # (N,q) -> {"d0":us, "d1":us, "n_d0":, "n_d1":, "src":}

    # 1) formal 3×3（该 CSV 表头 5 列但数据 6 列：path,N,q,iters,p50,p95 —— 按位切分）
    with open(os.path.join(NV, "a800_4gpu/phaseb_formal_20260902_232036",
                           "d_family_medians_partial_20260903.csv")) as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6 or parts[0] in ("path", "d1_vs_d0"):
                continue
            path_name, n, q, _iters, p50, _p95 = parts[0], int(parts[1][1:]), \
                int(parts[2][1:]), parts[3], float(parts[4]), float(parts[5])
            cells.setdefault((n, q), {"src": "formal"})
            if path_name == "d0":
                cells[(n, q)]["d0"] = p50; cells[(n, q)]["n_d0"] = 400
            elif path_name == "d1":
                cells[(n, q)]["d1"] = p50; cells[(n, q)]["n_d1"] = 400

    # 2) matched q16 两格（现算，池化全部 rep×iter）
    matched_root = os.path.join(NV, "matched_hygon_20260902T163728Z")
    for (n, q) in [(2048, 16), (8192, 16)]:
        entry = cells.setdefault((n, q), {"src": "matched"})
        entry["src"] = "matched"
        for path_name in ("d0", "d1"):
            dirs = sorted(glob.glob(os.path.join(
                matched_root, f"mh_{path_name}_C0_DEFAULT_N{n}_q{q}_rep*")))
            med, count = pooled_median(dirs)
            entry[path_name] = med
            entry[f"n_{path_name}"] = count

    # 3) locked_rerun 10-rep 敏感性（不进主拟合，只对照）
    locked_root = os.path.join(NV, "locked_rerun_20260903T013721Z")
    locked = {}
    for (n, q) in [(2048, 8), (4096, 8)]:
        entry = {"d0": None, "d1": None}
        for path_name in ("d0", "d1"):
            dirs = sorted(glob.glob(os.path.join(
                locked_root, f"lr_{path_name}_N{n}_q{q}_rep*")))
            med, count = pooled_median(dirs)
            entry[path_name] = med
            entry["n"] = count
        if entry["d0"] and entry["d1"]:
            locked[(n, q)] = (entry["d1"] - entry["d0"]) / entry["d0"] * 100

    # ---- 拟合（最小二乘，无 numpy 依赖：4 参数正规方程手解） ----
    rows = []
    for (n, q), entry in sorted(cells.items()):
        if "d0" not in entry or "d1" not in entry or not entry.get("d0"):
            continue
        cols = n // q
        p_meas = (entry["d1"] - entry["d0"]) / entry["d0"] * 100
        rows.append({"N": n, "q": q, "cols": cols, "d0_us": entry["d0"], "d1_us": entry["d1"],
                     "P_measured_pt": p_meas, "src": entry["src"]})
    rows.sort(key=lambda r: (r["q"], r["N"]))

    def design(r):
        return [1.0, float(r["q"]), float(r["cols"]), r["q"] * r["cols"]]

    def solve(pts):
        m = len(pts)
        A = [design(r) for r in pts]
        y = [r["P_measured_pt"] for r in pts]
        # normal equations (AtA)x = Aty
        k = 4
        ata = [[sum(A[i][a] * A[i][b] for i in range(m)) for b in range(k)] for a in range(k)]
        aty = [sum(A[i][a] * y[i] for i in range(m)) for a in range(k)]
        # gaussian elimination
        for col in range(k):
            pivot = max(range(col, k), key=lambda r2: abs(ata[r2][col]))
            ata[col], ata[pivot] = ata[pivot], ata[col]
            aty[col], aty[pivot] = aty[pivot], aty[col]
            for r2 in range(col + 1, k):
                factor = ata[r2][col] / ata[col][col]
                for c2 in range(col, k):
                    ata[r2][c2] -= factor * ata[col][c2]
                aty[r2] -= factor * aty[col]
        x = [0.0] * k
        for r2 in reversed(range(k)):
            x[r2] = (aty[r2] - sum(ata[r2][c2] * x[c2] for c2 in range(r2 + 1, k))) / ata[r2][r2]
        return x

    beta = solve(rows)

    def predict(beta_vec, r):
        d_ = design(r)
        return sum(b * v for b, v in zip(beta_vec, d_))

    mean_y = statistics.mean(r["P_measured_pt"] for r in rows)
    ss_tot = sum((r["P_measured_pt"] - mean_y) ** 2 for r in rows)
    ss_res = sum((r["P_measured_pt"] - predict(beta, r)) ** 2 for r in rows)
    r2 = 1 - ss_res / ss_tot

    # LOO
    loo_errs = []
    sign_hits = 0
    for i, held in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        loo_beta = solve(rest)
        loo_errs.append(abs(predict(loo_beta, held) - held["P_measured_pt"]))
        if (predict(loo_beta, held) > 0) == (held["P_measured_pt"] > 0):
            sign_hits += 1
    loo_rmse = (sum(e ** 2 for e in loo_errs) / len(loo_errs)) ** 0.5

    for r in rows:
        r["P_fitted_pt"] = predict(beta, r)
        r["residual_pt"] = r["P_measured_pt"] - r["P_fitted_pt"]

    a, b, c, d = beta

    # 边界 N*(q)：解 P=0 → cols* = −(a+b·q)/(c+d·q)（若存在正根）
    def n_star(qv):
        denom = c + d * qv
        if denom == 0:
            return None
        cols_star = -(a + b * qv) / denom
        if cols_star <= 0:
            return None
        return cols_star * qv

    # 弯曲诊断：q8 三点线性 vs 实测
    q8 = sorted([r for r in rows if r["q"] == 8], key=lambda r: r["cols"])
    seg = []
    for i in range(len(q8) - 1):
        slope = (q8[i + 1]["P_measured_pt"] - q8[i]["P_measured_pt"]) / \
                (q8[i + 1]["cols"] - q8[i]["cols"])
        seg.append(slope)
    curvature = "凸（斜率绝对值随 cols 收敛：{}）".format(
        " → ".join(f"{s:.4f}" for s in seg)) if len(seg) == 2 and abs(seg[1]) < abs(seg[0]) else \
        ("线性或凹：{} → {}".format(*(f"{s:.4f}" for s in seg)) if seg else "数据不足")

    # A800 闭式外推 N8192/q8
    extrapolated = predict(beta, {"q": 8, "cols": 1024})

    fields = ["N", "q", "cols", "src", "d0_us", "d1_us", "P_measured_pt", "P_fitted_pt",
              "residual_pt"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k])
                             for k in fields})

    print(f"A800 闭式: P = {a:.2f} + {b:.3f}·q + ({c:.5f} + {d:.5f}·q)·cols  [pt]")
    print(f"  R²={r2:.3f}  符号 LOO {sign_hits}/{len(rows)}  LOO-RMSE={loo_rmse:.1f}pt  n={len(rows)}")
    print(f"K500SM_AI 参照: a={HYGON['a']} b={HYGON['b']} c={HYGON['c']} d={HYGON['d']}")
    print(f"边界 N*(q8)={'不存在(P 恒正)' if n_star(8) is None else f'{n_star(8):.0f}'}"
          f"  N*(q16)={'不存在' if n_star(16) is None else f'{n_star(16):.0f}'}")
    print(f"q8 cols 斜率分段: {curvature}")
    print(f"A800 闭式外推 N8192/q8 (cols=1024): {extrapolated:.1f}pt")
    if locked:
        for k, v in sorted(locked.items()):
            main_p = next((r["P_measured_pt"] for r in rows if (r["N"], r["q"]) == k), None)
            print(f"locked_rerun(10rep) {k}: P={v:.1f}% vs 主拟合值 {main_p:.1f}% "
                  f"（Δ={v - main_p:+.1f}pt，符号{'一致' if (v > 0) == (main_p > 0) else '翻转'}）")
    print(f"csv -> {OUT_CSV}")


if __name__ == "__main__":
    main()
