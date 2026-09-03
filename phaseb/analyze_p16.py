#!/usr/bin/env python3
"""P16 summarizer — 边界定律外推三格裁决.

Reads every case dir under <result_root>/cases (manifest.csv + raw_global_samples.csv,
same e2e_max_us p50 口径 as analyze_phaseb.py), computes P=(T_d1−T_d0)/T_d0 per cell,
compares against the preregistered closed form, writes:

  p16_extrap_verdict.csv  per cell: d0/d1 p50, P_meas, P_fit, residual, E1/E2/E3 verdict
  p16_verdicts.txt        E1–E4 one-liners

Preregistration: phaseb/P16_预注册_边界外推_20260903.md (locked before launch).
Closed form: P ≈ −44.3 + 9.22·q + (0.032−0.0154·q)·cols  [pt, cols=N/q, sign: + = d1 slower]
"""
import argparse
import csv
import glob
import math
import os
import statistics

# (N, q) -> P_fit[pt]  (代入闭式，见预注册 §0)
PRED = {
    (4096, 32): 191.7,   # E3 只判单调: P > +80 且 T_d1 > T_d0
    (8192, 8): -63.9,    # E1: P < −30
    (8192, 16): -6.5,    # E2 主裁决格: −20 < P < +10
}


def load_case(case_dir):
    manifest_path = os.path.join(case_dir, "manifest.csv")
    samples_path = os.path.join(case_dir, "raw_global_samples.csv")
    if not (os.path.exists(manifest_path) and os.path.exists(samples_path)):
        return None
    try:
        with open(manifest_path, newline="") as handle:
            manifest = next(csv.DictReader(handle))
        with open(samples_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        e2e = [float(row["e2e_max_us"]) for row in rows
               if row.get("e2e_max_us") not in (None, "")]
        correctness = [row.get("correctness_all_ranks", "?") for row in rows]
        status = "PASS" if correctness and all(v == "PASS" for v in correctness) else "FAIL"
    except (StopIteration, OSError, csv.Error, ValueError):
        return None
    if not e2e:
        return None
    exit_path = os.path.join(case_dir, "exit_status.txt")
    exit_code = None
    if os.path.exists(exit_path):
        with open(exit_path) as handle:
            exit_code = handle.read().strip()
    return {"manifest": manifest, "p50": statistics.median(e2e), "status": status,
            "exit_code": exit_code}


def median_of(values):
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else float("nan")


def fmt(value):
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()

    cases_root = os.path.join(args.result_root, "cases")
    summary_root = os.path.join(args.result_root, "summary")
    os.makedirs(summary_root, exist_ok=True)

    groups = {}
    for case_dir in sorted(glob.glob(os.path.join(cases_root, "case*"))):
        if not os.path.isdir(case_dir):
            continue
        case = load_case(case_dir)
        if case is None:
            continue
        if case["exit_code"] not in (None, "0") or case["status"] != "PASS":
            continue
        m = case["manifest"]
        # manifest path 是全名（D0_FCOLLECT_SERIAL 等），查找用短名——键上归一化（同 analyze_p15 修复）
        SHORT = {"D0_FCOLLECT_SERIAL": "d0", "D1_PUSHSIG_OVERLAP": "d1",
                 "R1_EVENT_OVERLAP": "r1"}
        groups.setdefault((int(m["N"]), int(m["q"]), SHORT.get(m["path"], m["path"])),
                          []).append(case["p50"])

    out_rows, verdicts = [], []
    residuals_n8192 = []
    for (n, q) in sorted(PRED):
        d0 = median_of(groups.get((n, q, "d0"), []))
        d1 = median_of(groups.get((n, q, "d1"), []))
        p_meas = (d1 - d0) / d0 * 100 if (d0 == d0 and d1 == d1 and d0) else float("nan")
        p_fit = PRED[(n, q)]
        residual = p_meas - p_fit if p_meas == p_meas else float("nan")
        if n == 8192 and residual == residual:
            residuals_n8192.append(abs(residual))
        if (n, q) == (8192, 8):
            verdict = "HIT" if p_meas == p_meas and p_meas < -30 else (
                "MISS" if p_meas == p_meas else "NO_DATA")
            tag = "E1"
        elif (n, q) == (8192, 16):
            tag = "E2"
            if p_meas != p_meas:
                verdict = "NO_DATA"
            elif -20 < p_meas < 10:
                verdict = "HIT(带内)"
            elif p_meas <= -20:
                verdict = "符号HIT+幅度低估(边界右移,进模型修订)"
            else:
                verdict = "MISS(闭式该区失效)"
        else:
            tag = "E3"
            if p_meas != p_meas:
                verdict = "NO_DATA"
            elif p_meas > 80 and d1 == d1 and d0 == d0 and d1 > d0:
                verdict = "HIT(q爆炸延续到q32)"
            else:
                verdict = "UNDETERMINED-饱和(q爆炸存在上界,非MISS,同样可写)"
        out_rows.append({
            "N": n, "q": q, "cols": n // q, "tag": tag,
            "n_d0_reps": len(groups.get((n, q, "d0"), [])),
            "n_d1_reps": len(groups.get((n, q, "d1"), [])),
            "d0_p50_us": d0, "d1_p50_us": d1, "P_meas_pt": p_meas, "P_fit_pt": p_fit,
            "residual_pt": residual, "verdict": verdict,
        })
        verdicts.append(f"{tag} N{n}/q{q}: P_meas={fmt(p_meas)}pt "
                        f"(fit {p_fit:+.1f}, residual {fmt(residual)}) → {verdict}")

    if residuals_n8192:
        worst = max(residuals_n8192)
        verdicts.append(f"E4 N8192 两格 |residual|max={worst:.1f}pt → "
                        + ("同LOO量级(≤30),外推有效" if worst <= 30 else
                           ">30pt 系统性偏离,模型需N交互项(符号律不推翻)"))
    else:
        verdicts.append("E4 无 N8192 有效数据")

    fields = ["N", "q", "cols", "tag", "n_d0_reps", "n_d1_reps", "d0_p50_us", "d1_p50_us",
              "P_meas_pt", "P_fit_pt", "residual_pt", "verdict"]
    with open(os.path.join(summary_root, "p16_extrap_verdict.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    with open(os.path.join(summary_root, "p16_verdicts.txt"), "w") as handle:
        handle.write("\n".join(verdicts) + "\n")
    print("\n".join(verdicts))
    print(f"summary -> {summary_root}")


if __name__ == "__main__":
    main()
