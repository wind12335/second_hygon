#!/usr/bin/env python3
"""P15 summarizer — window_mult 曲线 + r1 家族轴复判.

Reads every case dir under <result_root>/cases (manifest.csv + raw_global_samples.csv,
same e2e_max_us p50 口径 as analyze_phaseb.py), produces in <result_root>/summary:

  p15_wm_curve.csv   per (cell, cfg): d1 p50/p95, in-batch d0, penalty P, delta vs wm1,
                     D1 direction verdict per cell + 主格落带
  p15_r1_family.csv  per cell: new r1 p50, old-root r1 (ref csv), drift, dc clean,
                     dc-r1 new sign vs old sign, R1/R2 verdict

Preregistration: phaseb/P15_预注册_window_mult与r1同根_20260903.md (locked before launch).
"""
import argparse
import csv
import glob
import math
import os
import statistics

WM_ROLES = {
    (2048, 8): "main",      # clean +29.5pt
    (512, 8): "sub",        # clean +40.0pt
    (2048, 16): "boom",     # clean +69.9pt
    (4096, 8): "win",       # clean -21.0pt
}


def load_case(case_dir):
    manifest_path = os.path.join(case_dir, "manifest.csv")
    samples_path = os.path.join(case_dir, "raw_global_samples.csv")
    if not (os.path.exists(manifest_path) and os.path.exists(samples_path)):
        return None
    try:
        with open(manifest_path, newline="") as handle:
            manifest = next(csv.DictReader(handle))
        e2e, e2e_p95_source = [], []
        with open(samples_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("e2e_max_us") not in (None, ""):
                e2e.append(float(row["e2e_max_us"]))
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
    return {
        "manifest": manifest,
        "p50": statistics.median(e2e),
        "p95": sorted(e2e)[min(len(e2e) - 1, max(0, int(round(0.95 * (len(e2e) - 1)))))],
        "status": status,
        "exit_code": exit_code,
    }


def median_of(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(values) if values else float("nan")


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--ref-csv", default="",
                        help="family_axis_dushmem_vs_rccl_20260902.csv (old r1 / dc clean)")
    args = parser.parse_args()

    cases_root = os.path.join(args.result_root, "cases")
    summary_root = os.path.join(args.result_root, "summary")
    os.makedirs(summary_root, exist_ok=True)

    cases = []
    for case_dir in sorted(glob.glob(os.path.join(cases_root, "case*"))):
        if not os.path.isdir(case_dir):
            continue
        loaded = load_case(case_dir)
        if loaded is None:
            continue
        loaded["case_id"] = os.path.basename(case_dir)
        cases.append(loaded)

    ok_cases = [c for c in cases if c["exit_code"] in (None, "0") and c["status"] == "PASS"]

    # manifest 的 path 是全名（D1_PUSHSIG_OVERLAP 等），Part A/B 查找用短名——在键上归一化
    SHORT = {"D0_FCOLLECT_SERIAL": "d0", "D1_PUSHSIG_OVERLAP": "d1",
             "R1_EVENT_OVERLAP": "r1"}

    def key_of(case):
        m = case["manifest"]
        return (int(m["N"]), int(m["q"]), SHORT.get(m["path"], m["path"]),
                int(m["window_mult"]))

    groups = {}
    for case in ok_cases:
        groups.setdefault(key_of(case), []).append(case)

    # ---- Part A: window_mult curve -----------------------------------------
    wm_rows = []
    for (n, q) in sorted(WM_ROLES):
        role = WM_ROLES[(n, q)]
        d0_p50 = median_of([c["p50"] for c in groups.get((n, q, "d0", 1), [])])
        t_by_wm = {}
        for wm in (1, 2, 4):
            t_by_wm[wm] = median_of([c["p50"] for c in groups.get((n, q, "d1", wm), [])])
        t1 = t_by_wm.get(1, float("nan"))
        for wm in (1, 2, 4):
            t = t_by_wm.get(wm, float("nan"))
            penalty = (t - d0_p50) / d0_p50 * 100 if (t and d0_p50) else float("nan")
            delta = (t / t1 - 1) * 100 if (t and t1) else float("nan")
            wm_rows.append({
                "N": n, "q": q, "role": role, "cfg": f"d1wm{wm}", "window_mult": wm,
                "n_reps": len(groups.get((n, q, "d1", wm), [])),
                "d1_p50_us": t, "d1_p95_us": median_of(
                    [c["p95"] for c in groups.get((n, q, "d1", wm), [])]),
                "d0_p50_us_inbatch": d0_p50,
                "P_penalty_pt": penalty, "delta_vs_wm1_pct": delta,
            })
        d0_rows = groups.get((n, q, "d0", 1), [])
        wm_rows.append({
            "N": n, "q": q, "role": role, "cfg": "d0", "window_mult": 1,
            "n_reps": len(d0_rows),
            "d1_p50_us": median_of([c["p50"] for c in d0_rows]),
            "d1_p95_us": median_of([c["p95"] for c in d0_rows]),
            "d0_p50_us_inbatch": d0_p50,
            "P_penalty_pt": 0.0, "delta_vs_wm1_pct": float("nan"),
        })

    verdicts = []
    for row in wm_rows:
        if row["cfg"] == "d1wm4":
            t1 = next((r["d1_p50_us"] for r in wm_rows
                       if (r["N"], r["q"], r["cfg"]) == (row["N"], row["q"], "d1wm1")),
                      float("nan"))
            ratio = row["d1_p50_us"] / t1 if (row["d1_p50_us"] and t1) else float("nan")
            if row["n_reps"] == 0 or ratio != ratio:
                row["d1_direction"] = "NO_DATA"
            else:
                row["d1_direction"] = "HIT" if ratio <= 1.02 else "MISS"
            row["wm4_over_wm1"] = (ratio - 1) * 100 if ratio == ratio else float("nan")
        else:
            row["d1_direction"] = ""
            row["wm4_over_wm1"] = float("nan")

    main_row = next((r for r in wm_rows if r["role"] == "main" and r["cfg"] == "d1wm4"), None)
    main_band = ""
    if main_row and main_row["wm4_over_wm1"] == main_row["wm4_over_wm1"]:
        d = main_row["wm4_over_wm1"]
        if -25 <= d <= -5:
            main_band = "credit一阶载体(D1强HIT)"
        elif d < -25:
            main_band = "强于预测(幅度超带,记录)"
        elif d <= 0:
            main_band = "部分载体"
        elif d <= 2:
            main_band = "噪声带内(平)"
        else:
            main_band = "反向=发现F同构(D1 MISS,通报NVIDIA)"

    fields = ["N", "q", "role", "cfg", "window_mult", "n_reps", "d1_p50_us", "d1_p95_us",
              "d0_p50_us_inbatch", "P_penalty_pt", "delta_vs_wm1_pct", "wm4_over_wm1",
              "d1_direction"]
    with open(os.path.join(summary_root, "p15_wm_curve.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in wm_rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})
    verdicts.append(f"D1 主格(N2048/q8) wm4/wm1 = "
                    f"{fmt(main_row['wm4_over_wm1']) if main_row else 'N/A'}% → {main_band or 'N/A'}")

    # ---- Part B: r1 family axis --------------------------------------------
    ref = {}
    if args.ref_csv and os.path.exists(args.ref_csv):
        with open(args.ref_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                ref[(int(row["N"]), int(row["q"]))] = row

    r1_rows = []
    sign_flips = 0
    for (n, q, path, wm), group in sorted(groups.items()):
        if path != "r1":
            continue
        new_p50 = median_of([c["p50"] for c in group])
        old = ref.get((n, q), {})
        old_p50 = float(old["r1_e2e_p50_us"]) if old.get("r1_e2e_p50_us") else float("nan")
        dc = float(old["dc_clean_e2e_p50_us"]) if old.get("dc_clean_e2e_p50_us") else float("nan")
        drift = (new_p50 / old_p50 - 1) * 100 if (new_p50 == new_p50 and old_p50 == old_p50
                                                  and old_p50) else float("nan")
        new_minus = (dc / new_p50 - 1) * 100 if (dc == dc and new_p50 == new_p50 and
                                                 new_p50) else float("nan")
        old_minus = float(old["dc_minus_r1_pct"]) if old.get("dc_minus_r1_pct") else float("nan")

        def sign(v):
            return "" if v != v else ("dc胜" if v > 0 else "r1胜")
        consistent = "" if (new_minus != new_minus or old_minus != old_minus) else \
            ("一致" if sign(new_minus) == sign(old_minus) else "翻转")
        if consistent == "翻转":
            sign_flips += 1
        if (n, q) == (4096, 16):
            r2 = "HIT(反超带不扩q16)" if new_minus > 0 else "MISS(dc反超扩入q16)"
        else:
            r2 = ""
        r1_rows.append({
            "N": n, "q": q, "n_reps": len(group), "r1_new_p50_us": new_p50,
            "r1_old_p50_us": old_p50, "drift_pct": drift,
            "r1_old_vs_new_R1": "HIT(<2%)" if (drift == drift and abs(drift) < 2) else
            ("超2%红线" if drift == drift else ""),
            "dc_clean_us": dc, "dc_minus_r1_new_pct": new_minus,
            "dc_minus_r1_old_pct": old_minus, "sign_check": consistent, "R2": r2,
        })

    r1_fields = ["N", "q", "n_reps", "r1_new_p50_us", "r1_old_p50_us", "drift_pct",
                 "r1_old_vs_new_R1", "dc_clean_us", "dc_minus_r1_new_pct",
                 "dc_minus_r1_old_pct", "sign_check", "R2"]
    with open(os.path.join(summary_root, "p15_r1_family.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=r1_fields)
        writer.writeheader()
        for row in r1_rows:
            writer.writerow({k: fmt(row.get(k)) for k in r1_fields})
    drift_ok = all(r["r1_old_vs_new_R1"] in ("HIT(<2%)", "") for r in r1_rows)
    verdicts.append(f"R1 漂移<2%红线: {'全部通过' if drift_ok else '存在超线格'}; "
                    f"符号翻转数={sign_flips}")

    meta_path = os.path.join(summary_root, "p15_verdicts.txt")
    with open(meta_path, "w") as handle:
        handle.write("\n".join(verdicts) + "\n")
    print("\n".join(verdicts))
    print(f"summary -> {summary_root}")


if __name__ == "__main__":
    main()
