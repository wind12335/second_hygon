#!/usr/bin/env python3
"""Analyze the targeted K500SM_AI/gfx928 Phase-2 confirmation experiment."""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


def percentile(values, point):
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * point
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    if not values:
        return {"n": 0, "p05": float("nan"), "p50": float("nan"), "p95": float("nan"),
                "mean": float("nan"), "std": float("nan"), "cv": float("nan")}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "p05": percentile(values, 0.05), "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95), "mean": mean, "std": std,
            "cv": 100.0 * std / mean if mean else 0.0}


def number(value):
    return f"{value:.6f}" if not math.isnan(value) else ""


def main(root):
    output = os.path.join(root, "summary")
    os.makedirs(output, exist_ok=True)
    manifest_path = os.path.join(root, "phase2_case_manifest.tsv")
    manifests = {}
    with open(manifest_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            manifests[row["case_dir"]] = row

    case_rows = []
    audit_rows = []
    for raw_path in sorted(glob.glob(os.path.join(root, "cases", "*", "raw_global_samples.csv"))):
        case_dir = os.path.dirname(raw_path)
        manifest = manifests.get(case_dir)
        if not manifest:
            audit_rows.append([raw_path, "NO", "no matching case manifest"])
            continue
        with open(raw_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"e2e_max_us", "t_done_max_us", "t_release_first_max_us", "correctness_all_ranks"}
        if not rows or not required.issubset(rows[0]):
            audit_rows.append([raw_path, "NO", "unexpected raw_global schema"])
            continue
        valid = [row for row in rows if row["correctness_all_ranks"] == "PASS"]
        metrics = {
            key: describe([float(row[field]) for row in valid])
            for key, field in (
                ("e2e", "e2e_max_us"), ("done", "t_done_max_us"),
                ("release_first", "t_release_first_max_us"),
                ("release_last", "t_release_last_max_us"),
                ("gemm_interval", "gemm_interval_max_us"),
            )
        }
        record = dict(manifest)
        record["raw_global_csv"] = raw_path
        record["sample_count"] = len(rows)
        record["pass_count"] = len(valid)
        record.update({f"{name}_{stat}": value for name, stats in metrics.items()
                       for stat, value in stats.items()})
        case_rows.append(record)
        audit_rows.append([raw_path, "YES", "matching manifest and valid raw_global schema"])

    case_fields = [
        "case_id", "phase", "repetition", "shape_id", "m_local", "n", "k", "q", "path",
        "candidate", "status", "sample_count", "pass_count", "raw_global_csv",
        "e2e_p05", "e2e_p50", "e2e_p95", "e2e_mean", "e2e_std", "e2e_cv",
        "done_p05", "done_p50", "done_p95", "release_first_p50", "release_last_p50",
        "gemm_interval_p50",
    ]
    with open(os.path.join(output, "phase2_case_level_summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        for record in sorted(case_rows, key=lambda item: item["case_id"]):
            out = {}
            for field in case_fields:
                value = record.get(field, "")
                out[field] = number(value) if isinstance(value, float) else value
            writer.writerow(out)

    def aggregate(path, candidate, q, metric="e2e_p50"):
        selected = [row[metric] for row in case_rows if row["path"] == path and
                    row["candidate"] == candidate and row["q"] == str(q)]
        return describe(selected)

    reversal_fields = [
        "M", "N", "K", "q", "comm_p50_by_candidate", "isolated_comm_best",
        "isolated_comm_best_p50_us", "h0_p50_by_candidate", "h0_best",
        "h0_best_p50_us", "h0_isolated_best_p50_us", "h0_margin_vs_isolated_best_percent",
        "h0_per_process_p50_by_candidate", "h0_winner_beats_comm_winner_all_repetitions",
        "b1_h0_c0_gain_percent", "b1_h0_c2_gain_percent", "evidence_status", "caveat",
    ]
    reversal_rows = []
    candidates = ["C0_DEFAULT", "C1_RING_SIMPLE_CH4", "C2_RING_SIMPLE_CH8"]
    for q in (2, 8):
        comm = {candidate: aggregate("comm", candidate, q, "done_p50") for candidate in candidates}
        h0 = {candidate: aggregate("h0", candidate, q, "e2e_p50") for candidate in candidates}
        if any(stats["n"] != 5 for stats in comm.values()) or any(stats["n"] != 5 for stats in h0.values()):
            continue
        comm_best = min(candidates, key=lambda candidate: comm[candidate]["p50"])
        h0_best = min(candidates, key=lambda candidate: h0[candidate]["p50"])
        per_rep = {}
        for candidate in candidates:
            per_rep[candidate] = {row["repetition"]: row["e2e_p50"] for row in case_rows
                                  if row["path"] == "h0" and row["candidate"] == candidate and
                                  row["q"] == str(q)}
        common_reps = sorted(set(per_rep[h0_best]).intersection(per_rep[comm_best]))
        all_wins = h0_best != comm_best and bool(common_reps) and all(
            per_rep[h0_best][rep] < per_rep[comm_best][rep] for rep in common_reps)
        margin = 100.0 * (h0[comm_best]["p50"] / h0[h0_best]["p50"] - 1.0)
        b1_c0 = aggregate("b1", "C0_DEFAULT", q, "e2e_p50")
        b1_c2 = aggregate("b1", "C2_RING_SIMPLE_CH8", q, "e2e_p50")
        gain_c0 = 100.0 * (b1_c0["p50"] / h0["C0_DEFAULT"]["p50"] - 1.0)
        gain_c2 = 100.0 * (b1_c2["p50"] / h0["C2_RING_SIMPLE_CH8"]["p50"] - 1.0)
        if h0_best != comm_best and all_wins and margin >= 5.0:
            evidence = "CONFIRMED_STRONG_REVERSAL"
        elif h0_best != comm_best and all_wins:
            evidence = "CONFIRMED_SMALL_REVERSAL"
        else:
            evidence = "NO_REVERSAL_CONTROL"
        reversal_rows.append({
            "M": "2048", "N": "2048", "K": "2048", "q": q,
            "comm_p50_by_candidate": ";".join(f"{candidate}:{comm[candidate]['p50']:.3f}" for candidate in candidates),
            "isolated_comm_best": comm_best,
            "isolated_comm_best_p50_us": number(comm[comm_best]["p50"]),
            "h0_p50_by_candidate": ";".join(f"{candidate}:{h0[candidate]['p50']:.3f}" for candidate in candidates),
            "h0_best": h0_best,
            "h0_best_p50_us": number(h0[h0_best]["p50"]),
            "h0_isolated_best_p50_us": number(h0[comm_best]["p50"]),
            "h0_margin_vs_isolated_best_percent": number(margin),
            "h0_per_process_p50_by_candidate": ";".join(
                f"{candidate}:" + "/".join(f"r{rep}={per_rep[candidate][rep]:.3f}"
                                                   for rep in sorted(per_rep[candidate]))
                for candidate in candidates),
            "h0_winner_beats_comm_winner_all_repetitions": "YES" if all_wins else "NO",
            "b1_h0_c0_gain_percent": number(gain_c0),
            "b1_h0_c2_gain_percent": number(gain_c2),
            "evidence_status": evidence,
            "caveat": "Synthetic engineering shape; release curve and repeat evidence, not a final model trace result.",
        })
    with open(os.path.join(output, "phase2_reversal_analysis.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reversal_fields)
        writer.writeheader()
        writer.writerows(reversal_rows)

    slice_groups = defaultdict(list)
    for raw_path in sorted(glob.glob(os.path.join(root, "cases", "*", "release_slices_global.csv"))):
        case_dir = os.path.dirname(raw_path)
        if case_dir not in manifests:
            continue
        with open(raw_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["correctness_all_ranks"] != "PASS":
                    continue
                key = (row["path"], row["candidate"], row["M"], row["N"], row["K"], row["q"],
                       row["slice_index"], row["slice_bytes"])
                slice_groups[key].append(row)
    curve_fields = [
        "path", "candidate", "M", "N", "K", "q", "slice_index", "slice_bytes", "sample_count",
        "release_p05_us", "release_p50_us", "release_p95_us", "gemm_start_p50_us",
        "gemm_end_p50_us", "gemm_duration_p50_us", "consumer_wait_p50_us",
    ]
    with open(os.path.join(output, "phase2_release_curve_summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for key, rows in sorted(slice_groups.items()):
            release = [float(row["t_release_max_us"]) for row in rows]
            start = [float(row["t_gemm_start_max_us"]) for row in rows]
            end = [float(row["t_gemm_end_max_us"]) for row in rows]
            duration = [float(row["t_gemm_duration_max_us"]) for row in rows]
            waits = [start_value - release_value for start_value, release_value in zip(start, release)]
            row = dict(zip(["path", "candidate", "M", "N", "K", "q", "slice_index", "slice_bytes"], key))
            row.update({
                "sample_count": len(rows), "release_p05_us": number(percentile(release, 0.05)),
                "release_p50_us": number(percentile(release, 0.50)),
                "release_p95_us": number(percentile(release, 0.95)),
                "gemm_start_p50_us": number(percentile(start, 0.50)),
                "gemm_end_p50_us": number(percentile(end, 0.50)),
                "gemm_duration_p50_us": number(percentile(duration, 0.50)),
                "consumer_wait_p50_us": number(percentile(waits, 0.50)),
            })
            writer.writerow(row)

    with open(os.path.join(output, "phase2_input_audit.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_global_csv", "accepted", "reason"])
        writer.writerows(audit_rows)

    report_path = os.path.join(output, "phase2_analysis.md")
    with open(report_path, "w") as report:
        report.write("# Phase 2 Targeted Confirmation Analysis\n\n")
        report.write("Platform: K500SM_AI / gfx928 / 4 GPUs / PCIe. All values are PASS-only max-rank device-event samples.\n\n")
        report.write(f"- Accepted raw global CSVs: {len(case_rows)}\n")
        report.write(f"- Expected target cases: 91\n")
        report.write("- COMM_ONLY and H0 have five independent MPI process repetitions per candidate and q.\n")
        report.write("- Per-slice release/start/end samples are in `release_slices_global.csv` per case and aggregated in `phase2_release_curve_summary.csv`.\n\n")
        for row in reversal_rows:
            report.write(f"## q={row['q']}\n\n")
            report.write(f"- COMM_ONLY p50: {row['comm_p50_by_candidate']}; best `{row['isolated_comm_best']}`.\n")
            report.write(f"- H0 p50: {row['h0_p50_by_candidate']}; best `{row['h0_best']}`.\n")
            report.write(f"- H0 winner advantage over the isolated-communication winner: {row['h0_margin_vs_isolated_best_percent']}%.\n")
            report.write(f"- H0 process medians: {row['h0_per_process_p50_by_candidate']}.\n")
            report.write(f"- H0 winner is faster in every matched process repetition: {row['h0_winner_beats_comm_winner_all_repetitions']}.\n")
            report.write(f"- H0 vs B1 gain: C0={row['b1_h0_c0_gain_percent']}%, C2={row['b1_h0_c2_gain_percent']}%.\n")
            report.write(f"- Evidence status: `{row['evidence_status']}`.\n\n")
        report.write("## Scope Boundary\n\n")
        report.write("This confirms or rejects a controlled engineering counterexample. It does not by itself establish a new backend, a DUSHMEM result, or end-to-end model-training benefit. Those require a later valid-primitive test and trace-derived workloads.\n")
    print(os.path.join(output, "phase2_reversal_analysis.csv"))
    print(os.path.join(output, "phase2_release_curve_summary.csv"))
    print(report_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_phase2.py RESULT_ROOT")
    main(sys.argv[1])
