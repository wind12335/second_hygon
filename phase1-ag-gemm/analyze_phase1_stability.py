#!/usr/bin/env python3
"""Audit Phase-1 repeat stability without altering raw experiment data.

The benchmark has three independent MPI-process repetitions for B0/B1/H0.
This tool keeps those repetitions visible instead of treating all timed
iterations as independent process-level experiments.  COMM_ONLY is deliberately
reported as a single-process measurement in this phase and is labelled as such.
"""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


def percentile(values, percentile_value):
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def fmt(value, digits=6):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def summarize_rows(rows, metric):
    values = [float(row[metric]) for row in rows if row["correctness_all_ranks"] == "PASS"]
    if not values:
        return {"count": 0, "mean": float("nan"), "p05": float("nan"),
                "p50": float("nan"), "p95": float("nan"), "std": float("nan"),
                "cv": float("nan")}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "p05": percentile(values, 0.05),
            "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "std": std, "cv": 100.0 * std / mean if mean else 0.0}


def main(root):
    manifest_path = os.path.join(root, "phase1_case_manifest.tsv")
    output_dir = os.path.join(root, "summary")
    os.makedirs(output_dir, exist_ok=True)

    manifests = {}
    with open(manifest_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            manifests[row["case_dir"]] = row

    case_records = []
    missing = []
    for raw_path in sorted(glob.glob(os.path.join(root, "cases", "*", "raw_global_samples.csv"))):
        case_dir = os.path.dirname(raw_path)
        manifest = manifests.get(case_dir)
        if not manifest:
            missing.append(raw_path)
            continue
        with open(raw_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"e2e_max_us", "t_done_max_us", "t_release_first_max_us",
                    "t_release_last_max_us", "gemm_interval_max_us", "correctness_all_ranks"}
        if not rows or not required.issubset(rows[0]):
            missing.append(raw_path)
            continue
        record = dict(manifest)
        record["raw_global_csv"] = raw_path
        record["sample_count"] = len(rows)
        record["pass_count"] = sum(row["correctness_all_ranks"] == "PASS" for row in rows)
        for label, metric in (
            ("e2e", "e2e_max_us"),
            ("done", "t_done_max_us"),
            ("release_first", "t_release_first_max_us"),
            ("release_last", "t_release_last_max_us"),
            ("gemm_interval", "gemm_interval_max_us"),
        ):
            stats = summarize_rows(rows, metric)
            for name, value in stats.items():
                record[f"{label}_{name}"] = value
        ratios = []
        spans = []
        for row in rows:
            if row["correctness_all_ranks"] != "PASS":
                continue
            done = float(row["t_done_max_us"])
            if done > 0:
                ratios.append(100.0 * (done - float(row["t_release_first_max_us"])) / done)
                spans.append(100.0 * (float(row["t_release_last_max_us"]) -
                                      float(row["t_release_first_max_us"])) / done)
        record["release_window_ratio_p50"] = percentile(ratios, 0.50)
        record["release_span_ratio_p50"] = percentile(spans, 0.50)
        case_records.append(record)

    fields = [
        "case_id", "phase", "repetition", "ranks", "shape_id", "m_local", "n", "k", "q",
        "path", "candidate", "status", "sample_count", "pass_count", "raw_global_csv",
        "e2e_p05", "e2e_p50", "e2e_p95", "e2e_mean", "e2e_std", "e2e_cv",
        "done_p05", "done_p50", "done_p95", "release_first_p50", "release_last_p50",
        "gemm_interval_p50", "release_window_ratio_p50", "release_span_ratio_p50",
    ]
    case_summary_path = os.path.join(output_dir, "phase1_case_level_summary.csv")
    with open(case_summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(case_records, key=lambda value: value["case_id"]):
            out = {}
            for field in fields:
                value = record.get(field, "")
                out[field] = fmt(value) if isinstance(value, float) else value
            writer.writerow(out)

    by_key = defaultdict(list)
    for record in case_records:
        key = (record["m_local"], record["n"], record["k"], record["q"])
        by_key[key].append(record)

    stability_fields = [
        "M", "N", "K", "q", "isolated_comm_best", "isolated_comm_best_done_p50_us",
        "isolated_comm_p50_by_candidate", "isolated_comm_process_repetitions",
        "h0_best", "h0_best_aggregate_p50_us", "h0_isolated_best_aggregate_p50_us",
        "h0_margin_vs_isolated_best_percent", "h0_p50_by_candidate",
        "h0_per_process_p50_by_candidate", "h0_winner_beats_isolated_best_in_all_repetitions",
        "candidate_ranking_reversal", "release_window_c0_p50_percent",
        "release_span_c0_p50_percent", "evidence_status", "caveat",
    ]
    stability_rows = []
    for key, records in sorted(by_key.items()):
        m, n, k, q = key
        if int(m) < 1024:
            continue
        comm = {record["candidate"]: record for record in records if record["path"] == "comm"}
        h0_by_candidate = defaultdict(list)
        for record in records:
            if record["path"] == "h0":
                h0_by_candidate[record["candidate"]].append(record)
        candidates = ["C0_DEFAULT", "C1_RING_SIMPLE_CH4", "C2_RING_SIMPLE_CH8"]
        if any(candidate not in comm or candidate not in h0_by_candidate for candidate in candidates):
            continue
        comm_best = min(candidates, key=lambda candidate: comm[candidate]["done_p50"])
        h0_aggregate = {
            candidate: percentile(
                [record["e2e_p50"] for record in h0_by_candidate[candidate]], 0.50)
            for candidate in candidates
        }
        h0_best = min(candidates, key=lambda candidate: h0_aggregate[candidate])
        h0_per_rep = {
            candidate: {record["repetition"]: record["e2e_p50"] for record in h0_by_candidate[candidate]}
            for candidate in candidates
        }
        common_repetitions = sorted(set(h0_per_rep[h0_best]).intersection(h0_per_rep[comm_best]))
        all_repeat_wins = bool(common_repetitions) and all(
            h0_per_rep[h0_best][rep] < h0_per_rep[comm_best][rep]
            for rep in common_repetitions
        ) if h0_best != comm_best else False
        c0_records = h0_by_candidate["C0_DEFAULT"]
        release_windows = [record["release_window_ratio_p50"] for record in c0_records]
        release_spans = [record["release_span_ratio_p50"] for record in c0_records]
        reversal = comm_best != h0_best
        margin = 100.0 * (h0_aggregate[comm_best] / h0_aggregate[h0_best] - 1.0)
        status = "REPEAT_STABLE_REVERSAL" if reversal and all_repeat_wins else (
            "REVERSAL_NEEDS_REPEAT" if reversal else "NO_REVERSAL")
        stability_rows.append({
            "M": m, "N": n, "K": k, "q": q,
            "isolated_comm_best": comm_best,
            "isolated_comm_best_done_p50_us": fmt(comm[comm_best]["done_p50"]),
            "isolated_comm_p50_by_candidate": ";".join(
                f"{candidate}:{comm[candidate]['done_p50']:.3f}" for candidate in candidates),
            "isolated_comm_process_repetitions": "1 per candidate (Phase-1 limitation)",
            "h0_best": h0_best,
            "h0_best_aggregate_p50_us": fmt(h0_aggregate[h0_best]),
            "h0_isolated_best_aggregate_p50_us": fmt(h0_aggregate[comm_best]),
            "h0_margin_vs_isolated_best_percent": fmt(margin),
            "h0_p50_by_candidate": ";".join(
                f"{candidate}:{h0_aggregate[candidate]:.3f}" for candidate in candidates),
            "h0_per_process_p50_by_candidate": ";".join(
                f"{candidate}:" + "/".join(
                    f"r{rep}={h0_per_rep[candidate][rep]:.3f}"
                    for rep in sorted(h0_per_rep[candidate]))
                for candidate in candidates),
            "h0_winner_beats_isolated_best_in_all_repetitions": "YES" if all_repeat_wins else "NO",
            "candidate_ranking_reversal": "YES" if reversal else "NO",
            "release_window_c0_p50_percent": fmt(percentile(release_windows, 0.50)),
            "release_span_c0_p50_percent": fmt(percentile(release_spans, 0.50)),
            "evidence_status": status,
            "caveat": "COMM_ONLY has one independent process per candidate; Phase 2 repeats it before a final claim.",
        })

    stability_path = os.path.join(output_dir, "candidate_reversal_stability.csv")
    with open(stability_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stability_fields)
        writer.writeheader()
        writer.writerows(stability_rows)

    report_path = os.path.join(output_dir, "phase1_stability_analysis.md")
    with open(report_path, "w") as report:
        report.write("# Phase 1 Repeat-Stability Analysis\n\n")
        report.write("This report separates process-level repetitions from timed iterations. "
                     "All entries use max-rank device-event timing and PASS-only samples.\n\n")
        report.write(f"- Raw global CSVs accepted: {len(case_records)}\n")
        report.write(f"- Raw global CSVs missing a matching manifest or schema: {len(missing)}\n")
        report.write("- H0/B1/B0 use three independent MPI process repetitions in the discovery matrix.\n")
        report.write("- COMM_ONLY uses one independent MPI process per candidate in Phase 1; its ranking must be repeated in Phase 2.\n\n")
        for row in stability_rows:
            report.write(f"## M={row['M']} N={row['N']} K={row['K']} q={row['q']}\n\n")
            report.write(f"- Isolated COMM_ONLY: {row['isolated_comm_p50_by_candidate']}; "
                         f"best `{row['isolated_comm_best']}`.\n")
            report.write(f"- H0 aggregate: {row['h0_p50_by_candidate']}; "
                         f"best `{row['h0_best']}`.\n")
            report.write(f"- H0 process medians: {row['h0_per_process_p50_by_candidate']}.\n")
            report.write(f"- Reversal: {row['candidate_ranking_reversal']}; "
                         f"H0 margin over isolated-communication winner: "
                         f"{row['h0_margin_vs_isolated_best_percent']}%.\n")
            report.write(f"- Winner beats the isolated-communication winner in every H0 repetition: "
                         f"{row['h0_winner_beats_isolated_best_in_all_repetitions']}.\n")
            report.write(f"- C0 first-release window: {row['release_window_c0_p50_percent']}%; "
                         f"first-to-last release span: {row['release_span_c0_p50_percent']}% of T_done.\n")
            report.write(f"- Evidence status: `{row['evidence_status']}`.\n\n")
        report.write("## Interpretation Boundary\n\n")
        report.write("A repeat-stable H0 reversal is a motivation result, not a complete paper claim. "
                     "Phase 2 must repeat COMM_ONLY, export every slice release time, and rerun the selected "
                     "counterexamples before attributing the effect to release semantics, resource contention, "
                     "or a specific communication backend mechanism.\n")

    audit_path = os.path.join(output_dir, "phase1_stability_input_audit.csv")
    with open(audit_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_global_csv", "accepted", "reason"])
        for record in case_records:
            writer.writerow([record["raw_global_csv"], "YES", "matching Phase-1 manifest and expected raw schema"])
        for raw_path in missing:
            writer.writerow([raw_path, "NO", "missing matching manifest or expected raw schema"])
    print(case_summary_path)
    print(stability_path)
    print(report_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_phase1_stability.py RESULT_ROOT")
    main(sys.argv[1])
