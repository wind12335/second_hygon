#!/usr/bin/env python3
"""Create auditable decision tables from Phase-1 raw global samples."""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def median_metric(rows, field):
    values = [float(row[field]) for row in rows if row["correctness_all_ranks"] == "PASS"]
    return quantile(values, 0.5)


def sample_count(rows):
    return sum(row["correctness_all_ranks"] == "PASS" for row in rows)


def main(root):
    raw_paths = sorted(glob.glob(os.path.join(root, "cases", "**", "raw_global_samples.csv"), recursive=True))
    rows = []
    ignored = []
    required = {"path", "candidate", "M", "N", "K", "q", "e2e_max_us", "t_done_max_us"}
    for path in raw_paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                ignored.append(path)
                continue
            for row in reader:
                row["_source"] = path
                rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["path"], row["candidate"], row["M"], row["N"], row["K"], row["q"])].append(row)

    summary = {}
    for key, samples in groups.items():
        summary[key] = {
            "samples": sample_count(samples),
            "e2e_p50": median_metric(samples, "e2e_max_us"),
            "done_p50": median_metric(samples, "t_done_max_us"),
            "first_p50": median_metric(samples, "t_release_first_max_us"),
            "last_p50": median_metric(samples, "t_release_last_max_us"),
            "gemm_p50": median_metric(samples, "gemm_interval_max_us"),
            "e2e_p05": quantile([float(x["e2e_max_us"]) for x in samples], 0.05),
            "e2e_p95": quantile([float(x["e2e_max_us"]) for x in samples], 0.95),
        }

    shapes = sorted({(key[2], key[3], key[4], key[5]) for key in summary})
    output_dir = os.path.join(root, "summary")
    os.makedirs(output_dir, exist_ok=True)
    decision_path = os.path.join(output_dir, "decision_analysis.csv")
    fields = [
        "M", "N", "K", "q", "isolated_comm_candidates", "isolated_comm_best_candidate",
        "isolated_comm_best_t_done_p50_us", "h0_candidates", "h0_best_candidate",
        "h0_best_e2e_p50_us", "candidate_ranking_reversal", "c0_comm_t_done_p50_us",
        "c0_h0_e2e_p50_us", "c0_b1_e2e_p50_us", "h0_gain_vs_b1_percent",
        "b1_fragmentation_vs_b0_percent", "gemm_fragmentation_vs_q1_percent",
        "release_first_over_done_percent", "h0_p05_us", "h0_p95_us",
        "h0_sample_count", "evidence_status", "notes"
    ]
    evidence_rows = []
    with open(decision_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m, n, k, q in shapes:
            candidates = ["C0_DEFAULT", "C1_RING_SIMPLE_CH4", "C2_RING_SIMPLE_CH8"]
            comm = [(candidate, summary[("COMM_ONLY", candidate, m, n, k, q)])
                    for candidate in candidates
                    if ("COMM_ONLY", candidate, m, n, k, q) in summary]
            h0 = [(candidate, summary[("H0_EVENT_OVERLAP", candidate, m, n, k, q)])
                  for candidate in candidates
                  if ("H0_EVENT_OVERLAP", candidate, m, n, k, q) in summary]
            if not comm or not h0:
                continue
            comm_best = min(comm, key=lambda item: item[1]["done_p50"])
            h0_best = min(h0, key=lambda item: item[1]["e2e_p50"])
            c0_comm = summary.get(("COMM_ONLY", "C0_DEFAULT", m, n, k, q))
            c0_h0 = summary.get(("H0_EVENT_OVERLAP", "C0_DEFAULT", m, n, k, q))
            c0_b1 = summary.get(("B1_SLICE_SERIAL", "C0_DEFAULT", m, n, k, q))
            b0 = summary.get(("B0_FULL_SERIAL", "C0_DEFAULT", m, n, k, "1"))
            gemm_q = summary.get(("GEMM_ONLY", "C0_DEFAULT", m, n, k, q))
            gemm_q1 = summary.get(("GEMM_ONLY", "C0_DEFAULT", m, n, k, "1"))
            gain = float("nan")
            if c0_h0 and c0_b1 and c0_h0["e2e_p50"] > 0:
                gain = 100.0 * (c0_b1["e2e_p50"] / c0_h0["e2e_p50"] - 1.0)
            b1_frag = float("nan")
            if c0_b1 and b0 and b0["e2e_p50"] > 0:
                b1_frag = 100.0 * (c0_b1["e2e_p50"] / b0["e2e_p50"] - 1.0)
            gemm_frag = float("nan")
            if gemm_q and gemm_q1 and gemm_q1["gemm_p50"] > 0:
                gemm_frag = 100.0 * (gemm_q["gemm_p50"] / gemm_q1["gemm_p50"] - 1.0)
            release_window = float("nan")
            if c0_h0 and c0_h0["done_p50"] > 0:
                release_window = 100.0 * (c0_h0["done_p50"] - c0_h0["first_p50"]) / c0_h0["done_p50"]
            reversal = comm_best[0] != h0_best[0]
            status = "CANDIDATE_REVERSAL" if reversal else "SAME_RANKING_OR_INSUFFICIENT_MARGIN"
            evidence = {
                "M": m, "N": n, "K": k, "q": q,
                "isolated_comm_candidates": ";".join(
                    f"{name}:{stats['done_p50']:.3f}" for name, stats in comm),
                "isolated_comm_best_candidate": comm_best[0],
                "isolated_comm_best_t_done_p50_us": f"{comm_best[1]['done_p50']:.6f}",
                "h0_candidates": ";".join(f"{name}:{stats['e2e_p50']:.3f}" for name, stats in h0),
                "h0_best_candidate": h0_best[0],
                "h0_best_e2e_p50_us": f"{h0_best[1]['e2e_p50']:.6f}",
                "candidate_ranking_reversal": str(reversal).upper(),
                "c0_comm_t_done_p50_us": f"{c0_comm['done_p50']:.6f}" if c0_comm else "",
                "c0_h0_e2e_p50_us": f"{c0_h0['e2e_p50']:.6f}" if c0_h0 else "",
                "c0_b1_e2e_p50_us": f"{c0_b1['e2e_p50']:.6f}" if c0_b1 else "",
                "h0_gain_vs_b1_percent": f"{gain:.6f}" if not math.isnan(gain) else "",
                "b1_fragmentation_vs_b0_percent": f"{b1_frag:.6f}" if not math.isnan(b1_frag) else "",
                "gemm_fragmentation_vs_q1_percent": f"{gemm_frag:.6f}" if not math.isnan(gemm_frag) else "",
                "release_first_over_done_percent": f"{release_window:.6f}" if not math.isnan(release_window) else "",
                "h0_p05_us": f"{h0_best[1]['e2e_p05']:.6f}",
                "h0_p95_us": f"{h0_best[1]['e2e_p95']:.6f}",
                "h0_sample_count": h0_best[1]["samples"],
                "evidence_status": status,
                "notes": "Synthetic Phase-1 engineering shape; not trace-derived paper evidence.",
            }
            writer.writerow(evidence)
            evidence_rows.append(evidence)

    report_path = os.path.join(output_dir, "decision_analysis.md")
    with open(report_path, "w") as report:
        report.write("# Phase 1 Decision Analysis\n\n")
        report.write("This report is generated from per-iteration, max-rank raw samples. "
                     "The three shapes are synthetic engineering controls, not final trace-derived workloads.\n\n")
        report.write(f"- Valid raw global files: {len(raw_paths) - len(ignored)}\n")
        report.write(f"- Ignored malformed or superseded raw files: {len(ignored)}\n")
        report.write(f"- Complete shape/q decision rows: {len(evidence_rows)}\n\n")
        for row in evidence_rows:
            report.write(f"## M={row['M']} N={row['N']} K={row['K']} q={row['q']}\n\n")
            report.write(f"- Isolated communication: {row['isolated_comm_candidates']}; "
                         f"best {row['isolated_comm_best_candidate']}.\n")
            report.write(f"- H0 end-to-end: {row['h0_candidates']}; "
                         f"best {row['h0_best_candidate']}.\n")
            report.write(f"- Ranking reversal: {row['candidate_ranking_reversal']}. "
                         f"H0-vs-B1 gain for C0: {row['h0_gain_vs_b1_percent']}%.\n")
            report.write(f"- B1 fragmentation vs B0: {row['b1_fragmentation_vs_b0_percent']}%; "
                         f"GEMM q-vs-q1 fragmentation: {row['gemm_fragmentation_vs_q1_percent']}%.\n")
            report.write(f"- First legal release window: {row['release_first_over_done_percent']}% of C0 T_done.\n\n")

    audit_path = os.path.join(output_dir, "analysis_input_audit.csv")
    with open(audit_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_global_csv", "accepted", "reason"])
        for path in raw_paths:
            writer.writerow([path, "NO" if path in ignored else "YES",
                             "missing required CSV columns" if path in ignored else "valid Phase-1 raw schema"])
    print(decision_path)
    print(report_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_phase1.py RESULT_ROOT")
    main(sys.argv[1])
