#!/usr/bin/env python3
"""Aggregate raw_global_samples.csv files without changing any raw result."""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main(root):
    files = sorted(glob.glob(os.path.join(root, "cases", "**", "raw_global_samples.csv"), recursive=True))
    rows = []
    for path in files:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "candidate", "M", "N", "K", "q", "e2e_max_us"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                print(f"skip malformed or superseded CSV: {path}", file=sys.stderr)
                continue
            rows.extend(reader)
    key_fields = ["path", "candidate", "M", "N", "K", "q"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    output = os.path.join(root, "summary", "phase1_summary.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fields = key_fields + [
        "sample_count", "pass_count", "fail_count", "e2e_mean_us", "e2e_p50_us", "e2e_p05_us",
        "e2e_p95_us", "e2e_std_us", "e2e_cv_percent", "t_done_mean_us", "release_first_mean_us",
        "release_last_mean_us", "gemm_interval_mean_us", "gemm_tflops_mean", "notes"
    ]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, samples in sorted(groups.items()):
            e2e = [float(sample["e2e_max_us"]) for sample in samples]
            done = [float(sample["t_done_max_us"]) for sample in samples]
            first = [float(sample["t_release_first_max_us"]) for sample in samples]
            last = [float(sample["t_release_last_max_us"]) for sample in samples]
            gemm = [float(sample["gemm_interval_max_us"]) for sample in samples]
            tflops = [float(sample["gemm_tflops_min"]) for sample in samples]
            average = statistics.mean(e2e)
            std = statistics.stdev(e2e) if len(e2e) > 1 else 0.0
            writer.writerow(dict(zip(key_fields, key),
                                    sample_count=len(samples),
                                    pass_count=sum(sample["correctness_all_ranks"] == "PASS" for sample in samples),
                                    fail_count=sum(sample["correctness_all_ranks"] != "PASS" for sample in samples),
                                    e2e_mean_us=f"{average:.6f}",
                                    e2e_p50_us=f"{percentile(e2e, 0.50):.6f}",
                                    e2e_p05_us=f"{percentile(e2e, 0.05):.6f}",
                                    e2e_p95_us=f"{percentile(e2e, 0.95):.6f}",
                                    e2e_std_us=f"{std:.6f}",
                                    e2e_cv_percent=f"{(100.0 * std / average) if average else 0.0:.6f}",
                                    t_done_mean_us=f"{statistics.mean(done):.6f}",
                                    release_first_mean_us=f"{statistics.mean(first):.6f}",
                                    release_last_mean_us=f"{statistics.mean(last):.6f}",
                                    gemm_interval_mean_us=f"{statistics.mean(gemm):.6f}",
                                    gemm_tflops_mean=f"{statistics.mean(tflops):.6f}",
                                    notes="raw global samples aggregated; synthetic Phase-1 shapes only"))
    print(output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_phase1.py RESULT_ROOT")
    main(sys.argv[1])
