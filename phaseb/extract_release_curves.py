#!/usr/bin/env python3
"""Extract per-slice release / GEMM pacing curves from Phase B cases.

Input : <result_root>/cases/case*/release_slices_rank{0..3}.csv
Output: <result_root>/summary/
  phaseb_release_curve_summary.csv  one row per case (pacing statistics)
  phaseb_release_curves_long.csv    long-format per-slice median curves (F6 plots)

Conventions: per (iteration, slice) the metric is the MAX across ranks
(consistent with the max-rank aggregation used everywhere else). Rows with
t_release==0 AND t_gemm_start==0 are placeholder rows for paths that only
instrument slice 0 (r0/d0/fc) and are skipped.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict


def load_case_slices(case_dir):
    """{iteration: {slice: {metric: max_across_ranks}}}, plus manifest dict."""
    manifest_path = os.path.join(case_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        return None, None
    try:
        with open(manifest_path, newline="") as handle:
            manifest = next(csv.DictReader(handle))
    except (StopIteration, OSError, csv.Error):
        return None, None
    iters = defaultdict(dict)
    for rank_path in sorted(glob.glob(os.path.join(case_dir, "release_slices_rank*.csv"))):
        try:
            with open(rank_path, newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        rel = float(row["t_release_us"])
                        gs = float(row["t_gemm_start_us"])
                    except (TypeError, ValueError):
                        continue
                    if rel == 0.0 and gs == 0.0:
                        continue  # uninstrumented placeholder slice
                    key = int(row["slice_index"])
                    cur = iters[int(row["iteration_index"])].get(key)
                    entry = {"t_release_us": rel, "t_gemm_start_us": gs,
                             "t_gemm_end_us": float(row["t_gemm_end_us"] or 0.0),
                             "t_gemm_duration_us": float(row["t_gemm_duration_us"] or 0.0)}
                    if cur is None:
                        iters[int(row["iteration_index"])][key] = entry
                    else:
                        for name, value in entry.items():
                            cur[name] = max(cur[name], value)
        except (OSError, csv.Error):
            continue
    return manifest, iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    cases_root = os.path.join(args.result_root, "cases")
    summary_root = os.path.join(args.result_root, "summary")
    os.makedirs(summary_root, exist_ok=True)

    summary_rows = []
    long_rows = []
    for case_dir in sorted(glob.glob(os.path.join(cases_root, "case*"))):
        if not os.path.isdir(case_dir):
            continue
        manifest, iters = load_case_slices(case_dir)
        if manifest is None or not iters:
            continue
        q = int(manifest.get("q", 0))
        # per-slice medians across iterations
        per_slice = defaultdict(lambda: defaultdict(list))
        release_first, release_last = [], []
        release_intervals, delays, gemm_durs = [], [], []
        for _, slices in iters.items():
            keys = sorted(slices)
            if not keys:
                continue
            release_first.append(slices[keys[0]]["t_release_us"])
            release_last.append(slices[keys[-1]]["t_release_us"])
            for idx in range(1, len(keys)):
                release_intervals.append(slices[keys[idx]]["t_release_us"] -
                                         slices[keys[idx - 1]]["t_release_us"])
            for key in keys:
                entry = slices[key]
                for name in ("t_release_us", "t_gemm_start_us",
                             "t_gemm_end_us", "t_gemm_duration_us"):
                    per_slice[key][name].append(entry[name])
                delays.append(entry["t_gemm_start_us"] - entry["t_release_us"])
                gemm_durs.append(entry["t_gemm_duration_us"])
        case_id = os.path.basename(case_dir)
        rep = case_id.rsplit("_rep", 1)[1] if "_rep" in case_id else "1"

        def med(values):
            return statistics.median(values) if values else float("nan")

        summary_rows.append({
            "case_id": case_id,
            "path": manifest.get("path", ""), "family": manifest.get("family", ""),
            "candidate": manifest.get("candidate", ""),
            "N": manifest.get("N", ""), "q": manifest.get("q", ""), "rep": rep,
            "iterations": len(iters),
            "release_first_med_us": f"{med(release_first):.1f}",
            "release_last_med_us": f"{med(release_last):.1f}",
            "release_pacing_med_us": f"{med(release_intervals):.1f}" if release_intervals else "",
            "release_to_gemm_delay_med_us": f"{med(delays):.1f}" if delays else "",
            "gemm_dur_med_us": f"{med(gemm_durs):.1f}" if gemm_durs else "",
        })
        for key in sorted(per_slice):
            long_rows.append({
                "case_id": case_id,
                "path": manifest.get("path", ""),
                "candidate": manifest.get("candidate", ""),
                "N": manifest.get("N", ""), "q": manifest.get("q", ""), "rep": rep,
                "slice_index": key,
                "t_release_med_us": f"{statistics.median(per_slice[key]['t_release_us']):.1f}",
                "t_gemm_start_med_us": f"{statistics.median(per_slice[key]['t_gemm_start_us']):.1f}",
                "t_gemm_end_med_us": f"{statistics.median(per_slice[key]['t_gemm_end_us']):.1f}",
                "t_gemm_duration_med_us": f"{statistics.median(per_slice[key]['t_gemm_duration_us']):.1f}",
            })

    summary_path = os.path.join(summary_root, "phaseb_release_curve_summary.csv")
    long_path = os.path.join(summary_root, "phaseb_release_curves_long.csv")
    for path, rows in ((summary_path, summary_rows), (long_path, long_rows)):
        if not rows:
            continue
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"release curves -> {summary_path} ({len(summary_rows)} cases) "
          f"+ {long_path} ({len(long_rows)} slice rows)")

    # console digest: release->gemm delay + pacing by path family
    by_path = defaultdict(list)
    for row in summary_rows:
        if row["release_to_gemm_delay_med_us"]:
            by_path[(row["path"], row["N"], row["q"])].append(
                (float(row["release_to_gemm_delay_med_us"]),
                 float(row["release_pacing_med_us"]) if row["release_pacing_med_us"] else float("nan")))
    print(f"{'path':<22} {'N':>5} {'q':>3} {'rel->gemm us':>13} {'pacing us':>10}")
    for (path, n, q) in sorted(by_path.keys(), key=lambda k: (int(k[1]), int(k[2]), k[0])):
        delays = [v[0] for v in by_path[(path, n, q)]]
        pacings = [v[1] for v in by_path[(path, n, q)] if v[1] == v[1]]
        print(f"{path:<22} {n:>5} {q:>3} {statistics.median(delays):>13.1f} "
              f"{(statistics.median(pacings) if pacings else float('nan')):>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
