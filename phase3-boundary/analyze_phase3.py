#!/usr/bin/env python3
"""Analyze Phase 3 K500SM_AI/gfx928 AllGather-GEMM boundary measurements."""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


CANDIDATES = ["C0_DEFAULT", "C1_RING_SIMPLE_CH4", "C2_RING_SIMPLE_CH8"]


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
    if isinstance(value, float):
        return f"{value:.6f}" if not math.isnan(value) else ""
    return str(value)


def write_dict_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: number(row.get(field, "")) for field in fields})


def case_key(row):
    return (row["m_local"], row["n"], row["k"], row["q"])


def main(root):
    output = os.path.join(root, "summary")
    os.makedirs(output, exist_ok=True)
    manifest_path = os.path.join(root, "phase3_case_manifest.tsv")
    manifest_by_dir = {}
    with open(manifest_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            manifest_by_dir[row["case_dir"]] = row

    case_rows = []
    audit_rows = []
    required = {"e2e_max_us", "t_done_max_us", "t_release_first_max_us",
                "t_release_last_max_us", "gemm_interval_max_us", "correctness_all_ranks"}
    for raw_path in sorted(glob.glob(os.path.join(root, "cases", "*", "raw_global_samples.csv"))):
        case_dir = os.path.dirname(raw_path)
        manifest = manifest_by_dir.get(case_dir)
        if not manifest:
            audit_rows.append({"raw_global_csv": raw_path, "accepted": "NO",
                               "reason": "no matching case manifest"})
            continue
        with open(raw_path, newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        if not raw_rows or not required.issubset(raw_rows[0]):
            audit_rows.append({"raw_global_csv": raw_path, "accepted": "NO",
                               "reason": "unexpected or empty raw_global schema"})
            continue
        valid = [row for row in raw_rows if row["correctness_all_ranks"] == "PASS"]
        metrics = {}
        for name, column in (("e2e", "e2e_max_us"), ("done", "t_done_max_us"),
                             ("release_first", "t_release_first_max_us"),
                             ("release_last", "t_release_last_max_us"),
                             ("gemm_interval", "gemm_interval_max_us")):
            metrics[name] = describe([float(row[column]) for row in valid])
        row = dict(manifest)
        row.update({"raw_global_csv": raw_path, "sample_count": len(raw_rows), "pass_count": len(valid)})
        for name, stats in metrics.items():
            row.update({f"{name}_{stat}": value for stat, value in stats.items()})
        case_rows.append(row)
        reason = "status=0 and all raw samples PASS" if manifest["status"] == "0" and len(valid) == len(raw_rows) else \
                 "raw CSV retained but status or correctness requires review"
        audit_rows.append({"raw_global_csv": raw_path,
                           "accepted": "YES" if manifest["status"] == "0" and len(valid) == len(raw_rows) else "NO",
                           "reason": reason})

    case_fields = [
        "case_id", "phase", "repetition", "shape_id", "m_local", "n", "k", "q", "path",
        "candidate", "status", "sample_count", "pass_count", "raw_global_csv",
        "e2e_p05", "e2e_p50", "e2e_p95", "e2e_mean", "e2e_std", "e2e_cv",
        "done_p05", "done_p50", "done_p95", "release_first_p50", "release_last_p50",
        "gemm_interval_p50",
    ]
    write_dict_csv(os.path.join(output, "phase3_case_level_summary.csv"), case_fields,
                   sorted(case_rows, key=lambda row: row["case_id"]))
    write_dict_csv(os.path.join(output, "phase3_input_audit.csv"),
                   ["raw_global_csv", "accepted", "reason"], audit_rows)

    def selected(path, candidate, shape, metric):
        return [row[metric] for row in case_rows if row["path"] == path and
                row["candidate"] == candidate and case_key(row) == shape and
                row["status"] == "0" and row["pass_count"] == row["sample_count"]]

    shapes = sorted({case_key(row) for row in case_rows if row["path"] in {"comm", "h0"}},
                    key=lambda item: tuple(map(int, item)))
    selection_rows = []
    for shape in shapes:
        comm = {candidate: describe(selected("comm", candidate, shape, "done_p50"))
                for candidate in CANDIDATES}
        h0 = {candidate: describe(selected("h0", candidate, shape, "e2e_p50"))
              for candidate in CANDIDATES}
        if any(comm[candidate]["n"] != 5 or h0[candidate]["n"] != 5 for candidate in CANDIDATES):
            evidence = "INCOMPLETE_OR_INVALID"
            comm_best = ""
            h0_best = ""
            margin = float("nan")
            all_wins = "NO"
            per_rep_text = ""
        else:
            comm_best = min(CANDIDATES, key=lambda candidate: comm[candidate]["p50"])
            h0_best = min(CANDIDATES, key=lambda candidate: h0[candidate]["p50"])
            margin = 100.0 * (h0[comm_best]["p50"] / h0[h0_best]["p50"] - 1.0)
            by_rep = {}
            for candidate in CANDIDATES:
                by_rep[candidate] = {row["repetition"]: row["e2e_p50"] for row in case_rows
                                     if row["path"] == "h0" and row["candidate"] == candidate and
                                     case_key(row) == shape and row["status"] == "0" and
                                     row["pass_count"] == row["sample_count"]}
            common = sorted(set(by_rep[h0_best]).intersection(by_rep[comm_best]))
            stable = h0_best != comm_best and len(common) == 5 and all(
                by_rep[h0_best][rep] < by_rep[comm_best][rep] for rep in common)
            all_wins = "YES" if stable else "NO"
            if stable and margin >= 5.0:
                evidence = "STRONG_REVERSAL"
            elif stable:
                evidence = "REPEAT_STABLE_REVERSAL"
            elif h0_best != comm_best:
                evidence = "UNSTABLE_REVERSAL"
            else:
                evidence = "NO_REVERSAL_CONTROL"
            per_rep_text = ";".join(
                f"{candidate}:" + "/".join(f"r{rep}={by_rep[candidate][rep]:.3f}"
                                               for rep in sorted(by_rep[candidate]))
                for candidate in CANDIDATES)
        selection_rows.append({
            "M_local": shape[0], "N": shape[1], "K": shape[2], "q": shape[3],
            "comm_p50_by_candidate_us": ";".join(
                f"{candidate}:{comm[candidate]['p50']:.3f}" for candidate in CANDIDATES),
            "isolated_comm_best": comm_best,
            "isolated_comm_best_p50_us": comm[comm_best]["p50"] if comm_best else float("nan"),
            "h0_p50_by_candidate_us": ";".join(
                f"{candidate}:{h0[candidate]['p50']:.3f}" for candidate in CANDIDATES),
            "h0_best": h0_best,
            "h0_best_p50_us": h0[h0_best]["p50"] if h0_best else float("nan"),
            "h0_at_isolated_best_p50_us": h0[comm_best]["p50"] if comm_best else float("nan"),
            "h0_margin_vs_isolated_best_percent": margin,
            "h0_per_process_p50_by_candidate_us": per_rep_text,
            "h0_winner_beats_comm_winner_all_5_repetitions": all_wins,
            "evidence_status": evidence,
        })
    selection_fields = [
        "M_local", "N", "K", "q", "comm_p50_by_candidate_us", "isolated_comm_best",
        "isolated_comm_best_p50_us", "h0_p50_by_candidate_us", "h0_best", "h0_best_p50_us",
        "h0_at_isolated_best_p50_us", "h0_margin_vs_isolated_best_percent",
        "h0_per_process_p50_by_candidate_us", "h0_winner_beats_comm_winner_all_5_repetitions",
        "evidence_status",
    ]
    write_dict_csv(os.path.join(output, "phase3_selection_boundary.csv"), selection_fields, selection_rows)

    slice_groups = defaultdict(list)
    for raw_path in sorted(glob.glob(os.path.join(root, "cases", "*", "release_slices_global.csv"))):
        case_dir = os.path.dirname(raw_path)
        manifest = manifest_by_dir.get(case_dir)
        if not manifest or manifest["status"] != "0" or manifest["path"] != "h0":
            continue
        with open(raw_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["correctness_all_ranks"] != "PASS":
                    continue
                key = (row["M"], row["N"], row["K"], row["q"], row["candidate"], row["slice_index"])
                slice_groups[key].append(row)

    per_slice_rows = []
    for key, rows in sorted(slice_groups.items(), key=lambda item: tuple(map(int, item[0][:4])) +
                             (item[0][4], int(item[0][5]))):
        release = [float(row["t_release_max_us"]) for row in rows]
        start = [float(row["t_gemm_start_max_us"]) for row in rows]
        duration = [float(row["t_gemm_duration_max_us"]) for row in rows]
        waits = [start_value - release_value for start_value, release_value in zip(start, release)]
        per_slice_rows.append({
            "M_local": key[0], "N": key[1], "K": key[2], "q": key[3], "candidate": key[4],
            "slice_index": key[5], "sample_count": len(rows),
            "release_p50_us": percentile(release, 0.50), "gemm_start_p50_us": percentile(start, 0.50),
            "gemm_duration_p50_us": percentile(duration, 0.50),
            "consumer_wait_p50_us": percentile(waits, 0.50),
        })
    slice_fields = ["M_local", "N", "K", "q", "candidate", "slice_index", "sample_count",
                    "release_p50_us", "gemm_start_p50_us", "gemm_duration_p50_us", "consumer_wait_p50_us"]
    write_dict_csv(os.path.join(output, "phase3_release_curve_summary.csv"), slice_fields, per_slice_rows)

    pipeline_rows = []
    grouped_slices = defaultdict(list)
    for row in per_slice_rows:
        grouped_slices[(row["M_local"], row["N"], row["K"], row["q"], row["candidate"])].append(row)
    for key, rows in sorted(grouped_slices.items(), key=lambda item: tuple(map(int, item[0][:4])) + (item[0][4],)):
        rows = sorted(rows, key=lambda row: int(row["slice_index"]))
        waits = [row["consumer_wait_p50_us"] for row in rows]
        durations = [row["gemm_duration_p50_us"] for row in rows]
        pipeline_rows.append({
            "M_local": key[0], "N": key[1], "K": key[2], "q": key[3], "candidate": key[4],
            "slice_sample_count": min(row["sample_count"] for row in rows),
            "first_release_p50_us": rows[0]["release_p50_us"],
            "last_release_p50_us": rows[-1]["release_p50_us"],
            "first_consumer_wait_p50_us": waits[0],
            "last_consumer_wait_p50_us": waits[-1],
            "max_consumer_wait_p50_us": max(waits),
            "consumer_wait_growth_us": waits[-1] - waits[0],
            "mean_consumer_wait_p50_us": statistics.mean(waits),
            "first_gemm_duration_p50_us": durations[0],
            "last_gemm_duration_p50_us": durations[-1],
            "mean_gemm_duration_p50_us": statistics.mean(durations),
        })
    pipeline_fields = [
        "M_local", "N", "K", "q", "candidate", "slice_sample_count", "first_release_p50_us",
        "last_release_p50_us", "first_consumer_wait_p50_us", "last_consumer_wait_p50_us",
        "max_consumer_wait_p50_us", "consumer_wait_growth_us", "mean_consumer_wait_p50_us",
        "first_gemm_duration_p50_us", "last_gemm_duration_p50_us", "mean_gemm_duration_p50_us",
    ]
    write_dict_csv(os.path.join(output, "phase3_pipeline_features.csv"), pipeline_fields, pipeline_rows)

    controls = []
    for n in ("512", "2048", "4096"):
        b0_shape = ("2048", n, "2048", "1")
        b0 = describe(selected("b0", "C0_DEFAULT", b0_shape, "e2e_p50"))
        for q in (("2", "4", "8") if n != "2048" else ("2", "4", "8", "16")):
            shape = ("2048", n, "2048", q)
            gemm = describe(selected("gemm", "C0_DEFAULT", shape, "e2e_p50"))
            for candidate in ("C0_DEFAULT", "C2_RING_SIMPLE_CH8"):
                h0 = describe(selected("h0", candidate, shape, "e2e_p50"))
                b1 = describe(selected("b1", candidate, shape, "e2e_p50"))
                gain = 100.0 * (b1["p50"] / h0["p50"] - 1.0) if b1["n"] and h0["n"] else float("nan")
                controls.append({
                    "M_local": "2048", "N": n, "K": "2048", "q": q, "candidate": candidate,
                    "b0_full_serial_p50_us": b0["p50"], "gemm_only_p50_us": gemm["p50"],
                    "b1_slice_serial_p50_us": b1["p50"], "h0_overlap_p50_us": h0["p50"],
                    "h0_vs_b1_gain_percent": gain,
                })
    control_fields = ["M_local", "N", "K", "q", "candidate", "b0_full_serial_p50_us",
                      "gemm_only_p50_us", "b1_slice_serial_p50_us", "h0_overlap_p50_us",
                      "h0_vs_b1_gain_percent"]
    write_dict_csv(os.path.join(output, "phase3_control_summary.csv"), control_fields, controls)

    accepted = sum(row["accepted"] == "YES" for row in audit_rows)
    strong = [row for row in selection_rows if row["evidence_status"] == "STRONG_REVERSAL"]
    stable = [row for row in selection_rows if row["evidence_status"] in
              {"STRONG_REVERSAL", "REPEAT_STABLE_REVERSAL"}]
    report_path = os.path.join(output, "phase3_analysis.md")
    with open(report_path, "w") as report:
        report.write("# Phase 3 Boundary-Mapping Analysis\n\n")
        report.write("Platform: `K500SM_AI / gfx928 / 4 GPUs / PCIe`. Metrics are PASS-only, max-rank HIP device-event values.\n\n")
        report.write("## Audit\n\n")
        report.write(f"- Case manifest entries: {len(manifest_by_dir)}\n")
        report.write(f"- Accepted raw global CSVs: {accepted}/{len(audit_rows)}\n")
        report.write("- A comparison is eligible only when each candidate/path has five independent, all-PASS process runs.\n\n")
        report.write("## Selection Boundary\n\n")
        report.write("| N | q | isolated winner | H0 winner | H0 gap vs isolated winner | evidence |\n")
        report.write("|---:|---:|---|---|---:|---|\n")
        for row in selection_rows:
            report.write(f"| {row['N']} | {row['q']} | {row['isolated_comm_best'] or 'N/A'} | "
                         f"{row['h0_best'] or 'N/A'} | {number(row['h0_margin_vs_isolated_best_percent']) or 'N/A'}% | "
                         f"{row['evidence_status']} |\n")
        report.write("\n")
        if strong:
            report.write("Strong repeat-stable reversals occurred at: " + ", ".join(
                f"N={row['N']}, q={row['q']} ({number(row['h0_margin_vs_isolated_best_percent'])}%)"
                for row in strong) + ".\n\n")
        elif stable:
            report.write("Repeat-stable reversals occurred, but none reached the preregistered 5% strong threshold.\n\n")
        else:
            report.write("No repeat-stable candidate-ranking reversal was observed in the completed matrix.\n\n")
        report.write("## How To Interpret The Tables\n\n")
        report.write("- `phase3_selection_boundary.csv` is the main decision result: it compares isolated `T_done` selection with H0 `T_e2e` selection.\n")
        report.write("- `phase3_release_curve_summary.csv` retains per-slice release, compute start, duration and consumer wait.\n")
        report.write("- `phase3_pipeline_features.csv` condenses queue buildup: positive consumer-wait growth means later released slices wait longer before GEMM begins.\n")
        report.write("- `phase3_control_summary.csv` separates B1-to-H0 overlap benefit and GEMM-only fragmentation cost from the selection result.\n\n")
        report.write("## Causal Boundary\n\n")
        report.write("The timing data can show that earlier collective completion/release does not necessarily imply lower dependent `T_e2e`, and can associate a reversal with consumer backlog or changed GEMM duration. It cannot by itself prove a particular CU/cache/memory-controller mechanism. Hardware-counter traces are a subsequent validation step. This synthetic controlled matrix is not a DUSHMEM result or a trace-derived model-workload result.\n")

    print(os.path.join(output, "phase3_selection_boundary.csv"))
    print(os.path.join(output, "phase3_release_curve_summary.csv"))
    print(os.path.join(output, "phase3_pipeline_features.csv"))
    print(os.path.join(output, "phase3_control_summary.csv"))
    print(report_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_phase3.py RESULT_ROOT")
    main(sys.argv[1])
