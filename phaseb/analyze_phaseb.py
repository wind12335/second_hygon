#!/usr/bin/env python3
"""Phase B analyzer: cross-substrate selection-boundary tables.

Reads every case directory under <result_root>/cases, aggregates per-case p50
statistics from raw_global_samples.csv, then builds:

  phaseb_case_summary.csv        one row per case (p50/p05/p95, correctness)
  phaseb_cell_matrix.csv         one row per (N, q, candidate, path) cell
  phaseb_selection_boundary.csv  isolated winner vs e2e winner per (N, q)
  phaseb_control_table.csv       overlap gains / fragmentation costs
  phaseb_analysis.md             human-readable summary of the above

Judgement columns follow the phase2/3 vocabulary (REVERSAL = isolated ranking
disagrees with end-to-end ranking).
"""

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

ISOLATED_PATHS = ("COMM_ONLY", "FC_FCOLLECT_ONLY", "DC_PUSHSIG_ONLY")
E2E_PATHS = ("R0_FULL_SERIAL", "RS_SLICE_SERIAL", "R1_EVENT_OVERLAP",
             "D0_FCOLLECT_SERIAL", "DS_PUSHSIG_SERIAL", "D1_PUSHSIG_OVERLAP",
             "D1W_WAITSTREAM_OVERLAP")


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def load_case(case_dir):
    samples_path = os.path.join(case_dir, "raw_global_samples.csv")
    manifest_path = os.path.join(case_dir, "manifest.csv")
    if not os.path.exists(samples_path) or not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, newline="") as handle:
            manifest = next(csv.DictReader(handle))
    except (StopIteration, OSError, csv.Error):
        return None
    rows = []
    try:
        with open(samples_path, newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    except (OSError, csv.Error):
        return None
    if not rows:
        return None

    def column(name):
        return [float(row[name]) for row in rows if row.get(name) not in (None, "")]

    stats = {}
    for name in ("t_release_first_max_us", "t_release_last_max_us", "t_done_max_us",
                 "gemm_first_start_max_us", "gemm_last_end_max_us",
                 "gemm_interval_max_us", "e2e_max_us"):
        values = column(name)
        stats[name] = {
            "p50": statistics.median(values) if values else float("nan"),
            "p05": percentile(values, 0.05),
            "p95": percentile(values, 0.95),
        }
    correctness = [row.get("correctness_all_ranks", "?") for row in rows]
    status = "PASS" if correctness and all(value == "PASS" for value in correctness) else "FAIL"
    rep_fail = sum(1 for value in correctness if value != "PASS")
    return {
        "manifest": manifest,
        "iterations": len(rows),
        "status": status,
        "iteration_failures": rep_fail,
        "stats": stats,
    }


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
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

    case_rows = []
    for case_dir in sorted(glob.glob(os.path.join(cases_root, "case*"))):
        if not os.path.isdir(case_dir):
            continue
        loaded = load_case(case_dir)
        if loaded is None:
            exit_path = os.path.join(case_dir, "exit_status.txt")
            exit_code = "unknown"
            if os.path.exists(exit_path):
                with open(exit_path) as handle:
                    exit_code = handle.read().strip()
            case_rows.append({"case_dir": case_dir, "manifest": None, "exit_code": exit_code,
                              "status": "NO_DATA", "iterations": 0, "stats": {}})
            continue
        loaded["case_dir"] = case_dir
        loaded["exit_code"] = "0"
        case_rows.append(loaded)

    # ---- per-case summary --------------------------------------------------
    case_summary_path = os.path.join(summary_root, "phaseb_case_summary.csv")
    fields = ["case_id", "path", "family", "candidate", "M_local", "N", "K", "q", "rep",
              "iterations", "iteration_failures", "status"] + \
             [f"{col}_{agg}" for col in ("t_release_first_max_us", "t_release_last_max_us",
                                         "t_done_max_us", "gemm_first_start_max_us",
                                         "gemm_last_end_max_us", "gemm_interval_max_us",
                                         "e2e_max_us") for agg in ("p50", "p05", "p95")]
    with open(case_summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in case_rows:
            manifest = row.get("manifest") or {}
            out = {
                "case_id": os.path.basename(row["case_dir"]),
                "path": manifest.get("path", ""),
                "family": manifest.get("family", ""),
                "candidate": manifest.get("candidate", ""),
                "M_local": manifest.get("M", ""),
                "N": manifest.get("N", ""),
                "K": manifest.get("K", ""),
                "q": manifest.get("q", ""),
                "rep": "",
                "iterations": row.get("iterations", 0),
                "iteration_failures": row.get("iteration_failures", ""),
                "status": row.get("status", ""),
            }
            case_id = out["case_id"]
            if "_rep" in case_id:
                out["rep"] = case_id.rsplit("_rep", 1)[1]
            for col in ("t_release_first_max_us", "t_release_last_max_us", "t_done_max_us",
                        "gemm_first_start_max_us", "gemm_last_end_max_us",
                        "gemm_interval_max_us", "e2e_max_us"):
                values = row.get("stats", {}).get(col, {})
                for agg in ("p50", "p05", "p95"):
                    out[f"{col}_{agg}"] = fmt(values.get(agg))
            writer.writerow(out)

    # ---- cell matrix --------------------------------------------------------
    cells = defaultdict(list)
    for row in case_rows:
        manifest = row.get("manifest")
        if not manifest or row.get("status") != "PASS":
            continue
        key = (manifest.get("candidate"), int(manifest.get("N", 0)),
               int(manifest.get("q", 0)), manifest.get("path"))
        cells[key].append(row)
    cell_matrix_path = os.path.join(summary_root, "phaseb_cell_matrix.csv")
    cell_fields = ["candidate", "N", "q", "path", "family", "reps",
                   "t_done_p50_us", "e2e_p50_us", "release_last_p50_us",
                   "e2e_p05_us", "e2e_p95_us", "release_gap_us", "comm_compute_overlap_us",
                   "e2e_p50_stdev_us"]
    cell_rows = []
    for key, rows in sorted(cells.items()):
        candidate, n, q, path = key
        family = rows[0]["manifest"].get("family", "")
        done_p50s = [row["stats"]["t_done_max_us"]["p50"] for row in rows]
        e2e_p50s = [row["stats"]["e2e_max_us"]["p50"] for row in rows]
        rel_last = [row["stats"]["t_release_last_max_us"]["p50"] for row in rows]
        gemm_last_end = [row["stats"]["gemm_last_end_max_us"]["p50"] for row in rows]
        e2e_all = [value for row in rows for value in
                   (row["stats"]["e2e_max_us"]["p50"],)]
        done_med = statistics.median(done_p50s) if done_p50s else float("nan")
        e2e_med = statistics.median(e2e_p50s) if e2e_p50s else float("nan")
        rel_med = statistics.median(rel_last) if rel_last else float("nan")
        cell_rows.append({
            "candidate": candidate, "N": n, "q": q, "path": path, "family": family,
            "reps": len(rows),
            "t_done_p50_us": fmt(done_med),
            "e2e_p50_us": fmt(e2e_med),
            "release_last_p50_us": fmt(rel_med),
            "e2e_p05_us": fmt(percentile(e2e_all, 0.05)),
            "e2e_p95_us": fmt(percentile(e2e_all, 0.95)),
            "release_gap_us": fmt(e2e_med - done_med),
            "comm_compute_overlap_us": fmt(e2e_med - done_med),
            "e2e_p50_stdev_us": fmt(statistics.pstdev(e2e_p50s) if len(e2e_p50s) > 1
                                    else 0.0),
        })
    with open(cell_matrix_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    # ---- selection boundary --------------------------------------------------
    by_cell = defaultdict(dict)
    for row in cell_rows:
        by_cell[(row["candidate"], row["N"], row["q"])][row["path"]] = row

    boundary_path = os.path.join(summary_root, "phaseb_selection_boundary.csv")
    boundary_fields = ["candidate", "N", "q",
                       "isolated_winner_path", "isolated_runner_up", "isolated_gap_pct",
                       "e2e_winner_path", "e2e_runner_up", "e2e_gap_pct",
                       "isolated_winner_family", "e2e_winner_family",
                       "isolated_best_us", "e2e_best_us",
                       "reversal_flag", "notes"]
    boundary_rows = []
    for (candidate, n, q), paths in sorted(by_cell.items()):
        isolated = {path: row for path, row in paths.items() if path in ISOLATED_PATHS}
        e2e = {path: row for path, row in paths.items() if path in E2E_PATHS}

        def best(group, metric):
            entries = []
            for path, row in group.items():
                value = row.get(metric)
                if value != "" and not math.isnan(float(value)):
                    entries.append((float(value), path))
            entries.sort()
            return entries

        iso_entries = best(isolated, "t_done_p50_us")
        e2e_entries = best(e2e, "e2e_p50_us")
        if not iso_entries or not e2e_entries:
            continue
        iso_best_v, iso_best_p = iso_entries[0]
        iso_second_v, iso_second_p = iso_entries[1] if len(iso_entries) > 1 else (float("nan"), "")
        e2e_best_v, e2e_best_p = e2e_entries[0]
        e2e_second_v, e2e_second_p = e2e_entries[1] if len(e2e_entries) > 1 else (float("nan"), "")

        def family_of(path):
            return "DUSHMEM" if path.startswith(("D", "FC", "DC")) else "RCCL"

        iso_family = family_of(iso_best_p)
        e2e_family = family_of(e2e_best_p)
        # Cross-substrate reversal: the substrate that wins isolated comm does
        # not field the end-to-end winner strategy.
        reversal = "REVERSAL" if iso_family != e2e_family else "CONSISTENT"
        iso_gap = 100.0 * (iso_second_v - iso_best_v) / iso_second_v if iso_entries else float("nan")
        e2e_gap = 100.0 * (e2e_second_v - e2e_best_v) / e2e_second_v if e2e_entries else float("nan")
        boundary_rows.append({
            "candidate": candidate, "N": n, "q": q,
            "isolated_winner_path": iso_best_p,
            "isolated_runner_up": iso_second_p,
            "isolated_gap_pct": fmt(iso_gap),
            "e2e_winner_path": e2e_best_p,
            "e2e_runner_up": e2e_second_p,
            "e2e_gap_pct": fmt(e2e_gap),
            "isolated_winner_family": iso_family,
            "e2e_winner_family": e2e_family,
            "isolated_best_us": fmt(iso_best_v),
            "e2e_best_us": fmt(e2e_best_v),
            "reversal_flag": reversal,
            "notes": "",
        })
    with open(boundary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=boundary_fields)
        writer.writeheader()
        writer.writerows(boundary_rows)

    # ---- cross-candidate boundary (RCCL-internal C0/C2 reversal carried in) ----
    # Phase 2/3 proved an RCCL-internal reversal: C2 (Ring/Simple/ch8) wins the
    # isolated collective but loses end-to-end at q=8. The per-candidate table
    # above cannot see it because comm_C0/comm_C2/r1_C0/r1_C2 land in separate
    # cells. Here we pool isolated entries {comm_C0, comm_C2, fc_C0, dc_C0} and
    # e2e entries {r1_C0, r1_C2, r0, rs, d0, ds, d1} across candidates per (N,q).
    xcand_path = os.path.join(summary_root, "phaseb_boundary_xcand.csv")
    xcand_fields = ["N", "q",
                    "isolated_ranking_top3", "isolated_winner_label", "isolated_best_us",
                    "e2e_ranking_top3", "e2e_winner_label", "e2e_best_us",
                    "iso_winner_family", "e2e_winner_family",
                    "substrate_reversal_flag",
                    "comm_C2_vs_C0_iso_pct", "r1_C2_vs_C0_e2e_pct", "rccl_config_flag"]
    xcand_rows = []
    by_nq = defaultdict(dict)
    for row in cell_rows:
        by_nq[(row["N"], row["q"])][(row["path"], row["candidate"])] = row

    def cell_value(row, metric):
        raw = row.get(metric)
        return float(raw) if raw != "" else None

    for (n, q), labels in sorted(by_nq.items()):
        iso_pool = {f"{path.lower()}_{cand.lower()}": cell_value(row, "t_done_p50_us")
                    for (path, cand), row in labels.items()
                    if path in ISOLATED_PATHS}
        e2e_pool = {f"{path.lower()}_{cand.lower()}": cell_value(row, "e2e_p50_us")
                    for (path, cand), row in labels.items()
                    if path in E2E_PATHS}
        iso_rank = sorted((v, k) for k, v in iso_pool.items() if v is not None)
        e2e_rank = sorted((v, k) for k, v in e2e_pool.items() if v is not None)
        if not iso_rank or not e2e_rank:
            continue
        iso_winner = iso_rank[0][1]
        e2e_winner = e2e_rank[0][1]

        def label_family(label):
            return "DUSHMEM" if label.split("_")[0] in ("fc", "dc", "d0", "ds", "d1", "d1w") else "RCCL"

        iso_fam, e2e_fam = label_family(iso_winner), label_family(e2e_winner)
        substrate_flag = "REVERSAL" if iso_fam != e2e_fam else "CONSISTENT"

        # RCCL-internal config reversal: does C2 beat C0 isolated but lose e2e?
        iso_c2 = iso_pool.get("comm_only_c2_ring_simple_ch8")
        iso_c0 = iso_pool.get("comm_only_c0_default")
        e2e_c2 = e2e_pool.get("r1_event_overlap_c2_ring_simple_ch8")
        e2e_c0 = e2e_pool.get("r1_event_overlap_c0_default")
        # (pool keys are lowercased above)
        config_flag = "N/A"
        iso_delta = e2e_delta = None
        if None not in (iso_c2, iso_c0, e2e_c2, e2e_c0) and iso_c0 > 0 and e2e_c0 > 0:
            iso_delta = 100.0 * (iso_c0 - iso_c2) / iso_c0   # + = C2 faster isolated
            e2e_delta = 100.0 * (e2e_c0 - e2e_c2) / e2e_c0   # + = C2 faster e2e
            if iso_delta * e2e_delta < 0 and min(abs(iso_delta), abs(e2e_delta)) >= 1.0:
                config_flag = "RCCL_CONFIG_REVERSAL"
            else:
                config_flag = "CONSISTENT"
        xcand_rows.append({
            "N": n, "q": q,
            "isolated_ranking_top3": " < ".join(f"{k}:{v:.0f}" for v, k in iso_rank[:3]),
            "isolated_winner_label": iso_winner,
            "isolated_best_us": fmt(iso_rank[0][0]),
            "e2e_ranking_top3": " < ".join(f"{k}:{v:.0f}" for v, k in e2e_rank[:3]),
            "e2e_winner_label": e2e_winner,
            "e2e_best_us": fmt(e2e_rank[0][0]),
            "iso_winner_family": iso_fam,
            "e2e_winner_family": e2e_fam,
            "substrate_reversal_flag": substrate_flag,
            "comm_C2_vs_C0_iso_pct": fmt(iso_delta) if iso_delta is not None else "",
            "r1_C2_vs_C0_e2e_pct": fmt(e2e_delta) if e2e_delta is not None else "",
            "rccl_config_flag": config_flag,
        })
    with open(xcand_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=xcand_fields)
        writer.writeheader()
        writer.writerows(xcand_rows)

    # ---- control table --------------------------------------------------------
    control_path = os.path.join(summary_root, "phaseb_control_table.csv")
    control_fields = ["candidate", "N", "q",
                      "r1_vs_rs_gain_pct", "r1_vs_r0_gain_pct", "r0_vs_rs_gain_pct",
                      "d1_vs_ds_gain_pct", "d1_vs_d0_gain_pct", "d0_vs_ds_gain_pct",
                      "r1_vs_d1_gain_pct", "d1_vs_dc_done_stretch_pct", "gemm_only_vs_r0_gemm_pct"]
    control_rows = []
    for (candidate, n, q), paths in sorted(by_cell.items()):
        def value(path, metric):
            row = paths.get(path)
            if row is None:
                return None
            raw = row.get(metric)
            return float(raw) if raw != "" else None

        def gain(winner, loser, metric="e2e_p50_us"):
            a = value(winner, metric)
            b = value(loser, metric)
            if a is None or b is None or b == 0:
                return ""
            return fmt(100.0 * (b - a) / b)

        gemm_metric = "gemm_interval_p50_us"
        # semantics stretch: how much d1's t_done exceeds pure dc transport
        # (same per-slice put-signal moves). Positive = the release protocol
        # (credit / self-WAR polling) plus comm/compute interference stretches
        # the transport shadow; this is the price paid for per-slice release.
        d1_done = value("D1_PUSHSIG_OVERLAP", "t_done_p50_us")
        dc_done = value("DC_PUSHSIG_ONLY", "t_done_p50_us")
        control_rows.append({
            "candidate": candidate, "N": n, "q": q,
            "r1_vs_rs_gain_pct": gain("R1_EVENT_OVERLAP", "RS_SLICE_SERIAL"),
            "r1_vs_r0_gain_pct": gain("R1_EVENT_OVERLAP", "R0_FULL_SERIAL"),
            "r0_vs_rs_gain_pct": gain("R0_FULL_SERIAL", "RS_SLICE_SERIAL"),
            "d1_vs_ds_gain_pct": gain("D1_PUSHSIG_OVERLAP", "DS_PUSHSIG_SERIAL"),
            "d1_vs_d0_gain_pct": gain("D1_PUSHSIG_OVERLAP", "D0_FCOLLECT_SERIAL"),
            "d0_vs_ds_gain_pct": gain("D0_FCOLLECT_SERIAL", "DS_PUSHSIG_SERIAL"),
            "r1_vs_d1_gain_pct": gain("R1_EVENT_OVERLAP", "D1_PUSHSIG_OVERLAP"),
            "d1_vs_dc_done_stretch_pct": (fmt(100.0 * (d1_done - dc_done) / dc_done)
                                          if d1_done is not None and dc_done
                                          else ""),
            "gemm_only_vs_r0_gemm_pct": "",
        })
    with open(control_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=control_fields)
        writer.writeheader()
        writer.writerows(control_rows)

    # ---- markdown summary ------------------------------------------------------
    report_path = os.path.join(summary_root, "phaseb_analysis.md")
    total = len(case_rows)
    passed = sum(1 for row in case_rows if row.get("status") == "PASS")
    lines = [
        "# Phase B analysis (cross-substrate AG-GEMM)",
        "",
        f"- result root: `{args.result_root}`",
        f"- cases: {total} (PASS {passed}, other {total - passed})",
        "",
        "## Selection boundary (isolated winner vs e2e winner)",
        "",
        "| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in boundary_rows:
        lines.append(
            f"| {row['candidate']} | {row['N']} | {row['q']} | {row['isolated_winner_path']} "
            f"({row['isolated_best_us']}us) | {row['e2e_winner_path']} ({row['e2e_best_us']}us) "
            f"| {row['isolated_winner_family']} | {row['e2e_winner_family']} | {row['reversal_flag']} |")
    lines += [
        "",
        "## Control table (positive = listed path faster)",
        "",
        "| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in control_rows:
        lines.append(
            f"| {row['candidate']} | {row['N']} | {row['q']} | {row['r1_vs_rs_gain_pct']} "
            f"| {row['r1_vs_r0_gain_pct']} | {row['d1_vs_ds_gain_pct']} | {row['d1_vs_d0_gain_pct']} "
            f"| {row['r1_vs_d1_gain_pct']} | {row['d1_vs_dc_done_stretch_pct']} |")
    reversals = sum(1 for row in boundary_rows if row["reversal_flag"] == "REVERSAL")
    lines += [
        "",
        f"Cross-substrate reversal cells (per-candidate table): {reversals} / {len(boundary_rows)}",
        "",
        "## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)",
        "",
        "| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in xcand_rows:
        lines.append(
            f"| {row['N']} | {row['q']} | {row['isolated_ranking_top3']} | {row['e2e_ranking_top3']} "
            f"| {row['iso_winner_family']} | {row['e2e_winner_family']} | {row['substrate_reversal_flag']} "
            f"| {row['comm_C2_vs_C0_iso_pct']} / {row['r1_C2_vs_C0_e2e_pct']} | {row['rccl_config_flag']} |")
    substrate_rev = sum(1 for row in xcand_rows if row["substrate_reversal_flag"] == "REVERSAL")
    config_rev = sum(1 for row in xcand_rows if row["rccl_config_flag"] == "RCCL_CONFIG_REVERSAL")
    lines += [
        "",
        f"Cross-substrate reversal cells: {substrate_rev} / {len(xcand_rows)}; "
        f"RCCL-internal C0/C2 reversal cells: {config_rev} / {len(xcand_rows)}",
        "",
        "## Case status",
        "",
    ]
    for row in case_rows:
        if row.get("status") != "PASS":
            lines.append(f"- {os.path.basename(row['case_dir'])}: status={row.get('status')} "
                         f"exit={row.get('exit_code')}")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"analysis written to {summary_root}")
    print(f"cases: {total} pass: {passed} boundary cells: {len(boundary_rows)} "
          f"reversals: {reversals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
