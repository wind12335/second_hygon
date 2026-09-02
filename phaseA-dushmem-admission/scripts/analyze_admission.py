#!/usr/bin/env python3
"""Generate auditable CSV and Markdown summaries from one admission run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else "nan"


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    if not path.exists():
        return manifest
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            manifest[row["case_id"]] = row
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    raw_rows: list[dict[str, str]] = []
    for raw_csv in sorted(run_root.glob("cases/*/raw/rank_*.csv")):
        with raw_csv.open(newline="") as handle:
            raw_rows.extend(csv.DictReader(handle))

    manifest = read_manifest(run_root / "manifest.csv")
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_case[row["case_id"]].append(row)

    global_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    all_case_ids = sorted(set(manifest) | set(rows_by_case))
    for case_id in all_case_ids:
        rows = rows_by_case.get(case_id, [])
        process = manifest.get(case_id, {})
        expected_epochs = int(process.get("epochs", "0") or 0)
        expected_ranks = int(process.get("expected_pes", "4") or 4)
        by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_epoch[as_int(row, "epoch")].append(row)

        pass_release_times: list[float] = []
        pass_comm_times: list[float] = []
        pass_checked_times: list[float] = []
        pass_release_after_comm_times: list[float] = []
        pass_checksum_after_release_times: list[float] = []
        global_pass_epochs = 0
        missing_rank_epochs = 0
        failed_epochs = 0
        for epoch in sorted(by_epoch):
            epoch_rows = by_epoch[epoch]
            rank_set = {as_int(row, "rank") for row in epoch_rows}
            complete = len(rank_set) == expected_ranks
            status = "PASS"
            if not complete:
                status = "FAIL_MISSING_RANK"
                missing_rank_epochs += 1
            elif any(row["iteration_status"] != "PASS" for row in epoch_rows):
                status = "FAIL_ITERATION"
                failed_epochs += 1
            checked_max = max(as_float(row, "issue_to_checked_us") for row in epoch_rows)
            release_max = max(as_float(row, "issue_to_release_us") for row in epoch_rows)
            comm_max = max(as_float(row, "issue_to_comm_stream_complete_us") for row in epoch_rows)
            checksum_mismatches = sum(as_int(row, "checksum_mismatches") for row in epoch_rows)
            if status == "PASS":
                global_pass_epochs += 1
                pass_release_times.append(release_max)
                pass_comm_times.append(comm_max)
                pass_checked_times.append(checked_max)
                pass_release_after_comm_times.append(release_max - comm_max)
                pass_checksum_after_release_times.append(checked_max - release_max)
            global_rows.append(
                {
                    "case_id": case_id,
                    "epoch": epoch,
                    "observed_ranks": len(rank_set),
                    "expected_ranks": expected_ranks,
                    "max_rank_release_us": f"{release_max:.3f}",
                    "max_rank_comm_stream_complete_us": f"{comm_max:.3f}",
                    "max_rank_checked_us": f"{checked_max:.3f}",
                    "max_rank_release_after_comm_us": f"{release_max - comm_max:.3f}",
                    "max_rank_checksum_after_release_us": f"{checked_max - release_max:.3f}",
                    "checksum_mismatches": checksum_mismatches,
                    "global_status": status,
                }
            )

        exit_status = process.get("exit_status", "MISSING")
        observed_epochs = len(by_epoch)
        case_pass = (
            exit_status == "0"
            and expected_epochs > 0
            and observed_epochs == expected_epochs
            and global_pass_epochs == expected_epochs
            and missing_rank_epochs == 0
            and failed_epochs == 0
        )
        summary_rows.append(
            {
                "case_id": case_id,
                "mode": process.get("mode", rows[0]["mode"] if rows else "UNKNOWN"),
                "payload_bytes": process.get("payload_bytes", rows[0]["payload_bytes"] if rows else ""),
                "epochs_expected": expected_epochs,
                "epochs_observed": observed_epochs,
                "epochs_pass": global_pass_epochs,
                "missing_rank_epochs": missing_rank_epochs,
                "failed_epochs": failed_epochs,
                "process_exit_status": exit_status,
                "p50_max_rank_release_us": fmt(percentile(pass_release_times, 0.50)),
                "p05_max_rank_release_us": fmt(percentile(pass_release_times, 0.05)),
                "p95_max_rank_release_us": fmt(percentile(pass_release_times, 0.95)),
                "p50_max_rank_comm_stream_complete_us": fmt(percentile(pass_comm_times, 0.50)),
                "p50_max_rank_checked_us": fmt(percentile(pass_checked_times, 0.50)),
                "p05_max_rank_checked_us": fmt(percentile(pass_checked_times, 0.05)),
                "p95_max_rank_checked_us": fmt(percentile(pass_checked_times, 0.95)),
                "mean_max_rank_checked_us": fmt(statistics.fmean(pass_checked_times)) if pass_checked_times else "nan",
                "p50_max_rank_release_after_comm_us": fmt(
                    percentile(pass_release_after_comm_times, 0.50)),
                "p50_max_rank_checksum_after_release_us": fmt(
                    percentile(pass_checksum_after_release_times, 0.50)),
                "admission_status": "PASS" if case_pass else "FAIL",
            }
        )

    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    global_path = analysis_dir / "global_iteration_max.csv"
    with global_path.open("w", newline="") as handle:
        fields = [
            "case_id", "epoch", "observed_ranks", "expected_ranks", "max_rank_release_us",
            "max_rank_comm_stream_complete_us", "max_rank_checked_us",
            "max_rank_release_after_comm_us", "max_rank_checksum_after_release_us",
            "checksum_mismatches", "global_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows)

    summary_path = analysis_dir / "case_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fields = [
            "case_id", "mode", "payload_bytes", "epochs_expected", "epochs_observed",
            "epochs_pass", "missing_rank_epochs", "failed_epochs", "process_exit_status",
            "p50_max_rank_release_us", "p05_max_rank_release_us",
            "p95_max_rank_release_us", "p50_max_rank_comm_stream_complete_us",
            "p50_max_rank_checked_us", "p05_max_rank_checked_us",
            "p95_max_rank_checked_us", "mean_max_rank_checked_us",
            "p50_max_rank_release_after_comm_us",
            "p50_max_rank_checksum_after_release_us", "admission_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_by_id = {row["case_id"]: row for row in summary_rows}
    paired_rows: list[dict[str, object]] = []
    for base_id, process in sorted(manifest.items()):
        if (process.get("mode") != "put_signal" or process.get("quiet") != "0" or
                process.get("credit") != "0" or process.get("slots") != "1"):
            continue
        base = summary_by_id.get(base_id)
        if not base or base["admission_status"] != "PASS":
            continue
        payload = process.get("payload_bytes")
        comparisons = [
            ("put_signal_vs_quiet", candidate_id)
            for candidate_id, candidate in manifest.items()
            if candidate.get("mode") == "put_signal" and candidate.get("quiet") == "1" and
            candidate.get("credit") == "0" and candidate.get("slots") == "1" and
            candidate.get("payload_bytes") == payload
        ]
        comparisons.extend(
            ("put_signal_vs_fcollect", candidate_id)
            for candidate_id, candidate in manifest.items()
            if candidate.get("mode") == "fcollect" and candidate.get("payload_bytes") == payload
        )
        for comparison, candidate_id in comparisons:
            candidate = summary_by_id.get(candidate_id)
            if not candidate or candidate["admission_status"] != "PASS":
                continue
            base_checked = float(base["p50_max_rank_checked_us"])
            candidate_checked = float(candidate["p50_max_rank_checked_us"])
            change_pct = 100.0 * (candidate_checked - base_checked) / base_checked
            paired_rows.append(
                {
                    "comparison": comparison,
                    "payload_bytes": payload,
                    "baseline_case": base_id,
                    "candidate_case": candidate_id,
                    "baseline_p50_release_us": base["p50_max_rank_release_us"],
                    "candidate_p50_release_us": candidate["p50_max_rank_release_us"],
                    "baseline_p50_checked_us": base["p50_max_rank_checked_us"],
                    "candidate_p50_checked_us": candidate["p50_max_rank_checked_us"],
                    "candidate_minus_baseline_checked_us": fmt(candidate_checked - base_checked),
                    "candidate_minus_baseline_checked_pct": fmt(change_pct),
                    "faster_checked_path": (candidate_id if candidate_checked < base_checked else base_id),
                }
            )

    paired_path = analysis_dir / "paired_strategy_comparison.csv"
    with paired_path.open("w", newline="") as handle:
        fields = [
            "comparison", "payload_bytes", "baseline_case", "candidate_case",
            "baseline_p50_release_us", "candidate_p50_release_us",
            "baseline_p50_checked_us", "candidate_p50_checked_us",
            "candidate_minus_baseline_checked_us", "candidate_minus_baseline_checked_pct",
            "faster_checked_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    analyzer_source = Path(__file__).resolve()
    with analyzer_source.open("rb") as handle:
        analyzer_digest = hashlib.sha256(handle.read()).hexdigest()
    (analysis_dir / "analysis_provenance.txt").write_text(
        f"analyzer={analyzer_source}\nsha256={analyzer_digest}\n")

    report_path = analysis_dir / "admission_report.md"
    with report_path.open("w") as report:
        report.write("# DUSHMEM Phase A Admission Report\n\n")
        report.write("Target platform: `K500SM_AI / gfx928 / 4 GPUs / PCIe`.\n\n")
        report.write("A case is PASS only when its process exits with zero, every expected epoch "
                     "contains four rank rows, and every row reports a successful full-payload check.\n\n")
        report.write("`release` means the receiver's wait stream has observed all required remote "
                     "signals. `comm stream complete` means the local sender has completed its "
                     "enqueued communication work. `checked` additionally includes the complete "
                     "GPU payload verifier. These are deliberately separate measurements.\n\n")
        report.write("| Case | Mode | Bytes | Epochs pass/expected | p50 release us | p50 comm-stream us | p50 checked us | Status |\n")
        report.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            report.write(
                f"| {row['case_id']} | {row['mode']} | {row['payload_bytes']} | "
                f"{row['epochs_pass']}/{row['epochs_expected']} | "
                f"{row['p50_max_rank_release_us']} | "
                f"{row['p50_max_rank_comm_stream_complete_us']} | "
                f"{row['p50_max_rank_checked_us']} | {row['admission_status']} |\n"
            )
        if paired_rows:
            report.write("\n## Matched Strategy Controls\n\n")
            report.write("| Comparison | Bytes | Baseline | Candidate | Candidate checked-time change | Faster checked path |\n")
            report.write("|---|---:|---|---|---:|---|\n")
            for row in paired_rows:
                report.write(
                    f"| {row['comparison']} | {row['payload_bytes']} | {row['baseline_case']} | "
                    f"{row['candidate_case']} | {row['candidate_minus_baseline_checked_pct']}% | "
                    f"{row['faster_checked_path']} |\n")
        report.write("\nA FAIL is evidence, not a discarded sample. Inspect the corresponding "
                     "`cases/<case_id>/stdout_stderr.log` and raw rank CSV files before modifying a protocol.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
