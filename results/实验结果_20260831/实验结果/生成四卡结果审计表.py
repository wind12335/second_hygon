#!/usr/bin/env python3
"""Generate auditable four-GPU summaries from the preserved raw CSV exports.

The inputs are intentionally left untouched.  RCCL channel records have repeated
case IDs, so this tool averages repeated measurements and exposes their span instead
of silently retaining an arbitrary row.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
NCCL_INPUT = ROOT / "nccl_rtx4090_4gpu_formal_summary.csv"
RCCL_INPUT = ROOT / "rccl_k500sm_ai_4gpu_formal_summary.csv"
RCCL_AUDIT_OUTPUT = ROOT / "rccl_4gpu_caseid_audit.csv"
DEFAULT_OUTPUT = ROOT / "四卡默认策略跨平台对齐汇总.csv"
CHANNEL_OUTPUT = ROOT / "四卡RingSimple通道扩展汇总.csv"

TARGET_SIZES = (1 << 20, 8 << 20, 64 << 20, 256 << 20, 1 << 30)
CHANNELS = (1, 2, 4, 8)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_field(rows: list[dict[str, str]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def collective_key(name: str) -> str:
    return name.lower().replace("_", "")


def rccl_case_tokens(case_id: str) -> list[str]:
    return case_id.lower().split("_")


def rccl_case_matches(
    row: dict[str, str], phase: str, collective: str, size: int, *tokens: str
) -> bool:
    case_id = row["case_id"].lower()
    return (
        row["collective"] == collective
        and int(row["ranks"]) == 4
        and int(row["size_bytes"]) == size
        and case_id.startswith(f"{phase}_")
        and all(token in rccl_case_tokens(case_id) for token in tokens)
    )


def valid_nccl(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if int(row["ranks"]) == 4
        and row["status"] == "0"
        and row["correctness"] == "PASS"
        and int(row["wrong_count"]) == 0
    ]


def auditable_rccl(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if int(row["ranks"]) == 4
        and int(row["wrong_count"]) == 0
    ]


def valid_rccl(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "0"]


def write_rccl_audit(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)

    fieldnames = [
        "case_id",
        "collective",
        "size_bytes",
        "original_row_count",
        "status_set",
        "mean_time_us",
        "mean_algbw_gbps",
        "mean_busbw_gbps",
        "min_busbw_gbps",
        "max_busbw_gbps",
        "busbw_span_gbps",
        "busbw_relative_span_percent",
        "action",
    ]
    with RCCL_AUDIT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_id in sorted(grouped):
            case_rows = grouped[case_id]
            busbws = [float(row["busbw_gbps"]) for row in case_rows]
            mean_busbw = fmean(busbws)
            span = max(busbws) - min(busbws)
            relative_span = 100 * span / mean_busbw if mean_busbw else 0.0
            if any(row["status"] != "0" for row in case_rows):
                action = "CHECK_UNKNOWN_STATUS_IN_SOURCE_LOG"
            elif len(case_rows) == 1:
                action = "SINGLE_VALID_RECORD"
            elif relative_span >= 10:
                action = "RERUN_REQUIRED_HIGH_VARIATION"
            else:
                action = "AVERAGED_REPEATED_RECORDS"
            writer.writerow(
                {
                    "case_id": case_id,
                    "collective": case_rows[0]["collective"],
                    "size_bytes": case_rows[0]["size_bytes"],
                    "original_row_count": len(case_rows),
                    "status_set": ";".join(sorted({row["status"] for row in case_rows})),
                    "mean_time_us": f"{mean_field(case_rows, 'time_us'):.6f}",
                    "mean_algbw_gbps": f"{mean_field(case_rows, 'algbw_gbps'):.6f}",
                    "mean_busbw_gbps": f"{mean_busbw:.6f}",
                    "min_busbw_gbps": f"{min(busbws):.6f}",
                    "max_busbw_gbps": f"{max(busbws):.6f}",
                    "busbw_span_gbps": f"{span:.6f}",
                    "busbw_relative_span_percent": f"{relative_span:.6f}",
                    "action": action,
                }
            )


def select_nccl_default(
    rows: list[dict[str, str]], collective: str, size: int
) -> list[dict[str, str]]:
    if size == 1 << 30:
        phase = "one_gib"
    else:
        phase = "representative"
    return [
        row
        for row in rows
        if row["phase"] == phase
        and row["collective"] == collective
        and int(row["size_bytes"]) == size
        and row["requested_algo"] == "DEFAULT"
        and row["requested_proto"] == "DEFAULT"
        and row["requested_channels"] == "DEFAULT"
    ]


def select_rccl_default(
    rows: list[dict[str, str]], collective: str, size: int
) -> list[dict[str, str]]:
    if size == 1 << 30:
        return [
            row
            for row in rows
            if row["case_id"].lower() == collective_key(collective)
        ]
    return [
        row
        for row in rows
        if rccl_case_matches(row, "rep", collective, size, "default", "chdefault")
    ]


def summary_fields(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    if not rows:
        return "", "", "0"
    return (
        f"{mean_field(rows, 'time_us'):.6f}",
        f"{mean_field(rows, 'busbw_gbps'):.6f}",
        str(len(rows)),
    )


def write_default_comparison(
    nccl_rows: list[dict[str, str]], rccl_rows: list[dict[str, str]]
) -> None:
    fieldnames = [
        "collective",
        "size_bytes",
        "nccl_time_us_mean",
        "nccl_busbw_gbps_mean",
        "nccl_raw_records",
        "rccl_time_us_mean",
        "rccl_busbw_gbps_mean",
        "rccl_raw_records",
        "nccl_over_rccl_busbw_ratio",
        "comparison_scope",
    ]
    with DEFAULT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for collective in ("AllGather", "AllReduce", "ReduceScatter"):
            for size in TARGET_SIZES:
                nccl = select_nccl_default(nccl_rows, collective, size)
                rccl = select_rccl_default(rccl_rows, collective, size)
                nccl_time, nccl_busbw, nccl_count = summary_fields(nccl)
                rccl_time, rccl_busbw, rccl_count = summary_fields(rccl)
                ratio = ""
                if nccl_busbw and rccl_busbw and float(rccl_busbw):
                    ratio = f"{float(nccl_busbw) / float(rccl_busbw):.6f}"
                writer.writerow(
                    {
                        "collective": collective,
                        "size_bytes": size,
                        "nccl_time_us_mean": nccl_time,
                        "nccl_busbw_gbps_mean": nccl_busbw,
                        "nccl_raw_records": nccl_count,
                        "rccl_time_us_mean": rccl_time,
                        "rccl_busbw_gbps_mean": rccl_busbw,
                        "rccl_raw_records": rccl_count,
                        "nccl_over_rccl_busbw_ratio": ratio,
                        "comparison_scope": "Different accelerator, topology, transport, and software stacks; not a library-only comparison.",
                    }
                )


def select_nccl_channel(
    rows: list[dict[str, str]], collective: str, size: int, channel: int
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["phase"] == "channels"
        and row["collective"] == collective
        and int(row["size_bytes"]) == size
        and row["requested_algo"].upper() == "RING"
        and row["requested_proto"].upper() == "SIMPLE"
        and row["requested_channels"] == str(channel)
    ]


def select_rccl_channel(
    rows: list[dict[str, str]], collective: str, size: int, channel: int
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if rccl_case_matches(
            row, "channels", collective, size, "ring", "simple", f"ch{channel}"
        )
    ]


def write_channel_scaling(
    nccl_rows: list[dict[str, str]], rccl_rows: list[dict[str, str]]
) -> None:
    fieldnames = [
        "backend_platform",
        "collective",
        "size_bytes",
        "channel",
        "raw_record_count",
        "mean_busbw_gbps",
        "relative_to_channel_1",
        "aggregation_note",
    ]
    with CHANNEL_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for backend, rows, selector, note in (
            (
                "NCCL_RTX4090",
                nccl_rows,
                select_nccl_channel,
                "Three requested repeats are averaged when present.",
            ),
            (
                "RCCL_K500SM_AI",
                rccl_rows,
                select_rccl_channel,
                "Repeated case IDs are averaged; inspect rccl_4gpu_caseid_audit.csv for span.",
            ),
        ):
            for collective in ("AllGather", "AllReduce", "ReduceScatter"):
                for size in TARGET_SIZES[:-1]:
                    ch1_rows = selector(rows, collective, size, 1)
                    ch1 = mean_field(ch1_rows, "busbw_gbps") if ch1_rows else 0.0
                    for channel in CHANNELS:
                        selected = selector(rows, collective, size, channel)
                        if not selected:
                            continue
                        mean_busbw = mean_field(selected, "busbw_gbps")
                        writer.writerow(
                            {
                                "backend_platform": backend,
                                "collective": collective,
                                "size_bytes": size,
                                "channel": channel,
                                "raw_record_count": len(selected),
                                "mean_busbw_gbps": f"{mean_busbw:.6f}",
                                "relative_to_channel_1": f"{mean_busbw / ch1:.6f}" if ch1 else "",
                                "aggregation_note": note,
                            }
                        )


def main() -> None:
    nccl_rows = valid_nccl(read_csv(NCCL_INPUT))
    rccl_audit_rows = auditable_rccl(read_csv(RCCL_INPUT))
    rccl_rows = valid_rccl(rccl_audit_rows)
    write_rccl_audit(rccl_audit_rows)
    write_default_comparison(nccl_rows, rccl_rows)
    write_channel_scaling(nccl_rows, rccl_rows)


if __name__ == "__main__":
    main()
