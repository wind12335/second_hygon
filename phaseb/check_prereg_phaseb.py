#!/usr/bin/env python3
"""Pre-registration verdicts for Phase B (design doc §12), evaluated from CSVs.

Every prediction in PHASEB_EXPERIMENT_DESIGN.md §12 was written down BEFORE
the data landed. This script turns each into a mechanical HIT / MISS /
PENDING line so the paper can quote an honest scorecard instead of a
hand-picked one. The q16/dsfix-dependent items stay PENDING until the
corresponding summary dirs exist.

Usage:
  check_prereg_phaseb.py --summary-dir <formal_root>/summary \
      [--dsfix-summary <dsfix_root>/summary] \
      [--nvidia-summary <nvidia_root>/summary]   # P8-P13 (sealed §14)

Verdicts:
  P1  N4096/q4  r1 e2e delta(C2 vs C0) in [-2.5, +0.5]
  P2  N4096/q8  delta in [-9, -3]                      (the famous MISS)
  P3  N2048/q16 delta < -5.5   (est -8..-12)
  P4  N2048/q16 stretch(d1 vs dc) > 90
  P5  dsfix: ds slower than d1 in EVERY cell (d1_vs_ds_gain_pct > 0)
  P6  dsfix: d1w first release ~= dc scale (not d1 scale) at N512/q8,
      and d1w_vs_d0 positive again at q8 cells
  P7  N4096/q8  d1_vs_d0 sign — explicitly NO prediction (observation only)
  DX  dsfix: N4096/q16 delta stays positive (B3: q has no independent
      effect at fixed ratio 0.59) — the discriminator for the balance law
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def e2e_delta(matrix_rows, n, q):
    """(c0 - c2)/c0 * 100 on r1 e2e for a cell; + : C2 faster."""
    vals = {}
    for r in matrix_rows:
        if int(r["N"]) == n and int(r["q"]) == q \
                and r["path"] == "R1_EVENT_OVERLAP" and r["e2e_p50_us"]:
            vals[r["candidate"]] = float(r["e2e_p50_us"])
    if "C0_DEFAULT" in vals and "C2_RING_SIMPLE_CH8" in vals:
        return 100.0 * (vals["C0_DEFAULT"] - vals["C2_RING_SIMPLE_CH8"]) \
            / vals["C0_DEFAULT"]
    return None


def control_value(rows, n, q, column):
    for r in rows:
        if int(r["N"]) == n and int(r["q"]) == q and r.get(column) not in \
                (None, "", "nan"):
            return float(r[column])
    return None


def first_release(rows, n, q, path, candidate="C0_DEFAULT"):
    per_rep = [float(r["release_first_med_us"]) for r in rows
               if r.get("path") == path and r.get("candidate") == candidate
               and r.get("N") and int(r["N"]) == n and r.get("q")
               and int(r["q"]) == q and r.get("release_first_med_us")]
    return statistics.median(per_rep) if per_rep else None


def emit_print(pid, pred, obs, state, note=""):
    """Same format as emit(), but prints immediately (NVIDIA section)."""
    obs_s = "n/a" if obs is None else f"{obs:+.3f}"
    print(f"{pid:<4} {state:<7} predicted {pred:<42} "
          f"observed {obs_s:>9}   {note}")


def verdict(ok):
    return "HIT" if ok else "MISS"


def short(path):
    return {"R0_FULL_SERIAL": "r0", "RS_SLICE_SERIAL": "rs",
            "R1_EVENT_OVERLAP": "r1", "D0_FCOLLECT_SERIAL": "d0",
            "D1_PUSHSIG_OVERLAP": "d1", "D1W_WAITSTREAM_OVERLAP": "d1w",
            "COMM_ONLY": "comm", "GEMM_ONLY": "gemm",
            "FC_FCOLLECT_ONLY": "fc", "DC_PUSHSIG_ONLY": "dc"}[path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True,
                        help="formal root's summary directory")
    parser.add_argument("--dsfix-summary", default=None,
                        help="dsfix root's summary directory (for P5/P6/DX)")
    parser.add_argument("--nvidia-summary", default=None,
                        help="NVIDIA root's summary directory (for P8-P13, "
                             "pre-registered in design doc §14)")
    args = parser.parse_args()

    matrix = read_rows(os.path.join(args.summary_dir,
                                    "phaseb_cell_matrix.csv"))
    control = read_rows(os.path.join(args.summary_dir,
                                     "phaseb_control_table.csv"))
    release = read_rows(os.path.join(args.summary_dir,
                                     "phaseb_release_curve_summary.csv"))
    dsfix_matrix, dsfix_control, dsfix_release = [], [], []
    if args.dsfix_summary:
        dsfix_matrix = read_rows(os.path.join(args.dsfix_summary,
                                              "phaseb_cell_matrix.csv"))
        dsfix_control = read_rows(os.path.join(args.dsfix_summary,
                                               "phaseb_control_table.csv"))
        dsfix_release = read_rows(os.path.join(args.dsfix_summary,
                                               "phaseb_release_curve_summary.csv"))
        # dc was never buggy: reuse the formal root's dc release curve
        dsfix_release += [r for r in release if r.get("path") == "DC_PUSHSIG_ONLY"]

    lines = []
    total = {"HIT": 0, "MISS": 0, "PENDING": 0, "OBSERVED": 0}

    def emit(pid, pred, obs, state, note=""):
        total[state] += 1
        obs_s = "n/a" if obs is None else f"{obs:+.3f}"
        lines.append(f"{pid:<4} {state:<7} predicted {pred:<42} "
                     f"observed {obs_s:>9}   {note}")

    d = e2e_delta(matrix, 4096, 4)
    emit("P1", "N4096/q4 delta in [-2.5,+0.5]", d,
         "PENDING" if d is None else verdict(-2.5 <= d <= 0.5))
    d = e2e_delta(matrix, 4096, 8)
    emit("P2", "N4096/q8 delta in [-9,-3]", d,
         "PENDING" if d is None else verdict(-9.0 <= d <= -3.0),
         "MISS -> balance law (design §13)")
    d = e2e_delta(matrix, 2048, 16)
    emit("P3", "N2048/q16 delta < -5.5 (est -8..-12)", d,
         "PENDING" if d is None else verdict(d < -5.5))
    s = control_value(control, 2048, 16, "d1_vs_dc_done_stretch_pct")
    emit("P4", "N2048/q16 stretch(d1/dc) > 90", s,
         "PENDING" if s is None else verdict(s > 90.0))

    # P5: dsfix, every cell d1 faster than (true-serial) ds
    if dsfix_control:
        gains = {(int(r["N"]), int(r["q"])): control_value(dsfix_control,
                                                           int(r["N"]),
                                                           int(r["q"]),
                                                           "d1_vs_ds_gain_pct")
                 for r in dsfix_control}
        gains = {k: v for k, v in gains.items() if v is not None}
        if gains:
            bad = [k for k, v in gains.items() if v <= 0]
            state = verdict(not bad)
            note = f"{len(gains)} cells; violations: {bad or 'none'}"
            emit("P5", "ds slower than d1 in EVERY cell (gain>0)",
                 min(gains.values()), state, note)
        else:
            emit("P5", "ds slower than d1 in EVERY cell (gain>0)", None,
                 "PENDING")
    else:
        emit("P5", "ds slower than d1 in EVERY cell (gain>0)", None,
             "PENDING", "needs --dsfix-summary")

    # P6: d1w first release at dc scale + d1w_vs_d0 positive at q8
    if dsfix_release and dsfix_control:
        fr = {p: first_release(dsfix_release, 512, 8, p) for p in
              ("DC_PUSHSIG_ONLY", "D1_PUSHSIG_OVERLAP",
               "D1W_WAITSTREAM_OVERLAP")}
        if all(fr.values()):
            scale_ok = fr["D1W_WAITSTREAM_OVERLAP"] < \
                0.5 * fr["D1_PUSHSIG_OVERLAP"]
            note = (f"dc={fr['DC_PUSHSIG_ONLY']:.0f}us "
                    f"d1={fr['D1_PUSHSIG_OVERLAP']:.0f}us "
                    f"d1w={fr['D1W_WAITSTREAM_OVERLAP']:.0f}us")
            q8 = [control_value(dsfix_control, n, 8, "d1w_vs_d0_gain_pct")
                  for n in (512, 2048, 4096)]
            q8 = [x for x in q8 if x is not None]
            sign_ok = bool(q8) and all(x > 0 for x in q8)
            state = verdict(scale_ok and sign_ok)
            note += f"; d1w_vs_d0@q8={['%+.1f' % x for x in q8]}"
            emit("P6", "d1w 1st release ~ dc scale & d1w>d0 at q8",
                 q8[0] if q8 else None, state, note)
        else:
            emit("P6", "d1w 1st release ~ dc scale & d1w>d0 at q8", None,
                 "PENDING", f"have { {k: (v and round(v)) for k, v in fr.items()} }")
    else:
        emit("P6", "d1w 1st release ~ dc scale & d1w>d0 at q8", None,
             "PENDING", "needs --dsfix-summary")

    # P7: no prediction — record the vacuum point
    v = control_value(control, 4096, 8, "d1_vs_d0_gain_pct")
    emit("P7", "N4096/q8 d1_vs_d0 — NO prediction (vacuum)", v,
         "PENDING" if v is None else "OBSERVED")

    # DX: N4096/q16 stays positive (B3: q has no independent effect)
    src = dsfix_matrix or matrix
    d = e2e_delta(src, 4096, 16)
    emit("DX", "N4096/q16 delta > 0 (B3 discriminator)", d,
         "PENDING" if d is None else verdict(d > 0.0),
         "if negative, B3 needs a q-term")

    print(f"pre-registration scorecard  ({args.summary_dir}"
          + (f" + {args.dsfix_summary}" if args.dsfix_summary else "") + ")")
    for line in lines:
        print(line)
    print(f"\ntotals: HIT={total['HIT']}  MISS={total['MISS']}  "
          f"PENDING={total['PENDING']}  (P7 excluded: no prediction)")

    # ---- NVIDIA-side pre-registrations (sealed §14, before any 4090 data) ----
    if args.nvidia_summary:
        print(f"\nNVIDIA pre-registrations (sealed before any 4090 data):")
        nvidia(matrix, control, args.nvidia_summary, emit_print, total)
        print(f"N-section totals: HIT={total['HIT']}  MISS={total['MISS']}  "
              f"PENDING={total['PENDING']}")
    return 0


def nvidia(matrix, control, nvidia_summary, emit, total):
    """P8-P13, sealed 2026-09-02 before any NVIDIA data landed (§14)."""
    nmat = read_rows(os.path.join(nvidia_summary, "phaseb_cell_matrix.csv"))
    nctl = read_rows(os.path.join(nvidia_summary, "phaseb_control_table.csv"))
    nrel = read_rows(os.path.join(nvidia_summary,
                                  "phaseb_release_curve_summary.csv"))

    def ratios_gaps(rows):
        comm, gemm, iso = {}, {}, {}
        for r in rows:
            key = (int(r["N"]), int(r["q"]))
            if r["path"] == "COMM_ONLY" and r["t_done_p50_us"]:
                comm[key] = float(r["t_done_p50_us"])
                if r["candidate"] == "C0_DEFAULT":
                    iso[key] = float(r["t_done_p50_us"])
            if r["path"] == "GEMM_ONLY" and r["candidate"] == "C0_DEFAULT" \
                    and r["t_done_p50_us"]:
                gemm[key] = float(r["t_done_p50_us"])
        iso2 = {}
        for r in rows:
            if r["path"] == "COMM_ONLY" \
                    and r["candidate"] == "C2_RING_SIMPLE_CH8" \
                    and r["t_done_p50_us"]:
                iso2[(int(r["N"]), int(r["q"]))] = float(r["t_done_p50_us"])
        ratios = {k: comm[k] / gemm[k] for k in comm.keys() & gemm.keys()
                  if k in iso}
        gaps = {k: 100.0 * (iso[k] - iso2[k]) / iso[k]
                for k in iso.keys() & iso2.keys()}
        return ratios, gaps

    k_ratios, _ = ratios_gaps(matrix + [])
    n_ratios, n_gaps = ratios_gaps(nmat)

    # P8: band shift — NVIDIA ratios smaller; reversal (if any) at N < 2048
    if k_ratios and n_ratios:
        shared = k_ratios.keys() & n_ratios.keys()
        shifted = sum(1 for k in shared if n_ratios[k] < k_ratios[k]) \
            >= (len(shared) + 1) // 2
        reversals = [k for k in n_ratios
                     if e2e_delta(nmat, *k) is not None and e2e_delta(nmat, *k) < 0]
        n_smaller = all(n < 2048 for n, _ in reversals) if reversals else True
        obs = statistics.median([n_ratios[k] for k in shared]) - \
            statistics.median([k_ratios[k] for k in shared])
        emit("P8", "ratio_4090 < ratio_K500; reversal N < 2048",
             obs, verdict(shifted and n_smaller),
             f"shifted {shifted}; reversals {reversals or 'none'}")
    else:
        emit("P8", "ratio_4090 < ratio_K500; reversal N < 2048", None,
             "PENDING", "missing COMM/GEMM cells")

    # P9: strategy axis non-degenerate — some cell where winner != r1
    cells = defaultdict(dict)
    for r in nmat:
        if r["candidate"] == "C0_DEFAULT" and r["e2e_p50_us"]:
            key = (int(r["N"]), int(r["q"]))
            if r["path"] in ("R0_FULL_SERIAL", "RS_SLICE_SERIAL",
                             "R1_EVENT_OVERLAP", "D0_FCOLLECT_SERIAL",
                             "D1_PUSHSIG_OVERLAP", "D1W_WAITSTREAM_OVERLAP"):
                cells[key][short(r["path"])] = float(r["e2e_p50_us"])
    if cells:
        non_r1 = [k for k, d in cells.items() if d and min(d, key=d.get) != "r1"]
        emit("P9", "at least one cell with e2e winner != r1",
             len(non_r1), verdict(bool(non_r1)), f"cells {non_r1 or 'none'}")
    else:
        emit("P9", "at least one cell with e2e winner != r1", None, "PENDING")

    # P10: wait placement transplants — d1w beats d1 at q>=8, first release
    # at dc scale
    if nctl:
        gains = [control_value(nctl, n, 8, "d1w_vs_d1_gain_pct")
                 for n in (512, 2048, 4096)]
        gains = [g for g in gains if g is not None]
        scale_ok = True
        if nrel:
            for n in (512, 2048, 4096):
                fr = {p: first_release(nrel, n, 8, p) for p in
                      ("DC_PUSHSIG_ONLY", "D1_PUSHSIG_OVERLAP",
                       "D1W_WAITSTREAM_OVERLAP")}
                if all(fr.values()):
                    scale_ok = scale_ok and fr["D1W_WAITSTREAM_OVERLAP"] < \
                        0.5 * fr["D1_PUSHSIG_OVERLAP"]
        if gains:
            emit("P10", "d1w > d1 at q8 & d1w release ~ dc scale",
                 min(gains), verdict(bool(gains) and all(g > 0 for g in gains)
                                     and scale_ok),
                 f"gains@q8={['%+.1f' % g for g in gains]} scale_ok={scale_ok}")
        else:
            emit("P10", "d1w > d1 at q8 & d1w release ~ dc scale", None,
                 "PENDING", "d1w_vs_d1 missing (control table)")
    else:
        emit("P10", "d1w > d1 at q8 & d1w release ~ dc scale", None, "PENDING")

    # P11: NCCL env-tuning headroom thinner — median iso gap < 1.5%
    if n_gaps:
        med = statistics.median(n_gaps.values())
        emit("P11", "median iso_gap(C2 vs C0) on 4090 < 1.5%", med,
             verdict(med < 1.5), f"{len(n_gaps)} cells")
    else:
        emit("P11", "median iso_gap(C2 vs C0) on 4090 < 1.5%", None, "PENDING")

    # P12: B3 with K500-calibrated thresholds transfers — top1 >= 8/9
    hits, cells_b3 = 0, 0
    for key, _ in sorted(n_ratios.items()):
        d = e2e_delta(nmat, *key)
        if d is None:
            continue
        n, q = key
        gap = n_gaps.get(key, float("nan"))
        pred = "c0" if (q >= 8 and 0.9 <= n_ratios[key] <= 1.35
                        and gap == gap and gap <= 2.0) else "c2"
        true_cfg = "c0" if d < 0 else "c2"
        cells_b3 += 1
        hits += (pred == true_cfg)
    if cells_b3:
        emit("P12", "K500-calibrated B3 top1 >= 8 on 4090", hits,
             verdict(hits >= 8 and cells_b3 >= 9), f"{hits}/{cells_b3} cells")
    else:
        emit("P12", "K500-calibrated B3 top1 >= 8 on 4090", None, "PENDING")

    # P13: capability vector differs — fc/comm and dc/comm ratios shift >10%
    def family_ratio(rows, path):
        comm, other = {}, {}
        for r in rows:
            key = (int(r["N"]), int(r["q"]))
            if r["candidate"] != "C0_DEFAULT" or not r["t_done_p50_us"]:
                continue
            if r["path"] == "COMM_ONLY":
                comm[key] = float(r["t_done_p50_us"])
            if r["path"] == path:
                other[key] = float(r["t_done_p50_us"])
        return {k: other[k] / comm[k] for k in comm.keys() & other.keys()}

    diffs = {}
    for path in ("FC_FCOLLECT_ONLY", "DC_PUSHSIG_ONLY"):
        k_r, n_r = family_ratio(matrix, path), family_ratio(nmat, path)
        shared = k_r.keys() & n_r.keys()
        if shared:
            diffs[short(path)] = statistics.median(
                [abs(n_r[k] - k_r[k]) / k_r[k] for k in shared])
    if diffs:
        obs = max(diffs.values())
        emit("P13", "fc/comm or dc/comm ratio shifts > 10% vs K500",
             100.0 * obs, verdict(obs > 0.10),
             f"per-family median shift: "
             f"{ {k: round(100 * v, 1) for k, v in diffs.items()} } %")
    else:
        emit("P13", "fc/comm or dc/comm ratio shifts > 10% vs K500", None,
             "PENDING")


if __name__ == "__main__":
    sys.exit(main())
