#!/usr/bin/env python3
"""Phase B significance tests: are the reversal / overlap-gain claims real?

Pure-stdlib (no scipy) Mann-Whitney U with tie-corrected normal approximation,
run on per-iteration samples pooled across reps, PLUS a rep-level direction
consistency check (all-iters p-values overstate confidence when iterations
within a process are autocorrelated; the rep-consistency criterion is the
conservative evidence).

Pairs tested per (candidate=C0, N, q) cell:
  isolated  : comm(C2) vs comm(C0)          on t_done   (resource axis, isolated)
  e2e       : r1(C2)   vs r1(C0)            on e2e      (resource axis, e2e -> reversal core)
  structure : r1 vs rs / r1 vs r0           on e2e      (overlap gain, RCCL)
  structure : d1 vs d0                      on e2e      (release semantics, DUSHMEM)
  structure : d1 vs ds                      on e2e      (FLAGGED: ds serial gate was missing
                                                       in phaseb_formal_20260902_160115; treat
                                                       as noise-characterisation only)
  cross     : r1 vs d1                      on e2e      (substrate gap under overlap)

Verdicts (two-sided p):  SIG p<1e-4 | WEAK p<0.05 | NS otherwise.
"""

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


def mann_whitney(x, y):
    """Return (U, two-sided p) via normal approximation with tie correction."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan"), float("nan")
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y], key=lambda t: t[0])
    n = len(pooled)
    ranks = [0.0] * n
    tie_sum = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        tie_sum += t ** 3 - t
        i = j + 1
    r_x = sum(r for r, tag in zip(ranks, (tag for _, tag in pooled)) if tag == 0)
    u = r_x - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    var = nx * ny / 12.0 * ((n + 1) - tie_sum / (n * (n - 1.0))) if n > 1 else 0.0
    if var <= 0:
        return u, 1.0
    z = (u - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return u, max(p, 0.0)


def load_samples(case_dir, metric):
    """Return list of per-iteration metric values, or []."""
    path = os.path.join(case_dir, "raw_global_samples.csv")
    if not os.path.exists(path):
        return []
    values = []
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                raw = row.get(metric)
                if raw not in (None, ""):
                    values.append(float(raw))
    except (OSError, csv.Error):
        return []
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    cases_root = os.path.join(args.result_root, "cases")
    summary_root = os.path.join(args.result_root, "summary")
    os.makedirs(summary_root, exist_ok=True)

    # (path, candidate, N, q) -> {rep: [samples]}
    groups = defaultdict(dict)
    for case_dir in sorted(glob.glob(os.path.join(cases_root, "case*"))):
        if not os.path.isdir(case_dir):
            continue
        manifest_path = os.path.join(case_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, newline="") as handle:
                manifest = next(csv.DictReader(handle))
        except (StopIteration, OSError, csv.Error):
            continue
        case_id = os.path.basename(case_dir)
        rep = case_id.rsplit("_rep", 1)[1] if "_rep" in case_id else "1"
        key = (manifest.get("path"), manifest.get("candidate"),
               int(manifest.get("N", 0)), int(manifest.get("q", 0)))
        groups[key][rep] = case_dir

    # pairs: (label, path_a, cand_a, path_b, cand_b, metric, note)
    def pairs_for(n, q):
        return [
            ("comm_C2_vs_C0", "COMM_ONLY", "C2_RING_SIMPLE_CH8",
             "COMM_ONLY", "C0_DEFAULT", "t_done_max_us", "resource axis, isolated"),
            ("r1_C2_vs_C0", "R1_EVENT_OVERLAP", "C2_RING_SIMPLE_CH8",
             "R1_EVENT_OVERLAP", "C0_DEFAULT", "e2e_max_us",
             "resource axis, e2e (reversal core)"),
            ("r1_vs_rs", "R1_EVENT_OVERLAP", "C0_DEFAULT",
             "RS_SLICE_SERIAL", "C0_DEFAULT", "e2e_max_us", "overlap gain, RCCL"),
            ("r1_vs_r0", "R1_EVENT_OVERLAP", "C0_DEFAULT",
             "R0_FULL_SERIAL", "C0_DEFAULT", "e2e_max_us", "overlap gain vs bulk, RCCL"),
            ("d1_vs_d0", "D1_PUSHSIG_OVERLAP", "C0_DEFAULT",
             "D0_FCOLLECT_SERIAL", "C0_DEFAULT", "e2e_max_us",
             "release semantics flip, DUSHMEM"),
            ("d1_vs_ds", "D1_PUSHSIG_OVERLAP", "C0_DEFAULT",
             "DS_PUSHSIG_SERIAL", "C0_DEFAULT", "e2e_max_us",
             "INVALID if ds serial gate missing (bug): noise characterisation"),
            ("d1w_vs_d1", "D1W_WAITSTREAM_OVERLAP", "C0_DEFAULT",
             "D1_PUSHSIG_OVERLAP", "C0_DEFAULT", "e2e_max_us",
             "wait-placement effect (dedicated wait stream)"),
            ("r1_vs_d1", "R1_EVENT_OVERLAP", "C0_DEFAULT",
             "D1_PUSHSIG_OVERLAP", "C0_DEFAULT", "e2e_max_us",
             "substrate gap under overlap"),
        ]

    out_path = os.path.join(summary_root, "phaseb_significance.csv")
    fields = ["N", "q", "pair", "axis_note", "metric",
              "median_a_us", "median_b_us", "delta_pct_pos_a_faster",
              "reps_a", "reps_b", "rep_direction_consistency",
              "mw_p_all_iters", "verdict"]
    rows = []
    cells = sorted({(n, q) for (_, _, n, q) in groups})
    for (n, q) in cells:
        for label, pa, ca, pb, cb, metric, note in pairs_for(n, q):
            ga = groups.get((pa, ca, n, q))
            gb = groups.get((pb, cb, n, q))
            if not ga or not gb:
                continue
            samples_a, samples_b, rep_meds_a, rep_meds_b = [], [], [], []
            for rep in sorted(ga):
                vals = load_samples(ga[rep], metric)
                if vals:
                    samples_a.extend(vals)
                    rep_meds_a.append(statistics.median(vals))
            for rep in sorted(gb):
                vals = load_samples(gb[rep], metric)
                if vals:
                    samples_b.extend(vals)
                    rep_meds_b.append(statistics.median(vals))
            if not samples_a or not samples_b:
                continue
            med_a = statistics.median(samples_a)
            med_b = statistics.median(samples_b)
            delta = 100.0 * (med_b - med_a) / med_b  # + : a faster
            n_pairs = min(len(rep_meds_a), len(rep_meds_b))
            consistent = 0
            for ma, mb in zip(rep_meds_a, rep_meds_b):
                if (med_a <= med_b and ma <= mb) or (med_a >= med_b and ma >= mb):
                    consistent += 1
            _, p = mann_whitney(samples_a, samples_b)
            verdict = "SIG" if p < 1e-4 else ("WEAK" if p < 0.05 else "NS")
            rows.append({
                "N": n, "q": q, "pair": label, "axis_note": note, "metric": metric,
                "median_a_us": f"{med_a:.1f}", "median_b_us": f"{med_b:.1f}",
                "delta_pct_pos_a_faster": f"{delta:.3f}",
                "reps_a": len(rep_meds_a), "reps_b": len(rep_meds_b),
                "rep_direction_consistency": f"{consistent}/{n_pairs}",
                "mw_p_all_iters": (f"{p:.2e}" if p == p else "nan"),
                "verdict": verdict,
            })
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"significance table -> {out_path} ({len(rows)} rows)")
    # console digest: only the claim-bearing pairs
    focus = ("r1_C2_vs_C0", "comm_C2_vs_C0", "d1_vs_d0", "r1_vs_rs")
    print(f"{'N':>5} {'q':>3} {'pair':<14} {'delta%':>9} {'reps':>7} {'p':>10} {'verdict':>8}")
    for row in rows:
        if row["pair"] in focus:
            print(f"{row['N']:>5} {row['q']:>3} {row['pair']:<14} "
                  f"{row['delta_pct_pos_a_faster']:>9} "
                  f"{row['rep_direction_consistency']:>7} "
                  f"{row['mw_p_all_iters']:>10} {row['verdict']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
