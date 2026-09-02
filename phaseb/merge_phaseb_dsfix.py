#!/usr/bin/env python3
"""Cross-root merged analysis: formal + dsfix.

The formal run (phaseb_formal_*) measured ds/d1 with the binary whose DS
serial gate was missing, so its ds cells are invalid and its d1 cells should
be re-paired with the fixed binary's ds. The dsfix run (phaseb_dsfix_*)
re-measures ds/d1/d1w with the fixed binary and adds the N4096/q16 config
quartet. This script merges BOTH roots at the case level:

  - case index   : dsfix preferred for {DS, D1, D1W} and for any (path,cell)
                   only dsfix has (N4096/q16 quartet); formal otherwise.
  - significance : same pair set as significance_phaseb.py plus the d1w and
                   serial-baseline pairs that only make sense post-fix
                   (d1w_vs_ds, d1w_vs_d0, ds_vs_d0, d1w_vs_dc stretch).
  - cell matrix  : merged with a `source` column.
  - control table: recomputed on the merged matrix, incl. d1w variants.

Usage:
  python3 merge_phaseb_dsfix.py --formal-root <phaseb_formal_*> \
                                --dsfix-root  <phaseb_dsfix_*>
Writes <formal-root>/summary/phaseb_{significance,cell_matrix,control}_merged.csv
(the formal summary dir is treated as the canonical analysis home).
Pure stdlib; imports mann_whitney / load_samples from significance_phaseb.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from significance_phaseb import load_samples, mann_whitney  # noqa: E402

# paths whose measurements must come from the fixed binary
FIXED_PATHS = {"DS_PUSHSIG_SERIAL", "D1_PUSHSIG_OVERLAP",
               "D1W_WAITSTREAM_OVERLAP"}

PAIRS = [
    # (label, path_a, path_b, metric, note)   candidate is C0_DEFAULT for all
    ("comm_C2_vs_C0", "COMM_ONLY:C2_RING_SIMPLE_CH8", "COMM_ONLY:C0_DEFAULT",
     "t_done_max_us", "resource axis, isolated"),
    ("r1_C2_vs_C0", "R1_EVENT_OVERLAP:C2_RING_SIMPLE_CH8",
     "R1_EVENT_OVERLAP:C0_DEFAULT", "e2e_max_us", "resource axis, e2e"),
    ("r1_vs_rs", "R1_EVENT_OVERLAP:C0_DEFAULT", "RS_SLICE_SERIAL:C0_DEFAULT",
     "e2e_max_us", "overlap gain, RCCL"),
    ("r1_vs_r0", "R1_EVENT_OVERLAP:C0_DEFAULT", "R0_FULL_SERIAL:C0_DEFAULT",
     "e2e_max_us", "overlap gain vs bulk, RCCL"),
    ("d1_vs_ds", "D1_PUSHSIG_OVERLAP:C0_DEFAULT", "DS_PUSHSIG_SERIAL:C0_DEFAULT",
     "e2e_max_us", "overlap gain vs TRUE serial (post-fix)"),
    ("d1w_vs_ds", "D1W_WAITSTREAM_OVERLAP:C0_DEFAULT",
     "DS_PUSHSIG_SERIAL:C0_DEFAULT", "e2e_max_us",
     "wait-stream overlap gain vs TRUE serial"),
    ("d1_vs_d0", "D1_PUSHSIG_OVERLAP:C0_DEFAULT", "D0_FCOLLECT_SERIAL:C0_DEFAULT",
     "e2e_max_us", "release semantics flip, DUSHMEM"),
    ("d1w_vs_d0", "D1W_WAITSTREAM_OVERLAP:C0_DEFAULT",
     "D0_FCOLLECT_SERIAL:C0_DEFAULT", "e2e_max_us",
     "wait-stream vs bulk fcollect"),
    ("d1w_vs_d1", "D1W_WAITSTREAM_OVERLAP:C0_DEFAULT",
     "D1_PUSHSIG_OVERLAP:C0_DEFAULT", "e2e_max_us",
     "wait-placement effect"),
    ("ds_vs_d0", "DS_PUSHSIG_SERIAL:C0_DEFAULT",
     "D0_FCOLLECT_SERIAL:C0_DEFAULT", "e2e_max_us",
     "true serial baselines, put-signal vs fcollect"),
    ("r1_vs_d1", "R1_EVENT_OVERLAP:C0_DEFAULT", "D1_PUSHSIG_OVERLAP:C0_DEFAULT",
     "e2e_max_us", "substrate gap under overlap"),
    ("r1_vs_d1w", "R1_EVENT_OVERLAP:C0_DEFAULT",
     "D1W_WAITSTREAM_OVERLAP:C0_DEFAULT", "e2e_max_us",
     "substrate gap under fixed overlap"),
    ("d1_stretch_vs_dc", "D1_PUSHSIG_OVERLAP:C0_DEFAULT",
     "DC_PUSHSIG_ONLY:C0_DEFAULT", "t_done_max_us",
     "sliced-release stretch (d1 done vs pure transport)"),
    ("d1w_stretch_vs_dc", "D1W_WAITSTREAM_OVERLAP:C0_DEFAULT",
     "DC_PUSHSIG_ONLY:C0_DEFAULT", "t_done_max_us",
     "sliced-release stretch post wait-placement fix"),
]


def index_root(root):
    """(path, candidate, N, q) -> {rep: case_dir} for one result root."""
    groups = defaultdict(dict)
    for case_dir in sorted(glob.glob(os.path.join(root, "cases", "case*"))):
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
    return groups


def merge_groups(formal, dsfix):
    """dsfix wins for FIXED_PATHS and for keys absent from formal."""
    merged = defaultdict(dict)
    provenance = {}
    for key, reps in formal.items():
        merged[key] = dict(reps)
        provenance[key] = "formal"
    for key, reps in dsfix.items():
        if key not in merged or key[0] in FIXED_PATHS:
            merged[key] = dict(reps)
            provenance[key] = "dsfix"
    return merged, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--dsfix-root", required=True)
    # dc reference curves: default the formal root (legacy behaviour). Pass the
    # d0dc root once available — its binary matches the dsfix batch, so the F6
    # dc-vs-d1/d1w comparison is same-binary instead of cross-root
    # (measured dc drift formal->d0dc: -0.7%..+4.8%, above machine noise).
    parser.add_argument("--dc-root", default=None,
                        help="root to source DC_PUSHSIG_ONLY curves from "
                             "(default: --formal-root)")
    args = parser.parse_args()
    out_dir = os.path.join(args.formal_root, "summary")

    formal = index_root(args.formal_root)
    dsfix = index_root(args.dsfix_root)
    merged, prov = merge_groups(formal, dsfix)
    n_dsfix_used = sum(1 for v in prov.values() if v == "dsfix")
    print(f"case index: formal={len(formal)} dsfix={len(dsfix)} "
          f"merged={len(merged)} (dsfix-sourced cells: {n_dsfix_used})")

    # ---- merged significance ----
    fields = ["N", "q", "pair", "axis_note", "metric", "source_a", "source_b",
              "median_a_us", "median_b_us", "delta_pct_pos_a_faster",
              "reps_a", "reps_b", "rep_direction_consistency",
              "mw_p_all_iters", "verdict"]
    rows = []
    cells = sorted({(n, q) for (_, _, n, q) in merged})
    for (n, q) in cells:
        for label, pa, pb, metric, note in PAIRS:
            path_a, cand_a = pa.split(":")
            path_b, cand_b = pb.split(":")
            ga = merged.get((path_a, cand_a, n, q))
            gb = merged.get((path_b, cand_b, n, q))
            if not ga or not gb:
                continue
            samples_a, samples_b, rep_a, rep_b = [], [], [], []
            for rep in sorted(ga):
                vals = load_samples(ga[rep], metric)
                if vals:
                    samples_a.extend(vals)
                    rep_a.append(statistics.median(vals))
            for rep in sorted(gb):
                vals = load_samples(gb[rep], metric)
                if vals:
                    samples_b.extend(vals)
                    rep_b.append(statistics.median(vals))
            if not samples_a or not samples_b:
                continue
            med_a = statistics.median(samples_a)
            med_b = statistics.median(samples_b)
            delta = 100.0 * (med_b - med_a) / med_b
            n_pairs = min(len(rep_a), len(rep_b))
            consistent = 0
            for ma, mb in zip(rep_a, rep_b):
                if (med_a <= med_b and ma <= mb) or (med_a >= med_b and ma >= mb):
                    consistent += 1
            _, p = mann_whitney(samples_a, samples_b)
            verdict = "SIG" if p < 1e-4 else ("WEAK" if p < 0.05 else "NS")
            rows.append({
                "N": n, "q": q, "pair": label, "axis_note": note,
                "metric": metric,
                "source_a": prov.get((path_a, cand_a, n, q), "?"),
                "source_b": prov.get((path_b, cand_b, n, q), "?"),
                "median_a_us": f"{med_a:.1f}", "median_b_us": f"{med_b:.1f}",
                "delta_pct_pos_a_faster": f"{delta:.3f}",
                "reps_a": len(rep_a), "reps_b": len(rep_b),
                "rep_direction_consistency": f"{consistent}/{n_pairs}",
                "mw_p_all_iters": (f"{p:.2e}" if p == p else "nan"),
                "verdict": verdict,
            })
    sig_path = os.path.join(out_dir, "phaseb_significance_merged.csv")
    with open(sig_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged significance -> {sig_path} ({len(rows)} rows)")

    # ---- merged cell matrix ----
    matrix = {}  # (cand, N, q, path) -> row dict
    for root in (args.formal_root, args.dsfix_root):
        path = os.path.join(root, "summary", "phaseb_cell_matrix.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["candidate"], int(row["N"]), int(row["q"]),
                       row["path"])
                src = "dsfix" if root == args.dsfix_root else "formal"
                if key not in matrix or (src == "dsfix" and
                                         key[3] in FIXED_PATHS):
                    row = dict(row)
                    row["source"] = src
                    matrix[key] = row
    if matrix:
        sample_row = next(iter(matrix.values()))
        cols = [c for c in sample_row if c != "source"] + ["source"]
        mat_path = os.path.join(out_dir, "phaseb_cell_matrix_merged.csv")
        with open(mat_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for key in sorted(matrix):
                writer.writerow(matrix[key])
        print(f"merged cell matrix -> {mat_path} ({len(matrix)} rows)")

    # ---- merged control table (p50-based, D-family focus) ----
    def cell(cand, n, q, path, field):
        row = matrix.get((cand, n, q, path))
        if row and row.get(field):
            try:
                return float(row[field])
            except ValueError:
                return None
        return None

    def gain(a, b):  # + : a faster
        return None if a is None or b is None or not b else 100.0 * (b - a) / b

    ctrl_rows = []
    metrics = [
        ("r1_vs_rs_gain_pct", ("C0_DEFAULT", "R1_EVENT_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "RS_SLICE_SERIAL", "e2e_p50_us")),
        ("r1_vs_r0_gain_pct", ("C0_DEFAULT", "R1_EVENT_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "R0_FULL_SERIAL", "e2e_p50_us")),
        ("d1_vs_ds_gain_pct", ("C0_DEFAULT", "D1_PUSHSIG_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "DS_PUSHSIG_SERIAL", "e2e_p50_us")),
        ("d1w_vs_ds_gain_pct", ("C0_DEFAULT", "D1W_WAITSTREAM_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "DS_PUSHSIG_SERIAL", "e2e_p50_us")),
        ("d1_vs_d0_gain_pct", ("C0_DEFAULT", "D1_PUSHSIG_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "D0_FCOLLECT_SERIAL", "e2e_p50_us")),
        ("d1w_vs_d0_gain_pct", ("C0_DEFAULT", "D1W_WAITSTREAM_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "D0_FCOLLECT_SERIAL", "e2e_p50_us")),
        ("d1w_vs_d1_gain_pct", ("C0_DEFAULT", "D1W_WAITSTREAM_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "D1_PUSHSIG_OVERLAP", "e2e_p50_us")),
        ("r1_vs_d1w_gain_pct", ("C0_DEFAULT", "R1_EVENT_OVERLAP", "e2e_p50_us"),
         ("C0_DEFAULT", "D1W_WAITSTREAM_OVERLAP", "e2e_p50_us")),
        ("d1_vs_dc_done_stretch_pct",
         ("C0_DEFAULT", "D1_PUSHSIG_OVERLAP", "t_done_p50_us"),
         ("C0_DEFAULT", "DC_PUSHSIG_ONLY", "t_done_p50_us")),
        ("d1w_vs_dc_done_stretch_pct",
         ("C0_DEFAULT", "D1W_WAITSTREAM_OVERLAP", "t_done_p50_us"),
         ("C0_DEFAULT", "DC_PUSHSIG_ONLY", "t_done_p50_us")),
    ]
    for (n, q) in cells:
        row = {"candidate": "C0_DEFAULT", "N": n, "q": q}
        for name, (ca, pa, fa), (cb, pb, fb) in metrics:
            va = cell(ca, n, q, pa, fa)
            vb = cell(cb, n, q, pb, fb)
            g = gain(va, vb)
            row[name] = f"{g:.3f}" if g is not None else ""
        ctrl_rows.append(row)
    ctrl_cols = ["candidate", "N", "q"] + [m[0] for m in metrics]
    ctrl_path = os.path.join(out_dir, "phaseb_control_merged.csv")
    with open(ctrl_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ctrl_cols)
        writer.writeheader()
        writer.writerows(ctrl_rows)
    print(f"merged control table -> {ctrl_path} ({len(ctrl_rows)} rows)")

    # ---- release curves: dc root (default formal; pass d0dc for same-binary)
    # contributes dc, dsfix contributes ds/d1/d1w — F6 needs them side by side
    # for the wait-placement comparison ----
    dc_root = args.dc_root or args.formal_root
    rel_out = os.path.join(out_dir, "phaseb_release_curves_long_merged.csv")
    rel_rows, seen = [], set()
    for src_root, paths in ((args.dsfix_root, ("DS_PUSHSIG_SERIAL",
                                               "D1_PUSHSIG_OVERLAP",
                                               "D1W_WAITSTREAM_OVERLAP")),
                            (dc_root, ("DC_PUSHSIG_ONLY",))):
        src = os.path.join(src_root, "summary",
                           "phaseb_release_curves_long.csv")
        if not os.path.exists(src):
            continue
        with open(src, newline="") as handle:
            for r in csv.DictReader(handle):
                if r.get("path") not in paths:
                    continue
                key = (r.get("path"), r.get("candidate"), r.get("N"),
                       r.get("q"), r.get("rep"), r.get("slice_index"))
                if key in seen:
                    continue
                seen.add(key)
                rel_rows.append(r)
    if rel_rows:
        with open(rel_out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rel_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rel_rows)
        print(f"merged release curves -> {rel_out} ({len(rel_rows)} rows; "
              f"dc from {os.path.basename(dc_root)}, ds/d1/d1w from dsfix)")

    # ---- console digest: the claim-bearing pairs post-fix ----
    focus = ("d1_vs_ds", "d1w_vs_ds", "d1_vs_d0", "d1w_vs_d0", "d1w_vs_d1",
             "d1w_stretch_vs_dc")
    print(f"\n{'N':>5} {'q':>3} {'pair':>18} {'delta%':>9} {'cons':>6} "
          f"{'p':>10} {'verdict':>8}")
    for row in rows:
        if row["pair"] in focus:
            print(f"{row['N']:>5} {row['q']:>3} {row['pair']:>18} "
                  f"{row['delta_pct_pos_a_faster']:>9} "
                  f"{row['rep_direction_consistency']:>6} "
                  f"{row['mw_p_all_iters']:>10} {row['verdict']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
