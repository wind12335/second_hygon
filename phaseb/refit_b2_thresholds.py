#!/usr/bin/env python3
"""Nested leave-one-cell-out refit of the B2 config-selector thresholds.

Track B of selector_phaseb.py predicts, within strategy r1, whether RCCL
config C0 (default) or C2 (Ring/Simple/ch8) wins e2e, from two observables
available to an isolated microbenchmark: the slice count q and the isolated
comm gap  gap = 100*(t_c0 - t_c2)/t_c0  (+ : C2 faster). The hand-picked
rule was `c0 if (q >= 8 and gap <= 2.0) else c2`; this script refits both
thresholds honestly: for each held-out cell, thresholds are chosen by
optimising on the REMAINING cells only, then used to predict the held-out
one. Reports the nested-LOO regret of (a) the best worst-regret rule,
(b) the best mean-regret rule, (c) the simplest rule within 0.3% of the
best worst-regret (largest q-threshold and loosest gap-threshold, i.e.
closest to the always-C2 default), and (d) the original hand-picked rule.

Usage: python3 refit_b2_thresholds.py --summary-dir <root>/summary
Input : phaseb_cell_matrix.csv.  Pure stdlib.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

Q_GRID = [2, 4, 8, 16]                 # candidate q thresholds
GAP_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, float("inf")]
INF = float("inf")


def load_cells(summary_dir):
    """(N,q) -> {"c0"/"c2": e2e_p50, "@gap": isolated gap %}."""
    cells = defaultdict(dict)
    path = os.path.join(summary_dir, "phaseb_cell_matrix.csv")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["N"]), int(row["q"]))
            if row["candidate"] == "C0_DEFAULT":
                if row["path"] == "R1_EVENT_OVERLAP" and row["e2e_p50_us"]:
                    cells[key]["c0"] = float(row["e2e_p50_us"])
                if row["path"] == "COMM_ONLY" and row["t_done_p50_us"]:
                    cells[key]["@iso_c0"] = float(row["t_done_p50_us"])
            elif row["candidate"] == "C2_RING_SIMPLE_CH8":
                if row["path"] == "R1_EVENT_OVERLAP" and row["e2e_p50_us"]:
                    cells[key]["c2"] = float(row["e2e_p50_us"])
                if row["path"] == "COMM_ONLY" and row["t_done_p50_us"]:
                    cells[key]["@iso_c2"] = float(row["t_done_p50_us"])
    out = []
    for key, d in sorted(cells.items()):
        if "c0" in d and "c2" in d and "@iso_c0" in d and "@iso_c2" in d:
            c0, c2 = d["@iso_c0"], d["@iso_c2"]
            if c0 and c2:
                out.append((key, d["c0"], d["c2"],
                            100.0 * (c0 - c2) / c0))
    return out


def predict(q, gap, q_thr, gap_thr):
    return "c0" if (q >= q_thr and gap <= gap_thr) else "c2"


def evaluate(cells, q_thr, gap_thr):
    """top1 / mean / worst regret (%) of a rule on the given cells."""
    hits, regrets = 0, []
    for (n, q), c0, c2, gap in cells:
        winner = "c0" if c0 < c2 else "c2"
        pick = predict(q, gap, q_thr, gap_thr)
        best = min(c0, c2)
        regret = 100.0 * ((c0 if pick == "c0" else c2) - best) / best
        regrets.append(max(regret, 0.0))
        hits += pick == winner
    return hits, sum(regrets) / len(regrets), max(regrets)


def fmt_thr(v):
    return "inf" if v == INF else f"{v:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()
    cells = load_cells(args.summary_dir)
    if len(cells) < 4:
        print(f"only {len(cells)} labelled cells — too few to refit")
        return 0

    print(f"labelled cells: {len(cells)}")
    for (n, q), c0, c2, gap in cells:
        winner = "c0" if c0 < c2 else "c2"
        print(f"  N{n:<5} q{q:<3} iso_gap={gap:+.2f}%   "
              f"e2e c0={c0:.0f}  c2={c2:.0f}  -> winner {winner}")
    print()

    # ---- grid of all candidate rules, full-sample fit (descriptive) ----
    all_rules = []
    for q_thr in Q_GRID:
        for gap_thr in GAP_GRID:
            hits, mean_r, worst_r = evaluate(cells, q_thr, gap_thr)
            all_rules.append((q_thr, gap_thr, hits, mean_r, worst_r))
    by_worst = sorted(all_rules, key=lambda r: (r[4], r[3]))
    print("full-sample top-5 by worst regret "
          "(descriptive only — nested LOO below is the honest number):")
    for q_thr, gap_thr, hits, mean_r, worst_r in by_worst[:5]:
        print(f"  q>={fmt_thr(q_thr):>4} gap<={fmt_thr(gap_thr):>4}  "
              f"top1={hits}/{len(cells)}  mean={mean_r:.2f}%  "
              f"worst={worst_r:.2f}%")

    # ---- nested LOO: thresholds re-fit on train cells per fold ----
    picks, regrets = [], []
    for i in range(len(cells)):
        train = [c for j, c in enumerate(cells) if j != i]
        held = cells[i]
        best_rule, best_key = None, None
        for q_thr in Q_GRID:
            for gap_thr in GAP_GRID:
                _, mean_r, worst_r = evaluate(train, q_thr, gap_thr)
                key = (round(worst_r, 6), round(mean_r, 6),
                       -q_thr if q_thr != INF else 1,  # prefer larger q_thr
                       gap_thr)                        # prefer looser gap
                if best_key is None or key < best_key:
                    best_key, best_rule = key, (q_thr, gap_thr)
        (n, q), c0, c2, gap = held
        winner = "c0" if c0 < c2 else "c2"
        q_thr, gap_thr = best_rule
        pick = predict(q, gap, q_thr, gap_thr)
        best = min(c0, c2)
        regret = max(100.0 * ((c0 if pick == "c0" else c2) - best) / best, 0.0)
        picks.append(pick == winner)
        regrets.append(regret)
        print(f"LOO N{n}/q{q}: fitted q>={fmt_thr(q_thr)} "
              f"gap<={fmt_thr(gap_thr)} -> picked {pick} "
              f"(winner {winner}, regret {regret:.2f}%)")
    print(f"\nnested LOO: top1={sum(picks)}/{len(cells)}  "
          f"mean={sum(regrets) / len(regrets):.2f}%  "
          f"worst={max(regrets):.2f}%")

    # ---- reference points on the full sample ----
    hits, mean_r, worst_r = evaluate(cells, 8, 2.0)
    print(f"hand-picked (q>=8, gap<=2) full-sample: top1={hits}/{len(cells)}  "
          f"mean={mean_r:.2f}%  worst={worst_r:.2f}%")
    hits, mean_r, worst_r = evaluate(cells, INF, INF)  # never fires -> always c2
    print(f"always-C2 baseline           : top1={hits}/{len(cells)}  "
          f"mean={mean_r:.2f}%  worst={worst_r:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
