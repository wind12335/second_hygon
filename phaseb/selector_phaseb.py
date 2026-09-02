#!/usr/bin/env python3
"""Release-aware selector prototype + leave-one-cell-out evaluation (F5).

Reads phaseb_cell_matrix.csv and evaluates two selection problems:

Track A (strategy): which e2e strategy from {r0, rs, r1, d0, d1, d1w} wins?
  P0 majority   : the globally most frequent winner (sanity floor)
  P1 nearest    : winner of the nearest labelled cell in (log q, comm/gemm
                  ratio) space — the "operating point" heuristic
  P2 feature    : naive transport-domination rule (r >= 2 -> bulk) kept as
                  the bandwidth-intuition baseline; on K500 it loses ~14%
                  mean regret — strategy-axis selection is degenerate here
                  (r1 dominates), which is itself a finding

Track B (config): within strategy r1, which RCCL config — C0 (default) or
C2 (Ring/Simple/ch8) — wins e2e? This is where isolated-vs-e2e reversal
bites: C2 is the isolated comm winner in every cell, but NOT the e2e winner
in every cell (e.g. N2048/q8: C0 wins by ~5.5%). Predictors:
  B0 isolated   : pick the config whose isolated comm t_done is smaller —
                  the bandwidth-naive baseline the paper argues against
  B1 majority   : most frequent e2e winner across cells
  B2 feature    : rule on (q, isolated gap %) — the release-aware fix

Reports top-1 accuracy and mean regret (e2e of the predicted strategy vs the
true winner, in % of the winner's e2e). Pure stdlib.

The comm/gemm ratio uses t_done of COMM_ONLY(C0) over the gemm interval of
GEMM_ONLY from the SAME cell — i.e. features observable by an isolated
microbenchmark, never by running the full e2e matrix.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

E2E_POOL = ("R0_FULL_SERIAL", "RS_SLICE_SERIAL", "R1_EVENT_OVERLAP",
            "D0_FCOLLECT_SERIAL", "DS_PUSHSIG_SERIAL", "D1_PUSHSIG_OVERLAP",
            "D1W_WAITSTREAM_OVERLAP")
# strategies the selector may actually recommend (exclude known-invalid ds
# from the formal run once dsfix replaces it; keep configurable via --pool)
DEFAULT_POOL = ("r0", "rs", "r1", "d0", "d1", "d1w")


def short(path):
    return {"R0_FULL_SERIAL": "r0", "RS_SLICE_SERIAL": "rs",
            "R1_EVENT_OVERLAP": "r1", "D0_FCOLLECT_SERIAL": "d0",
            "DS_PUSHSIG_SERIAL": "ds", "D1_PUSHSIG_OVERLAP": "d1",
            "D1W_WAITSTREAM_OVERLAP": "d1w",
            "COMM_ONLY": "comm", "GEMM_ONLY": "gemm"}[path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True,
                        help="directory containing phaseb_cell_matrix.csv")
    parser.add_argument("--pool", default=",".join(DEFAULT_POOL),
                        help="comma-separated recommendable strategies")
    args = parser.parse_args()
    pool = set(args.pool.split(","))

    matrix_path = os.path.join(args.summary_dir, "phaseb_cell_matrix.csv")
    cells = defaultdict(dict)  # (N,q) -> {strategy: e2e_p50}
    config_cells = defaultdict(dict)  # (N,q) -> {"c0"/"c2": e2e, "@iso_c0"/"@iso_c2": t_done}
    with open(matrix_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["N"]), int(row["q"]))
            if row["candidate"] == "C0_DEFAULT":
                if row["path"] in E2E_POOL:
                    value = row["e2e_p50_us"]
                    if value:
                        cells[key][short(row["path"])] = float(value)
                # features
                if row["path"] == "COMM_ONLY":
                    cells[key]["@comm_done"] = float(row["t_done_p50_us"] or "nan")
                    config_cells[key]["@iso_c0"] = float(row["t_done_p50_us"] or "nan")
                if row["path"] == "GEMM_ONLY":
                    cells[key]["@gemm_int"] = float(row["t_done_p50_us"] or "nan")
                    config_cells[key]["@gemm"] = float(row["t_done_p50_us"] or "nan")
                if row["path"] == "R1_EVENT_OVERLAP" and row["e2e_p50_us"]:
                    config_cells[key]["c0"] = float(row["e2e_p50_us"])
            elif row["candidate"] == "C2_RING_SIMPLE_CH8":
                if row["path"] == "R1_EVENT_OVERLAP" and row["e2e_p50_us"]:
                    config_cells[key]["c2"] = float(row["e2e_p50_us"])
                if row["path"] == "COMM_ONLY":
                    config_cells[key]["@iso_c2"] = float(row["t_done_p50_us"] or "nan")

    labelled = []
    for key, strategies in sorted(cells.items()):
        available = {s: v for s, v in strategies.items()
                     if s in pool and not s.startswith("@")}
        if available:
            labelled.append((key, available, strategies))

    if len(labelled) < 3:
        print("not enough labelled cells yet")
        return 0

    def feature_ratio(strategies):
        comm = strategies.get("@comm_done", float("nan"))
        gemm = strategies.get("@gemm_int", float("nan"))
        return comm / gemm if gemm and gemm == gemm and comm == comm else float("nan")

    def predict_majority(train, key=None):
        counts = defaultdict(int)
        for _, avail, _ in train:
            counts[min(avail, key=avail.get)] += 1
        return max(counts, key=counts.get)

    def predict_nearest(train, key):
        n0, q0 = key
        best, best_dist = None, float("inf")
        for (n, q), avail, _ in train:
            dist = (math.log(q) - math.log(q0)) ** 2 + \
                   0.1 * (math.log(n) - math.log(n0)) ** 2
            if dist < best_dist:
                best_dist, best = dist, min(avail, key=avail.get)
        return best or predict_majority(train, key)

    def predict_feature(train, key):
        """Naive transport-domination heuristic — deliberately adversarial.

        Observables: r = comm/gemm ratio (r>1 comm-dominated), q (slices).
        Rule: r >= 2.0 -> bulk (r0); r <= 0.5 -> r1 if q >= 4 else r0;
        else r1 if q >= 8 else r0.

        On the K500 substrate this rule LOSES badly (mean regret ~14%) even
        though r1 dominates every cell: it is the bandwidth-intuition
        baseline, kept to quantify the cost of applying isolated-thinking to
        the strategy axis. The discriminative release-aware selection on this
        substrate is Track B (config axis) — see below.
        """
        n, q = key
        # own features are observables: using own cell's isolated
        # microbenchmarks is allowed — only e2e labels are hidden in LOO
        own = next(((k, a, s) for k, a, s in labelled if k == key), None)
        if own is None:
            return predict_majority(train, key)
        r = feature_ratio(own[2])
        if r != r:
            return predict_majority(train, key)
        if r >= 2.0:
            return "r0"
        if r <= 0.5:
            return "r1" if q >= 4 else "r0"
        return "r1" if q >= 8 else "r0"

    predictors = [("P0_majority", predict_majority),
                  ("P1_nearest", lambda train, key: predict_nearest(train, key)),
                  ("P2_feature", predict_feature)]

    print(f"labelled cells: {len(labelled)}  pool: {sorted(pool)}")
    for name, predict in predictors:
        hits, regrets = 0, []
        for key, avail, _ in labelled:
            train = [row for row in labelled if row[0] != key]
            prediction = predict(train, key)
            winner = min(avail, key=avail.get)
            if prediction in avail:
                regret = 100.0 * (avail[prediction] - avail[winner]) / avail[winner]
                regrets.append(max(regret, 0.0))
                if prediction == winner:
                    hits += 1
            else:
                regrets.append(float("nan"))
        valid = [x for x in regrets if x == x]
        mean_regret = sum(valid) / len(valid) if valid else float("nan")
        print(f"{name:<12} top1={hits}/{len(labelled)} "
              f"({100.0 * hits / len(labelled):.0f}%)  "
              f"mean_regret={mean_regret:.1f}%  worst={max(valid) if valid else float('nan'):.1f}%")

    # ---- Track B: RCCL config selection (C0 default vs C2 Ring/Simple/ch8) ----
    cfg_labelled = []
    for key, d in sorted(config_cells.items()):
        if "c0" in d and "c2" in d:
            cfg_labelled.append((key, d))
    print()
    if len(cfg_labelled) < 3:
        print("track B (config): not enough cells with both r1_C0 and r1_C2 yet")
        return 0

    def iso_gap(d):
        """Isolated advantage of C2 over C0 in % (+ : C2 faster)."""
        c0, c2 = d.get("@iso_c0"), d.get("@iso_c2")
        if not c0 or not c2 or c0 != c0 or c2 != c2:
            return float("nan")
        return 100.0 * (c0 - c2) / c0

    def b0_isolated(train, key):
        """Bandwidth-naive: pick the isolated comm winner (always C2 here)."""
        own = next((d for k, d in cfg_labelled if k == key), None)
        if own is None:
            return "c2"
        gap = iso_gap(own)
        if gap != gap:
            return "c2"
        return "c2" if gap > 0 else "c0"

    def b1_majority(train, key):
        counts = defaultdict(int)
        for _, d in train:
            counts["c2" if d["c2"] < d["c0"] else "c0"] += 1
        return max(counts, key=counts.get)

    def b2_feature(train, key):
        """Release-aware config rule (q + iso-gap sensors).

        Observables: q and the isolated comm gap (both from microbenchmarks).
        The isolated advantage of C2 decays as slicing deepens (channel setup
        cost stops amortising); once the gap is thin AND slices are many, the
        e2e picture flips to C0. Superseded by b3 on the full matrix: gap is
        N-blind (1.64/1.65/1.72 at q8 across N) and mis-fires on N4096/q8.
        """
        n, q = key
        own = next((d for k, d in cfg_labelled if k == key), None)
        if own is None:
            return b1_majority(train, key)
        gap = iso_gap(own)
        if gap != gap:
            return b1_majority(train, key)
        return "c0" if (q >= 8 and gap <= 2.0) else "c2"

    def balance_ratio(d):
        """Isolated comm/gemm ratio — the N-sensor (workload balance)."""
        comm, gemm = d.get("@iso_c0"), d.get("@gemm")
        if not comm or not gemm or comm != comm or gemm != gemm or not gemm:
            return float("nan")
        return comm / gemm

    def b3_balance(train, key):
        """Balance-band rule (the full-matrix finding, design doc §13).

        The reversal is a resonance of the true-overlap regime: it needs BOTH
        resources saturated (ratio ~ 1) AND deep slicing (q >= 8) AND a thin
        isolated gap. Comm-dominated (ratio >> 1) propagates C2's bandwidth
        edge to e2e; compute-dominated (ratio << 1) shadows the config choice
        entirely. All three observables come from isolated microbenchmarks.
        Mechanism-derived thresholds; with a single observed reversal cell
        this is prior + calibration, NOT LOO-learnable — see refit script.
        """
        n, q = key
        own = next((d for k, d in cfg_labelled if k == key), None)
        if own is None:
            return b1_majority(train, key)
        gap, ratio = iso_gap(own), balance_ratio(own)
        if gap != gap or ratio != ratio:
            return b1_majority(train, key)
        return "c0" if (q >= 8 and 0.9 <= ratio <= 1.35 and gap <= 2.0) else "c2"

    print(f"track B (config c0 vs c2, strategy r1): {len(cfg_labelled)} cells")
    for name, predict in [("B0_isolated", b0_isolated),
                          ("B1_majority", b1_majority),
                          ("B2_feature", b2_feature),
                          ("B3_balance", b3_balance)]:
        hits, regrets = 0, []
        for key, d in cfg_labelled:
            train = [row for row in cfg_labelled if row[0] != key]
            prediction = predict(train, key)
            winner = "c2" if d["c2"] < d["c0"] else "c0"
            if prediction in d:
                regret = 100.0 * (d[prediction] - d[winner]) / d[winner]
                regrets.append(max(regret, 0.0))
                if prediction == winner:
                    hits += 1
        valid = [x for x in regrets if x == x]
        mean_regret = sum(valid) / len(valid) if valid else float("nan")
        print(f"{name:<12} top1={hits}/{len(cfg_labelled)} "
              f"({100.0 * hits / len(cfg_labelled):.0f}%)  "
              f"mean_regret={mean_regret:.2f}%  worst={max(valid) if valid else float('nan'):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
