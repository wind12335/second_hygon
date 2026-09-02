#!/usr/bin/env python3
"""Phase B paper figures — numbering aligned with 论文骨架与创新点主张.md.

Reads the summary CSVs produced by analyze_phaseb.py + significance_phaseb.py
+ extract_release_curves.py and renders:

  fig_F1_money.png        : per-cell paired bars — isolated comm gap (C2 vs C0,
                            always >0) against e2e delta (flips negative) —
                            the money figure.
  fig_F2a_winner_map.png  : (N,q) plane, config-axis e2e winner per cell
                            (RCCL_CONFIG_REVERSAL cells highlighted).
  fig_F2b_decay.png       : C2-vs-C0 delta vs q, one line per N, isolated as
                            dashed reference — the decay/crossing view.
  fig_F3_stretch.png      : d1-vs-dc done stretch vs q per N — the price of
                            sliced release semantics.
  fig_F4_crossover.png    : overlap-gain crossovers vs q per N (r1 vs rs/r0,
                            d1 vs d0, r1 vs d1) from the control table.
  fig_F5b_capability.png  : isolated comm/fc/dc t_done bars per (N,q) cell —
                            substrate capability vector B.
  fig_F6_release.png      : per-slice release absolute times, dc vs d1 (vs d1w
                            when present) — the wait-placement mechanism.

Usage: python3 plot_phaseb_figures.py --result-root <root>
Output: <root>/summary/figures/*.png (+ .pdf). Pure matplotlib, Agg backend.
NOTE: d1/d1w rows come from the dsfix batch once merged; the F6 panel picks
them up automatically when their curves exist in the release-curves CSV.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

COLORS = {"512": "#1f77b4", "2048": "#ff7f0e", "4096": "#d62728"}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def fig_f1_money(summary_dir, out_dir):
    """Money figure: per-cell isolated gap vs e2e delta, paired bars."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_boundary_xcand.csv"))
    cells = []
    for r in rows:
        try:
            iso = float(r["comm_C2_vs_C0_iso_pct"])
            e2e = float(r["r1_C2_vs_C0_e2e_pct"])
        except (KeyError, ValueError):
            continue
        cells.append((int(r["N"]), int(r["q"]), iso, e2e,
                      r.get("rccl_config_flag") == "RCCL_CONFIG_REVERSAL"))
    if not cells:
        return False
    cells.sort()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = range(len(cells))
    iso_vals = [c[2] for c in cells]
    e2e_vals = [c[3] for c in cells]
    ax.bar([x - 0.2 for x in xs], iso_vals, width=0.38, color="#bbbbbb",
           edgecolor="#888888", linewidth=0.5,
           label="isolated comm gap (C2 vs C0)")
    ax.bar([x + 0.2 for x in xs], e2e_vals, width=0.38,
           color=["#d62728" if v < 0 else "#1f77b4" for v in e2e_vals],
           label="e2e delta (r1 carrier)")
    # two colour entries for the e2e bars (sign = winner)
    ax.bar([float("nan")], [float("nan")], color="#1f77b4",
           label="   e2e: C2 wins")
    ax.bar([float("nan")], [float("nan")], color="#d62728",
           label="   e2e: C0 wins (reversal)")
    for i, (n, q, iso, e2e, rev) in enumerate(cells):
        if rev:
            ax.annotate(f"{e2e:+.1f}%", (i + 0.2, e2e), xytext=(0, -10),
                        textcoords="offset points", ha="center", fontsize=6,
                        color="#d62728")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"N{n}\nq{q}" for n, q, *_ in cells], fontsize=7)
    ax.set_ylabel("C2 vs C0  (%)  + : C2 faster")
    ax.set_title("Isolated winner (grey, always >0) vs e2e outcome per cell:\n"
                 "same substrate, opposite ranking", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F1_money.{ext}"))
    plt.close(fig)
    return True


def fig_f2b_decay(summary_dir, out_dir):
    """Resource-axis reversal: e2e delta (C2 vs C0) vs q per N; iso reference."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_significance.csv"))
    e2e = defaultdict(dict)   # N -> q -> delta%
    iso = defaultdict(dict)
    for r in rows:
        if r["pair"] == "r1_C2_vs_C0":
            e2e[int(r["N"])][int(r["q"])] = float(r["delta_pct_pos_a_faster"])
        elif r["pair"] == "comm_C2_vs_C0":
            iso[int(r["N"])][int(r["q"])] = float(r["delta_pct_pos_a_faster"])
    if not e2e:
        return False
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    handles = []
    for n in sorted(e2e):
        color = COLORS.get(str(n))
        qs = sorted(e2e[n])
        ax.plot(qs, [e2e[n][q] for q in qs], "o-", color=color,
                markersize=4.5)
        if n in iso:
            qi = sorted(iso[n])
            # slight x-offset so iso markers do not sit under e2e markers
            ax.plot([q - 0.12 for q in qi], [iso[n][q] for q in qi], "o--",
                    color=color, alpha=0.5, linewidth=1.2, markersize=3.5)
            handles.append(plt.Line2D([], [], color=color,
                                      label=f"N={n}: e2e (solid) / isolated (dash)"))
        else:
            handles.append(plt.Line2D([], [], color=color, label=f"N={n}: e2e"))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("slices q")
    ax.set_ylabel("C2 vs C0  (%)  + : C2 faster")
    ax.set_title("Isolated advantage (dash, always >0) vs e2e (solid):\n"
                 "decays with q, crosses zero — ranking reverses",
                 fontsize=8)
    ax.legend(handles=handles, fontsize=6.5, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F2b_decay.{ext}"))
    plt.close(fig)
    return True


def fig_f2a_winner_map(summary_dir, out_dir):
    """Config-axis winner map on the (N,q) plane."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_boundary_xcand.csv"))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    xs = sorted({int(r["q"]) for r in rows})
    ns = sorted({int(r["N"]) for r in rows})
    for r in rows:
        n, q = int(r["N"]), int(r["q"])
        try:
            e2e_pct = float(r["r1_C2_vs_C0_e2e_pct"])
        except (KeyError, ValueError):
            continue
        reversed_cell = r.get("rccl_config_flag") == "RCCL_CONFIG_REVERSAL"
        ax.scatter(q, n, s=420, marker="s",
                   c="white", edgecolors="k", linewidths=0.8, zorder=2)
        label = "C2" if e2e_pct > 0 else "C0"
        ax.annotate(label, (q, n), ha="center", va="center", fontsize=8,
                    fontweight="bold" if reversed_cell else "normal",
                    color="#d62728" if reversed_cell else "k", zorder=3)
        if reversed_cell:
            ax.annotate(f"{e2e_pct:+.1f}%", (q, n), xytext=(0, -16),
                        textcoords="offset points", ha="center",
                        fontsize=6, color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_yticks(ns)
    # pad in log space so squares/ticks are not clipped by the spines
    ax.set_xlim(xs[0] * 0.65, xs[-1] * 1.55)
    ax.set_ylim(ns[0] * 0.65, ns[-1] * 1.55)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("slices q (log2)")
    ax.set_ylabel("N (log2)")
    ax.set_title("e2e winner per cell, r1 carrier (label C2/C0)\n"
                 "red = isolated winner C2 loses e2e; % = e2e delta C2 vs C0",
                 fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F2a_winner_map.{ext}"))
    plt.close(fig)
    return True


def fig_f2c_balance(summary_dir, out_dir):
    """Balance law: e2e delta vs isolated comm/gemm ratio (the N-sensor)."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_cell_matrix.csv"))
    comm = {}   # (N,q) -> t_done of COMM_ONLY C0
    gemm = {}   # (N,q) -> t_done of GEMM_ONLY C0
    e2e = defaultdict(dict)  # (N,q) -> {"c0"/"c2": e2e}
    for r in rows:
        key = (int(r["N"]), int(r["q"]))
        if r["candidate"] == "C0_DEFAULT":
            if r["path"] == "COMM_ONLY" and r["t_done_p50_us"]:
                comm[key] = float(r["t_done_p50_us"])
            if r["path"] == "GEMM_ONLY" and r["t_done_p50_us"]:
                gemm[key] = float(r["t_done_p50_us"])
            if r["path"] == "R1_EVENT_OVERLAP" and r["e2e_p50_us"]:
                e2e[key]["c0"] = float(r["e2e_p50_us"])
        elif (r["candidate"] == "C2_RING_SIMPLE_CH8"
              and r["path"] == "R1_EVENT_OVERLAP" and r["e2e_p50_us"]):
            e2e[key]["c2"] = float(r["e2e_p50_us"])
    pts = []
    for key in sorted(set(comm) & set(gemm) & set(e2e)):
        n, q = key
        if "c0" not in e2e[key] or "c2" not in e2e[key]:
            continue
        delta = 100.0 * (e2e[key]["c0"] - e2e[key]["c2"]) / e2e[key]["c0"]
        pts.append((comm[key] / gemm[key], delta, n, q))
    if not pts:
        return False
    q_marker = {2: "o", 4: "s", 8: "^", 16: "D"}
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.axvspan(0.9, 1.35, color="#ff7f0e", alpha=0.12, zorder=1)
    for ratio, delta, n, q in pts:
        ax.scatter(ratio, delta, s=46, marker=q_marker.get(q, "o"),
                   color=COLORS.get(str(n)), edgecolors="k", linewidths=0.5,
                   zorder=3)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xscale("log")
    ratios = sorted({p[0] for p in pts})
    ticks = [t for t in (0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0)
             if ratios[0] * 0.9 <= t <= ratios[-1] * 1.1]
    if ticks:
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.set_xlabel("isolated comm / gemm  (log)  — workload balance")
    ax.set_ylabel("e2e delta C2 vs C0  (%)  + : C2 faster")
    n_handles = [plt.Line2D([], [], color=COLORS.get(str(n)), marker="o",
                            linestyle="", label=f"N={n}")
                 for n in sorted({p[2] for p in pts})]
    q_handles = [plt.Line2D([], [], color="k", marker=m, linestyle="",
                            label=f"q={q}")
                 for q, m in q_marker.items() if q in {p[3] for p in pts}]
    handles = n_handles + q_handles
    handles.append(plt.Rectangle((0, 0), 0, 0, color="#ff7f0e", alpha=0.25,
                                 label="balance band"))
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", ncol=2)
    ax.set_title("Reversal lives only in the true-overlap band:\n"
                 "both resources saturated (ratio ~ 1) AND deep slicing",
                 fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F2c_balance.{ext}"))
    plt.close(fig)
    return True


def fig_f3_stretch(summary_dir, out_dir):
    """Price of sliced release: d1 t_done stretch over pure dc transport."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_control_table.csv"))
    series = defaultdict(dict)  # N -> q -> stretch%
    for r in rows:
        if r["candidate"] != "C0_DEFAULT" or not r["d1_vs_dc_done_stretch_pct"]:
            continue
        series[int(r["N"])][int(r["q"])] = float(r["d1_vs_dc_done_stretch_pct"])
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for n in sorted(series):
        qs = sorted(series[n])
        ax.plot(qs, [series[n][q] for q in qs], "o-",
                color=COLORS.get(str(n)), label=f"N={n}")
        for q in qs:
            ax.annotate(f"{series[n][q]:.0f}", (q, series[n][q]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6,
                        color=COLORS.get(str(n)))
    ax.axhline(100, color="k", linewidth=0.7, linestyle=":")
    ax.set_xlabel("slices q")
    ax.set_ylabel("d1 t_done / dc t_done  (%)")
    ax.set_title("Stretch of sliced release vs pure transport\n"
                 "(100 = no protocol/interference cost)", fontsize=8)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F3_stretch.{ext}"))
    plt.close(fig)
    return True


def fig_f4_crossover(summary_dir, out_dir):
    """Overlap-gain crossovers vs q, from the control table."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_control_table.csv"))
    metrics = [("r1_vs_rs_gain_pct", "r1 vs rs  (RCCL overlap vs serial)"),
               ("r1_vs_r0_gain_pct", "r1 vs r0  (overlap vs bulk)"),
               ("d1_vs_d0_gain_pct", "d1 vs d0  (DUSHMEM overlap vs serial)"),
               ("r1_vs_d1_gain_pct", "r1 vs d1  (cross-substrate)")]
    data = {m: defaultdict(dict) for m, _ in metrics}
    for r in rows:
        if r["candidate"] != "C0_DEFAULT":
            continue
        for m, _ in metrics:
            if r[m]:
                data[m][int(r["N"])][int(r["q"])] = float(r[m])
    if not any(data[m] for m, _ in metrics):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
    for ax, (m, label) in zip(axes.flat, metrics):
        for n in sorted(data[m]):
            qs = sorted(data[m][n])
            ax.plot(qs, [data[m][n][q] for q in qs], "o-",
                    color=COLORS.get(str(n)), label=f"N={n}")
        ax.axhline(0, color="k", linewidth=0.7)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("slices q")
        ax.set_ylabel("gain (%)")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("Gain crossovers vs slicing depth (C0 carrier, e2e p50)",
                 fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F4_crossover.{ext}"))
    plt.close(fig)
    return True


def fig_f6(summary_dir, out_dir):
    """Wait-placement mechanism: per-slice release times, dc vs d1 vs d1w."""
    # prefer the cross-root merged curves when present (dc from the formal
    # root, ds/d1/d1w from the dsfix root — merge_phaseb_dsfix.py output)
    rows = read_csv(os.path.join(summary_dir,
                                 "phaseb_release_curves_long_merged.csv")) \
        or read_csv(os.path.join(summary_dir,
                                 "phaseb_release_curves_long.csv"))
    if not rows:
        return False
    cells = sorted({(int(r["N"]), int(r["q"])) for r in rows})
    # informative cells in priority order: clean protocol (512/q2), deepest
    # slicing (512/q8), and the strong-reversal cell (2048/q8) once present
    wanted = [(512, 2), (512, 8), (2048, 8)]
    picks = [c for c in wanted if c in cells] or cells[:2]
    styles = {"DC_PUSHSIG_ONLY": ("#2ca02c", "o", "dc (wait on wait_stream)"),
              "D1_PUSHSIG_OVERLAP": ("#d62728", "s", "d1 (wait on compute_stream)"),
              "D1W_WAITSTREAM_OVERLAP": ("#9467bd", "^", "d1w (wait on wait_stream)")}
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.0),
                             squeeze=False)
    handles = []
    plotted = False
    for ax, (n, q) in zip(axes[0], picks):
        for path, (color, marker, label) in styles.items():
            # aggregate across reps: one median point per slice — plotting
            # per-rep points draws saw-tooth artefacts that read as signal
            per_slice = defaultdict(list)
            for r in rows:
                if (int(r["N"]) == n and int(r["q"]) == q
                        and r["path"] == path
                        and r["candidate"] == "C0_DEFAULT"
                        and r["t_release_med_us"]):
                    per_slice[int(r["slice_index"])].append(
                        float(r["t_release_med_us"]))
            if not per_slice:
                continue
            pts = sorted((s, statistics.median(v)) for s, v in per_slice.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker=marker, markersize=3.5, color=color, linewidth=1.4)
            if not any(h.get_label() == label for h in handles):
                handles.append(plt.Line2D([], [], color=color, marker=marker,
                                          linestyle="-", label=label))
            plotted = True
        ax.set_title(f"N={n}  q={q}", fontsize=9)
        ax.set_xlabel("slice index")
    if not plotted:
        plt.close(fig)
        return False
    axes[0][0].set_ylabel("t_release  (us, p50 max-rank)")
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               fontsize=7, columnspacing=1.6)
    fig.suptitle("When does each slice become consumable? (wait placement)",
                 fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F6_release.{ext}"))
    plt.close(fig)
    return True


def fig_f5b_capability(summary_dir, out_dir):
    """Substrate capability vector: isolated t_done of comm/fc/dc per cell."""
    rows = read_csv(os.path.join(summary_dir, "phaseb_cell_matrix.csv"))
    groups = defaultdict(dict)  # (N,q) -> {family: t_done}
    for r in rows:
        if r["candidate"] != "C0_DEFAULT":
            continue
        key = (int(r["N"]), int(r["q"]))
        if r["path"] == "COMM_ONLY" and r["t_done_p50_us"]:
            groups[key]["comm (RCCL AG)"] = float(r["t_done_p50_us"])
        elif r["path"] == "FC_FCOLLECT_ONLY" and r["t_done_p50_us"]:
            groups[key]["fc (DUSHMEM fcollect)"] = float(r["t_done_p50_us"])
        elif r["path"] == "DC_PUSHSIG_ONLY" and r["t_done_p50_us"]:
            groups[key]["dc (DUSHMEM put-sig)"] = float(r["t_done_p50_us"])
    if not groups:
        return False
    keys = sorted(groups)
    families = ["comm (RCCL AG)", "fc (DUSHMEM fcollect)", "dc (DUSHMEM put-sig)"]
    colors = ["#1f77b4", "#9467bd", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.8 / len(families)
    all_ys = []
    for fi, fam in enumerate(families):
        xs, ys = [], []
        for ci, key in enumerate(keys):
            if fam in groups[key]:
                xs.append(ci + (fi - 1) * width)
                ys.append(groups[key][fam])
        all_ys.extend(ys)
        ax.bar(xs, ys, width=width, color=colors[fi], label=fam)
    ax.set_ylim(0, max(all_ys) * 1.08 if all_ys else 1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"N{n}/q{q}" for n, q in keys], rotation=45,
                       fontsize=6, ha="right")
    ax.set_ylabel("isolated t_done p50 (us)")
    ax.set_title("Substrate capability vector B (same payload, same slicing)",
                 fontsize=9)
    # legend below the rotated tick labels — the tallest bars reach ~11.5k in
    # every group, so any in-axes position would cover data
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_F5b_capability.{ext}"))
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    summary_dir = os.path.join(args.result_root, "summary")
    out_dir = os.path.join(summary_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in [("F1_money", fig_f1_money),
                     ("F2a_winner_map", fig_f2a_winner_map),
                     ("F2b_decay", fig_f2b_decay),
                     ("F2c_balance", fig_f2c_balance),
                     ("F3_stretch", fig_f3_stretch),
                     ("F4_crossover", fig_f4_crossover),
                     ("F5b_capability", fig_f5b_capability),
                     ("F6_release", fig_f6)]:
        try:
            ok = fn(summary_dir, out_dir)
            print(f"{name}: {'written' if ok else 'skipped (no data)'}")
        except Exception as exc:  # noqa: BLE001 — figure must not kill the batch
            print(f"{name}: FAILED ({exc})")
    print(f"figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
