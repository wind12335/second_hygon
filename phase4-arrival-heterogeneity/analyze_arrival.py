#!/usr/bin/env python3
"""Print a human-readable summary from one or more phase4 arrival JSON files.

Usage: python3 analyze_arrival.py --input a.json [b.json ...]
"""
import argparse
import json


def show(path):
    with open(path) as f:
        r = json.load(f)
    print("=" * 78)
    print(path)
    print("=" * 78)
    print("timestamp : {}   host: {}".format(r.get("timestamp"), r.get("hostname")))
    t = r.get("topology", {})
    print("topology  : {} nodes x {} gpus (world {})".format(
        t.get("nodes"), t.get("gpus_per_node"), t.get("world_size")))
    m = r.get("measurement", {})
    if m:
        print("method    : {} (degraded rounds: {})".format(
            m.get("method"), m.get("degraded_rounds")))
    g = r.get("gemm", {})
    print("gemm      : {} fp16, median {:.3f} ms".format(g.get("shape"), g.get("median_ms", 0)))
    print()
    print(f"{'rank':>6s} {'label':>12s} {'med(ms)':>9s} {'std(ms)':>8s} "
          f"{'min(ms)':>8s} {'max(ms)':>8s} {'avgpos':>7s}")
    for k, v in sorted(r.get("arrival_times", {}).items(), key=lambda x: int(x[0].split("_")[1])):
        print("{:>6s} {:>12s} {:9.3f} {:8.3f} {:8.3f} {:8.3f} {:7.1f}".format(
            k, v["label"], v["median_ms"], v["std_ms"], v["min_ms"], v["max_ms"],
            v.get("mean_arrival_position", float("nan"))))
    s = r.get("summary", {})
    print("-" * 78)
    print("same-node {:.3f} ms | cross-node {:.3f} ms | spread {:.3f} ms | "
          "ratio {:.1f}x | vs GEMM {:.1f}%  =>  {}".format(
              s.get("same_node_median_ms", 0), s.get("cross_node_median_ms", 0),
              s.get("spread_ms", 0), s.get("heterogeneity_ratio", 0),
              s.get("spread_vs_gemm_pct", 0), s.get("conclusion")))
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    a = p.parse_args()
    for path in a.input:
        show(path)
