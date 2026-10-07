#!/usr/bin/env python3
"""
[v2] BW1000 dual-node arrival heterogeneity measurement — hardened variant.

Methodology and JSON schema identical to measure_arrival.py (v1, NVIDIA 侧),
with three hardening changes for the BW1000/RCCL environment:

1. PREFLIGHT: before the measured rounds, one trial round checks whether
   Work.is_completed() polling actually fires on this backend. (On this
   torch build's gloo backend it never fires -> v1 would hang forever; RCCL
   could not be verified locally because RCCL refuses 2 ranks on 1 GPU.)
   If polling works -> "poll" mode (exactly v1 behaviour). If not ->
   "wait_fallback" mode: irecv all sources, then wait() each in posting
   order (order-biased, clearly flagged in JSON), so the job always
   terminates and still yields per-source latencies + GEMM ratio.
2. WATCHDOG: every round is bounded by --poll-timeout-s; leftover requests
   are drained with wait() and the round is flagged degraded.
3. PROGRESS + ORDER STATS: rank 0 prints one heartbeat line per round
   (liveness in pod logs), and the mean arrival POSITION of each source is
   recorded — the scheduling-value metric is precisely arrival order vs
   consumption order (rank order) mismatch, which v1 discarded.

Output JSON is a strict superset of v1's: every v1 field keeps its
name/meaning; extras live under "measurement", "platform", per-source
"mean_arrival_position", plus a "heterogeneity_summary" alias matching the
field names in README.md's example.
"""
import argparse
import json
import os
import socket
import statistics
import time
from collections import defaultdict

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-mb", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gemm-shape", type=int, nargs=3, default=[8192, 4096, 8192])
    parser.add_argument("--output", type=str, default="arrival_results_v2.json")
    parser.add_argument("--poll-timeout-s", type=float, default=20.0,
                        help="per-round watchdog; leftover irecvs drained via wait()")
    parser.add_argument("--preflight-timeout-s", type=float, default=5.0,
                        help="trial-round budget for deciding poll vs wait_fallback")
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl")  # RCCL on BW1000
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    hostname = socket.gethostname()
    node_id = rank // 8

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"World: {world} ranks across {world // 8} nodes")
        print(f"Chunk: {args.chunk_mb} MB per producer")
        print(f"Iterations: {args.iterations} (+{args.warmup} warmup)")
        print()

    # ---- platform facts (all nodes) --------------------------------------
    hosts = [None for _ in range(world)]
    dist.all_gather_object(hosts, hostname)
    node_names = []
    for h in hosts:
        if h not in node_names:
            node_names.append(h)
    if rank == 0:
        platform_info = {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpus_per_node_visible": torch.cuda.device_count(),
            "nodes_actual": node_names,
            "hostname_rank0": hostname,
        }

    # ================================================================
    # Part 1: GEMM time (every rank measures independently)
    # ================================================================
    M, N, K = args.gemm_shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    gemm_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        gemm_times.append((time.perf_counter() - t0) * 1000)
    del a, b, c
    torch.cuda.empty_cache()

    # Gather GEMM times to rank 0
    gemm_tensor = torch.tensor([statistics.median(gemm_times)], device="cuda")
    gemm_all = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(gemm_all, gemm_tensor)
    if rank == 0:
        gemm_result = {
            "shape": args.gemm_shape,
            "median_ms": gemm_all[0].item(),
            "all_ranks_ms": [g.item() for g in gemm_all],
        }
        print(f"GEMM {M}x{N}x{K} fp16: {gemm_result['median_ms']:.2f} ms (rank 0)")
    dist.barrier()

    # ================================================================
    # Part 2: Arrival time measurement
    # ================================================================
    chunk_elems = int(args.chunk_mb * 1024 * 1024) // 2  # fp16 elements
    data = torch.ones(chunk_elems, dtype=torch.float16, device="cuda")

    def post_requests():
        reqs = []
        for src in range(1, world):
            buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
            reqs.append((src, dist.irecv(buf, src=src), buf))
        return reqs

    # Warmup rounds (blocking receives, as v1)
    for _ in range(args.warmup):
        if rank == 0:
            for src in range(1, world):
                buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
                dist.recv(buf, src=src)
        else:
            dist.send(data, dst=0)
        dist.barrier()

    # ---- PREFLIGHT: does is_completed() polling work on this backend? ----
    degraded_rounds = 0
    if rank == 0:
        round_start = time.perf_counter()
        requests = post_requests()
        completed = set()
        while len(completed) < world - 1 and \
                (time.perf_counter() - round_start) < args.preflight_timeout_s:
            for src, req, buf in requests:
                if src not in completed and req.is_completed():
                    completed.add(src)
            time.sleep(0.0001)
        mode = "poll" if len(completed) == world - 1 else "wait_fallback"
        if len(completed) < world - 1:
            for src, req, buf in requests:
                if src not in completed:
                    req.wait()
        print(f"Preflight: is_completed polling "
              f"{'WORKS -> poll mode (v1 behaviour)' if mode == 'poll' else 'DOES NOT FIRE -> wait_fallback mode'}"
              f" ({len(completed)}/{world - 1} completions in {args.preflight_timeout_s}s)",
              flush=True)
    else:
        dist.send(data, dst=0)
    dist.barrier()
    mode_holder = [mode] if rank == 0 else [None]
    dist.broadcast_object_list(mode_holder, src=0)
    mode = mode_holder[0]
    dist.barrier()

    # Measurement rounds
    arrival_data = defaultdict(list)   # src_rank -> [latency_ms, ...]
    position_data = defaultdict(list)  # src_rank -> [1-based completion position, ...]

    for it in range(args.iterations):
        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            round_start = time.perf_counter()
            requests = post_requests()
            completed = set()
            degraded = False

            if mode == "poll":
                while len(completed) < world - 1:
                    if (time.perf_counter() - round_start) > args.poll_timeout_s:
                        degraded = True
                        break
                    for src, req, buf in requests:
                        if src not in completed and req.is_completed():
                            lat = (time.perf_counter() - round_start) * 1000
                            arrival_data[src].append(lat)
                            position_data[src].append(len(completed) + 1)
                            completed.add(src)
                    time.sleep(0.0001)  # 0.1ms polling granularity (v1 spec)
                if degraded:
                    for src, req, buf in requests:
                        if src not in completed:
                            req.wait()
                            lat = (time.perf_counter() - round_start) * 1000
                            arrival_data[src].append(lat)
                            position_data[src].append(len(completed) + 1)
                            completed.add(src)
                    degraded_rounds += 1
            else:  # wait_fallback
                for src, req, buf in requests:
                    req.wait()
                    lat = (time.perf_counter() - round_start) * 1000
                    arrival_data[src].append(lat)
                    position_data[src].append(len(completed) + 1)
                    completed.add(src)

            if (it + 1) % 10 == 0 or it == 0 or args.iterations <= 10:
                lats = [arrival_data[s][-1] for s in completed]
                print(f"  round {it + 1}/{args.iterations} done, "
                      f"last-round arrivals min/med/max = "
                      f"{min(lats):.2f}/{statistics.median(lats):.2f}/{max(lats):.2f} ms",
                      flush=True)
        else:
            dist.send(data, dst=0)

        dist.barrier()

    # ================================================================
    # Part 3: Analyze and output
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 75)
        print("ARRIVAL TIME MEASUREMENT RESULTS")
        print("=" * 75)
        print(f"{'Rank':>5s} {'Node':>5s} {'Label':>12s} {'Median(ms)':>10s} "
              f"{'Std(ms)':>8s} {'Min(ms)':>8s} {'Max(ms)':>8s} {'AvgPos':>6s}")
        print("-" * 82)

        arrival_stats = {}
        for src in sorted(arrival_data.keys()):
            times = arrival_data[src]
            src_node = src // 8
            label = "SAME-NODE" if src_node == 0 else "CROSS-NODE"
            med = statistics.median(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            mn, mx = min(times), max(times)
            pos = statistics.mean(position_data[src]) if position_data[src] else -1
            print(f"{src:5d} {src_node:5d} {label:>12s} {med:10.3f} "
                  f"{std:8.3f} {mn:8.3f} {mx:8.3f} {pos:6.1f}")
            arrival_stats[f"rank_{src}"] = {
                "node": src_node,
                "label": label,
                "median_ms": round(med, 3),
                "std_ms": round(std, 3),
                "min_ms": round(mn, 3),
                "max_ms": round(mx, 3),
                "mean_arrival_position": round(pos, 2),
                "n": len(times),
            }

        # Summary metrics (v1 definitions)
        same_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "SAME-NODE"]
        cross_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "CROSS-NODE"]

        same_med = statistics.median(same_node) if same_node else 0
        cross_med = statistics.median(cross_node) if cross_node else 0
        spread = cross_med - same_med
        ratio = cross_med / same_med if same_med > 0.001 else float("inf")
        gemm_ms = gemm_result["median_ms"]
        vs_gemm = spread / gemm_ms * 100 if gemm_ms > 0 else 0

        print("-" * 82)
        print(f"Same-node median:  {same_med:.3f} ms")
        print(f"Cross-node median: {cross_med:.3f} ms")
        print(f"Spread:            {spread:.3f} ms")
        print(f"Heterogeneity ratio: {ratio:.1f}×")
        print(f"GEMM time:         {gemm_ms:.2f} ms")
        print(f"Spread vs GEMM:    {vs_gemm:.1f}%")
        print()

        if vs_gemm > 20:
            conclusion = "SCHEDULING_VALUABLE"
            print("→ ✓ Scheduling is VALUABLE (spread > 20% of GEMM time)")
        elif vs_gemm > 5:
            conclusion = "SCHEDULING_MARGINAL"
            print("→ ~ Scheduling is MARGINAL (5-20% of GEMM time)")
        else:
            conclusion = "SCHEDULING_NOT_NEEDED"
            print("→ ✗ Scheduling NOT needed (spread < 5% of GEMM time)")

        summary = {
            "same_node_median_ms": round(same_med, 3),
            "cross_node_median_ms": round(cross_med, 3),
            "spread_ms": round(spread, 3),
            "heterogeneity_ratio": round(ratio, 1),
            "spread_vs_gemm_pct": round(vs_gemm, 1),
            "conclusion": conclusion,
        }

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": hostname,
            "script": "measure_arrival_v2.py",
            "topology": {
                "nodes": world // 8,
                "gpus_per_node": 8,
                "world_size": world,
                "note": "Please fill in interconnect type and bandwidth",
            },
            "config": {
                "chunk_mb": args.chunk_mb,
                "iterations": args.iterations,
                "gemm_shape": args.gemm_shape,
            },
            "measurement": {
                "method": mode,
                "method_note": (
                    "identical to v1 polling"
                    if mode == "poll" else
                    "is_completed() never fired on this backend; per-source "
                    "latency recorded at wait() completion in posting order — "
                    "ORDER BIASED, see mean_arrival_position with care"
                ),
                "poll_granularity_ms": 0.1,
                "degraded_rounds": degraded_rounds,
                "poll_timeout_s": args.poll_timeout_s,
            },
            "platform": platform_info,
            "gemm": gemm_result,
            "arrival_times": arrival_stats,
            "summary": summary,
            # alias matching README.md example field names (same values)
            "heterogeneity_summary": dict(summary),
        }
        if not cross_node:
            results["measurement"]["warning"] = (
                "no cross-node sources in this run (world=%d) — run with "
                "--nnodes=2 x 8 for the dual-node measurement" % world)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
