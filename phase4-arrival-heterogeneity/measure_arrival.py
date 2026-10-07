#!/usr/bin/env python3
"""
BW1000 dual-node arrival heterogeneity measurement.

Each rank sends a chunk to rank 0. Rank 0 timestamps each arrival.
Repeats N times for statistics. Also measures GEMM time for ratio.

Design for SCNet: 2 nodes × 8 BW1000 GPUs, DUSHMEM for cross-node comm.

Usage (on login node):
  # Submit via job scheduler:
  sbatch scnet_bw1000_job.sh

  # Or run directly (must have 16 GPUs across 2 nodes):
  python3 measure_arrival.py --output results.json
"""
import argparse
import json
import os
import socket
import statistics
import struct
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
    parser.add_argument("--output", type=str, default="arrival_results.json")
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl")  # or "gloo" fallback
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

    # Warmup rounds
    for _ in range(args.warmup):
        if rank == 0:
            for src in range(1, world):
                buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
                dist.recv(buf, src=src)
        else:
            dist.send(data, dst=0)
        dist.barrier()

    # Measurement rounds
    # rank 0 records: for each source rank, the time from round start to receipt
    arrival_data = defaultdict(list)  # src_rank -> [latency_ms, ...]

    for it in range(args.iterations):
        dist.barrier()  # synchronize all ranks
        torch.cuda.synchronize()

        if rank == 0:
            round_start = time.perf_counter()
            # Receive from all other ranks in arbitrary order
            # Use non-blocking receives to get true arrival order
            requests = []
            buffers = {}
            for src in range(1, world):
                buf = torch.empty(chunk_elems, dtype=torch.float16, device="cuda")
                req = dist.irecv(buf, src=src)
                requests.append((src, req, buf))

            # Wait for all and record arrival time
            completed = set()
            while len(completed) < world - 1:
                for src, req, buf in requests:
                    if src not in completed:
                        if req.is_completed():
                            lat = (time.perf_counter() - round_start) * 1000
                            arrival_data[src].append(lat)
                            completed.add(src)
                            # Mark the tensor to prevent premature deallocation
                            buffers[src] = buf
                time.sleep(0.0001)  # 0.1ms polling granularity

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
              f"{'Std(ms)':>8s} {'Min(ms)':>8s} {'Max(ms)':>8s}")
        print("-" * 75)

        arrival_stats = {}
        for src in sorted(arrival_data.keys()):
            times = arrival_data[src]
            src_node = src // 8
            label = "SAME-NODE" if src_node == 0 else "CROSS-NODE"
            med = statistics.median(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            mn, mx = min(times), max(times)
            print(f"{src:5d} {src_node:5d} {label:>12s} {med:10.3f} "
                  f"{std:8.3f} {mn:8.3f} {mx:8.3f}")
            arrival_stats[f"rank_{src}"] = {
                "node": src_node,
                "label": label,
                "median_ms": round(med, 3),
                "std_ms": round(std, 3),
                "min_ms": round(mn, 3),
                "max_ms": round(mx, 3),
                "n": len(times),
            }

        # Summary metrics
        same_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "SAME-NODE"]
        cross_node = [v["median_ms"] for v in arrival_stats.values() if v["label"] == "CROSS-NODE"]

        same_med = statistics.median(same_node) if same_node else 0
        cross_med = statistics.median(cross_node) if cross_node else 0
        spread = cross_med - same_med
        ratio = cross_med / same_med if same_med > 0.001 else float("inf")
        gemm_ms = gemm_result["median_ms"]
        vs_gemm = spread / gemm_ms * 100 if gemm_ms > 0 else 0

        print("-" * 75)
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

        # Save results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": hostname,
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
            "gemm": gemm_result,
            "arrival_times": arrival_stats,
            "summary": {
                "same_node_median_ms": round(same_med, 3),
                "cross_node_median_ms": round(cross_med, 3),
                "spread_ms": round(spread, 3),
                "heterogeneity_ratio": round(ratio, 1),
                "spread_vs_gemm_pct": round(vs_gemm, 1),
                "conclusion": conclusion,
            },
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
