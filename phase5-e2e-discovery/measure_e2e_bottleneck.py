#!/usr/bin/env python3
"""
BW1000 双机端到端 AG-GEMM 瓶颈发现实验。

不预设结论，跑完看数据。回答四个问题：
  Q1: 串行 AG-GEMM 时间分解——通信/计算/同步各占多少？
  Q2: 朴素重叠收益——不等全部到齐就开始算，能省多少？
  Q3: chunk 粒度影响——拆成不同大小分批发，行为怎么变？
  Q4: 多集合通信竞争——两个 AG 同时跑，到达时间恶化多少？

用法（SCNet 双机 16 卡）：
  torchrun --nnodes=2 --nproc-per-node=8 \
    --node_rank=<0或1> --master_addr=<IP> --master_port=29500 \
    measure_e2e_bottleneck.py --output e2e_results.json
"""
import argparse
import json
import os
import socket
import time
import statistics
from collections import defaultdict

import torch
import torch.distributed as dist


def init():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


# ==========================================================================
# Q1: 串行 AG-GEMM 时间分解
# ==========================================================================

def q1_serial_breakdown(A_local, B, M, N, K, rank, world, reps=20):
    """
    把串行 AG-GEMM 拆成三段计时：
      1. allgather 通信时间
      2. GEMM 计算时间
      3. barrier/同步开销
    """
    M_per_rank = A_local.shape[0]
    device = A_local.device

    # Warmup
    for _ in range(5):
        gathered = [torch.empty_like(A_local) for _ in range(world)]
        dist.all_gather(gathered, A_local)
        A_full = torch.cat(gathered, dim=0)
        C = A_full @ B
    torch.cuda.synchronize()
    dist.barrier()

    ag_times = []
    gemm_times = []
    sync_times = []
    total_times = []

    for rep in range(reps):
        gathered = [torch.empty_like(A_local) for _ in range(world)]

        torch.cuda.synchronize()
        dist.barrier()  # 对齐所有 rank
        t_start = time.perf_counter()

        # Phase 1: AllGather
        dist.all_gather(gathered, A_local)
        torch.cuda.synchronize()
        t_ag_done = time.perf_counter()

        # Phase 2: GEMM
        A_full = torch.cat(gathered, dim=0)
        C = A_full @ B
        torch.cuda.synchronize()
        t_gemm_done = time.perf_counter()

        # Phase 3: Sync
        dist.barrier()
        t_end = time.perf_counter()

        ag_times.append((t_ag_done - t_start) * 1000)
        gemm_times.append((t_gemm_done - t_ag_done) * 1000)
        sync_times.append((t_end - t_gemm_done) * 1000)
        total_times.append((t_end - t_start) * 1000)

    return {
        "ag_median_ms": statistics.median(ag_times),
        "gemm_median_ms": statistics.median(gemm_times),
        "sync_median_ms": statistics.median(sync_times),
        "total_median_ms": statistics.median(total_times),
        "ag_pct": statistics.median(ag_times) / statistics.median(total_times) * 100,
        "gemm_pct": statistics.median(gemm_times) / statistics.median(total_times) * 100,
        "sync_pct": statistics.median(sync_times) / statistics.median(total_times) * 100,
        "all_ranks_ag_ms": gather_scalar(statistics.median(ag_times), rank, world),
        "all_ranks_total_ms": gather_scalar(statistics.median(total_times), rank, world),
    }


def gather_scalar(value, rank, world):
    """Gather a scalar from all ranks to rank 0."""
    t = torch.tensor([value], device="cuda")
    all_t = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(all_t, t)
    return [v.item() for v in all_t]


# ==========================================================================
# Q2: 朴素重叠收益
# ==========================================================================

def q2_naive_overlap(A_local, B, M, N, K, rank, world, reps=20):
    """
    最朴素的重叠：把 allgather 拆成 per-rank 的 broadcast/isend-irecv，
    收到哪块数据就开始算哪块的 GEMM。

    对比：
      a) 串行：等全部到齐 → 一次 GEMM
      b) 朴素重叠：收到哪块算哪块（分块 GEMM）
    """
    M_per_rank = A_local.shape[0]
    device = A_local.device

    # ---- 方式 a: 串行 ----
    serial_times = []
    for rep in range(reps):
        gathered = [torch.empty_like(A_local) for _ in range(world)]
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_gather(gathered, A_local)
        torch.cuda.synchronize()
        A_full = torch.cat(gathered, dim=0)
        C = A_full @ B
        torch.cuda.synchronize()
        dist.barrier()
        serial_times.append((time.perf_counter() - t0) * 1000)

    # ---- 方式 b: 朴素重叠（逐块收 + 逐块算）----
    overlap_times = []
    for rep in range(reps):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        # 用 point-to-point 逐块接收
        chunks = [torch.empty_like(A_local) for _ in range(world)]
        chunks[rank] = A_local  # 本地数据立即可用

        # 发送本地数据给所有 peer
        send_reqs = []
        for peer in range(world):
            if peer != rank:
                send_reqs.append(dist.isend(A_local, dst=peer))

        # 立即开始算本地块的 GEMM（重叠开始）
        C_local = A_local @ B

        # 逐块接收并计算
        recv_reqs = []
        for peer in range(world):
            if peer != rank:
                recv_reqs.append((peer, dist.irecv(chunks[peer], src=peer)))

        for peer, req in recv_reqs:
            req.wait()
            # 收到哪块就算哪块
            C_peer = chunks[peer] @ B

        for req in send_reqs:
            req.wait()

        torch.cuda.synchronize()
        dist.barrier()
        overlap_times.append((time.perf_counter() - t0) * 1000)

    serial_med = statistics.median(serial_times)
    overlap_med = statistics.median(overlap_times)
    saving = serial_med - overlap_med

    return {
        "serial_median_ms": serial_med,
        "overlap_median_ms": overlap_med,
        "saving_ms": saving,
        "saving_pct": saving / serial_med * 100 if serial_med > 0 else 0,
    }


# ==========================================================================
# Q3: chunk 粒度影响
# ==========================================================================

def q3_chunk_granularity(A_local, B, M, N, K, rank, world, reps=10):
    """
    把 allgather 拆成不同大小的 chunk 分批收发，每收到一批就开始算一批。
    测试 chunk 数量：1（=串行）、2、4、8、16。
    """
    M_per_rank = A_local.shape[0]
    device = A_local.device
    K_chunk = A_local.shape[1]

    results = {}
    for num_chunks in [1, 2, 4, 8, 16]:
        chunk_rows = M_per_rank // num_chunks
        if chunk_rows < 128:  # 至少一个 tile 高
            continue

        times = []
        for rep in range(reps):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()

            # 分块 allgather + 分块 GEMM
            output = torch.empty(M, N, device=device, dtype=A_local.dtype)

            for c in range(num_chunks):
                row_start = c * chunk_rows
                row_end = row_start + chunk_rows

                # AllGather 这一块
                chunk_local = A_local[row_start:row_end]
                chunk_gathered = [torch.empty_like(chunk_local) for _ in range(world)]
                dist.all_gather(chunk_gathered, chunk_local)

                # 立即算这一块
                chunk_full = torch.cat(chunk_gathered, dim=0)
                output[row_start * world:row_end * world] = chunk_full @ B

            torch.cuda.synchronize()
            dist.barrier()
            times.append((time.perf_counter() - t0) * 1000)

        results[f"chunks_{num_chunks}"] = {
            "median_ms": statistics.median(times),
            "chunk_rows": chunk_rows,
        }

    return results


# ==========================================================================
# Q4: 多集合通信竞争
# ==========================================================================

def q4_contention(A_local, B, M, N, K, rank, world, reps=10):
    """
    同时跑 1 个、2 个、4 个 allgather，观察到达时间恶化。
    模拟真实 LLM 中多层同时通信的场景。
    """
    M_per_rank = A_local.shape[0]
    device = A_local.device

    results = {}
    for num_concurrent in [1, 2, 4]:
        # 创建多份输入
        inputs = [torch.randn_like(A_local) for _ in range(num_concurrent)]

        times = []
        for rep in range(reps):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()

            # 同时启动多个 allgather
            all_gathered = []
            for i in range(num_concurrent):
                gathered = [torch.empty_like(inputs[i]) for _ in range(world)]
                dist.all_gather(gathered, inputs[i])
                all_gathered.append(gathered)

            torch.cuda.synchronize()
            dist.barrier()
            times.append((time.perf_counter() - t0) * 1000)

        # 单个 AG 时间（无竞争）
        single_times = []
        for rep in range(reps):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            gathered = [torch.empty_like(A_local) for _ in range(world)]
            dist.all_gather(gathered, A_local)
            torch.cuda.synchronize()
            single_times.append((time.perf_counter() - t0) * 1000)

        total_med = statistics.median(times)
        single_med = statistics.median(single_times)
        per_ag = total_med / num_concurrent

        results[f"concurrent_{num_concurrent}"] = {
            "total_ms": total_med,
            "per_ag_ms": per_ag,
            "single_ag_ms": single_med,
            "degradation_pct": (per_ag - single_med) / single_med * 100,
        }

    return results


# ==========================================================================
# 主函数
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=[8192, 4096, 8192],
                       help="M_global N K")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", default="e2e_results.json")
    parser.add_argument("--skip", type=str, default="",
                       help="Comma-separated question numbers to skip (e.g. '3,4')")
    args = parser.parse_args()

    rank, world, local_rank = init()
    M, N, K = args.shape
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    skip = set(args.skip.split(",")) if args.skip else set()

    M_per_rank = M // world
    torch.manual_seed(42 + rank)
    A_local = torch.randn(M_per_rank, K, device=f"cuda:{local_rank}", dtype=dtype)
    B = torch.randn(K, N, device=f"cuda:{local_rank}", dtype=dtype)

    if rank == 0:
        hostname = socket.gethostname()
        print(f"Shape: M={M} N={N} K={K} dtype={args.dtype}")
        print(f"World: {world} ranks ({world//8} nodes × 8 GPUs)")
        print(f"Running on: {hostname}")
        print()

    results = {"config": vars(args), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Q1
    if "1" not in skip:
        if rank == 0:
            print("=" * 60)
            print("Q1: 串行 AG-GEMM 时间分解")
            print("=" * 60)
        r = q1_serial_breakdown(A_local, B, M, N, K, rank, world, args.reps)
        results["q1"] = r
        if rank == 0:
            print(f"  AllGather:  {r['ag_median_ms']:.2f} ms ({r['ag_pct']:.0f}%)")
            print(f"  GEMM:       {r['gemm_median_ms']:.2f} ms ({r['gemm_pct']:.0f}%)")
            print(f"  Sync:       {r['sync_median_ms']:.2f} ms ({r['sync_pct']:.0f}%)")
            print(f"  Total:      {r['total_median_ms']:.2f} ms")
            ag_all = r['all_ranks_ag_ms']
            print(f"  AG per-rank: min={min(ag_all):.2f} max={max(ag_all):.2f} "
                  f"spread={max(ag_all)-min(ag_all):.2f} ms")
            print()

    # Q2
    if "2" not in skip:
        if rank == 0:
            print("=" * 60)
            print("Q2: 朴素重叠收益")
            print("=" * 60)
        r = q2_naive_overlap(A_local, B, M, N, K, rank, world, args.reps)
        results["q2"] = r
        if rank == 0:
            print(f"  Serial:     {r['serial_median_ms']:.2f} ms")
            print(f"  Overlap:    {r['overlap_median_ms']:.2f} ms")
            print(f"  Saving:     {r['saving_ms']:.2f} ms ({r['saving_pct']:.1f}%)")
            print()

    # Q3
    if "3" not in skip:
        if rank == 0:
            print("=" * 60)
            print("Q3: Chunk 粒度影响")
            print("=" * 60)
        r = q3_chunk_granularity(A_local, B, M, N, K, rank, world, max(args.reps//2, 5))
        results["q3"] = r
        if rank == 0:
            for k, v in sorted(r.items()):
                print(f"  {k}: {v['median_ms']:.2f} ms (chunk_rows={v['chunk_rows']})")
            print()

    # Q4
    if "4" not in skip:
        if rank == 0:
            print("=" * 60)
            print("Q4: 多集合通信竞争")
            print("=" * 60)
        r = q4_contention(A_local, B, M, N, K, rank, world, max(args.reps//2, 5))
        results["q4"] = r
        if rank == 0:
            for k, v in sorted(r.items()):
                print(f"  {k}: per-AG={v['per_ag_ms']:.2f} ms "
                      f"(single={v['single_ag_ms']:.2f} ms, "
                      f"degradation={v['degradation_pct']:+.1f}%)")
            print()

    # Save
    if rank == 0:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
