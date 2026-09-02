# K500SM_AI / gfx928 / 4-GPU formal experiment suite

All experiments use `HIP_VISIBLE_DEVICES=0,1,2,3`, `HSA_FORCE_FINE_GRAIN_PCIE=1`,
float32, correctness checking, 10 warmups, 50 timed iterations, and max-rank
time (`-a 3`).  The fixed master log is:

```text
results/k500sm_ai_gfx928_4gpu/formal_master.log
```

Each RCCL-tests case additionally keeps its own complete stdout/stderr log,
CSV, command file, and status file.  The compact machine-readable summary is
written at the end to `formal_summary.csv`.

## Stages

1. `platform`: DTK/HIP/RCCL versions, `rocminfo`, topology, NUMA, MPI, and
   dynamic library provenance.
2. `preflight`: AllGather, AllReduce, and ReduceScatter at 4 KiB, 1 MiB, and
   64 MiB with correctness checking.
3. `representative`: 4-GPU default, Ring/Simple, Ring/LL, Tree/Simple, and
   Tree/LL at 4 KiB, 64 KiB, 1 MiB, 8 MiB, 64 MiB, and 256 MiB; three runs
   per case.  This is the primary isolated-communication matrix.
4. `channels`: 4-GPU channel values 1/2/4/8 at 1 MiB, 8 MiB, and 64 MiB for
   all three collectives and four forced algorithm/protocol pairs; three runs.
5. `mapping`: natural and two alternative rank/device mappings at 1 MiB and
   64 MiB for AllGather and AllReduce; three runs.
6. `2gpu_focused`: two-GPU default and four Ring/Tree protocol choices at
   64 KiB, 1 MiB, 8 MiB, and 64 MiB; two runs per case.
7. `1g_pilot`: one 1-GiB correctness/performance point for each mandatory
   collective.  A timeout or allocation failure is retained as evidence.
8. `overlap_dushmem_rccl`: existing RCCL-vs-DUSHMEM AG-GEMM probe with
   `TARGET_CHUNKS=1,2,4,8`, `NDIM=2`, five repetitions.  These logs are an
   exploratory overlap signal, not a substitute for a future common NCCL/RCCL
   chunked harness.
9. `ll128_probe`: separate INFO probes for Ring/LL128 and Tree/LL128 at 1 MiB,
   8 MiB, and 64 MiB, two runs; unsupported or fallback behavior is retained.

Raw case directories are append-only and are never overwritten.
