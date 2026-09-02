# Phase 2: Targeted Release-Curve Confirmation

## Fixed Platform

- Platform name: `K500SM_AI`
- GPU architecture: `gfx928`
- Process layout: four MPI processes, one process per GPU
- Interconnect scope: the observed PCIe four-GPU environment
- Libraries: RCCL, HIP, rocBLAS from the locally recorded DTK installation

This document intentionally does not use `gfx936` or `K100AI`. They are not names for this experiment.

## Why This Exists

Phase 1 found a controlled counterexample at `M_local=N=K=2048`, `q=8`:

1. `C2_RING_SIMPLE_CH8` was the fastest isolated RCCL AllGather candidate.
2. `C0_DEFAULT` had lower H0 AllGather-GEMM end-to-end p50.
3. The H0 preference for C0 appeared in all three independent process repetitions and the Phase-1 median gap was about 7.2%.

That is evidence for the hypothesis that isolated collective completion time does not always select the best dependent AllGather-GEMM strategy. It is not yet a final causal attribution because Phase 1 used one independent `COMM_ONLY` process per candidate and only retained first/last slice release timestamps.

## Hypotheses

- H1: At `q=8`, the `COMM_ONLY` winner remains `C2_RING_SIMPLE_CH8`, while H0 selects a different candidate by a material margin.
- H2: At `q=2`, the isolated and H0 winners agree. This is a negative control and defines a q-dependent decision boundary.
- H3: The per-slice release/start/end curves explain whether the H0 difference is principally release timing, GEMM fragmentation, or comm/compute contention.

## Candidate Set

- `C0_DEFAULT`: RCCL defaults; explicit tuning variables removed.
- `C1_RING_SIMPLE_CH4`: `Ring`, `Simple`, exactly four channels.
- `C2_RING_SIMPLE_CH8`: `Ring`, `Simple`, exactly eight channels.

## Matrix

For each `q in {2, 8}` at `M_local=N=K=2048`:

- `COMM_ONLY`: 3 candidates x 5 independent MPI process repetitions x 80 timed iterations.
- `H0_EVENT_OVERLAP`: 3 candidates x 5 repetitions x 80 timed iterations.
- `B1_SLICE_SERIAL`: C0 and C2 x 5 repetitions x 80 timed iterations.
- `GEMM_ONLY`: C0 x 3 repetitions x 80 timed iterations.

Additionally, `B0_FULL_SERIAL` runs five times with `q=1` to retain the full-collective/full-GEMM reference.

Total: 91 case directories and 7,280 timed distributed iterations, excluding warmup.

## Fairness Conditions

For B1 and H0 with the same q and candidate, the following are identical:

- input, weights, FP32 dtype, matrix dimensions, rank-major output layout;
- AllGather API, slice bytes, RCCL candidate, rocBLAS GEMM, scatter kernel;
- warmup, timed iteration count and correctness check.

Only the stream dependency graph changes. B1 makes AllGather slice `i+1` wait for GEMM slice `i`; H0 allows AllGather slice `i+1` to proceed after slice `i` is released to the compute stream.

## Outputs

Every case directory retains `command.txt`, `stdout_stderr.log`, `exit_status.txt`, `manifest.csv`, four per-rank timing CSVs, one max-rank raw timing CSV, four per-rank slice CSVs, one max-rank slice CSV, RCCL logs and an artifact inventory.

The result root also retains a case manifest, a master log, platform facts, source snapshot, hashes, analysis input audit, case-level summary, candidate-reversal table, release-curve summary and Markdown analysis.

## Decision Rule

Treat a strong counterexample as supported only if all are true:

1. all relevant correctness checks pass;
2. five `COMM_ONLY` repetitions retain the isolated candidate ranking;
3. five H0 repetitions retain the opposite ranking;
4. the H0 benefit is at least 5%; and
5. per-slice data shows a coherent timing explanation.

Anything weaker remains a useful boundary observation but is not sufficient for the central paper claim.
