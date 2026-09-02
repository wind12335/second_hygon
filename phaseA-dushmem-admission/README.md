# DUSHMEM Phase A Admission

This directory is an isolated, correctness-first experiment for:

```text
K500SM_AI / gfx928 / 4 GPUs / PCIe
```

It does not modify RCCL. It does not claim AG-GEMM acceleration. Its purpose is to decide whether
the DUSHMEM primitives needed by a later DUSHMEM AG-GEMM data path are semantically correct and
measurable on this exact platform.

The build deliberately uses the DTK 26.04 two-stage `hipcc -fgpu-rdc` flow. This is required for
the DUSHMEM v3.2.5 static device archive on this host; a generic CMake HIP link is not equivalent.

The tested paths are:

```text
put_signal:
  every rank pushes an epoch-tagged payload to every other rank
  -> remote ready signal
  -> local stream waits for all producer signals
  -> GPU checksum validates every received payload

fcollect:
  every rank invokes DUSHMEM fcollect on the same stream
  -> GPU checksum validates the AllGather-equivalent destination layout
```

The completed Phase A result interpretation and the precise Phase B AG-GEMM plan are in
`PhaseA_DUSHMEM原语准入实验结果与下一阶段设计.md`.

`--credit 1` adds bounded-slot reuse. A receiver emits a monotonically increasing credit only after
the payload has passed its GPU checksum. The next producer reuse waits for that credit. This avoids
the ABA problem caused by reusing a binary ready flag.

Run the short compiler and semantic smoke test first:

```bash
cd /root/private_data/lyc/2ndpaper/dushmem_phase4_admission
./scripts/run_admission.sh --smoke
```

Run the longer formal matrix only after the smoke matrix passes:

```bash
./scripts/run_admission.sh --formal
```

Each invocation creates a new result directory below `results/`. It refuses to overwrite old
results. Important files are:

```text
manifest.csv                         every requested case and its process exit status
cases/<case>/command.txt             exact replay command
cases/<case>/stdout_stderr.log       complete MPI and DUSHMEM output
cases/<case>/exit_status.txt         process exit status
cases/<case>/raw/rank_*.csv          one raw row per rank and epoch
cases/<case>/raw/capability_*.csv    P2P and dushmem_ptr capability observations
analysis/global_iteration_max.csv    max-rank time per epoch
analysis/case_summary.csv            PASS/FAIL admission summary
analysis/admission_report.md         concise human-readable summary
```

An admission case is PASS only when the process exits successfully, every requested epoch has four
rank records, and every rank verifies the complete payload without an epoch or checksum failure.
