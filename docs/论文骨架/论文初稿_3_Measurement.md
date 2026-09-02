# Paper draft — §3 Measurement（v0，2026-09-02）

> 写作口径：英文正文（论文语言），数据位用 ⟨formal:…⟩ 标注等全矩阵后回填。
> 平台命名铁律：`K500SM_AI / gfx928 / 4 GPUs / PCIe`（DTK 26.04, RCCL 2.22.3, DUSHMEM 3.2.5）
> 与 `RTX 4090 / sm_89 / 4 GPUs / PCIe`（NCCL/NVSHMEM 版本待环境报告）。绝不写旧机器型号。

## 3. Measurement: A Unified Substrate for Isolated and End-to-End Ranking

### 3.1 Why a new benchmark (design forces)

Existing collective benchmarks (nccl-tests / rccl-tests) answer "which primitive
configuration moves bytes fastest in isolation". Overlap-strategy selection
needs three more things they cannot provide: (i) a data-dependent consumer
(GEMM) on the same buffers the collective writes; (ii) per-slice *release*
observability — the moment a slice becomes consumable, not the moment the
transport retires; and (iii) both transport substrates (collective library and
remote-memory API) behind *identical* buffer layout, slicing, and correctness
checks, so that rankings differ for substantive reasons rather than
implementation accident. Phase-1 measurements confirmed the premise negatively:
on both substrates, isolated tuning of the collective alone approaches the
oracle bandwidth, leaving no headroom for a bandwidth-only contribution — and
yet, as §5 shows, the isolated ranking fails to predict end-to-end winners.

### 3.2 Benchmark structure and path taxonomy

All experiments run in a single binary per platform. Every rank owns a row
block `x_local` of an AllGather-GEMM pipeline (m_local = K = 2048, fp32;
N ∈ {512, 2048, 4096}; payload per rank 16 MiB). The gathered operand lives in
one symmetric heap (`dushmem_malloc` on the DCU side, its NVIDIA-port
equivalent on the 4090), so RCCL- and remote-memory-path experiments feed the
*same addresses* to the *same* vendor GEMM (rocBLAS / cuBLAS) — buffer
provenance is held fixed across the comparison.

The workload is sliced into q ∈ {2, 4, 8} (boundary probe q = 16 at N = 2048)
slices, and eleven paths cover the design space (naming used throughout):

| path | family | semantics |
|---|---|---|
| `comm` | RCCL | sliced AllGather, communication only (isolated reference) |
| `gemm` | — | sliced GEMM, compute only (fragmentation reference) |
| `r0` | RCCL | bulk: full AG, then full GEMM |
| `rs` | RCCL | sliced serial: AG(i) → GEMM(i) → AG(i+1) |
| `r1` | RCCL | sliced overlap: AG(i) → event → GEMM(i) concurrent |
| `fc` | DUSHMEM | fcollect, communication only (isolated reference) |
| `dc` | DUSHMEM | sliced put-with-signal, wait placed on a dedicated stream — measures pure transport plus release timing |
| `d0` | DUSHMEM | bulk fcollect, then full GEMM |
| `ds` | DUSHMEM | sliced put-signal serial (fair serial baseline for d1) |
| `d1` | DUSHMEM | sliced put-signal overlap, ready-flag driven (credit slot reuse, monotone epochs — protocol validated in the Phase-A admission suite) |
| `d1w` | DUSHMEM | as d1, but consumer waits on the dedicated wait stream with an event bridge into the compute stream |

Two release-semantics design variables are thus *separated by construction*:
where the consumer waits (compute stream vs. dedicated stream: d1 vs. d1w) and
how completion is delivered (collective-retire vs. per-slice signal). RCCL
configurations C0 (default) and C2 (Ring/Simple, 8 channels) ride the `comm`
and `r1` carriers, carrying the resource axis into the unified matrix.

Fair-baseline rule (learned the hard way, §3.6): sliced paths are only
compared against equally sliced baselines (`ds`, `rs`); the bulk baselines
(`r0`, `d0`) anchor the fragmentation cost via `gemm`.

### 3.3 Instrumentation: t_issue / t_done / t_release / e2e

Per iteration, each path records device-side timestamps on the streams that
own the work: issue time on the communication stream (`t_issue`), transport
retirement (`t_done`), and — the quantity conventional benchmarks lack — the
per-slice **release time** (`t_release`): when the consumer-side wait for
slice i actually resolves, together with the GEMM interval it gates
(`t_gemm_start/end`). End-to-end latency (`e2e`) is taken on the compute
stream from iteration start to final GEMM retirement. Aggregation is
max-rank (slowest participant), matching how a collective operation actually
bounds a step; p50 is the headline statistic with p05/p95 and across-rep
stability reported alongside. The gap between `t_release` and the
corresponding `t_done` slice is the *release lag* — the microsecond-scale
price of "completed" versus "consumable" that Phase A measured at 10–23 us
per slice on put-signal / fcollect paths.

### 3.4 Correctness and statistics

Every iteration validates the full payload elementwise (abs 1e-2 / rel 1e-4)
against an in-process reference (full AG + full GEMM of the same operands).
Each (path, config, N, q) cell is an independent process group (5 process
launches for formal runs), 20 warmup + 80 measured iterations, barrier-aligned
per iteration. Claims of ranking differences are tested with a
tie-corrected Mann–Whitney U (normal approximation, pure-stdlib
implementation, no SciPy dependency on the measurement host) across the five
replicate medians, and additionally require direction consistency in 5/5
replicates. We call a reversal *strong* when the sign flips with at least 5%
magnitude under both criteria.

### 3.5 Substrate capability vector

For each platform we extract a capability vector B from isolated cells only:
{collective t_done (comm), one-sided fcollect t_done (fc), put-signal t_done
(dc), release-lag profile, interference shape}. B is what a *deployable*
selector may observe — none of it requires running the end-to-end matrix.
The paper's question is precisely how much of the end-to-end ranking B
determines, and where it misleads.

### 3.6 Provenance and the audit trail

Every run lands in an immutable timestamped root (command line, manifest,
platform facts, source snapshot + sha256, per-case stdout/exit status); runs
are never overwritten and failed cases are retained. This protocol earned its
keep during the campaign: a mid-campaign audit revealed that the `ds` serial
gate had compiled to an empty block — making `ds` operationally identical to
`d1` — invisible in outputs and timings alone. The defect, its detection, the
source fix (single-loop restructure with the gate enqueued *after* the slice's
GEMM, because stream-wait primitives snapshot the most recent event record at
enqueue time), and the dedicated re-measurement batch are documented as a
worked example of why serial baselines in overlap papers need exactly this
kind of operational audit. Formal results use the fixed binary for all
ds/d1/d1w cells.

---

### 回填清单（formal + dsfix 完成后）

- [ ] §3.2 表格里 d1w 一行的实测角色（修复验证 or 独立档位）
- [ ] §3.3 release lag 数值区间（formal 全矩阵 + q16）
- [ ] §3.4 全矩阵 MW p 值表引用（T1）
- [ ] §3.6 ds 缺陷日期与 dsfix 根目录名
- [ ] NVIDIA 侧对称描述（NVSHMEM API 名、NCCL 版本）等环境报告
