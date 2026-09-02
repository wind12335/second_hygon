# Paper draft — §1 Introduction + §2 Background & Related Work（v0，2026-09-02 深夜）

> 英文正文、⟨⟩ 回填位；贡献编号与骨架 C1-C4 一致；相关工作锚点取自
> `20260902_文献调研2025-2026/`（A/B/C/D 四份），引用条目仅列锚点论文名。

## 1. Introduction

Distributed LLM serving and training constantly move tensor-parallel state
between GEMMs: AllGather before the column-parallel multiply, ReduceScatter
after the row-parallel one, all-to-all dispatch in MoE layers. The standard
answer to the resulting communication exposure is *overlap* — slice the
tensor, pipeline the transfers against per-slice GEMMs — and the standard way
to pick among overlap implementations and configurations is to rank the
communication primitives by *isolated* bandwidth and pick the fastest.

This paper is about why that ranking fails, *where* it fails, and what to
measure instead. Prior work has already cracked the resource axis: tuning
NCCL parameters in isolation can lose end-to-end when compute is the
bottleneck [Lagom]. We show the failure is broader and more structured than
resource contention: **the completion semantics of the release mechanism —
when, and on which stream, a consumer may begin — is an independent selection
axis with its own reversals**, and the two axes interact through a balance
condition on the workload.

We build one benchmark binary that hosts eleven execution paths spanning both
stacks of a heterogeneous fleet — RCCL collectives and DUSHMEM one-sided
put-signal/fcollect on ⟨K500SM_AI/gfx928⟩ DCUs, NCCL and NVSHMEM on RTX 4090
— under identical addressing, identical GEMM, and per-iteration correctness.
On the 10-cell (N, q) matrix ⟨×5 reps⟩ we find two classes of ranking
reversal, each statistically significant (Mann-Whitney p<1e-4, 5/5 rep
direction):

1. **Config axis (within a strategy).** The tuned RCCL configuration C2
   (Ring/Simple/8 channels) wins the isolated ranking in *every* cell
   (+0.7–3.2%), yet loses end-to-end in exactly the cells where the workload
   is balanced (isolated comm/gemm ratio ≈ 1) and slicing is deep (q ≥ 8):
   −5.5% at N=2048/q8, sustained −5.4% at q16. The reversal is a *bounded
   resonance band*, not a monotone surface: three of our pre-registered
   monotone extrapolations failed — the effect *rebounds* at N=4096/q8
   (+1.8%), *saturates* at q16, and the protocol's cost curve *peaks* and
   falls. A three-sensor rule (q, ratio, gap) read purely from isolated
   microbenchmarks picks the end-to-end winner in 10/10 cells with zero
   worst-case regret, where the bandwidth-naive rule pays 5.5%.

2. **Strategy axis (across mechanisms).** Within the DUSHMEM family, the
   "overlapping" put-signal path D1 beats its serial sibling D0 by +21–30%
   at q≤4 on every size — and *loses* by −19% to −45% at q=8 on every size.
   The mechanism is wait placement: D1's consumer waits on the stream that
   carries the GEMMs, which defers the first release until after *all*
   transfers complete (14.4ms vs 1.4ms for the wait-stream placement at
   N=512/q8) — the "overlap" path structurally cannot overlap. Moving the
   wait to a dedicated stream (D1W) is a three-line change whose effect is
   ⟨dsfix: pending measurement⟩; the point is that no isolated bandwidth
   number distinguishes D1 from D0.

The measurement methodology is itself a contribution: every quantitative
claim above was pre-registered before its data landed (§12/§14 of the
experiment design; 13 sealed predictions, ⟨N⟩ evaluated), and the misses —
not the hits — are what forced the bounded-band model. We release the first
public systems characterization of DUSHMEM, the one-sided stack of ⟨Hygon
DCU⟩, including a discovered defect in our own serial baseline (empty
`if(serial)` gates) whose fix-and-remeasure cycle we document as a case study
in porting audit.

Contributions:
- **C1**: a reversal atlas on two axes with a balance law — the failure
  pattern of isolated ranking is itself predictable from isolated sensors;
- **C2**: completion-semantics cost measurement — release deferral, per-slice
  release curves, and the stretch of sliced-release protocols made
  quantitative and non-monotone;
- **C3**: a release-aware selector (three comparisons, mechanism-derived
  thresholds, honestly evaluated by nested leave-one-cell-out that it
  *cannot* be learned from one substrate);
- **C4**: the first public measurement of DUSHMEM, with the same harness as
  its NVIDIA counterpart for cross-substrate capability comparison.

## 2. Background and Related Work

### 2.1 Collective-GEMM overlap

Chunk-centric automatic overlap [Syncopate/AutoOverlap, OSDI'26] unifies CE
memcpy, NVSHMEM send/recv and SM load/store behind a chunk state machine and
compiles fused kernels; FlashOverlap [EuroSys'26] triggers communication
from inside the GEMM on tile readiness; production MoE/token pipelines
(TokenWeave, COMET) overlap dispatch/combine against expert GEMMs. These
works *build* one overlap mechanism per platform and evaluate it against
non-overlapped baselines; none treats the choice *among* release structures
as the object of study, and none measures whether the isolated ranking of
underlying primitives survives composition.

### 2.2 Collective tuning and strategy selection

The AutoCCL lineage tunes NCCL parameters; Lagom [arXiv 2602.20656] shows
aggressive isolated tuning can lose end-to-end under compute-bound
workloads; Theseus [SIGCOMM'26] switches collective schedules at runtime.
All three vary the *resource axis* (algorithm, protocol, channels, schedule)
with the completion mechanism fixed. Our config-axis reversal replicates
their class of effect in a new stack, but our central claim is the
independent *semantics axis*: same resources, same transport, different
release structure — ranking flips (§5.3). Our selector differs from online
adaptation [Theseus] in being deployment-time, measurement-derived, and
pre-registered.

### 2.3 One-sided APIs and completion semantics

NVSHMEM's anatomy — VMM paths, quiet semantics, signal ordering — has been
characterized on H100/H200 [Demystifying NVSHMEM, arXiv 2606.05951];
OpenSHMEM semantics comparisons exist at API level [Unified OpenSHMEM API,
arXiv 2607.08006]; ISPASS'25 quantifies interference costs of overlapping
kernels on NV and AMD GPUs. None converts completion semantics into a
*selection variable*; none covers Chinese DCU stacks. We contribute the
DUSHMEM counterpart measurement (C4) and the first cross-stack comparison
under one binary (C1/C3). The wait-placement pathology of §5.4 is, to our
knowledge, the first published case where *stream assignment of a wait* —
not the primitive, not the data volume — decides whether overlap exists at
all.

### 2.4 Cross-vendor runtimes

HetCCL [CF'26] interoperates heterogeneous collectives in mixed groups;
Unified OpenSHMEM unifies the API while explicitly not unifying semantics;
TileLink/Triton-distributed [MLSys'25] compiles multi-backend PGAS (NVSHMEM,
rocSHMEM, NCCL). We inherit the multi-backend abstraction philosophy from
our own prior work but take the opposite stance on selection: each substrate
selects its own strategy from its own capability vector; we never mix groups
or claim semantic equivalence across vendors. The capability vector B
(§4.2/§5.6) is the portable object, not the primitives.

---

### 回填清单
- [ ] dsfix 结果 → D1W 段落（§1 第 2 条反转的"修复"半句）
- [ ] NVIDIA 数据 → §1 平台句 + §2.3 双基座声明复核
- [ ] 预注册计数（"13 sealed, ⟨N⟩ evaluated"）
- [ ] Fig.1 = F1_money 前置引用
- [ ] 术语表（路径名 comm/r0/rs/r1/fc/dc/d0/ds/d1/d1w + C0/C2）→ §3 已有，Intro 提前 2-3 个
