# §Mechanism — English Skeleton v2（2026-09-04）

> **状态**：v2 = v1 + 对方【19】【20】内审采纳。修订四类：①A800 trace 数字全部按对方配位稿 v2.1 换血并标 provisional（corrected table 前只作 descriptive 层）；②H3 A800 行降 proxy/pending re-extraction；③H8 下界升 P21（>32768/>65536）、H5 e2e 口径换 P24（w\*=4）；④D1 共同决定、D2 局部仿射边际项+删"必须在线探"直推、D3 改名 rank-scaling divergence+删单向预测。本骨架为 Sol① 统一英文执笔的底稿（单一集成主笔制，M.0 术语锁与 D1/D2/D3 结构不动——我方三条件见锚点小节 v3 §7）。
> **依据**：`paper/JM锚点小节_draft_v3_20260904.md`（锚点注册表 JM-H1..H8）+ NVIDIA 配位稿 v2.1（3ceba83d）。我方数字已在锚点表对回 result file；对方数字以 `docs/nsys锚点提取_20260904/anchor_extraction_summary.json`（schema v2）为准。

---

## M.0 Terminology (locked for the merged draft)

Five-stage chain:

**sender accepted (S1) → remote visible (S2) → notification satisfied (S3) → consumer-legal release R_i (S4) → actual consume (S5)**

- **consumer-legal release R_i** — the earliest instant the consumer is permitted to consume slice *i*; determined by stream dependencies, credit gating, and wait placement.
- The three regularities of this section (substrate-conditioned mechanism regularities, not laws): **D1 release semantics · D2 protocol tax · D3 rank-scaling divergence**.
- Causal-verb discipline: traces alone get *shows* / *is consistent with*; *causes* / *explains* only for API contract + intervention + e2e multi-source closure.
- Path codes follow the benchmark: **d0** (fcollect serial), **d1** (per-slice put–signal overlap), **d1w** (d1 with the wait moved to a dedicated stream), **r0 / r1** (RCCL serial / event overlap).
- Depth is measured in rounds of *q* slices: the A800 axis **w** caps in-flight slices; the Hygon axis **wm** reuses slots across rounds. The two axes are reported side by side, never merged.
- P(X vs Y) = (t_X − t_Y)/t_Y; positive means X is slower.
- Substrates: **K500SM_AI** (4 GPUs), **BW1000** (8 GPUs), **A800** (4 GPUs).
- Evidence levels: direct / derived / proxy / pending.

## M.1 The five-stage chain, and what a bandwidth model cannot see

A bandwidth model of an all-gather–GEMM stage contains two events: the collective finishes moving data, and the GEMM consumes it. Between them sit at least three more stages the model does not represent. We make them explicit:

- **S1 sender accepted** — the communication API accepts the write request for slice *i* (putmem_signal_on_stream enqueued; NCCL group enqueued).
- **S2 remote visible** — slice *i* is readable in the peer's symmetric heap.
- **S3 notification satisfied** — the signaling predicate holds: signal_wait_until returns, or the NCCL event is set.
- **S4 consumer-legal release R_i** — the earliest instant the consumer may consume slice *i*.
- **S5 actual consume** — the slice GEMM begins.

X-Stage established S1→S2 [ref]. This section carries S3→S5, the stages that decide whether a completed transfer becomes realized overlap. Each claim below is made only where the claiming side holds direct evidence; every anchor row in Table M1 carries its level, preregistration ID, and result file.

The evidence organizes into three regularities — release semantics (D1, §M.2), protocol tax (D2, §M.3), rank-scaling divergence (D3, §M.4). They are claims about signs and structure. Magnitudes are substrate-specific: this is the mechanism-level form of the paper's meta-conclusion (regularity forms transfer, parameters calibrate per substrate). What this section claims is only that parameters do not transfer across substrates; the further case for probing online rather than calibrating offline is made in the selection section [ref], and §M.6 only recalls it.

## M.2 D1 — release semantics: local completion is not permission to consume

**Completion ≠ consumability (JM-H1, direct).** In DUSHMEM admission runs on K500SM_AI (14/14 cases, full payload verification), the consumer-side release trails communication-stream completion by 18–23µs for put–signal and 10–15µs for fcollect. An on-stream primitive guarantees stream order, not peer visibility. The moment a local stream reports the collective done is therefore not the moment consumption may begin.

**Wait placement decides S3→S4 (JM-H2; semantics direct, effect from paired e2e).** Where the consumer's wait lives determines how early R_i can occur. In d1 the wait sits on the compute stream, so the first release is postponed until all transfers complete — the overlap path effectively serializes. Moving the wait to a dedicated stream with an event bridge (d1w) restores release placement but costs the bridge: on K500SM_AI and A800 the d1w–d1 difference stays within ±3% (predictions P6 and P10, preregistered separately on the two substrates, both missed). On BW1000 at eight ranks the same cell (N2048/q8) breaks out of that band: d1w runs 11.05% slower than d1 (median of five runs, 17421.7 vs 15688.2µs; +4.1% at four ranks). Wait placement is a necessary condition whose price becomes visible as rank count grows, not a free repair.

> **[NSYS-SLOT-1 — NVIDIA; v2.1 numbers, provisional]** Trace 04 (d1, N4096/q8): barrier_on_stream makes consumer-stream blocking kernel-visible — 34 instances per rank (aggregate across four ranks; one representative sealed trace per cell), p50 7.8–15.3µs, per-rank max 0.34/9.0/8.2/13.1ms: heavy-tailed and rank-asymmetric. Consistent with the API/source contract of wait-induced blocking; the trace alone does not claim to prove legal release. Plus the d1 first-release postponement visualization.

**The S4→S5 ladder (JM-H3; derived on K500SM_AI; A800 proxy/pending re-extraction).** Per-slice release curves on K500SM_AI separate release→GEMM latency into family-level rungs: d1 5.4µs, r1 10–47µs, d0/r0 16.2µs. The A800 kernel-timestamp row is held at proxy: the initial extraction (two direct rungs) was withdrawn by NVIDIA's internal audit — the pairing semantics is unproven (which AG kernel maps to which slice GEMM cannot be confirmed; the stability threshold was post hoc) and the v1 numbers were cross-device contaminated. Device-0 audit counts (d0 p50 14.2µs over 15 pairs; r1 fast mode p10 16.8/14.0µs) stand as descriptive audit numbers, not rung values; what the timeline supports is regime-level structure only — a coarse-serial and a fine-pipelined mode both exist. The d1 rung yields a single stable device-0 pair, so the derived 5.4µs stands rather than a forced number. The claim is the rung *order* on K500SM_AI, set by each family's synchronization semantics; magnitudes are calibration and are not claimed to coincide across substrates.

> **[NSYS-SLOT-2 — optional, NVIDIA to place; v2.1 number, provisional]** The r1 p50 of 514.7µs is pipeline occupancy, not a rung value; if used, it belongs with the §M.3 serialization evidence.

## M.3 D2 — protocol tax: chunking trades one synchronization for q handshakes

**First-order in q, depth-independent (JM-H4).** Chunking one collective into q slices trades a single synchronization for q handshakes — aggregate per-slice protocol work, not a constant tax per slice. Two K500SM_AI results fix the tax as first-order in q. First, sweeping slot-reuse depth wm ∈ {1,2,4} rounds moves no anchor cell by more than 3.6pt; the explosion cell (N2048/q16), where the preregistered rescue prediction was largest, is untouched on K500SM_AI (+0.0pt) and inside 3-rep noise on BW1000 (+2.9pt, the single verdict disagreement across substrates, recorded in the anchor table); only the transitional main cell responds at all, and by −2.7pt / −3.6pt. The predicted rescue ordering failed. Second, an RCCL allgather control decomposes as a local affine fit, 3938 + 38.1·q µs, with no N dependence — within the tested q range the marginal term is ≈38.1µs per slice; the fit supports a per-slice marginal over the measured range, not a universal constant. Depth is not the first-order carrier of the d-family penalty.

On A800, kernel traces of the r1 family show the same regularity at kernel granularity:

> **[NSYS-SLOT-3 — NVIDIA; v2.1 numbers, provisional]** Traces 02/03, isomorphic loser/winner pair (per card: 121 slice GEMMs and 129 AG_LL kernels, one representative sealed trace per cell): N2048/q8 — per-slice AG p50 108µs with spikes to 1,211µs; AG-end to next-slice GEMM p50 514.7µs, equal to the per-slice GEMM (495µs): alternation in practice, concurrent coverage 10.7%. N512/q8, same structure: coverage 37.5% (3.5×). Where the tax exceeds per-slice compute, communication and computation alternate; where it does not, they overlap.

**Depth is a threshold variable (JM-H5).** Depth matters only below one round in flight. The A800 w axis supplies the harm zone, read from the P24 e2e verdict (93/93, randomized blocks): capping in-flight slices at w1 compresses the B2/B1 overlap gains to within ±0.2% of the serial baseline, while w4 restores them to 22.9–32.2% — w\*=4 is the operational threshold for this shape and substrate, not a universal constant. Recovery is visible from half a round (w8 at q16, half the slices in flight), and unbounded w0 is the platform each curve returns to.

> **[NSYS-SLOT-4 — NVIDIA; v2.1 numbers, provisional]** Kernel-level triple, traces 00 / 01-w1 / 08-w8 (496×496 per card, one representative sealed trace per cell): concurrent coverage 18.74% (w0) → 0.00% (w1 — concurrency collapses entirely, strict lockstep) → 18.65% (w8, back to the platform within 0.1pt) while per-AG p50 stays at ~45–47µs across all three — depth reshapes pipeline structure without touching unit cost.

The Hygon wm axis supplies the flat segment at and above one round (1/2/4 rounds; main cell −2.7pt on K500SM_AI, −3.6pt on BW1000-np8, remaining cells within 1pt on K500SM_AI) and replicates the cell ordering exactly: boom > sub > main > win is conserved from K500SM_AI to BW1000 while the absolute tax level shifts ×2.6. Structure transfers; calibration does not. The two axes differ in semantics — in-flight cap versus slot reuse — and are reported side by side.

**Falsifiable exit.** Whether the tax lives in signal count, stream switches, or remote writes is testable: a grouped-signaling variant amortizes the per-slice signal while metering per-slice signal counts and bytes. If the tax thins, it tracks signal count; if not, it tracks the other two. Perseus owns the amortization idea [ref]; we preregister the intervention (DUSHMEM admission branch) with the metering in place before launch and report either outcome.

## M.4 D3 — rank-scaling divergence: two scaling laws move the boundary

**Two scaling laws (JM-H6, direct).** The families pay rank count differently. d0 issues one fcollect whose floor scales with np: 3.35→7.09 ms from four to eight ranks (2.12×). d1 pushes every slice to np−1 peers, so its cost scales with q·(np−1) — but the multiplier when ranks double is workload-dependent: 5.16× at N512/q8 (super-linear collapse), 1.40× at N2048/q8, 1.55× at N4096/q8 (sub-linear). The crossing between the two cost curves therefore moves as ranks are added, and the movement is not single-direction: small cells push the crossing to larger N, large cells pull it back. At (N4096, q8) the sign flips outright: d1 is 14.2% slower than d0 at np4 and 16.1% faster at np8 — same binary in both runs. A caveat on absolute cross-np ratios: the GEMM_ONLY control at the same (N,q) also inflates ~2× from np4 to np8 with no telemetry explanation, so within-batch pairings are trusted while cross-np absolute multipliers carry system-state risk; no direction of further crossing movement is predicted (np has only two levels).

**One mechanism, two projections (JM-H8).** The divergence projects onto the boundary genealogy at N8192/q16 as three observed same-cell anchors plus one missing configuration: K500SM_AI +4.6 (near its crossing, which empirically sits at N≈8192); BW1000-np8 +26.2 (same sign, amplified; at np4 the q-axis crossing lies inside the tested grid — the BW1000-np4 N8192 cell is the missing configuration); A800 +16.5 (early matched/formal evidence, not a P19/P21 randomized-block cell — the figure caption states provenance), with randomized-block bracketing extended by P21 (45/45) to N=32768 (q8) and N=65536 (q16) finding no crossing — hence lower bounds N\*(q8) > 32768 and N\*(q16) > 65536, right-censored, magnitudes converging non-monotonically [ref P21]. Substrates with an in-domain crossing and substrates without are two projections of one cost geometry, not two phenomena. We accordingly fit per-substrate forms and no global closed form.

*Honesty note.* A800 is a single four-rank point. D3's direct evidence is BW1000's np4→np8 flip; A800 contributes the far-side projection, not scaling data. (A card-migration replication series on A800 — 163 launches, 8/8 serial-wins, regret ≤0.078% — strengthens that row's stability; it is not np evidence.)

## M.5 What we do not claim

S2→S3 — remote visible → notification satisfied — holds no direct trace evidence on either side. What exists is a proxy chain: the API contract of on-stream primitives, the A800 admission floor at 47µs (derived), and the completion→release gaps of §M.2, which measure the whole span rather than its interior. We mark S2→S3 pending and treat it as the open problem this paper leaves after X-Stage. Two further limits: rung values are family medians, not per-cell constants; and the w and wm axes are never merged into one curve.

## M.6 Why this forces a probe

The three regularities bound what any bandwidth-only model can deliver. (i) Isolated timings cannot see S3→S5: release semantics, gating, and joint contention exist only inside the dependent path — on BW1000 the isolated winner, COMM_ONLY, leads by 70–90% in every cell, while end-to-end gaps run 2–38%. (ii) Structure transfers but calibration does not: three substrates yield three different best zero-knowledge defaults (A800 → always-d0; K500SM_AI → always-r1; BW1000 → none; its best default still costs 9–22% p95 regret). What follows is that parameters cannot be moved across substrates — nothing stronger: within one substrate an offline calibration can even short-circuit (K500SM_AI always-r1, 9/9), and whether a deployment should probe online rather than calibrate offline is decided by the hidden-residual and probe-value arguments of the selection section [ref P25], not re-argued here. (iii) Boundaries move with rank count, so even a valid calibration ages with deployment shape. The generic answer is not a better prior but a cheap robust probe: at k≥3, p95 regret stays between 0.00% and 0.06% on all three substrates.

---

## Table M1 — condensed anchor registry (from JM-H2 v2, to be typeset)

| anchor | stage | level | one-line evidence |
|---|---|---|---|
| JM-H1 | S1→S3 (span) | direct | release trails comm-complete +18–23µs (put–signal) / +10–15µs (fcollect), 14/14 |
| JM-H2 | S3→S4 | direct+derived | d1 wait on compute stream serializes; d1w ±3% (K500SM_AI, A800), +11.05% at BW1000-np8 |
| JM-H3 | S4→S5 | derived (A800 proxy/pending) | rungs d1 5.4 / r1 10–47 / d0,r0 16.2µs (K500SM_AI derived); A800 = device-0 audit counts only, regime-level structure |
| JM-H4 | S3→S4 cost | direct | wm-insensitivity; allgather local affine 3938+38.1·q µs (tested q range); A800 coverage 10.7% vs 37.5% (provisional) |
| JM-H5 | credit gating | direct | harm zone <1 round, flat ≥1 round; P24 w\*=4 operational threshold; cell order conserved, tax ×2.6 |
| JM-H6 | transport structure | direct | d0 ∝ np vs d1 ∝ q·(np−1), multiplier workload-dependent; (4096,q8) flips with np; crossing moves both directions |
| JM-H7 | S5 | proxy (→direct) | release_slices paired with kernel timestamps |
| JM-H8 | e2e projection | direct | N8192/q16: three anchors +4.6 / +16.5 / +26.2 (+ BW1000-np4 missing cell); P21 lower bounds N\*(q8)>32768, N\*(q16)>65536 |

## Figures

- **F-JM1** (Hygon): wm flat curves, K500SM_AI vs BW1000, four cells normalized.
- **F-JM2** (Hygon): the two scaling laws, np4→np8, with the (4096,q8) flip annotated.
- **F-JM3** (NVIDIA): ladder + first-release postponement + barrier spikes — **deferred until the corrected table** (per-device/iteration-aware re-extraction); drawn from paper-grade numbers only, nothing that a one-line recomputation could overturn.
- **F-JM4** (Hygon; A800 coverage sentence reviewed by NVIDIA): lineage cell N8192/q16, three observed columns + BW1000-np4 missing-cell marker + P21 lower-bound arrows (to 65536/q16, 32768/q8); A800 column caption states provenance (early matched/formal evidence, not a P21 randomized-block cell).

## Slot inventory for the merged draft

NSYS-SLOT-1 (M.2, trace 04 barrier) · NSYS-SLOT-2 (M.2, optional r1 clarification) · NSYS-SLOT-3 (M.3, loser/winner pair) · NSYS-SLOT-4 (M.3, w-triple). All numbers inside slots are quoted from the sealed-trace re-extraction, schema v2 (`docs/nsys锚点提取_20260904/anchor_extraction_summary.json`, second_nvidia repo; device-0, per-card audit counts) and are **provisional** until the corrected table (NVTX/iteration-aligned) — until then they appear in the main text at descriptive level only, µs values confined to footnotes.
