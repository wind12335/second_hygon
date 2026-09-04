# §Mechanism — English Skeleton v1（2026-09-04）

> **状态**：兑现 09-05 承诺（提前）。海光侧拥有的段落已写全；NVIDIA 的 nsys 段落进四个标记槽位 `[NSYS-SLOT-1..4]`。§M 为节号占位。M.0 的术语为合并稿锁定术语，双方沿用不另造。
> **依据**：`paper/JM锚点小节_draft_v2_20260904.md`（锚点注册表 JM-H1..H8）+ NVIDIA 配位稿 db28c77。所有数字已在锚点表对回 result file。

---

## M.0 Terminology (locked for the merged draft)

Five-stage chain:

**sender accepted (S1) → remote visible (S2) → notification satisfied (S3) → consumer-legal release R_i (S4) → actual consume (S5)**

- **consumer-legal release R_i** — the earliest instant the consumer is permitted to consume slice *i*; determined by stream dependencies, credit gating, and wait placement.
- The three regularities of this section: **D1 release semantics · D2 protocol tax · D3 rank-scaling bifurcation**.
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

The evidence organizes into three regularities — release semantics (D1, §M.2), protocol tax (D2, §M.3), rank-scaling bifurcation (D3, §M.4). They are claims about signs and structure. Magnitudes are substrate-specific: this is the mechanism-level form of the paper's meta-conclusion (law forms transfer, parameters calibrate per substrate), and it is why the selection layer probes rather than calibrates (§M.6).

## M.2 D1 — release semantics: local completion is not permission to consume

**Completion ≠ consumability (JM-H1, direct).** In DUSHMEM admission runs on K500SM_AI (14/14 cases, full payload verification), the consumer-side release trails communication-stream completion by 18–23µs for put–signal and 10–15µs for fcollect. An on-stream primitive guarantees stream order, not peer visibility. The moment a local stream reports the collective done is therefore not the moment consumption may begin.

**Wait placement decides S3→S4 (JM-H2; semantics direct, effect from paired e2e).** Where the consumer's wait lives determines how early R_i can occur. In d1 the wait sits on the compute stream, so the first release is postponed until all transfers complete — the overlap path effectively serializes. Moving the wait to a dedicated stream with an event bridge (d1w) restores release placement but costs the bridge: on K500SM_AI and A800 the d1w–d1 difference stays within ±3% (predictions P6 and P10, preregistered separately on the two substrates, both missed). On BW1000 at eight ranks the same cell (N2048/q8) breaks out of that band: d1w runs 11.05% slower than d1 (median of five runs, 17421.7 vs 15688.2µs; +4.1% at four ranks). Wait placement is a necessary condition whose price becomes visible as rank count grows, not a free repair.

> **[NSYS-SLOT-1 — NVIDIA]** Trace 04 (d1, N4096/q8): barrier_on_stream makes the consumer-stream block kernel-visible — 136 instances, p50 9.9µs, p90 338µs, max 13,106µs, six spikes ≥1ms; plus the d1 first-release postponement visualization.

**The S4→S5 ladder (JM-H3; derived on K500SM_AI, two rungs direct on A800).** Per-slice release curves on K500SM_AI separate release→GEMM latency into family-level rungs: d1 5.4µs, r1 10–47µs, d0/r0 16.2µs. On A800, kernel timestamps promote two rungs to direct — the d0 rung at p50 10.5µs (60 stable pairs) and the r1 fast-coupled mode at p10 11.5–15.8µs (512 stable pairs per trace). The rung order holds across substrates; the values do not coincide. The d1 rung resists clean extraction on A800 (4 of 84 pairs stable), and we keep the derived 5.4µs rather than force a number. The claim is the rung *order*, set by each family's synchronization semantics; the magnitudes are calibration.

> **[NSYS-SLOT-2 — optional, NVIDIA to place]** The r1 p50 of 510.8µs is pipeline occupancy, not a rung value; if used, it belongs with the §M.3 serialization evidence.

## M.3 D2 — protocol tax: every slice pays a fixed handshake

**First-order, per-slice, depth-independent (JM-H4).** Chunking one collective into q slices trades a single synchronization for q handshakes. Two K500SM_AI results fix the tax as first-order and per-slice fixed. First, sweeping slot-reuse depth wm ∈ {1,2,4} rounds moves no anchor cell by more than 3.6pt; the explosion cell (N2048/q16), where the preregistered rescue prediction was largest, is untouched on K500SM_AI (+0.0pt) and inside 3-rep noise on BW1000 (+2.9pt, the single verdict disagreement across substrates, recorded in the anchor table); only the transitional main cell responds at all, and by −2.7pt / −3.6pt. The predicted rescue ordering failed. Second, an RCCL allgather control decomposes as 3938 + 38.1·q µs with no N dependence — the per-slice synchronization cost is directly visible. Depth is not the first-order carrier of the d-family penalty.

On A800, kernel traces of the r1 family show the same regularity at kernel granularity:

> **[NSYS-SLOT-3 — NVIDIA]** Traces 02/03, isomorphic loser/winner pair: N2048/q8 — per-slice AG p50 107.6µs with spikes to 1,211µs; AG-end to next-slice GEMM p50 510.8µs, equal to the per-slice GEMM (493µs): alternation in practice, concurrent coverage 3.4%. N512/q8, same structure: coverage 12.1%. Where the tax exceeds per-slice compute, communication and computation alternate; where it does not, they overlap.

**Depth is a threshold variable (JM-H5).** Depth matters only below one round in flight. The A800 w axis supplies the harm zone: capping in-flight slices at w1/w2 costs 30–40% (both windows sit far below one round), recovery is visible from half a round (w8 at q16), and unbounded w0 is the platform each curve returns to.

> **[NSYS-SLOT-4 — NVIDIA]** Kernel-level triple, traces 00 / 01-w1 / 08-w8 (1984×1984 each): concurrent-coverage 10.7% (w0) → 3.2% (w1) → 11.3% (w8, back to the platform within 0.5pt) while per-AG p50 stays at 45–47µs across all three — depth reshapes pipeline structure without touching unit cost.

The Hygon wm axis supplies the flat segment at and above one round (1/2/4 rounds; main cell −2.7pt on K500SM_AI, −3.6pt on BW1000-np8, remaining cells within 1pt on K500SM_AI) and replicates the cell ordering exactly: boom > sub > main > win is conserved from K500SM_AI to BW1000 while the absolute tax level shifts ×2.6. Structure transfers; calibration does not. The two axes differ in semantics — in-flight cap versus slot reuse — and are reported side by side.

**Falsifiable exit.** Whether the tax lives in signal count, stream switches, or remote writes is testable: a grouped-signaling variant amortizes the per-slice signal while metering per-slice signal counts and bytes. If the tax thins, it tracks signal count; if not, it tracks the other two. Perseus owns the amortization idea [ref]; we preregister the intervention (DUSHMEM admission branch) with the metering in place before launch and report either outcome.

## M.4 D3 — rank-scaling bifurcation: two power laws move the boundary

**Two scaling laws (JM-H6, direct).** The families pay rank count differently. d0 issues one fcollect whose floor scales with np: 3.35→7.09 ms from four to eight ranks (2.12×). d1 pushes every slice to np−1 peers, so its cost scales with q·(np−1): when the rank count doubles, the d1 cost multiplies 5.16× at N512/q8 (super-linear collapse) but only 1.55× at N4096/q8. The crossing between the two cost curves therefore moves as ranks are added. At (N4096, q8) the sign flips outright: d1 is 14.2% slower than d0 at np4 and 16.1% faster at np8 — same binary in both runs.

**One mechanism, two projections (JM-H8).** The bifurcation projects onto the boundary genealogy at N8192/q16: K500SM_AI +4.6 (near its crossing, which empirically sits at N≈8192); BW1000-np8 +26.2 (same sign, amplified; at np4 the q-axis crossing lies inside the tested grid); A800 +16.5, with bracketing coverage to N=32768 finding no crossing — hence lower bounds N\*(q8) > 16384 and N\*(q16) > 32768 [ref P19]. Substrates with an in-domain crossing and substrates without are two projections of one cost geometry, not two phenomena. We accordingly fit per-substrate forms and no global closed form.

*Honesty note.* A800 is a single four-rank point. D3's direct evidence is BW1000's np4→np8 flip; A800 contributes the far-side projection, not scaling data. (A card-migration replication series on A800 — 163 launches, 8/8 serial-wins, regret ≤0.078% — strengthens that row's stability; it is not np evidence.)

## M.5 What we do not claim

S2→S3 — remote visible → notification satisfied — holds no direct trace evidence on either side. What exists is a proxy chain: the API contract of on-stream primitives, the A800 admission floor at 47µs (derived), and the completion→release gaps of §M.2, which measure the whole span rather than its interior. We mark S2→S3 pending and treat it as the open problem this paper leaves after X-Stage. Two further limits: rung values are family medians, not per-cell constants; and the w and wm axes are never merged into one curve.

## M.6 Why this forces a probe

The three regularities close the paper's argument. (i) Isolated timings cannot see S3→S5: release semantics, gating, and joint contention exist only inside the dependent path — on BW1000 the isolated winner, COMM_ONLY, leads by 70–90% in every cell, while end-to-end gaps run 2–38%. (ii) Structure transfers but calibration does not: three substrates yield three different best zero-knowledge defaults (A800 → always-d0; K500SM_AI → always-r1; BW1000 → none; its best default still costs 9–22% p95 regret). (iii) Boundaries move with rank count, so even a valid calibration ages with deployment shape. The generic answer is not a better prior but a cheap robust probe: at k≥3, p95 regret stays between 0.00% and 0.06% on all three substrates.

---

## Table M1 — condensed anchor registry (from JM-H2 v2, to be typeset)

| anchor | stage | level | one-line evidence |
|---|---|---|---|
| JM-H1 | S1→S3 (span) | direct | release trails comm-complete +18–23µs (put–signal) / +10–15µs (fcollect), 14/14 |
| JM-H2 | S3→S4 | direct+derived | d1 wait on compute stream serializes; d1w ±3% (K500SM_AI, A800), +11.05% at BW1000-np8 |
| JM-H3 | S4→S5 | derived (+2 direct A800) | rungs d1 5.4 / r1 10–47 / d0,r0 16.2µs; A800 d0 10.5, r1 11.5–15.8µs |
| JM-H4 | S3→S4 cost | direct | wm-insensitivity; allgather 3938+38.1·q µs; A800 coverage 3.4% vs 12.1% |
| JM-H5 | credit gating | direct | harm zone <1 round, flat ≥1 round; cell order conserved, tax ×2.6 |
| JM-H6 | transport structure | direct | d0 ∝ np vs d1 ∝ q·(np−1); (4096,q8) flips with np |
| JM-H7 | S5 | proxy (→direct) | release_slices paired with kernel timestamps |
| JM-H8 | e2e projection | direct | N8192/q16: +4.6 / +16.5 / +26.2; P19 lower bounds |

## Figures

- **F-JM1** (Hygon): wm flat curves, K500SM_AI vs BW1000, four cells normalized.
- **F-JM2** (Hygon): the two scaling laws, np4→np8, with the (4096,q8) flip annotated.
- **F-JM3** (NVIDIA): ladder + first-release postponement + barrier spikes (log or broken axis; caption notes six ≥1ms spikes of 136).
- **F-JM4** (Hygon; A800 coverage sentence reviewed by NVIDIA): lineage cell N8192/q16, three columns + lower-bound arrow.

## Slot inventory for the merged draft

NSYS-SLOT-1 (M.2, trace 04 barrier) · NSYS-SLOT-2 (M.2, optional r1 clarification) · NSYS-SLOT-3 (M.3, loser/winner pair) · NSYS-SLOT-4 (M.3, w-triple). All numbers inside slots are quoted from the sealed-trace extraction (`docs/nsys锚点提取_20260904/anchor_extraction_summary.json`, second_nvidia repo).
