# Paper draft — §4 Design: Release-Aware Selection（v0，2026-09-02 深夜）

> 与 §3/§5 同口径：英文正文、⟨formal/dsfix/NVIDIA⟩ 回填位。
> 方法名沿用骨架 §3 候选（SemSelect / ROSS），下文用 **SemSelect** 占位。

## 4. Design

### 4.1 Problem statement and constraints

Given a workload point (N, q) and a substrate, choose (strategy, config) ∈
{r0, rs, r1, d0, d1, d1w} × {C0, C2} to minimize e2e latency. Two constraints
shape the design:

**C-observability.** The selector may only use quantities obtainable from
*isolated microbenchmarks* — the same class of measurements a tuner already
collects (bandwidth sweeps, kernel timing). Per-cell e2e of the full path
matrix is the ground truth we evaluate against, but is *not* an input: our
claim is precisely that the mapping isolated → e2e is where naive selection
breaks, so the selector must survive on isolated-side sensors plus a small
calibration.

**C-honesty.** With one substrate and a 10-cell matrix, any learned rule is
one experiment away from overfitting. We therefore *derive* the rule's
structure from the mechanism (§5.2) and treat its thresholds as calibration;
§4.6 quantifies, via nested leave-one-cell-out, exactly how much a data-driven
alternative could have learned — and could not.

### 4.2 Sensors: three numbers, all isolated-side

| sensor | definition | source | what it senses |
|---|---|---|---|
| `q` | slicing depth | workload | release granularity, protocol trips |
| `ratio` | t_comm(C0) / t_gemm (isolated, same cell) | COMM_ONLY, GEMM_ONLY | workload balance; which resource is the critical path (the *N-sensor*) |
| `gap` | (t_comm(C0) − t_comm(C2)) / t_comm(C0) | COMM_ONLY × {C0, C2} | how much bandwidth headroom the config axis offers (the *q-sensor*: thinens monotonically with q on K500: 2.98/2.12/1.65/0.73 at N=2048) |

The sensor naming reflects an empirical fact the paper leans on: **gap and
ratio are not interchangeable** (§5.2). iso_gap is N-blind (1.64/1.65/1.72
across N at q8) while ratio spans 2.3→0.6 over the same cells; conversely gap
carries the q-information that ratio lacks. A single-sensor rule is
principally unable to separate the reversal cell from its neighbors.

### 4.3 The balance-band rule (config axis)

The reversal of §5.2 is a *resonance of the true-overlap regime*: it appears
only where both resources saturate together (ratio ≈ 1), slicing is deep
enough to make release structure binding (q ≥ 8), and the config axis has thin
isolated headroom (gap ≤ 2%). Outside the band, the isolated winner
transmits (comm-dominated), or the config choice is shadowed by GEMM
(compute-dominated) — either way bandwidth thinking is safe. Inside the band
it is not: the bandwidth winner's extra channels steal time from the GEMM
critical path.

```
select_config(q, ratio, gap):
    if q >= 8 and 0.9 <= ratio <= 1.35 and gap <= 2.0:  return C0
    return C2
```

Thresholds: q≥8 from the within-band traversal (+5.65→+1.69→−5.49→−5.37 at
N=2048, ratio≈1.2 held fixed); the ratio window from the q8 row
(ratio 2.32/+0.2, 1.12/−5.5, 0.59/+1.8 — sign change inside [0.6, 2.3]);
gap≤2 from the thinnest gap that still reversed (1.65). All three are
mechanism-derived and single-point calibrated; §4.6 states this cost
honestly. On the 10-cell K500 matrix: 10/10 top-1, 0.00% worst regret, vs
8/10 and 5.49% for the bandwidth-naive rule (always pick the isolated
winner). The discriminator DX — ⟨N4096/q16, ratio pinned at 0.59 with q
doubled⟩ — tests whether q has any effect *outside* the band; the rule
predicts it does not.

### 4.4 Strategy axis: nearest-cell with a degeneracy check

Strategy selection uses nearest-labelled-cell lookup in (log q, log ratio)
space over the offline calibration matrix, with a *degeneracy guard*: if one
strategy holds the winner slot in every labelled cell (as r1 does on K500,
10/10), the selector emits it with a "degenerate" flag and no per-cell
confidence — the axis carries no information on this substrate, and the
calibration report says so rather than inventing structure. The
bandwidth-intuition rule (transport-dominated ⇒ bulk) is kept in the
evaluation as an adversarial baseline: on K500 it pays worst-case 33.7%
(§5.5) — evidence that heuristics *about* the strategy axis can be worse
than no selection at all. Whether the axis is discriminative is a substrate
property; ⟨NVIDIA/4090 pre-registered predictions in §5.6⟩.

### 4.5 Safety net and hysteresis

Borrowed from gain scheduling with hysteresis (control practice; §2
cross-domain survey): the rule fires C0 only inside the band; cells *near*
band edges (within ε of any threshold) fall back to the conservative config
(C0 default) *and* to the conservative strategy when the strategy axis is
unflagged-degenerate. The offline report lists, for every calibrated cell,
which sensor dominates the margin — giving the operator a dead-zone map
rather than a bare decision. Runtime cost is three comparisons; there is no
online measurement, model, or search.

Dead-zone map on the K500 matrix (margins to each threshold; the binding
sensor is the one closest to its edge):

| cell | q−8 | ratio to band | 2.0−gap | binding | decision (=winner) |
|---|---|---|---|---|---|
| N2048/q8 | **0** | +0.22 | +0.35 | **q — exactly on the threshold** | C0 ✓ |
| N2048/q16 | +8 | +0.12 | +1.27 | ratio | C0 ✓ |
| N512/q8 | 0 | −0.98 (out) | +0.36 | ratio | C2 ✓ |
| N4096/q8 | 0 | −0.31 (out) | +0.28 | ratio | C2 ✓ |
| all q≤4 cells | <0 | — | — | q | C2 ✓ |

The calibration cell sits *on* the q-threshold, which is the honest reading
of "single-point calibration": ε-dead-zone cells are precisely those like
N512/q8 (q on the threshold, ratio 0.98 outside) where the rule's C2 call is
correct but by a mechanism margin an operator should see flagged, not by a
comfortable margin. ⟨on NVIDIA: regenerate this table as B'-calibration⟩

### 4.6 Honest evaluation: nested LOO and the prior stance

We report *nested* leave-one-cell-out: in each fold, thresholds are re-fit on
the training cells only (grid over q-thresholds and gap thresholds; the ratio
window from mechanism), then applied to the held-out cell. Result on K500
(10 cells, 2 positives): 7/10 top-1, worst regret 5.49% — the re-fit rule
*loses* to the fixed mechanism-prior rule, because with two positive cells
the fitted q-threshold drifts to the deeper one (q≥16, gap≤1) and misses the
shallower reversal. We do not hide this: it is the quantitative statement
that this rule is a **mechanism prior with calibrated constants**, not a
learned model — and that a second substrate (§5.6) is the informative
replication, not more cells on one substrate.

### 4.7 Overhead

Selection is O(1) at deployment (three comparisons). The offline calibration
is the cost: ⟨10 paths + 2 configs⟩ × cells × 5 reps ≈ 4–5 h on 4 GPUs,
one-time per substrate — the same order as a conventional tuner's sweep,
reused here as both measurement study and selector training set.

---

### 回填清单
- [ ] DX 判别结果（N4096/q16 四件套，dsfix）
- [ ] 4.4 degeneracy 判定在 NVIDIA 侧的解除（或保持）
- [ ] 4.5 ε-dead-zone 的具体值与 regret 合并叙述
- [ ] 4.7 校准 wall-clock 实测
- [ ] 方法名定稿（SemSelect/ROSS/…）
