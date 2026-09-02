# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_q16fill_20260902_212009`
- cases: 25 (PASS 25, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 4096 | 16 | DC_PUSHSIG_ONLY (14402.254us) | D0_FCOLLECT_SERIAL (18357.384us) | DUSHMEM | DUSHMEM | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 4096 | 16 |  |  | -5.530 | -40.801 |  | 79.450 |

Cross-substrate reversal cells (per-candidate table): 0 / 1

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 4096 | 16 | dc_pushsig_only_c0_default:14402 | d0_fcollect_serial_c0_default:18357 < ds_pushsig_serial_c0_default:24493 < d1_pushsig_overlap_c0_default:25847 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |

Cross-substrate reversal cells: 0 / 1; RCCL-internal C0/C2 reversal cells: 0 / 1

## Case status

