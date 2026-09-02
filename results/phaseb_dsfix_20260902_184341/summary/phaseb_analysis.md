# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_dsfix_20260902_184341`
- cases: 170 (PASS 170, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 4096 | 16 | COMM_ONLY (4562.704us) | R1_EVENT_OVERLAP (8853.051us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 16 | COMM_ONLY (4523.346us) | R1_EVENT_OVERLAP (8911.101us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 |  |  | 0.270 |  |  |  |
| C0_DEFAULT | 512 | 4 |  |  | 0.221 |  |  |  |
| C0_DEFAULT | 512 | 8 |  |  | 1.051 |  |  |  |
| C0_DEFAULT | 2048 | 2 |  |  | 0.064 |  |  |  |
| C0_DEFAULT | 2048 | 4 |  |  | -0.069 |  |  |  |
| C0_DEFAULT | 2048 | 8 |  |  | -1.067 |  |  |  |
| C0_DEFAULT | 2048 | 16 |  |  | 1.580 |  |  |  |
| C0_DEFAULT | 4096 | 2 |  |  | -0.358 |  |  |  |
| C0_DEFAULT | 4096 | 4 |  |  | 0.009 |  |  |  |
| C0_DEFAULT | 4096 | 8 |  |  | -0.234 |  |  |  |
| C0_DEFAULT | 4096 | 16 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 16 |  |  |  |  |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 2

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 4096 | 16 | comm_only_c2_ring_simple_ch8:4523 < comm_only_c0_default:4563 | r1_event_overlap_c0_default:8853 < r1_event_overlap_c2_ring_simple_ch8:8911 | RCCL | RCCL | CONSISTENT | 0.863 / -0.656 | CONSISTENT |

Cross-substrate reversal cells: 0 / 1; RCCL-internal C0/C2 reversal cells: 0 / 1

## Case status

