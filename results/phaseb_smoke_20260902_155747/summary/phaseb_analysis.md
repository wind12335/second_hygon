# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_smoke_20260902_155747`
- cases: 12 (PASS 12, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 2048 | 8 | COMM_ONLY (4232.722us) | R1_EVENT_OVERLAP (5396.257us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 8 | COMM_ONLY (4164.331us) | R1_EVENT_OVERLAP (5688.881us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 2048 | 8 | 32.176 | 21.120 | 3.789 | -40.583 | 73.150 | 103.423 |
| C2_RING_SIMPLE_CH8 | 2048 | 8 |  |  |  |  |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 2

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | comm_only_c2_ring_simple_ch8:4164 < comm_only_c0_default:4233 < dc_pushsig_only_c0_default:9879 | r1_event_overlap_c0_default:5396 < r1_event_overlap_c2_ring_simple_ch8:5689 < r0_full_serial_c0_default:6841 | RCCL | RCCL | CONSISTENT | 1.616 / -5.423 | RCCL_CONFIG_REVERSAL |

Cross-substrate reversal cells: 0 / 1; RCCL-internal C0/C2 reversal cells: 1 / 1

## Case status

