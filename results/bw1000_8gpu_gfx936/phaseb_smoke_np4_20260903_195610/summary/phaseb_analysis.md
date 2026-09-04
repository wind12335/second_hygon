# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/bw1000-port/results/bw1000_8gpu/phaseb_smoke_np4_20260903_195610`
- cases: 11 (PASS 11, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 2048 | 8 | COMM_ONLY (825.040us) | R0_FULL_SERIAL (1899.999us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 2048 | 8 | 8.937 | -81.482 | 6.899 | -124.010 | 67.275 | 459.033 |

Cross-substrate reversal cells (per-candidate table): 0 / 1

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | comm_only_c0_default:825 < dc_pushsig_only_c0_default:1883 < fc_fcollect_only_c0_default:3337 | r0_full_serial_c0_default:1900 < r1_event_overlap_c0_default:3448 < rs_slice_serial_c0_default:3787 | RCCL | RCCL | CONSISTENT |  /  | N/A |

Cross-substrate reversal cells: 0 / 1; RCCL-internal C0/C2 reversal cells: 0 / 1

## Case status

