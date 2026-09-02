# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_d0dc_20260902_195753`
- cases: 100 (PASS 100, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | DC_PUSHSIG_ONLY (7461.508us) | D0_FCOLLECT_SERIAL (12575.456us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 512 | 4 | DC_PUSHSIG_ONLY (7583.026us) | D0_FCOLLECT_SERIAL (12567.907us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 512 | 8 | DC_PUSHSIG_ONLY (11037.698us) | D0_FCOLLECT_SERIAL (12568.333us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 2048 | 2 | DC_PUSHSIG_ONLY (7487.207us) | D0_FCOLLECT_SERIAL (14345.365us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 2048 | 4 | DC_PUSHSIG_ONLY (7581.987us) | D0_FCOLLECT_SERIAL (14332.968us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 2048 | 8 | DC_PUSHSIG_ONLY (10839.952us) | D0_FCOLLECT_SERIAL (14347.912us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 2048 | 16 | DC_PUSHSIG_ONLY (14956.376us) | D0_FCOLLECT_SERIAL (14339.047us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 4096 | 2 | DC_PUSHSIG_ONLY (7483.020us) | D0_FCOLLECT_SERIAL (18347.126us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 4096 | 4 | DC_PUSHSIG_ONLY (7558.708us) | D0_FCOLLECT_SERIAL (18335.866us) | DUSHMEM | DUSHMEM | CONSISTENT |
| C0_DEFAULT | 4096 | 8 | DC_PUSHSIG_ONLY (11361.280us) | D0_FCOLLECT_SERIAL (18347.323us) | DUSHMEM | DUSHMEM | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 |  |  |  |  |  |  |
| C0_DEFAULT | 512 | 4 |  |  |  |  |  |  |
| C0_DEFAULT | 512 | 8 |  |  |  |  |  |  |
| C0_DEFAULT | 2048 | 2 |  |  |  |  |  |  |
| C0_DEFAULT | 2048 | 4 |  |  |  |  |  |  |
| C0_DEFAULT | 2048 | 8 |  |  |  |  |  |  |
| C0_DEFAULT | 2048 | 16 |  |  |  |  |  |  |
| C0_DEFAULT | 4096 | 2 |  |  |  |  |  |  |
| C0_DEFAULT | 4096 | 4 |  |  |  |  |  |  |
| C0_DEFAULT | 4096 | 8 |  |  |  |  |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 10

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 512 | 2 | dc_pushsig_only_c0_default:7462 | d0_fcollect_serial_c0_default:12575 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 512 | 4 | dc_pushsig_only_c0_default:7583 | d0_fcollect_serial_c0_default:12568 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 512 | 8 | dc_pushsig_only_c0_default:11038 | d0_fcollect_serial_c0_default:12568 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 2048 | 2 | dc_pushsig_only_c0_default:7487 | d0_fcollect_serial_c0_default:14345 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 2048 | 4 | dc_pushsig_only_c0_default:7582 | d0_fcollect_serial_c0_default:14333 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 2048 | 8 | dc_pushsig_only_c0_default:10840 | d0_fcollect_serial_c0_default:14348 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 2048 | 16 | dc_pushsig_only_c0_default:14956 | d0_fcollect_serial_c0_default:14339 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 4096 | 2 | dc_pushsig_only_c0_default:7483 | d0_fcollect_serial_c0_default:18347 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 4096 | 4 | dc_pushsig_only_c0_default:7559 | d0_fcollect_serial_c0_default:18336 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |
| 4096 | 8 | dc_pushsig_only_c0_default:11361 | d0_fcollect_serial_c0_default:18347 | DUSHMEM | DUSHMEM | CONSISTENT |  /  | N/A |

Cross-substrate reversal cells: 0 / 10; RCCL-internal C0/C2 reversal cells: 0 / 10

## Case status

