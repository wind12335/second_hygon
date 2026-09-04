# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/bw1000-port/results/bw1000_8gpu/phaseb_wm_np8_20260904_145332`
- cases: 48 (PASS 48, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 8 |  |  |  | -112.127 |  |  |
| C0_DEFAULT | 2048 | 8 |  |  |  | -61.862 |  |  |
| C0_DEFAULT | 2048 | 16 |  |  |  | -189.269 |  |  |
| C0_DEFAULT | 4096 | 8 |  |  |  | 14.935 |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 0

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|

Cross-substrate reversal cells: 0 / 0; RCCL-internal C0/C2 reversal cells: 0 / 0

## Case status

