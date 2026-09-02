# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_smoke_20260902_155330`
- cases: 12 (PASS 0, other 12)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 |
|---|---|---|---|---|---|---|---|

Cross-substrate reversal cells: 0 / 0

## Case status

- case001_comm_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case002_comm_C2_RING_SIMPLE_CH8_N2048_q8_rep1: status=FAIL exit=0
- case003_gemm_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case004_r0_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case005_rs_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case006_r1_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case007_r1_C2_RING_SIMPLE_CH8_N2048_q8_rep1: status=FAIL exit=0
- case008_fc_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case009_dc_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case010_d0_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case011_ds_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
- case012_d1_C0_DEFAULT_N2048_q8_rep1: status=FAIL exit=0
