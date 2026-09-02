# Phase 2 Targeted Confirmation Analysis

Platform: K500SM_AI / gfx928 / 4 GPUs / PCIe. All values are PASS-only max-rank device-event samples.

- Accepted raw global CSVs: 91
- Expected target cases: 91
- COMM_ONLY and H0 have five independent MPI process repetitions per candidate and q.
- Per-slice release/start/end samples are in `release_slices_global.csv` per case and aggregated in `phase2_release_curve_summary.csv`.

## q=2

- COMM_ONLY p50: C0_DEFAULT:3962.817;C1_RING_SIMPLE_CH4:3963.377;C2_RING_SIMPLE_CH8:3889.781; best `C2_RING_SIMPLE_CH8`.
- H0 p50: C0_DEFAULT:5338.505;C1_RING_SIMPLE_CH4:5343.254;C2_RING_SIMPLE_CH8:5147.324; best `C2_RING_SIMPLE_CH8`.
- H0 winner advantage over the isolated-communication winner: 0.000000%.
- H0 process medians: C0_DEFAULT:r1=5342.022/r2=5336.739/r3=5341.059/r4=5338.505/r5=5337.941;C1_RING_SIMPLE_CH4:r1=5339.300/r2=5345.460/r3=5344.180/r4=5336.100/r5=5343.254;C2_RING_SIMPLE_CH8:r1=5147.324/r2=5147.324/r3=5148.427/r4=5148.030/r5=5144.992.
- H0 winner is faster in every matched process repetition: NO.
- H0 vs B1 gain: C0=20.122778%, C2=23.333280%.
- Evidence status: `NO_REVERSAL_CONTROL`.

## q=8

- COMM_ONLY p50: C0_DEFAULT:4204.244;C1_RING_SIMPLE_CH4:4202.644;C2_RING_SIMPLE_CH8:4168.873; best `C2_RING_SIMPLE_CH8`.
- H0 p50: C0_DEFAULT:5166.832;C1_RING_SIMPLE_CH4:5170.190;C2_RING_SIMPLE_CH8:5537.038; best `C0_DEFAULT`.
- H0 winner advantage over the isolated-communication winner: 7.165049%.
- H0 process medians: C0_DEFAULT:r1=5167.711/r2=5166.033/r3=5166.675/r4=5166.832/r5=5171.470;C1_RING_SIMPLE_CH4:r1=5185.790/r2=5172.590/r3=5169.310/r4=5170.190/r5=5165.070;C2_RING_SIMPLE_CH8:r1=5541.995/r2=5542.699/r3=5537.038/r4=5531.769/r5=5534.329.
- H0 winner is faster in every matched process repetition: YES.
- H0 vs B1 gain: C0=50.881696%, C2=40.129158%.
- Evidence status: `CONFIRMED_STRONG_REVERSAL`.

## Scope Boundary

This confirms or rejects a controlled engineering counterexample. It does not by itself establish a new backend, a DUSHMEM result, or end-to-end model-training benefit. Those require a later valid-primitive test and trace-derived workloads.
