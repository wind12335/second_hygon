# Phase 1 Repeat-Stability Analysis

This report separates process-level repetitions from timed iterations. All entries use max-rank device-event timing and PASS-only samples.

- Raw global CSVs accepted: 286
- Raw global CSVs missing a matching manifest or schema: 0
- H0/B1/B0 use three independent MPI process repetitions in the discovery matrix.
- COMM_ONLY uses one independent MPI process per candidate in Phase 1; its ranking must be repeated in Phase 2.

## M=2048 N=128 K=2048 q=1

- Isolated COMM_ONLY: C0_DEFAULT:3905.541;C1_RING_SIMPLE_CH4:3905.940;C2_RING_SIMPLE_CH8:3830.505; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4347.197;C1_RING_SIMPLE_CH4:4347.994;C2_RING_SIMPLE_CH8:4276.719; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4349.195/r2=4345.115/r3=4347.197;C1_RING_SIMPLE_CH4:r1=4340.800/r2=4347.994/r3=4348.442;C2_RING_SIMPLE_CH8:r1=4276.719/r2=4275.440/r3=4288.239.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 0.154323%; first-to-last release span: 0.000000% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=128 K=2048 q=2

- Isolated COMM_ONLY: C0_DEFAULT:3955.698;C1_RING_SIMPLE_CH4:3952.258;C2_RING_SIMPLE_CH8:3891.302; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4209.684;C1_RING_SIMPLE_CH4:4210.722;C2_RING_SIMPLE_CH8:4115.929; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4213.283/r2=4208.564/r3=4209.684;C1_RING_SIMPLE_CH4:r1=4213.229/r2=4209.604/r3=4210.722;C2_RING_SIMPLE_CH8:r1=4116.649/r2=4115.929/r3=4111.610.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 49.871925%; first-to-last release span: 49.723433% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=128 K=2048 q=4

- Isolated COMM_ONLY: C0_DEFAULT:4024.975;C1_RING_SIMPLE_CH4:4026.333;C2_RING_SIMPLE_CH8:3973.584; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4313.578;C1_RING_SIMPLE_CH4:4324.316;C2_RING_SIMPLE_CH8:4223.204; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4317.757/r2=4313.578/r3=4307.279;C1_RING_SIMPLE_CH4:r1=4325.038/r2=4315.114/r3=4324.316;C2_RING_SIMPLE_CH8:r1=4219.764/r2=4224.325/r3=4223.204.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 75.037930%; first-to-last release span: 74.886255% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=128 K=2048 q=8

- Isolated COMM_ONLY: C0_DEFAULT:4190.645;C1_RING_SIMPLE_CH4:4200.645;C2_RING_SIMPLE_CH8:4158.650; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4499.988;C1_RING_SIMPLE_CH4:4501.031;C2_RING_SIMPLE_CH8:4418.522; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4499.988/r2=4500.307/r3=4497.354;C1_RING_SIMPLE_CH4:r1=4495.268/r2=4505.106/r3=4501.031;C2_RING_SIMPLE_CH8:r1=4418.522/r2=4425.032/r3=4415.676.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 87.332525%; first-to-last release span: 87.184088% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=2048 K=2048 q=1

- Isolated COMM_ONLY: C0_DEFAULT:3912.664;C1_RING_SIMPLE_CH4:3910.980;C2_RING_SIMPLE_CH8:3842.028; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:6376.521;C1_RING_SIMPLE_CH4:6381.087;C2_RING_SIMPLE_CH8:6312.447; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=6376.521/r2=6373.328/r3=6389.736;C1_RING_SIMPLE_CH4:r1=6381.087/r2=6384.687/r3=6380.847;C2_RING_SIMPLE_CH8:r1=6312.447/r2=6310.286/r3=6474.516.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 0.157732%; first-to-last release span: 0.000000% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=2048 K=2048 q=2

- Isolated COMM_ONLY: C0_DEFAULT:3962.497;C1_RING_SIMPLE_CH4:3961.137;C2_RING_SIMPLE_CH8:3888.901; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:5341.861;C1_RING_SIMPLE_CH4:5340.581;C2_RING_SIMPLE_CH8:5143.870; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=5342.648/r2=5341.861/r3=5337.624;C1_RING_SIMPLE_CH4:r1=5339.386/r2=5344.182/r3=5340.581;C2_RING_SIMPLE_CH8:r1=5143.870/r2=5142.686/r3=5151.711.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 51.072935%; first-to-last release span: 50.919119% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=2048 K=2048 q=4

- Isolated COMM_ONLY: C0_DEFAULT:4035.694;C1_RING_SIMPLE_CH4:4034.099;C2_RING_SIMPLE_CH8:3984.648; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4853.407;C1_RING_SIMPLE_CH4:4852.047;C2_RING_SIMPLE_CH8:4842.288; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4853.407/r2=4858.715/r3=4847.571;C1_RING_SIMPLE_CH4:r1=4854.367/r2=4851.888/r3=4852.047;C2_RING_SIMPLE_CH8:r1=4842.288/r2=4841.973/r3=4844.389.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 74.792192%; first-to-last release span: 74.634192% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=2048 K=2048 q=8

- Isolated COMM_ONLY: C0_DEFAULT:4207.444;C1_RING_SIMPLE_CH4:4209.444;C2_RING_SIMPLE_CH8:4167.285; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:5167.551;C1_RING_SIMPLE_CH4:5174.434;C2_RING_SIMPLE_CH8:5539.774; best `C0_DEFAULT`.
- H0 process medians: C0_DEFAULT:r1=5183.389/r2=5167.551/r3=5162.274;C1_RING_SIMPLE_CH4:r1=5174.434/r2=5170.510/r3=5179.070;C2_RING_SIMPLE_CH8:r1=5539.774/r2=5536.330/r3=5545.931.
- Reversal: YES; H0 margin over isolated-communication winner: 7.203103%.
- Winner beats the isolated-communication winner in every H0 repetition: YES.
- C0 first-release window: 87.453420%; first-to-last release span: 87.318896% of T_done.
- Evidence status: `REPEAT_STABLE_REVERSAL`.

## M=2048 N=512 K=2048 q=1

- Isolated COMM_ONLY: C0_DEFAULT:3905.620;C1_RING_SIMPLE_CH4:3907.145;C2_RING_SIMPLE_CH8:3830.265; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4876.046;C1_RING_SIMPLE_CH4:4880.850;C2_RING_SIMPLE_CH8:4810.210; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=5136.680/r2=4876.046/r3=4873.326;C1_RING_SIMPLE_CH4:r1=4881.167/r2=4880.850/r3=4877.806;C2_RING_SIMPLE_CH8:r1=4810.210/r2=4805.490/r3=4815.720.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 0.156456%; first-to-last release span: 0.000000% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=512 K=2048 q=2

- Isolated COMM_ONLY: C0_DEFAULT:3956.898;C1_RING_SIMPLE_CH4:3959.858;C2_RING_SIMPLE_CH8:3883.945; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4435.271;C1_RING_SIMPLE_CH4:4430.551;C2_RING_SIMPLE_CH8:4342.157; best `C2_RING_SIMPLE_CH8`.
- H0 process medians: C0_DEFAULT:r1=4435.271/r2=4432.950/r3=4436.232;C1_RING_SIMPLE_CH4:r1=4435.807/r2=4430.152/r3=4430.551;C2_RING_SIMPLE_CH8:r1=4341.436/r2=4342.157/r3=4349.515.
- Reversal: NO; H0 margin over isolated-communication winner: 0.000000%.
- Winner beats the isolated-communication winner in every H0 repetition: NO.
- C0 first-release window: 49.861024%; first-to-last release span: 49.711223% of T_done.
- Evidence status: `NO_REVERSAL`.

## M=2048 N=512 K=2048 q=4

- Isolated COMM_ONLY: C0_DEFAULT:4029.454;C1_RING_SIMPLE_CH4:4028.734;C2_RING_SIMPLE_CH8:3978.646; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4295.838;C1_RING_SIMPLE_CH4:4295.678;C2_RING_SIMPLE_CH8:4339.197; best `C1_RING_SIMPLE_CH4`.
- H0 process medians: C0_DEFAULT:r1=4295.838/r2=4298.869/r3=4291.520;C1_RING_SIMPLE_CH4:r1=4295.038/r2=4295.678/r3=4302.318;C2_RING_SIMPLE_CH8:r1=4338.236/r2=4339.197/r3=4341.516.
- Reversal: YES; H0 margin over isolated-communication winner: 1.013076%.
- Winner beats the isolated-communication winner in every H0 repetition: YES.
- C0 first-release window: 74.554142%; first-to-last release span: 74.389255% of T_done.
- Evidence status: `REPEAT_STABLE_REVERSAL`.

## M=2048 N=512 K=2048 q=8

- Isolated COMM_ONLY: C0_DEFAULT:4196.485;C1_RING_SIMPLE_CH4:4197.765;C2_RING_SIMPLE_CH8:4165.286; best `C2_RING_SIMPLE_CH8`.
- H0 aggregate: C0_DEFAULT:4551.265;C1_RING_SIMPLE_CH4:4561.505;C2_RING_SIMPLE_CH8:4570.383; best `C0_DEFAULT`.
- H0 process medians: C0_DEFAULT:r1=4550.944/r2=4551.265/r3=4559.185;C1_RING_SIMPLE_CH4:r1=4568.705/r2=4559.264/r3=4561.505;C2_RING_SIMPLE_CH8:r1=4568.566/r2=4570.708/r3=4570.383.
- Reversal: YES; H0 margin over isolated-communication winner: 0.420081%.
- Winner beats the isolated-communication winner in every H0 repetition: YES.
- C0 first-release window: 87.333499%; first-to-last release span: 87.184281% of T_done.
- Evidence status: `REPEAT_STABLE_REVERSAL`.

## Interpretation Boundary

A repeat-stable H0 reversal is a motivation result, not a complete paper claim. Phase 2 must repeat COMM_ONLY, export every slice release time, and rerun the selected counterexamples before attributing the effect to release semantics, resource contention, or a specific communication backend mechanism.
