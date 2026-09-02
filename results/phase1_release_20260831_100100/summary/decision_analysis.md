# Phase 1 Decision Analysis

This report is generated from per-iteration, max-rank raw samples. The three shapes are synthetic engineering controls, not final trace-derived workloads.

- Valid raw global files: 286
- Ignored malformed or superseded raw files: 0
- Complete shape/q decision rows: 14

## M=128 N=128 K=128 q=1

- Isolated communication: C0_DEFAULT:83.996; best C0_DEFAULT.
- H0 end-to-end: C0_DEFAULT:150.231; best C0_DEFAULT.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: -1.809879%.
- B1 fragmentation vs B0: 0.875321%; GEMM q-vs-q1 fragmentation: 0.000000%.
- First legal release window: 44.098021% of C0 T_done.

## M=128 N=128 K=128 q=2

- Isolated communication: C0_DEFAULT:121.913; best C0_DEFAULT.
- H0 end-to-end: C0_DEFAULT:263.957; best C0_DEFAULT.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 7.822865%.
- B1 fragmentation vs B0: 94.626347%; GEMM q-vs-q1 fragmentation: 57.430635%.
- First legal release window: 68.404147% of C0 T_done.

## M=2048 N=128 K=2048 q=1

- Isolated communication: C0_DEFAULT:3905.541;C1_RING_SIMPLE_CH4:3905.940;C2_RING_SIMPLE_CH8:3830.505; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4347.116;C1_RING_SIMPLE_CH4:4345.106;C2_RING_SIMPLE_CH8:4280.319; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: -0.093855%.
- B1 fragmentation vs B0: 0.346520%; GEMM q-vs-q1 fragmentation: 0.000000%.
- First legal release window: 0.155254% of C0 T_done.

## M=2048 N=128 K=2048 q=2

- Isolated communication: C0_DEFAULT:3955.698;C1_RING_SIMPLE_CH4:3952.258;C2_RING_SIMPLE_CH8:3891.302; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4209.923;C1_RING_SIMPLE_CH4:4210.644;C2_RING_SIMPLE_CH8:4114.489; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 3.967555%.
- B1 fragmentation vs B0: 1.130200%; GEMM q-vs-q1 fragmentation: -7.842655%.
- First legal release window: 49.852315% of C0 T_done.

## M=2048 N=128 K=2048 q=4

- Isolated communication: C0_DEFAULT:4024.975;C1_RING_SIMPLE_CH4:4026.333;C2_RING_SIMPLE_CH8:3973.584; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4313.189;C1_RING_SIMPLE_CH4:4321.198;C2_RING_SIMPLE_CH8:4223.123; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 9.915287%.
- B1 fragmentation vs B0: 9.538155%; GEMM q-vs-q1 fragmentation: 73.012517%.
- First legal release window: 75.028241% of C0 T_done.

## M=2048 N=128 K=2048 q=8

- Isolated communication: C0_DEFAULT:4190.645;C1_RING_SIMPLE_CH4:4200.645;C2_RING_SIMPLE_CH8:4158.650; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4498.868;C1_RING_SIMPLE_CH4:4499.747;C2_RING_SIMPLE_CH8:4419.084; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 12.791918%.
- B1 fragmentation vs B0: 17.243862%; GEMM q-vs-q1 fragmentation: 108.510004%.
- First legal release window: 87.345716% of C0 T_done.

## M=2048 N=2048 K=2048 q=1

- Isolated communication: C0_DEFAULT:3912.664;C1_RING_SIMPLE_CH4:3910.980;C2_RING_SIMPLE_CH8:3842.028; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:6379.493;C1_RING_SIMPLE_CH4:6382.202;C2_RING_SIMPLE_CH8:6313.566; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 0.007454%.
- B1 fragmentation vs B0: 3.908604%; GEMM q-vs-q1 fragmentation: 0.000000%.
- First legal release window: 0.159647% of C0 T_done.

## M=2048 N=2048 K=2048 q=2

- Isolated communication: C0_DEFAULT:3962.497;C1_RING_SIMPLE_CH4:3961.137;C2_RING_SIMPLE_CH8:3888.901; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:5340.669;C1_RING_SIMPLE_CH4:5341.540;C2_RING_SIMPLE_CH8:5146.816; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 20.122030%.
- B1 fragmentation vs B0: 4.484371%; GEMM q-vs-q1 fragmentation: -0.681047%.
- First legal release window: 51.084371% of C0 T_done.

## M=2048 N=2048 K=2048 q=4

- Isolated communication: C0_DEFAULT:4035.694;C1_RING_SIMPLE_CH4:4034.099;C2_RING_SIMPLE_CH8:3984.648; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4853.167;C1_RING_SIMPLE_CH4:4852.529;C2_RING_SIMPLE_CH8:4842.939; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 38.402429%.
- B1 fragmentation vs B0: 9.396129%; GEMM q-vs-q1 fragmentation: 7.221601%.
- First legal release window: 74.792778% of C0 T_done.

## M=2048 N=2048 K=2048 q=8

- Isolated communication: C0_DEFAULT:4207.444;C1_RING_SIMPLE_CH4:4209.444;C2_RING_SIMPLE_CH8:4167.285; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:5171.470;C1_RING_SIMPLE_CH4:5174.270;C2_RING_SIMPLE_CH8:5541.455; best C0_DEFAULT.
- Ranking reversal: TRUE. H0-vs-B1 gain for C0: 50.682910%.
- B1 fragmentation vs B0: 26.914434%; GEMM q-vs-q1 fragmentation: 36.689559%.
- First legal release window: 87.451271% of C0 T_done.

## M=2048 N=512 K=2048 q=1

- Isolated communication: C0_DEFAULT:3905.620;C1_RING_SIMPLE_CH4:3907.145;C2_RING_SIMPLE_CH8:3830.265; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4900.284;C1_RING_SIMPLE_CH4:4879.565;C2_RING_SIMPLE_CH8:4811.009; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: -0.501155%.
- B1 fragmentation vs B0: 1.114825%; GEMM q-vs-q1 fragmentation: 0.000000%.
- First legal release window: 0.156770% of C0 T_done.

## M=2048 N=512 K=2048 q=2

- Isolated communication: C0_DEFAULT:3956.898;C1_RING_SIMPLE_CH4:3959.858;C2_RING_SIMPLE_CH8:3883.945; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4434.630;C1_RING_SIMPLE_CH4:4432.459;C2_RING_SIMPLE_CH8:4343.517; best C2_RING_SIMPLE_CH8.
- Ranking reversal: FALSE. H0-vs-B1 gain for C0: 8.884146%.
- B1 fragmentation vs B0: 0.137693%; GEMM q-vs-q1 fragmentation: -15.619437%.
- First legal release window: 49.882785% of C0 T_done.

## M=2048 N=512 K=2048 q=4

- Isolated communication: C0_DEFAULT:4029.454;C1_RING_SIMPLE_CH4:4028.734;C2_RING_SIMPLE_CH8:3978.646; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4295.518;C1_RING_SIMPLE_CH4:4296.479;C2_RING_SIMPLE_CH8:4339.038; best C0_DEFAULT.
- Ranking reversal: TRUE. H0-vs-B1 gain for C0: 15.304382%.
- B1 fragmentation vs B0: 2.715716%; GEMM q-vs-q1 fragmentation: -11.814150%.
- First legal release window: 74.557861% of C0 T_done.

## M=2048 N=512 K=2048 q=8

- Isolated communication: C0_DEFAULT:4196.485;C1_RING_SIMPLE_CH4:4197.765;C2_RING_SIMPLE_CH8:4165.286; best C2_RING_SIMPLE_CH8.
- H0 end-to-end: C0_DEFAULT:4553.505;C1_RING_SIMPLE_CH4:4561.184;C2_RING_SIMPLE_CH8:4569.664; best C0_DEFAULT.
- Ranking reversal: TRUE. H0-vs-B1 gain for C0: 32.020982%.
- B1 fragmentation vs B0: 24.670677%; GEMM q-vs-q1 fragmentation: 85.067436%.
- First legal release window: 87.315057% of C0 T_done.

