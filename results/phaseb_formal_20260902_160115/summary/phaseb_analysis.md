# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_formal_20260902_160115`
- cases: 585 (PASS 585, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | COMM_ONLY (4007.855us) | R1_EVENT_OVERLAP (4783.811us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 4 | COMM_ONLY (4072.895us) | R1_EVENT_OVERLAP (4553.264us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 8 | COMM_ONLY (4226.486us) | R1_EVENT_OVERLAP (4692.221us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 2 | COMM_ONLY (4018.579us) | R1_EVENT_OVERLAP (6086.778us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 4 | COMM_ONLY (4079.477us) | R1_EVENT_OVERLAP (5122.167us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 8 | COMM_ONLY (4233.552us) | R1_EVENT_OVERLAP (5400.203us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 16 | COMM_ONLY (4552.205us) | R1_EVENT_OVERLAP (5830.553us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 2 | COMM_ONLY (4027.054us) | R1_EVENT_OVERLAP (8798.786us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 4 | COMM_ONLY (4082.571us) | R1_EVENT_OVERLAP (8272.903us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 8 | COMM_ONLY (4238.481us) | R1_EVENT_OVERLAP (8772.738us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 2 | COMM_ONLY (3892.501us) | R1_EVENT_OVERLAP (4536.626us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 4 | COMM_ONLY (3985.617us) | R1_EVENT_OVERLAP (4476.549us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 8 | COMM_ONLY (4157.367us) | R1_EVENT_OVERLAP (4681.898us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 2 | COMM_ONLY (3898.821us) | R1_EVENT_OVERLAP (5742.637us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 4 | COMM_ONLY (3993.056us) | R1_EVENT_OVERLAP (5035.478us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 8 | COMM_ONLY (4163.527us) | R1_EVENT_OVERLAP (5696.881us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 16 | COMM_ONLY (4519.034us) | R1_EVENT_OVERLAP (6143.738us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 2 | COMM_ONLY (3900.101us) | R1_EVENT_OVERLAP (8438.165us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 4 | COMM_ONLY (3995.651us) | R1_EVENT_OVERLAP (8262.423us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 8 | COMM_ONLY (4165.526us) | R1_EVENT_OVERLAP (8617.604us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | 8.548 | 6.206 | -0.004 | 30.316 | 45.395 | 17.402 |
| C0_DEFAULT | 512 | 4 | 14.759 | 10.721 | -0.169 | 28.758 | 49.141 | 18.076 |
| C0_DEFAULT | 512 | 8 | 24.464 | 7.960 | -0.105 | -45.374 | 74.309 | 64.405 |
| C0_DEFAULT | 2048 | 2 | 12.918 | 11.117 | 0.005 | 26.469 | 42.274 | 40.967 |
| C0_DEFAULT | 2048 | 4 | 28.765 | 25.179 | -0.051 | 23.866 | 53.072 | 43.824 |
| C0_DEFAULT | 2048 | 8 | 32.202 | 21.124 | 0.592 | -41.107 | 73.306 | 92.908 |
| C0_DEFAULT | 2048 | 16 | 30.078 |  | -1.400 |  | 74.783 | 53.349 |
| C0_DEFAULT | 4096 | 2 | 19.710 | 18.750 | 0.145 | 21.424 | 38.964 | 92.464 |
| C0_DEFAULT | 4096 | 4 | 15.342 | 23.656 | 0.017 | 26.718 | 38.504 | 77.849 |
| C0_DEFAULT | 4096 | 8 | 20.368 | 18.975 | -0.843 | -19.436 | 59.994 | 97.329 |
| C2_RING_SIMPLE_CH8 | 512 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 512 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 512 | 8 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 8 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 16 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 8 |  |  |  |  |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 20

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 512 | 2 | comm_only_c2_ring_simple_ch8:3893 < comm_only_c0_default:4008 < dc_pushsig_only_c0_default:7460 | r1_event_overlap_c2_ring_simple_ch8:4537 < r1_event_overlap_c0_default:4784 < r0_full_serial_c0_default:5100 | RCCL | RCCL | CONSISTENT | 2.878 / 5.167 | CONSISTENT |
| 512 | 4 | comm_only_c2_ring_simple_ch8:3986 < comm_only_c0_default:4073 < dc_pushsig_only_c0_default:7582 | r1_event_overlap_c2_ring_simple_ch8:4477 < r1_event_overlap_c0_default:4553 < r0_full_serial_c0_default:5100 | RCCL | RCCL | CONSISTENT | 2.143 / 1.685 | CONSISTENT |
| 512 | 8 | comm_only_c2_ring_simple_ch8:4157 < comm_only_c0_default:4226 < dc_pushsig_only_c0_default:11109 | r1_event_overlap_c2_ring_simple_ch8:4682 < r1_event_overlap_c0_default:4692 < r0_full_serial_c0_default:5098 | RCCL | RCCL | CONSISTENT | 1.635 / 0.220 | CONSISTENT |
| 2048 | 2 | comm_only_c2_ring_simple_ch8:3899 < comm_only_c0_default:4019 < dc_pushsig_only_c0_default:7479 | r1_event_overlap_c2_ring_simple_ch8:5743 < r1_event_overlap_c0_default:6087 < r0_full_serial_c0_default:6848 | RCCL | RCCL | CONSISTENT | 2.980 / 5.654 | CONSISTENT |
| 2048 | 4 | comm_only_c2_ring_simple_ch8:3993 < comm_only_c0_default:4079 < dc_pushsig_only_c0_default:7588 | r1_event_overlap_c2_ring_simple_ch8:5035 < r1_event_overlap_c0_default:5122 < r0_full_serial_c0_default:6846 | RCCL | RCCL | CONSISTENT | 2.118 / 1.692 | CONSISTENT |
| 2048 | 8 | comm_only_c2_ring_simple_ch8:4164 < comm_only_c0_default:4234 < dc_pushsig_only_c0_default:10486 | r1_event_overlap_c0_default:5400 < r1_event_overlap_c2_ring_simple_ch8:5697 < r0_full_serial_c0_default:6846 | RCCL | RCCL | CONSISTENT | 1.654 / -5.494 | RCCL_CONFIG_REVERSAL |
| 2048 | 16 | comm_only_c2_ring_simple_ch8:4519 < comm_only_c0_default:4552 < dc_pushsig_only_c0_default:14925 | r1_event_overlap_c0_default:5831 < r1_event_overlap_c2_ring_simple_ch8:6144 < rs_slice_serial_c0_default:8339 | RCCL | RCCL | CONSISTENT | 0.729 / -5.371 | CONSISTENT |
| 4096 | 2 | comm_only_c2_ring_simple_ch8:3900 < comm_only_c0_default:4027 < dc_pushsig_only_c0_default:7490 | r1_event_overlap_c2_ring_simple_ch8:8438 < r1_event_overlap_c0_default:8799 < r0_full_serial_c0_default:10829 | RCCL | RCCL | CONSISTENT | 3.153 / 4.099 | CONSISTENT |
| 4096 | 4 | comm_only_c2_ring_simple_ch8:3996 < comm_only_c0_default:4083 < dc_pushsig_only_c0_default:7564 | r1_event_overlap_c2_ring_simple_ch8:8262 < r1_event_overlap_c0_default:8273 < rs_slice_serial_c0_default:9772 | RCCL | RCCL | CONSISTENT | 2.129 / 0.127 | CONSISTENT |
| 4096 | 8 | comm_only_c2_ring_simple_ch8:4166 < comm_only_c0_default:4238 < dc_pushsig_only_c0_default:11110 | r1_event_overlap_c2_ring_simple_ch8:8618 < r1_event_overlap_c0_default:8773 < r0_full_serial_c0_default:10827 | RCCL | RCCL | CONSISTENT | 1.721 / 1.768 | CONSISTENT |

Cross-substrate reversal cells: 0 / 10; RCCL-internal C0/C2 reversal cells: 1 / 10

## Case status

