# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/bw1000-port/results/bw1000_8gpu/phaseb_formal_np8_20260904_093927`
- cases: 645 (PASS 645, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | COMM_ONLY (1111.120us) | R0_FULL_SERIAL (1858.399us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 4 | COMM_ONLY (1215.199us) | R0_FULL_SERIAL (1858.961us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 8 | COMM_ONLY (1441.040us) | R0_FULL_SERIAL (1858.720us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 2 | COMM_ONLY (1178.561us) | R0_FULL_SERIAL (4117.358us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 4 | COMM_ONLY (1269.840us) | R0_FULL_SERIAL (4124.800us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 8 | COMM_ONLY (1495.600us) | R0_FULL_SERIAL (4128.159us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 16 | COMM_ONLY (1872.479us) | R1_EVENT_OVERLAP (6819.592us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 2 | COMM_ONLY (1216.959us) | R0_FULL_SERIAL (9157.035us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 4 | COMM_ONLY (1317.360us) | R1_EVENT_OVERLAP (8412.798us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 8 | COMM_ONLY (1560.720us) | R1_EVENT_OVERLAP (8375.922us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 16 | COMM_ONLY (1918.398us) | R1_EVENT_OVERLAP (12260.238us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 2 | COMM_ONLY (1347.120us) | R1_EVENT_OVERLAP (2553.198us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 4 | COMM_ONLY (1482.719us) | R1_EVENT_OVERLAP (2566.800us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 8 | COMM_ONLY (1767.280us) | R1_EVENT_OVERLAP (2332.079us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 2 | COMM_ONLY (1405.360us) | R1_EVENT_OVERLAP (4283.118us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 4 | COMM_ONLY (1535.043us) | R1_EVENT_OVERLAP (4292.568us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 8 | COMM_ONLY (1809.528us) | R1_EVENT_OVERLAP (6619.541us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 2 | COMM_ONLY (1462.402us) | R1_EVENT_OVERLAP (9332.809us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 4 | COMM_ONLY (1582.401us) | R1_EVENT_OVERLAP (8366.723us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 8 | COMM_ONLY (1856.960us) | R1_EVENT_OVERLAP (8174.240us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 16 | COMM_ONLY (2359.519us) | R1_EVENT_OVERLAP (12331.016us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | 9.291 | -28.106 | -1.261 | 32.049 | 55.360 | 37.502 |
| C0_DEFAULT | 512 | 4 | 15.338 | -26.918 | -0.161 | -55.817 | 80.726 | 132.012 |
| C0_DEFAULT | 512 | 8 | 20.915 | -15.009 | 0.553 | -112.427 | 87.179 | 19.330 |
| C0_DEFAULT | 2048 | 2 | -0.746 | -6.484 | -0.251 | 33.177 | 35.131 | 67.990 |
| C0_DEFAULT | 2048 | 4 | 8.536 | -4.282 | 1.195 | 28.227 | 40.787 | 41.489 |
| C0_DEFAULT | 2048 | 8 | 10.068 | -60.325 | 7.539 | -54.799 | 57.812 | 64.663 |
| C0_DEFAULT | 2048 | 16 | 14.241 |  | 0.001 |  | 76.842 | 70.149 |
| C0_DEFAULT | 4096 | 2 | 0.076 | -2.930 | -0.846 | 23.126 | 19.164 | 182.551 |
| C0_DEFAULT | 4096 | 4 | 3.011 | 8.331 | 0.315 | 27.464 | 23.599 | 86.960 |
| C0_DEFAULT | 4096 | 8 | 3.086 | 8.455 | -0.058 | 16.080 | 34.252 | 25.350 |
| C0_DEFAULT | 4096 | 16 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 512 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 512 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 512 | 8 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 2048 | 8 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 2 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 4 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 8 |  |  |  |  |  |  |
| C2_RING_SIMPLE_CH8 | 4096 | 16 |  |  |  |  |  |  |

Cross-substrate reversal cells (per-candidate table): 0 / 21

## Cross-candidate boundary (isolated vs e2e pooled across RCCL configs)

| N | q | isolated ranking (top3, us) | e2e ranking (top3, us) | iso fam | e2e fam | substrate flag | C2 vs C0 iso% / e2e% | config flag |
|---|---|---|---|---|---|---|---|---|
| 512 | 2 | comm_only_c0_default:1111 < comm_only_c2_ring_simple_ch8:1347 < dc_pushsig_only_c0_default:3874 | r0_full_serial_c0_default:1858 < r1_event_overlap_c0_default:2381 < r1_event_overlap_c2_ring_simple_ch8:2553 | RCCL | RCCL | CONSISTENT | -21.240 / -7.245 | CONSISTENT |
| 512 | 4 | comm_only_c0_default:1215 < comm_only_c2_ring_simple_ch8:1483 < dc_pushsig_only_c0_default:5273 | r0_full_serial_c0_default:1859 < r1_event_overlap_c0_default:2359 < r1_event_overlap_c2_ring_simple_ch8:2567 | RCCL | RCCL | CONSISTENT | -22.015 / -8.792 | CONSISTENT |
| 512 | 8 | comm_only_c0_default:1441 < comm_only_c2_ring_simple_ch8:1767 < fc_fcollect_only_c0_default:7062 | r0_full_serial_c0_default:1859 < r1_event_overlap_c0_default:2138 < r1_event_overlap_c2_ring_simple_ch8:2332 | RCCL | RCCL | CONSISTENT | -22.639 / -9.094 | CONSISTENT |
| 2048 | 2 | comm_only_c0_default:1179 < comm_only_c2_ring_simple_ch8:1405 < dc_pushsig_only_c0_default:4022 | r0_full_serial_c0_default:4117 < r1_event_overlap_c2_ring_simple_ch8:4283 < rs_slice_serial_c0_default:4352 | RCCL | RCCL | CONSISTENT | -19.244 / 2.308 | RCCL_CONFIG_REVERSAL |
| 2048 | 4 | comm_only_c0_default:1270 < comm_only_c2_ring_simple_ch8:1535 < dc_pushsig_only_c0_default:5129 | r0_full_serial_c0_default:4125 < r1_event_overlap_c2_ring_simple_ch8:4293 < r1_event_overlap_c0_default:4301 | RCCL | RCCL | CONSISTENT | -20.885 / 0.206 | CONSISTENT |
| 2048 | 8 | comm_only_c0_default:1496 < comm_only_c2_ring_simple_ch8:1810 < fc_fcollect_only_c0_default:7087 | r0_full_serial_c0_default:4128 < r1_event_overlap_c0_default:6618 < r1_event_overlap_c2_ring_simple_ch8:6620 | RCCL | RCCL | CONSISTENT | -20.990 / -0.016 | CONSISTENT |
| 2048 | 16 | comm_only_c0_default:1872 < dc_pushsig_only_c0_default:17303 | r1_event_overlap_c0_default:6820 < rs_slice_serial_c0_default:7952 < d1_pushsig_overlap_c0_default:29448 | RCCL | RCCL | CONSISTENT |  /  | N/A |
| 4096 | 2 | comm_only_c0_default:1217 < comm_only_c2_ring_simple_ch8:1462 < dc_pushsig_only_c0_default:4126 | r0_full_serial_c0_default:9157 < r1_event_overlap_c2_ring_simple_ch8:9333 < r1_event_overlap_c0_default:9425 | RCCL | RCCL | CONSISTENT | -20.169 / 0.982 | CONSISTENT |
| 4096 | 4 | comm_only_c0_default:1317 < comm_only_c2_ring_simple_ch8:1582 < dc_pushsig_only_c0_default:5886 | r1_event_overlap_c2_ring_simple_ch8:8367 < r1_event_overlap_c0_default:8413 < rs_slice_serial_c0_default:8674 | RCCL | RCCL | CONSISTENT | -20.119 / 0.548 | CONSISTENT |
| 4096 | 8 | comm_only_c0_default:1561 < comm_only_c2_ring_simple_ch8:1857 < fc_fcollect_only_c0_default:7118 | r1_event_overlap_c2_ring_simple_ch8:8174 < r1_event_overlap_c0_default:8376 < rs_slice_serial_c0_default:8643 | RCCL | RCCL | CONSISTENT | -18.981 / 2.408 | RCCL_CONFIG_REVERSAL |
| 4096 | 16 | comm_only_c0_default:1918 < comm_only_c2_ring_simple_ch8:2360 | r1_event_overlap_c0_default:12260 < r1_event_overlap_c2_ring_simple_ch8:12331 | RCCL | RCCL | CONSISTENT | -22.994 / -0.577 | CONSISTENT |

Cross-substrate reversal cells: 0 / 11; RCCL-internal C0/C2 reversal cells: 2 / 11

## Case status

