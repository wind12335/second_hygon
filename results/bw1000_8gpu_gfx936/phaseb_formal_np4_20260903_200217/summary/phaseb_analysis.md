# Phase B analysis (cross-substrate AG-GEMM)

- result root: `/root/private_data/lyc/bw1000-port/results/bw1000_8gpu/phaseb_formal_np4_20260903_200217`
- cases: 645 (PASS 645, other 0)

## Selection boundary (isolated winner vs e2e winner)

| cand | N | q | isolated winner | e2e winner | iso family | e2e family | flag |
|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | COMM_ONLY (535.761us) | R0_FULL_SERIAL (1206.319us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 4 | COMM_ONLY (620.000us) | R1_EVENT_OVERLAP (988.958us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 512 | 8 | COMM_ONLY (797.439us) | R0_FULL_SERIAL (1203.919us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 2 | COMM_ONLY (579.520us) | R0_FULL_SERIAL (1898.318us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 4 | COMM_ONLY (653.120us) | R0_FULL_SERIAL (1898.159us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 8 | COMM_ONLY (823.600us) | R0_FULL_SERIAL (1899.120us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 2048 | 16 | COMM_ONLY (1100.240us) | R1_EVENT_OVERLAP (2626.159us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 2 | COMM_ONLY (616.480us) | RS_SLICE_SERIAL (4285.279us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 4 | COMM_ONLY (688.880us) | R1_EVENT_OVERLAP (4099.522us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 8 | COMM_ONLY (853.361us) | R0_FULL_SERIAL (4360.957us) | RCCL | RCCL | CONSISTENT |
| C0_DEFAULT | 4096 | 16 | COMM_ONLY (1114.880us) | R1_EVENT_OVERLAP (6337.439us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 2 | COMM_ONLY (646.321us) | R1_EVENT_OVERLAP (1269.599us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 4 | COMM_ONLY (735.761us) | R1_EVENT_OVERLAP (1084.320us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 512 | 8 | COMM_ONLY (915.280us) | R1_EVENT_OVERLAP (1331.120us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 2 | COMM_ONLY (680.801us) | R1_EVENT_OVERLAP (2247.041us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 4 | COMM_ONLY (761.760us) | R1_EVENT_OVERLAP (3275.680us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 2048 | 8 | COMM_ONLY (936.961us) | R1_EVENT_OVERLAP (3444.720us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 2 | COMM_ONLY (717.120us) | R1_EVENT_OVERLAP (4301.360us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 4 | COMM_ONLY (796.160us) | R1_EVENT_OVERLAP (4052.002us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 8 | COMM_ONLY (962.321us) | R1_EVENT_OVERLAP (6049.999us) | RCCL | RCCL | CONSISTENT |
| C2_RING_SIMPLE_CH8 | 4096 | 16 | COMM_ONLY (1303.600us) | R1_EVENT_OVERLAP (6345.675us) | RCCL | RCCL | CONSISTENT |

## Control table (positive = listed path faster)

| cand | N | q | r1/rs | r1/r0 | d1/ds | d1/d0 | r1/d1 | d1-done vs dc-done (transport stretch) |
|---|---|---|---|---|---|---|---|---|
| C0_DEFAULT | 512 | 2 | 4.460 | -3.707 | -0.246 | 52.855 | 33.855 | 98.813 |
| C0_DEFAULT | 512 | 4 | 19.753 | 18.241 | -4.146 | 48.882 | 51.803 | 72.625 |
| C0_DEFAULT | 512 | 8 | 23.228 | -6.625 | 5.510 | 19.508 | 60.268 | 67.572 |
| C0_DEFAULT | 2048 | 2 | 1.937 | -17.970 | 0.242 | 40.339 | 20.095 | 193.527 |
| C0_DEFAULT | 2048 | 4 | 7.095 | -73.886 | 1.175 | 7.706 | 23.958 | 243.976 |
| C0_DEFAULT | 2048 | 8 | 9.783 | -79.936 | -2.932 | -138.095 | 69.475 | 476.597 |
| C0_DEFAULT | 2048 | 16 | 26.967 |  | 4.684 |  | 82.642 | 368.653 |
| C0_DEFAULT | 4096 | 2 | -1.878 | -0.908 | -0.821 | 33.474 | 8.724 | 391.418 |
| C0_DEFAULT | 4096 | 4 | 2.742 | 6.663 | -0.265 | 29.705 | 18.805 | 308.598 |
| C0_DEFAULT | 4096 | 8 | 7.960 | -39.063 | 0.429 | -14.248 | 26.194 | 317.824 |
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
| 512 | 2 | comm_only_c0_default:536 < comm_only_c2_ring_simple_ch8:646 < dc_pushsig_only_c0_default:950 | r0_full_serial_c0_default:1206 < r1_event_overlap_c0_default:1251 < r1_event_overlap_c2_ring_simple_ch8:1270 | RCCL | RCCL | CONSISTENT | -20.636 / -1.483 | CONSISTENT |
| 512 | 4 | comm_only_c0_default:620 < comm_only_c2_ring_simple_ch8:736 < dc_pushsig_only_c0_default:1187 | r1_event_overlap_c0_default:989 < r1_event_overlap_c2_ring_simple_ch8:1084 < r0_full_serial_c0_default:1210 | RCCL | RCCL | CONSISTENT | -18.671 / -9.643 | CONSISTENT |
| 512 | 8 | comm_only_c0_default:797 < comm_only_c2_ring_simple_ch8:915 < dc_pushsig_only_c0_default:1923 | r0_full_serial_c0_default:1204 < r1_event_overlap_c0_default:1284 < r1_event_overlap_c2_ring_simple_ch8:1331 | RCCL | RCCL | CONSISTENT | -14.777 / -3.696 | CONSISTENT |
| 2048 | 2 | comm_only_c0_default:580 < comm_only_c2_ring_simple_ch8:681 < dc_pushsig_only_c0_default:954 | r0_full_serial_c0_default:1898 < r1_event_overlap_c0_default:2239 < r1_event_overlap_c2_ring_simple_ch8:2247 | RCCL | RCCL | CONSISTENT | -17.477 / -0.339 | CONSISTENT |
| 2048 | 4 | comm_only_c0_default:653 < comm_only_c2_ring_simple_ch8:762 < dc_pushsig_only_c0_default:1261 | r0_full_serial_c0_default:1898 < r1_event_overlap_c2_ring_simple_ch8:3276 < r1_event_overlap_c0_default:3301 | RCCL | RCCL | CONSISTENT | -16.634 / 0.756 | CONSISTENT |
| 2048 | 8 | comm_only_c0_default:824 < comm_only_c2_ring_simple_ch8:937 < dc_pushsig_only_c0_default:1941 | r0_full_serial_c0_default:1899 < r1_event_overlap_c0_default:3417 < r1_event_overlap_c2_ring_simple_ch8:3445 | RCCL | RCCL | CONSISTENT | -13.764 / -0.805 | CONSISTENT |
| 2048 | 16 | comm_only_c0_default:1100 < dc_pushsig_only_c0_default:3228 | r1_event_overlap_c0_default:2626 < rs_slice_serial_c0_default:3596 < d1_pushsig_overlap_c0_default:15129 | RCCL | RCCL | CONSISTENT |  /  | N/A |
| 4096 | 2 | comm_only_c0_default:616 < comm_only_c2_ring_simple_ch8:717 < dc_pushsig_only_c0_default:973 | rs_slice_serial_c0_default:4285 < r1_event_overlap_c2_ring_simple_ch8:4301 < r0_full_serial_c0_default:4326 | RCCL | RCCL | CONSISTENT | -16.325 / 1.475 | RCCL_CONFIG_REVERSAL |
| 4096 | 4 | comm_only_c0_default:689 < comm_only_c2_ring_simple_ch8:796 < dc_pushsig_only_c0_default:1235 | r1_event_overlap_c2_ring_simple_ch8:4052 < r1_event_overlap_c0_default:4100 < rs_slice_serial_c0_default:4215 | RCCL | RCCL | CONSISTENT | -15.573 / 1.159 | RCCL_CONFIG_REVERSAL |
| 4096 | 8 | comm_only_c0_default:853 < comm_only_c2_ring_simple_ch8:962 < dc_pushsig_only_c0_default:1965 | r0_full_serial_c0_default:4361 < r1_event_overlap_c2_ring_simple_ch8:6050 < r1_event_overlap_c0_default:6064 | RCCL | RCCL | CONSISTENT | -12.768 / 0.239 | CONSISTENT |
| 4096 | 16 | comm_only_c0_default:1115 < comm_only_c2_ring_simple_ch8:1304 | r1_event_overlap_c0_default:6337 < r1_event_overlap_c2_ring_simple_ch8:6346 | RCCL | RCCL | CONSISTENT | -16.927 / -0.130 | CONSISTENT |

Cross-substrate reversal cells: 0 / 11; RCCL-internal C0/C2 reversal cells: 2 / 11

## Case status

