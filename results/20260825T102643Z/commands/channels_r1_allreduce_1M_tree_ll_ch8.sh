#!/usr/bin/env bash
# UTC 2026-08-25T10:32:57Z
cd /root/private_data/lyc/rccl-tests
env env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS HIP_VISIBLE_DEVICES=0\,1\,2\,3 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=ERROR NCCL_DEBUG_SUBSYS=INIT NCCL_ALGO=Tree NCCL_PROTO=LL NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 mpirun --allow-run-as-root -np 4 -mca coll \^hcoll /root/private_data/lyc/rccl-tests/build/all_reduce_perf -b 1M -e 1M -g 1 -w 10 -n 50 -c 1 -a 3 -d float -Z csv -x /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/20260825T102643Z/csv/channels_r1_allreduce_1M_tree_ll_ch8.csv -O 1
