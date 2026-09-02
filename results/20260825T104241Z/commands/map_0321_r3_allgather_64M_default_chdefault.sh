#!/usr/bin/env bash
# UTC 2026-08-25T10:43:29Z
cd /root/private_data/lyc/rccl-tests
env env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS HIP_VISIBLE_DEVICES=0\,3\,2\,1 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=ERROR NCCL_DEBUG_SUBSYS=INIT mpirun --allow-run-as-root -np 4 -mca coll \^hcoll /root/private_data/lyc/rccl-tests/build/all_gather_perf -b 64M -e 64M -g 1 -w 10 -n 50 -c 1 -a 3 -d float -Z csv -x /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/20260825T104241Z/csv/map_0321_r3_allgather_64M_default_chdefault.csv -O 1
