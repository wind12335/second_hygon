#!/bin/bash
# Phase 5: 端到端瓶颈发现实验（BW1000 双机 16 卡）
# 用法：两个 worker 同时执行这条命令
#
# 在 worker-0 上:
#   bash run_phase5.sh 0 <worker-0-IP>
# 在 worker-1 上:
#   bash run_phase5.sh 1 <worker-0-IP>

NODE_RANK=${1:-0}
MASTER_ADDR=${2:-$(hostname)}
MASTER_PORT=29500
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="phase5_e2e_${TIMESTAMP}.json"

echo "=============================================="
echo " Phase5: E2E Bottleneck Discovery"
echo " Node rank: $NODE_RANK"
echo " Master: $MASTER_ADDR:$MASTER_PORT"
echo " Date: $(date)"
echo "=============================================="

# 环境变量（BW1000 DTK）
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

torchrun --nnodes=2 --nproc-per-node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  measure_e2e_bottleneck.py \
  --shape 8192 4096 8192 \
  --reps 20 \
  --output "$OUTPUT" 2>&1 | tee "phase5_log_${TIMESTAMP}.log"

echo ""
echo "Done. Output: $OUTPUT"
