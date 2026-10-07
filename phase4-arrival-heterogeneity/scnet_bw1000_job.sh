#!/bin/bash
#SBATCH --job-name=arrival_hetero
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:15:00
#SBATCH --output=arrival_%j.log
#
# BW1000 双机 AG 到达异质性测量
# 在 SCNet 提交: sbatch scnet_bw1000_job.sh
# 输出: arrival_results_YYYYMMDD_HHMMSS.json
#

echo "=============================================="
echo " BW1000 Arrival Heterogeneity Measurement"
echo " Date: $(date)"
echo " Job ID: ${SLURM_JOB_ID:-manual}"
echo "=============================================="

# 记录节点信息
echo "Nodes assigned:"
if command -v scontrol &>/dev/null; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | tee /tmp/nodes.txt
else
    hostname | tee /tmp/nodes.txt
fi
echo ""

# 环境检查
echo "--- Environment Check ---"
echo "Python: $(python3 --version 2>&1)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA/HIP devices: $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>&1)"
echo "NCCL available: $(python3 -c 'import torch.distributed; print("yes")' 2>&1)"
echo ""

# 检查 GPU 可见性
python3 -c "
import torch
n = torch.cuda.device_count()
if n == 0:
    print('ERROR: No GPU visible!')
    exit(1)
for i in range(min(n, 3)):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory // 1024 // 1024} MB)')
if n >= 8:
    print(f'  ... and {n - 3} more')
" 2>&1
echo ""

# 设置分布式环境变量
export MASTER_ADDR=$(head -1 /tmp/nodes.txt 2>/dev/null || hostname)
export MASTER_PORT=29500
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

# 运行测量
echo "--- Running Measurement ---"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="arrival_results_${TIMESTAMP}.json"

# 方法1: 使用 torchrun (推荐)
if command -v torchrun &>/dev/null; then
    echo "Using torchrun..."
    torchrun --nproc_per_node=8 --nnodes=2 \
        --node_rank=${SLURM_NODEID:-0} \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        measure_arrival.py \
        --chunk-mb 8 \
        --iterations 100 \
        --warmup 10 \
        --gemm-shape 8192 4096 8192 \
        --output "$OUTPUT_FILE"
else
    # 方法2: 使用 srun
    echo "Using srun + torch.distributed..."
    srun python3 measure_arrival.py \
        --chunk-mb 8 \
        --iterations 100 \
        --warmup 10 \
        --gemm-shape 8192 4096 8192 \
        --output "$OUTPUT_FILE"
fi

echo ""
echo "--- Results ---"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file: $OUTPUT_FILE"
    echo ""
    # Print summary
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    r = json.load(f)
s = r.get('summary', {})
print('Same-node median:  {:.3f} ms'.format(s.get('same_node_median_ms', 0)))
print('Cross-node median: {:.3f} ms'.format(s.get('cross_node_median_ms', 0)))
print('Spread:            {:.3f} ms'.format(s.get('spread_ms', 0)))
print('vs GEMM:           {:.1f}%'.format(s.get('spread_vs_gemm_pct', 0)))
print('Conclusion:        {}'.format(s.get('conclusion', 'N/A')))
" 2>/dev/null || echo "(Could not parse results)"
else
    echo "WARNING: Output file not found!"
fi

echo ""
echo "=============================================="
echo " Done. $(date)"
echo "=============================================="
