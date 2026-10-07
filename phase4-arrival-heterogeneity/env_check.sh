#!/bin/bash
# 环境自检：在 SCNet 登录节点或计算节点上运行
# 确认 GPU 可见、PyTorch 正常、跨节点通信就绪

echo "=== BW1000 环境自检 ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "--- Python ---"
python3 --version
echo ""

echo "--- PyTorch ---"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/HIP available: {torch.cuda.is_available()}')
n = torch.cuda.device_count()
print(f'GPU count: {n}')
if n > 0:
    for i in range(min(n, 8)):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory // (1024*1024)} MB)')
else:
    print('WARNING: No GPU visible!')
"
echo ""

echo "--- Distributed ---"
python3 -c "
import torch.distributed as dist
print('torch.distributed available: yes')
print('Backends: nccl={}, gloo={}'.format(
    dist.is_nccl_available(), dist.is_gloo_available()))
" 2>&1
echo ""

echo "--- Network ---"
echo "Interconnect (if known): $BW1000_INTERCONNECT"
# Test hostname resolution
python3 -c "
import socket
try:
    print(f'Hostname resolves: {socket.gethostbyname(socket.gethostname())}')
except:
    print('WARNING: hostname does not resolve')
"
echo ""

echo "--- Disk space ---"
df -h . | tail -1
echo ""

echo "=== 检查完成 ==="
echo "如果一切正常，请运行: sbatch scnet_bw1000_job.sh"
