#!/usr/bin/env bash
# env_check.sh — NVIDIA 侧交接第一步：机器形态与工具链探测。
# 只读探测 + 编译运行 p2p_probe，不跑任何正式实验。
# 产出: env_check_report.txt —— 请把这个文件发回来，我确认后再跑 smoke。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=env_check_report.txt

{
  echo "=== env_check $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) user=$(whoami) ==="
  echo "=== os / kernel ==="
  cat /etc/os-release 2>/dev/null | head -5
  uname -a
  echo
  echo "=== gpu 列表 ==="
  nvidia-smi 2>&1 | head -30
  echo
  echo "=== 拓扑（P2P 判定的第一证据）==="
  nvidia-smi topo -m 2>&1
  echo
  echo "=== 驱动 ==="
  cat /proc/driver/nvidia/version 2>/dev/null
  echo
  echo "=== nvcc / CUDA ==="
  which nvcc && nvcc --version
  ls -d /usr/local/cuda* 2>/dev/null
  cat /usr/local/cuda/version.json 2>/dev/null | head -8
  echo
  echo "=== 库探测 (nccl / nvshmem / cublas) ==="
  ldconfig -p 2>/dev/null | grep -Ei 'libnccl|libnvshmem|libcublas' | head -20
  for d in /usr/local /opt /usr; do
    find "$d" -maxdepth 4 \( -name 'nvshmem.h' -o -name 'nccl.h' \) 2>/dev/null | head -10
  done
  echo "NVSHMEM_HOME=${NVSHMEM_HOME:-<unset>}"
  echo "NCCL_HOME=${NCCL_HOME:-<unset>}"
  echo
  echo "=== 头文件版本号 ==="
  for h in $(find /usr/local /opt /usr -maxdepth 5 -name 'nvshmem.h' 2>/dev/null | head -2); do
    echo "-- $h"; grep -E 'NVSHMEM_(MAJOR|MINOR|PATCH)' "$h" | head -5
  done
  for h in $(find /usr/local /opt /usr -maxdepth 5 -name 'nccl.h' 2>/dev/null | head -2); do
    echo "-- $h"; grep -E '#define NCCL_(MAJOR|MINOR|PATCH)' "$h" | head -5
  done
  echo
  echo "=== MPI ==="
  which mpirun && mpirun --version 2>&1 | head -3
  which mpicc
  echo
  echo "=== gcc / 磁盘 / 内存 ==="
  gcc --version 2>/dev/null | head -1
  df -h . | tail -1
  free -g | head -2
} > "$OUT" 2>&1

# P2P 实测（nvcc 存在才编译）
if command -v nvcc >/dev/null 2>&1; then
  echo >> "$OUT"
  echo "=== p2p_probe 编译+运行 ===" >> "$OUT"
  if nvcc -o p2p_probe p2p_probe.cu >> "$OUT" 2>&1; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./p2p_probe >> "$OUT" 2>&1
    echo "p2p_probe exit=$?" >> "$OUT"
  else
    echo "p2p_probe COMPILE FAILED（把错误发回来）" >> "$OUT"
  fi
else
  echo "nvcc not found —— 无法编译 p2p_probe" >> "$OUT"
fi

echo "done -> ${OUT} ($(wc -l < "$OUT") lines)"
echo "请把 ${OUT} 发回来给我确认，再进行下一步。"
