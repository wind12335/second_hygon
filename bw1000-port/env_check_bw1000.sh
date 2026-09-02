#!/usr/bin/env bash
# bw1000 8-GPU 环境体检：只读探测，不改任何东西。
# 产出 ./env_check_report_bw1000.txt —— 发回给分析侧确认后再继续后续步骤。
set -uo pipefail
OUT="env_check_report_bw1000.txt"
: > "${OUT}"

log()  { echo "$@" | tee -a "${OUT}"; }
sec()  { echo | tee -a "${OUT}"; echo "===== $1 =====" | tee -a "${OUT}"; }
run()  { # run <标题> <命令...>
  sec "$1"; shift; eval "$@" 2>&1 | tee -a "${OUT}" || true
}

log "bw1000 env check  started=$(date '+%F %T')  host=$(hostname)"

run "OS / CPU / 内存" "head -3 /etc/os-release; lscpu | grep -E '^(Model name|Socket|NUMA|CPU\(s\))' | head -8; free -g | head -2"

run "GPU 产品名 (rocm-smi 或 hy-smi)" "rocm-smi --showproductname --showdriverversion 2>&1 || hy-smi --showproductname 2>&1 || true"

run "GPU 数量与序号" "rocm-smi --showid 2>&1 || hy-smi --showid 2>&1 || true"

run "GPU 拓扑 (P2P/PCIe)" "rocm-smi --showtopo 2>&1 | head -20 || true; rocm-smi --showtopoweight 2>&1 | head -12 || true"

run "GPU 架构名 (rocminfo: 请从输出里抄 gfx 名字)" "rocminfo 2>/dev/null | grep -E 'Marketing|gfx' | head -10 || true"

run "hipcc / hipconfig" "which hipcc; hipcc --version 2>&1 | head -6; hipconfig --full 2>&1 | head -30"

run "DTK 目录" "ls -d /opt/dtk* /opt/rocm* 2>/dev/null; ls /opt/dtk/include 2>/dev/null | grep -iE 'dushmem|rccl|hip' | head; ls /opt/dtk/lib 2>/dev/null | grep -iE 'dushmem|rccl|rocblas' | head"

run "DUSHMEM 头文件与版本宏" "H=\$(find -L /opt/dtk -name 'dushmem.h' 2>/dev/null | head -1); echo header=\$H; [ -n \"\$H\" ] && grep -iE 'VERSION|version' \"\$H\" | head -5"

run "DUSHMEM 关键符号 (5 个 on-stream 原语)" "for L in \$(find -L /opt/dtk \( -name 'libdushmem_host.so*' -o -name 'libdushmem_device.*' \) 2>/dev/null | head -3); do echo \"-- \$L\"; nm -D \"\$L\" 2>/dev/null | grep -E 'putmem_signal_on_stream|signal_wait_until_on_stream|fcollectmem_on_stream|quiet_on_stream|signal_op_on_stream'; done"

run "RCCL 库与版本" "R=\$(find -L /opt/dtk -name 'librccl.so*' 2>/dev/null | head -1); echo lib=\$R; [ -n \"\$R\" ] && strings \"\$R\" | grep -iE '^2\\.[0-9]+' | head -3"

run "MPI" "which mpirun mpiexec 2>/dev/null; mpirun --version 2>&1 | head -2"

run "Python3" "python3 --version 2>&1"

# ---- 汇总判定 ----
sec "SUMMARY (自动检查)"
NGPU=$( (rocm-smi --showid 2>/dev/null || hy-smi --showid 2>/dev/null) | grep -cE '^(GPU|HCU)\[' || true)
NDUSH=$(find -L /opt/dtk \( -name 'libdushmem_device.*' -o -name 'libdushmem_host.*' \) 2>/dev/null | wc -l)
NRCCL=$(find -L /opt/dtk -name 'librccl.so*' 2>/dev/null | wc -l)
NMPI=$(which mpirun 2>/dev/null | wc -l)
GFX=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1)
{
  echo "gpu_count        = ${NGPU}   (期望 8)"
  echo "gfx_arch         = ${GFX:-未识别}   (编译时填进 make ARCH=)"
  echo "dushmem_present  = $([ "${NDUSH}" -ge 1 ] && echo YES || echo NO)   (NO 则 d 族路径不可跑, 见手册附录B)"
  echo "rccl_present     = $([ "${NRCCL}" -ge 1 ] && echo YES || echo NO)"
  echo "mpirun_present   = $([ "${NMPI}" -ge 1 ] && echo YES || echo NO)"
} | tee -a "${OUT}"

echo
echo "== 体检完成: 把整个文件发回去: ${OUT} =="
