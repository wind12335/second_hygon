#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

OUT_DIR="$1"
mkdir -p "${OUT_DIR}"

capture() {
  local name="$1"
  shift
  "$@" >"${OUT_DIR}/${name}" 2>&1 || true
}

capture date.txt date --iso-8601=seconds
capture uname.txt uname -a
capture os_release.txt cat /etc/os-release
capture lscpu.txt lscpu
capture hipcc_version.txt /opt/dtk/bin/hipcc --version
capture dushmem_info.txt /opt/dtk/dushmem/bin/dushmem-info
capture rocminfo.txt rocminfo
if command -v rocm-smi >/dev/null 2>&1; then
  capture rocm_smi.txt rocm-smi
fi
capture dushmem_host_ldd.txt ldd /opt/dtk/dushmem/lib/libdushmem_host.so
capture mpi_version.txt mpirun --version
capture dushmem_headers.txt rg -n 'putmem_signal|quiet_on_stream|signal_wait_until|fcollectmem_on_stream|dushmem_ptr' /opt/dtk/dushmem/include
printf 'target_platform=K500SM_AI / gfx928 / 4 GPUs / PCIe\n' >"${OUT_DIR}/target_platform.txt"
