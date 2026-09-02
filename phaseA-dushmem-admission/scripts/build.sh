#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# DTK 26.04 supplies this stable symlink, while retaining a versioned install
# directory underneath. Fail early with a useful error on a host without HIP.
if [[ ! -x /opt/dtk/bin/hipcc ]]; then
  echo "Missing HIP compiler: expected /opt/dtk/bin/hipcc" >&2
  exit 69
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DDUSHMEM_ADMISSION_ARCH=gfx928 \
  -DDTK_ROOT=/opt/dtk \
  -DMPI_ROOT=/opt/mpi
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-$(nproc)}"
