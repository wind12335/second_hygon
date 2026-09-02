#!/usr/bin/env bash
# Phase B runner — P16: 边界定律外推三格裁决 (2026-09-03).
# 预注册: phaseb/P16_预注册_边界外推_20260903.md（发射前已 commit）。
# 纪律: 不 make；二进制 sha 必须以 f17aae1d 开头；与 P15 不同结果根；
#       格序 (4096,32)→(8192,8)→(8192,16)——q32 首个 case 即天然冒烟；
#       rep 外层交错配对；时间戳结果根绝不覆盖；失败 case 保留。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="${PHASEB_RESULT_BASE:-$(dirname "${SCRIPT_DIR}")/results/k500sm_ai_gfx928_4gpu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_p16_extrap_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"
[[ -e "${LOCK}" ]] && { echo "refusing to reuse result root ${RESULT_ROOT}" >&2; exit 1; }

[[ -x "${BINARY}" ]] || { echo "binary missing" >&2; exit 1; }
EXPECTED_SHA_PREFIX="f17aae1d"
ACTUAL_SHA="$(sha256sum "${BINARY}" | awk '{print $1}')"
[[ "${ACTUAL_SHA}" == "${EXPECTED_SHA_PREFIX}"* ]] || {
  echo "BINARY SHA GUARD FAILED: expected prefix ${EXPECTED_SHA_PREFIX}, got ${ACTUAL_SHA}" >&2
  echo "二进制与终判表不同族 — 检查是否有人 make 过；本批必须用 f17aae1d 原二进制。" >&2
  exit 1; }

export HIP_VISIBLE_DEVICES=0,1,2,3
export DUSHMEM_SYMMETRIC_SIZE=1G
export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}

M_LOCAL=2048; K=2048
REPS=5; WARMUP=20; ITERS=80; CASE_TIMEOUT=1200
PATHS=(d0 d1)

# 格序 = 风险序: q32 超包络排首，天然冒烟
CELLS=(
  "4096 32"
  "8192 8"
  "8192 16"
)

{
  echo "mode=p16(extrap) np=4"
  echo "cells=${CELLS[*]}"
  echo "paths=${PATHS[*]}"
  echo "result_root=${RESULT_ROOT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "binary=${BINARY}"
  echo "binary_sha256=${ACTUAL_SHA}"
} > "${RESULT_ROOT}/run_metadata.txt"

{
  echo "=== rocm-smi ==="; rocm-smi --showproductname --showid 2>&1 || true
  echo "=== hipconfig ==="; hipconfig --full 2>&1 | head -40 || true
  echo "=== binary ldd ==="; ldd "${BINARY}" 2>&1 || true
} > "${RESULT_ROOT}/platform/platform_facts.txt"

cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" "${SCRIPT_DIR}/run_phaseb_p16.sh" \
   "${SCRIPT_DIR}/analyze_p16.py" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

CASE_SEQ=0; TOTAL_FAIL=0
run_case() { # path N Q rep
  local path="$1" n="$2" q="$3" rep="$4"
  CASE_SEQ=$((CASE_SEQ + 1))
  local case_id
  case_id="$(printf 'case%03d_%s_C0_DEFAULT_N%s_q%s_rep%d' "${CASE_SEQ}" "${path}" "${n}" "${q}" "${rep}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}"
  local status=0
  timeout "${CASE_TIMEOUT}" \
    mpirun --allow-run-as-root -np 4 -mca coll ^hcoll "${BINARY}" \
    --path "${path}" --m-local "${M_LOCAL}" --n "${n}" --k "${K}" --q "${q}" \
    --warmup "${WARMUP}" --iters "${ITERS}" --verify-every 1 --window-mult 1 \
    --output-dir "${case_dir}" --run-id "PHASEB_P16_${path}_N${n}_q${q}_rep${rep}" \
    --candidate C0_DEFAULT \
    < /dev/null > "${case_dir}/stdout_stderr.log" 2>&1 || status=$?
  echo "${status}" > "${case_dir}/exit_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "[FAIL] ${case_id} exit=${status}"
  else
    echo "[ ok ] ${case_id}"
  fi
}

for cell in "${CELLS[@]}"; do
  read -r n q <<< "${cell}"
  echo "-- cell N${n}/q${q} --"
  for ((rep = 1; rep <= REPS; rep++)); do
    for path in "${PATHS[@]}"; do
      run_case "${path}" "${n}" "${q}" "${rep}"
    done
  done
done

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb P16 complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
python3 "${SCRIPT_DIR}/analyze_p16.py" --result-root "${RESULT_ROOT}" || \
  echo "p16 summarizer failed; raw data intact under ${RESULT_ROOT}/cases"
