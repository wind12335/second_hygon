#!/usr/bin/env bash
# Phase B runner — q16 N-boundary fill-in (2026-09-02).
# 单一格子 N4096/q16 × d 族五路径，验证 q16 爆炸是否有 N 边界（预注册 P14）。
# 纪律：不 make（二进制必须保持 f17aae1d 与 dsfix/d0dc 同族）；时间戳根绝不覆盖。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="${PHASEB_RESULT_BASE:-$(dirname "${SCRIPT_DIR}")/results/k500sm_ai_gfx928_4gpu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_q16fill_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"
[[ -e "${LOCK}" ]] && { echo "refusing to reuse result root ${RESULT_ROOT}" >&2; exit 1; }

[[ -x "${BINARY}" ]] || { echo "binary missing — 先 make" >&2; exit 1; }

export HIP_VISIBLE_DEVICES=0,1,2,3
export DUSHMEM_SYMMETRIC_SIZE=1G
export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}

PATHS=(d0 dc d1 ds d1w)
M_LOCAL=2048; K=2048; N=4096; Q=16
REPS=5; WARMUP=20; ITERS=80; CASE_TIMEOUT=900

{
  echo "mode=q16fill(P14) np=4"
  echo "paths=${PATHS[*]}"
  echo "cell=N${N}/q${Q}"
  echo "result_root=${RESULT_ROOT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "binary=${BINARY}"
  echo "binary_sha256=$(sha256sum "${BINARY}" | awk '{print $1}')"
} > "${RESULT_ROOT}/run_metadata.txt"

{
  echo "=== rocm-smi ==="; rocm-smi --showproductname --showid 2>&1 || true
  echo "=== hipconfig ==="; hipconfig --full 2>&1 | head -40 || true
  echo "=== binary ldd ==="; ldd "${BINARY}" 2>&1 || true
} > "${RESULT_ROOT}/platform/platform_facts.txt"

cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" "${SCRIPT_DIR}/run_phaseb_q16fill.sh" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

CASE_SEQ=0; TOTAL_FAIL=0
run_case() {
  local path="$1" rep="$2"
  CASE_SEQ=$((CASE_SEQ + 1))
  local case_id
  case_id="$(printf 'case%03d_%s_C0_DEFAULT_N%s_q%s_rep%d' "${CASE_SEQ}" "${path}" "${N}" "${Q}" "${rep}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}"
  local status=0
  timeout "${CASE_TIMEOUT}" \
    mpirun --allow-run-as-root -np 4 -mca coll ^hcoll "${BINARY}" \
    --path "${path}" --m-local "${M_LOCAL}" --n "${N}" --k "${K}" --q "${Q}" \
    --warmup "${WARMUP}" --iters "${ITERS}" --verify-every 1 \
    --output-dir "${case_dir}" --run-id "PHASEB_Q16FILL_${path}_N${N}_q${Q}_rep${rep}" \
    --candidate C0_DEFAULT \
    < /dev/null > "${case_dir}/stdout_stderr.log" 2>&1 || status=$?
  echo "${status}" > "${case_dir}/exit_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "[FAIL] ${case_id} exit=${status}"
  else
    echo "[ ok ] ${case_id}"
  fi
}

for path in "${PATHS[@]}"; do
  for ((rep = 1; rep <= REPS; rep++)); do
    run_case "${path}" "${rep}"
  done
done

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb q16fill complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
