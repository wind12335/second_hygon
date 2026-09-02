#!/usr/bin/env bash
# Phase B formal runner: RCCL vs DUSHMEM AG-GEMM cross-substrate comparison on
# K500SM_AI / gfx928 / 4 GPUs / PCIe (DTK 26.04).
#
# Usage: bash run_phaseb.sh {smoke|formal}
#
# Conventions (inherited from phase3 runner):
#   - immutable timestamped result root, never overwrite (lock file guard)
#   - full platform facts + source snapshot + sha256 per run
#   - one directory per case with command.txt / stdout_stderr.log / exit_status.txt
#   - RCCL candidates selected purely via env (NCCL_ALGO/NCCL_PROTO/NCCL_*NCHANNELS)

set -euo pipefail

MODE="${1:-formal}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "formal" ]]; then
  echo "usage: $0 {smoke|formal}" >&2
  exit 1
fi

export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_${MODE}_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"

if [[ -e "${LOCK}" ]]; then
  echo "refusing to reuse result root ${RESULT_ROOT}" >&2
  exit 1
fi

{
  echo "mode=${MODE}"
  echo "result_root=${RESULT_ROOT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "binary=${BINARY}"
  echo "binary_sha256=$(sha256sum "${BINARY}" | awk '{print $1}')"
} > "${RESULT_ROOT}/run_metadata.txt"

# ---- platform facts --------------------------------------------------------
{
  echo "=== rocm-smi ==="
  rocm-smi --showproductname --showmeminfo vram 2>&1 || true
  echo "=== gfx arch (rocminfo subset) ==="
  rocminfo 2>/dev/null | grep -E "Marketing Name|Name:|gfx|Chip ID|PCIe" | head -80 || true
  echo "=== hipconfig ==="
  hipconfig --full 2>&1 || true
  echo "=== binary ldd ==="
  ldd "${BINARY}" 2>&1 || true
} > "${RESULT_ROOT}/platform/platform_facts.txt"

# ---- source snapshot -------------------------------------------------------
cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" \
   "${SCRIPT_DIR}/run_phaseb.sh" "${SCRIPT_DIR}/analyze_phaseb.py" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

# ---- matrix ----------------------------------------------------------------
M_LOCAL=2048
K=2048
REPS=5
WARMUP=20
ITERS=80
VERIFY_EVERY=1
CASE_TIMEOUT=900

declare -A CANDIDATES_FOR=(
  ["comm"]="C0_DEFAULT C2_RING_SIMPLE_CH8"
  ["r1"]="C0_DEFAULT C2_RING_SIMPLE_CH8"
)

if [[ "${MODE}" == "smoke" ]]; then
  NS=(2048)
  QS=(8)
  PATHS=(comm gemm r0 rs r1 fc dc d0 ds d1)
  REPS=1
  WARMUP=10
  ITERS=40
  CASE_TIMEOUT=300
else
  NS=(512 2048 4096)
  QS=(2 4 8)
  PATHS=(comm gemm r0 rs r1 fc dc d0 ds d1)
fi

candidate_env() {
  case "$1" in
    C0_DEFAULT)
      printf '%s\n' "env -u NCCL_ALGO -u NCCL_PROTO -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS" ;;
    C2_RING_SIMPLE_CH8)
      printf '%s\n' "env NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8" ;;
    *)
      echo "unknown candidate $1" >&2; exit 1 ;;
  esac
}

CASE_SEQ=0
TOTAL_FAIL=0
run_case() {
  local path="$1" cand="$2" n="$3" q="$4" rep="$5"
  CASE_SEQ=$((CASE_SEQ + 1))
  local case_id
  case_id="$(printf 'case%03d_%s_%s_N%s_q%s_rep%d' "${CASE_SEQ}" "${path}" "${cand}" "${n}" "${q}" "${rep}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}"
  local run_id="PHASEB_${MODE}_${path}_${cand}_N${n}_q${q}_rep${rep}"
  local launcher
  launcher="$(candidate_env "${cand}")"
  {
    echo "# ${case_id}"
    echo "cd ${SCRIPT_DIR}"
    echo "${launcher} mpirun --allow-run-as-root -np 4 -mca coll ^hcoll ./ag_gemm_phaseb \\"
    echo "  --path ${path} --m-local ${M_LOCAL} --n ${n} --k ${K} --q ${q} \\"
    echo "  --warmup ${WARMUP} --iters ${ITERS} --verify-every ${VERIFY_EVERY} \\"
    echo "  --output-dir ${case_dir} --run-id ${run_id} --candidate ${cand}"
  } > "${case_dir}/command.txt"

  local status=0
  ${launcher} timeout "${CASE_TIMEOUT}" \
    mpirun --allow-run-as-root -np 4 -mca coll ^hcoll "${BINARY}" \
    --path "${path}" --m-local "${M_LOCAL}" --n "${n}" --k "${K}" --q "${q}" \
    --warmup "${WARMUP}" --iters "${ITERS}" --verify-every "${VERIFY_EVERY}" \
    --output-dir "${case_dir}" --run-id "${run_id}" --candidate "${cand}" \
    < /dev/null > "${case_dir}/stdout_stderr.log" 2>&1 || status=$?
  echo "${status}" > "${case_dir}/exit_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    echo "[FAIL] ${case_id} exit=${status}"
  else
    echo "[ ok ] ${case_id}"
  fi
}

for n in "${NS[@]}"; do
  for q in "${QS[@]}"; do
    for path in "${PATHS[@]}"; do
      cands="${CANDIDATES_FOR[${path}]:-C0_DEFAULT}"
      for cand in ${cands}; do
        for ((rep = 1; rep <= REPS; rep++)); do
          run_case "${path}" "${cand}" "${n}" "${q}" "${rep}"
        done
      done
    done
  done
done

# q=16 boundary extension (formal only), N=2048.
if [[ "${MODE}" == "formal" ]]; then
  n=2048; q=16
  for path in comm gemm rs r1 dc ds d1; do
    cands="${CANDIDATES_FOR[${path}]:-C0_DEFAULT}"
    for cand in ${cands}; do
      for ((rep = 1; rep <= REPS; rep++)); do
        run_case "${path}" "${cand}" "${n}" "${q}" "${rep}"
      done
    done
  done
fi

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb ${MODE} complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
