#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_PATH="${ROOT_DIR}/configs/admission_smoke.csv"
RUN_ID="phase4_dushmem_admission_$(date -u +%Y%m%dT%H%M%SZ)"
TIMEOUT_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: run_admission.sh [options]

  --smoke                 Run configs/admission_smoke.csv (default).
  --formal                Run configs/admission_formal.csv.
  --matrix PATH           Run an explicit CSV matrix.
  --run-id ID             Use a fixed, non-existing result directory name.
  --timeout-seconds N     Override every case timeout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MATRIX_PATH="${ROOT_DIR}/configs/admission_smoke.csv" ;;
    --formal) MATRIX_PATH="${ROOT_DIR}/configs/admission_formal.csv" ;;
    --matrix) MATRIX_PATH="$2"; shift ;;
    --run-id) RUN_ID="$2"; shift ;;
    --timeout-seconds) TIMEOUT_OVERRIDE="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

if [[ ! -f "${MATRIX_PATH}" ]]; then
  echo "Matrix not found: ${MATRIX_PATH}" >&2
  exit 66
fi

RUN_ROOT="${ROOT_DIR}/results/${RUN_ID}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${RUN_ROOT}" >&2
  exit 73
fi

mkdir -p "${RUN_ROOT}/cases" "${RUN_ROOT}/platform" "${RUN_ROOT}/source_snapshot"
MASTER_LOG="${RUN_ROOT}/master.log"
log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${MASTER_LOG}"
}

log "run_root=${RUN_ROOT}"
log "matrix=${MATRIX_PATH}"
log "target_platform=K500SM_AI / gfx928 / 4 GPUs / PCIe"

set +e
"${ROOT_DIR}/scripts/build.sh" >"${RUN_ROOT}/build.log" 2>&1
BUILD_STATUS=$?
set -e
printf '%s\n' "${BUILD_STATUS}" >"${RUN_ROOT}/build_exit_status.txt"
if [[ ${BUILD_STATUS} -ne 0 ]]; then
  log "build failed; inspect build.log"
  exit "${BUILD_STATUS}"
fi

"${ROOT_DIR}/scripts/collect_platform.sh" "${RUN_ROOT}/platform"
cp "${ROOT_DIR}/CMakeLists.txt" "${ROOT_DIR}/src/dushmem_admission.cpp" "${RUN_ROOT}/source_snapshot/"
cp "${ROOT_DIR}/scripts/build.sh" "${ROOT_DIR}/scripts/collect_platform.sh" \
   "${ROOT_DIR}/scripts/run_admission.sh" "${ROOT_DIR}/scripts/analyze_admission.py" \
   "${RUN_ROOT}/source_snapshot/"
cp "${MATRIX_PATH}" "${RUN_ROOT}/source_snapshot/matrix.csv"
(cd "${RUN_ROOT}/source_snapshot" && sha256sum * > sha256.txt)

printf 'case_id,mode,payload_bytes,epochs,slots,credit,quiet,credit_quiet,expected_pes,timeout_seconds,exit_status,case_directory\n' >"${RUN_ROOT}/manifest.csv"

BIN="${ROOT_DIR}/build/dushmem_admission"
export DUSHMEM_SYMMETRIC_SIZE="${DUSHMEM_SYMMETRIC_SIZE:-2G}"
export LD_LIBRARY_PATH="/opt/dtk/dushmem/lib:${LD_LIBRARY_PATH:-}"

while IFS=, read -r case_id mode payload_bytes epochs slots credit quiet credit_quiet timeout_seconds; do
  [[ -z "${case_id}" || "${case_id}" == \#* ]] && continue
  if [[ -n "${TIMEOUT_OVERRIDE}" ]]; then timeout_seconds="${TIMEOUT_OVERRIDE}"; fi
  case_dir="${RUN_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}/raw"
  command=(
    timeout --foreground --kill-after=30s "${timeout_seconds}s"
    mpirun --allow-run-as-root --bind-to none -mca coll ^hcoll -np 4
    -x HIP_VISIBLE_DEVICES=0,1,2,3
    -x DUSHMEM_SYMMETRIC_SIZE
    -x LD_LIBRARY_PATH
    "${BIN}"
    --case-id "${case_id}"
    --mode "${mode}"
    --outdir "${case_dir}"
    --payload-bytes "${payload_bytes}"
    --epochs "${epochs}"
    --slots "${slots}"
    --credit "${credit}"
    --quiet "${quiet}"
    --credit-quiet "${credit_quiet}"
    --expected-pes 4
  )
  printf '%q ' "${command[@]}" >"${case_dir}/command.txt"
  printf '\n' >>"${case_dir}/command.txt"
  log "case=${case_id} start"
  set +e
  # Do not let mpirun inherit the matrix CSV on stdin. Open MPI otherwise
  # forwards it to rank 0 and consumes the remaining case definitions.
  "${command[@]}" </dev/null >"${case_dir}/stdout_stderr.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "${status}" >"${case_dir}/exit_status.txt"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,4,%s,%s,%s\n' \
    "${case_id}" "${mode}" "${payload_bytes}" "${epochs}" "${slots}" "${credit}" "${quiet}" \
    "${credit_quiet}" "${timeout_seconds}" "${status}" "${case_dir}" >>"${RUN_ROOT}/manifest.csv"
  if [[ ${status} -eq 0 ]]; then
    log "case=${case_id} PASS"
  else
    log "case=${case_id} EXIT_STATUS=${status}; raw logs retained"
  fi
done <"${MATRIX_PATH}"

python3 "${ROOT_DIR}/scripts/analyze_admission.py" "${RUN_ROOT}" >"${RUN_ROOT}/analysis.log" 2>&1
log "analysis=${RUN_ROOT}/analysis/admission_report.md"
log "completed; inspect manifest.csv and analysis/case_summary.csv"
