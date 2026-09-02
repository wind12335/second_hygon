#!/usr/bin/env bash
# Phase B runner — P15: d2 window_mult 深槽位曲线 + r1 同二进制补格 (2026-09-03).
# 预注册: phaseb/P15_预注册_window_mult与r1同根_20260903.md（发射前已 commit）。
# 纪律: 不 make；二进制 sha 必须以 f17aae1d 开头（与 dsfix/d0dc/q16fill/clean 终判表同族）；
#       同 run rep 外层交错配对；时间戳结果根绝不覆盖；失败 case 保留。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="${PHASEB_RESULT_BASE:-$(dirname "${SCRIPT_DIR}")/results/k500sm_ai_gfx928_4gpu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_p15_wmr1_${STAMP}"
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
REPS=5; WARMUP=20; ITERS=80; CASE_TIMEOUT=900

# ---- Part A: window_mult 曲线（4 格 × {d0, d1@wm1, d1@wm2, d1@wm4}） ----
WM_CELLS=(
  "2048 8   main"     # 主格 (clean +29.5pt)
  "512  8   sub"      # 副格 (clean +40.0pt)
  "2048 16  boom"     # 爆炸格 (clean +69.9pt)
  "4096 8   win"      # 已赢格 (clean -21.0pt)
)
# cfg: label | path | window_mult
WM_CFGS=(
  "d0    d0 1"
  "d1wm1 d1 1"
  "d1wm2 d1 2"
  "d1wm4 d1 4"
)

# ---- Part B: r1 家族轴 11 格（补 4096/q16 成 11/11） ----
R1_CELLS=(
  "512 2"  "512 4"  "512 8"
  "2048 2" "2048 4" "2048 8" "2048 16"
  "4096 2" "4096 4" "4096 8" "4096 16"
)

{
  echo "mode=p15(wmr1) np=4"
  echo "partA=window_mult cells=${WM_CELLS[*]}"
  echo "partB=r1 cells=${R1_CELLS[*]}"
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

cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" "${SCRIPT_DIR}/run_phaseb_p15.sh" \
   "${SCRIPT_DIR}/analyze_p15.py" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

CASE_SEQ=0; TOTAL_FAIL=0
run_case() { # label path N Q rep wm
  local label="$1" path="$2" n="$3" q="$4" rep="$5" wm="$6"
  CASE_SEQ=$((CASE_SEQ + 1))
  local case_id
  case_id="$(printf 'case%03d_%s_C0_DEFAULT_N%s_q%s_rep%d' "${CASE_SEQ}" "${label}" "${n}" "${q}" "${rep}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}"
  local status=0
  timeout "${CASE_TIMEOUT}" \
    mpirun --allow-run-as-root -np 4 -mca coll ^hcoll "${BINARY}" \
    --path "${path}" --m-local "${M_LOCAL}" --n "${n}" --k "${K}" --q "${q}" \
    --warmup "${WARMUP}" --iters "${ITERS}" --verify-every 1 --window-mult "${wm}" \
    --output-dir "${case_dir}" --run-id "PHASEB_P15_${label}_N${n}_q${q}_wm${wm}_rep${rep}" \
    --candidate C0_DEFAULT \
    < /dev/null > "${case_dir}/stdout_stderr.log" 2>&1 || status=$?
  echo "${status}" > "${case_dir}/exit_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "[FAIL] ${case_id} exit=${status}"
  else
    echo "[ ok ] ${case_id}"
  fi
}

echo "== Part A: window_mult curve (4 cells x 4 cfg x ${REPS} reps) =="
for cell in "${WM_CELLS[@]}"; do
  read -r n q role <<< "${cell}"
  echo "-- cell N${n}/q${q} (${role}) --"
  for ((rep = 1; rep <= REPS; rep++)); do
    for cfg in "${WM_CFGS[@]}"; do
      read -r label path wm <<< "${cfg}"
      run_case "${label}" "${path}" "${n}" "${q}" "${rep}" "${wm}"
    done
  done
done

echo "== Part B: r1 family axis (11 cells x ${REPS} reps) =="
for ((rep = 1; rep <= REPS; rep++)); do
  for cell in "${R1_CELLS[@]}"; do
    read -r n q <<< "${cell}"
    run_case "r1" "r1" "${n}" "${q}" "${rep}" 1
  done
done

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb P15 complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
python3 "${SCRIPT_DIR}/analyze_p15.py" --result-root "${RESULT_ROOT}" \
  --ref-csv "${SCRIPT_DIR}/family_axis_dushmem_vs_rccl_20260902.csv" || \
  echo "p15 summarizer failed; raw data intact under ${RESULT_ROOT}/cases"
