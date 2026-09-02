#!/usr/bin/env bash
# Phase B d0/dc same-binary patch batch (2026-09-02 深夜):
#   dsfix 批只跑了 ds/d1/d1w；d0/dc 只有 formal 旧二进制的数据。而 DS 修复
#   合并双循环改变了 d 族的 host enqueue 顺序（实测 d1 跨根偏移 3%~34%），
#   merge 表里 d1w_vs_d0 / d1w_stretch_vs_dc / d1_vs_d0 都是新旧二进制混比。
#   本批用当前（dsfix 同款）二进制补跑 d0/dc，使 d 族全部比较同二进制、
#   同 enqueue 语义；同时补上 formal 缺的 N2048/q16 d0/dc（结构轴 q16 侧写）。
#   - PATHS="d0 dc"，主矩阵 9 格 × 5 rep + N2048/q16 × 5 rep = 100 case
#   - 新时间戳结果根 phaseb_d0dc_<STAMP>，不覆盖任何旧数据
# 前置条件：GPU 空闲、二进制为 dsfix 同款（无需 make）。
# 用法: bash run_phaseb_d0dc.sh
set -euo pipefail

export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_d0dc_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"

if [[ -e "${LOCK}" ]]; then
  echo "refusing to reuse result root ${RESULT_ROOT}" >&2
  exit 1
fi

{
  echo "mode=d0dc"
  echo "reason=merge d-family comparisons need d0/dc on the dsfix binary (enqueue-order A/B)"
  echo "result_root=${RESULT_ROOT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "binary=${BINARY}"
  echo "binary_sha256=$(sha256sum "${BINARY}" | awk '{print $1}')"
} > "${RESULT_ROOT}/run_metadata.txt"

{
  echo "=== rocm-smi ==="
  rocm-smi --showproductname --showmeminfo vram 2>&1 || true
  echo "=== hipconfig ==="
  hipconfig --full 2>&1 || true
  echo "=== binary ldd ==="
  ldd "${BINARY}" 2>&1 || true
} > "${RESULT_ROOT}/platform/platform_facts.txt"

cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" \
   "${SCRIPT_DIR}/run_phaseb_d0dc.sh" "${SCRIPT_DIR}/analyze_phaseb.py" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

M_LOCAL=2048
K=2048
REPS=5
WARMUP=20
ITERS=80
VERIFY_EVERY=1
CASE_TIMEOUT=900

CASE_SEQ=0
TOTAL_FAIL=0
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
run_case() {
  local path="$1" n="$2" q="$3" rep="$4" cand="${5:-C0_DEFAULT}"
  CASE_SEQ=$((CASE_SEQ + 1))
  local case_id
  case_id="$(printf 'case%03d_%s_%s_N%s_q%s_rep%d' "${CASE_SEQ}" "${path}" "${cand}" "${n}" "${q}" "${rep}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  mkdir -p "${case_dir}"
  local run_id="PHASEB_D0DC_${path}_${cand}_N${n}_q${q}_rep${rep}"
  local launcher
  launcher="$(candidate_env "${cand}")"
  {
    echo "# ${case_id}"
    echo "cd ${SCRIPT_DIR}"
    echo "${launcher} \\"
    echo "mpirun --allow-run-as-root -np 4 -mca coll ^hcoll ./ag_gemm_phaseb \\"
    echo "  --path ${path} --m-local ${M_LOCAL} --n ${n} --k ${K} --q ${q} \\"
    echo "  --warmup ${WARMUP} --iters ${ITERS} --verify-every ${VERIFY_EVERY} \\"
    echo "  --output-dir ${case_dir} --run-id ${run_id} --candidate ${cand}"
  } > "${case_dir}/command.txt"

  local status=0
  ${launcher} \
    timeout "${CASE_TIMEOUT}" \
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

for n in 512 2048 4096; do
  for q in 2 4 8; do
    for path in d0 dc; do
      for ((rep = 1; rep <= REPS; rep++)); do
        run_case "${path}" "${n}" "${q}" "${rep}"
      done
    done
  done
done
# q=16 边界（N=2048）：补 formal 缺失的 d0/dc 结构轴侧写
n=2048; q=16
for path in d0 dc; do
  for ((rep = 1; rep <= REPS; rep++)); do
    run_case "${path}" "${n}" "${q}" "${rep}"
  done
done

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb d0dc complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
