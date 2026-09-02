#!/usr/bin/env bash
# Phase B ds-fix runner: DS 的串行门控在 2026-09-02 被发现缺失（两处
# if(serial) 块为空，DS 与 D1 发出完全相同的流操作序列），formal
# phaseb_formal_20260902_160115 里的 ds 数据实为 D1 行为。
# 本脚本用修复后的二进制补测 ds（并同批复测 d1 以获得同批配对）：
#   - 只跑 PATHS="ds d1 d1w"，矩阵与其余参数与 formal 完全一致
#   - 追加 N4096/q16 配置轴四件套（comm/r1 × C0/C2）：若 formal 的
#     N4096/q8 不反转，Track B 只剩 N2048/q8 一个正例；往深一档切片
#     探边界，作嵌套 LOO 可学习性的保险。
#   - 新时间戳结果根目录 phaseb_dsfix_<STAMP>，不覆盖任何旧数据
#
# 前置条件：formal 已结束且 ag_gemm_phaseb 已用修复后源码重新编译（make）。
# 用法: bash run_phaseb_dsfix.sh
set -euo pipefail

export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_dsfix_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"

if [[ -e "${LOCK}" ]]; then
  echo "refusing to reuse result root ${RESULT_ROOT}" >&2
  exit 1
fi

{
  echo "mode=dsfix"
  echo "reason=DS serial gate was missing in phaseb_formal_20260902_160115; ds there measured D1 behavior"
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
   "${SCRIPT_DIR}/run_phaseb_dsfix.sh" "${SCRIPT_DIR}/analyze_phaseb.py" \
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
  local run_id="PHASEB_DSFIX_${path}_${cand}_N${n}_q${q}_rep${rep}"
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
    for path in ds d1 d1w; do
      for ((rep = 1; rep <= REPS; rep++)); do
        run_case "${path}" "${n}" "${q}" "${rep}"
      done
    done
  done
done
# q=16 边界（N=2048）
n=2048; q=16
for path in ds d1 d1w; do
  for ((rep = 1; rep <= REPS; rep++)); do
    run_case "${path}" "${n}" "${q}" "${rep}"
  done
done
# N4096/q16 配置轴四件套（Track B 正例保险，见文件头注释）
n=4096; q=16
for path in comm r1; do
  for cand in C0_DEFAULT C2_RING_SIMPLE_CH8; do
    for ((rep = 1; rep <= REPS; rep++)); do
      run_case "${path}" "${n}" "${q}" "${rep}" "${cand}"
    done
  done
done

touch "${LOCK}"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "total_cases=${CASE_SEQ}"
  echo "failed_cases=${TOTAL_FAIL}"
} >> "${RESULT_ROOT}/run_metadata.txt"

echo "== phaseb dsfix complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
