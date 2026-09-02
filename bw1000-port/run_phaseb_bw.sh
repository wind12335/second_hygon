#!/usr/bin/env bash
# Phase B runner — bw1000 8-GPU port (2026-09-02).
# 用法:
#   bash run_phaseb_bw.sh smoke              # 冒烟 (~10-15 分钟)
#   bash run_phaseb_bw.sh formal             # 正式批 np=4 (对照批, 与 K500 4卡同口径)
#   NP=8 bash run_phaseb_bw.sh formal        # 正式批 np=8 (规模批)
#   PHASEB_PATHS="comm gemm r0 rs r1" bash run_phaseb_bw.sh formal   # 无 DUSHMEM 时的 r 族子集
# 结果根: <脚本目录>/results/bw1000_8gpu/phaseb_<mode>_np<NP>_<时间戳>/  (绝不覆盖旧数据)
# 纪律: 运行期间不要 make、不要跑其他 GPU 任务; 失败 case 原样保留。
set -euo pipefail

MODE="${1:-smoke}"
NP="${NP:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/ag_gemm_phaseb"
RESULT_BASE="${PHASEB_RESULT_BASE:-${SCRIPT_DIR}/results/bw1000_8gpu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_BASE}/phaseb_${MODE}_np${NP}_${STAMP}"
mkdir -p "${RESULT_ROOT}"/{cases,summary,platform,source_snapshot}
LOCK="${RESULT_ROOT}/.phaseb_initialized"
if [[ -e "${LOCK}" ]]; then
  echo "refusing to reuse result root ${RESULT_ROOT}" >&2
  exit 1
fi

[[ -x "${BINARY}" ]] || { echo "binary missing: ${BINARY} — 先 make" >&2; exit 1; }

# 设备可见性: 每个局部 rank 绑一张卡 (二进制读 OMPI_COMM_WORLD_LOCAL_RANK)
VIS="$(seq -s, 0 $((NP - 1)))"
export HIP_VISIBLE_DEVICES="${VIS}"
export DUSHMEM_SYMMETRIC_SIZE=1G   # 每 PE 对称堆 ~150MiB, 1G 留足余量; 未识别时无害
export PATH=/opt/dtk/bin:/opt/mpi/bin:${PATH}

ALL_PATHS=(comm gemm r0 rs r1 fc dc d0 ds d1 d1w)
if [[ -n "${PHASEB_PATHS:-}" ]]; then
  read -r -a PATHS <<< "${PHASEB_PATHS}"
else
  PATHS=("${ALL_PATHS[@]}")
fi
has_path() { [[ " ${PATHS[*]} " == *" $1 "* ]]; }

{
  echo "mode=${MODE} np=${NP}"
  echo "paths=${PATHS[*]}"
  echo "result_root=${RESULT_ROOT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "binary=${BINARY}"
  echo "binary_sha256=$(sha256sum "${BINARY}" | awk '{print $1}')"
} > "${RESULT_ROOT}/run_metadata.txt"

{
  echo "=== rocm-smi/hy-smi ==="; (rocm-smi --showproductname --showid 2>&1 || hy-smi --showproductname 2>&1) || true
  echo "=== hipconfig ==="; hipconfig --full 2>&1 | head -40 || true
  echo "=== binary ldd ==="; ldd "${BINARY}" 2>&1 || true
} > "${RESULT_ROOT}/platform/platform_facts.txt"

cp "${SCRIPT_DIR}/ag_gemm_phaseb.cpp" "${SCRIPT_DIR}/Makefile" \
   "${SCRIPT_DIR}/run_phaseb_bw.sh" "${SCRIPT_DIR}/analyze_phaseb.py" \
   "${RESULT_ROOT}/source_snapshot/" 2>/dev/null || true
( cd "${RESULT_ROOT}/source_snapshot" && sha256sum * > sha256sums.txt )

M_LOCAL=2048
K=2048
if [[ "${MODE}" == "smoke" ]]; then
  NS=(2048); QS=(8); REPS=1; WARMUP=10; ITERS=40; CASE_TIMEOUT=300
else
  NS=(512 2048 4096); QS=(2 4 8); REPS=5; WARMUP=20; ITERS=80; CASE_TIMEOUT=900
fi

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
  local run_id="PHASEB_BW_${path}_${cand}_N${n}_q${q}_rep${rep}"
  local launcher
  launcher="$(candidate_env "${cand}")"
  {
    echo "# ${case_id}"
    echo "cd ${SCRIPT_DIR}"
    echo "HIP_VISIBLE_DEVICES=${VIS} ${launcher} \\"
    echo "mpirun --allow-run-as-root -np ${NP} -mca coll ^hcoll ./ag_gemm_phaseb \\"
    echo "  --path ${path} --m-local ${M_LOCAL} --n ${n} --k ${K} --q ${q} \\"
    echo "  --warmup ${WARMUP} --iters ${ITERS} --verify-every 1 \\"
    echo "  --output-dir ${case_dir} --run-id ${run_id} --candidate ${cand}"
  } > "${case_dir}/command.txt"

  local status=0
  ${launcher} \
    timeout "${CASE_TIMEOUT}" \
    mpirun --allow-run-as-root -np "${NP}" -mca coll ^hcoll "${BINARY}" \
    --path "${path}" --m-local "${M_LOCAL}" --n "${n}" --k "${K}" --q "${q}" \
    --warmup "${WARMUP}" --iters "${ITERS}" --verify-every 1 \
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
      for ((rep = 1; rep <= REPS; rep++)); do
        run_case "${path}" "${n}" "${q}" "${rep}"
      done
    done
  done
done

if [[ "${MODE}" == "formal" ]]; then
  # 配置轴对照: comm/r1 × C2 (主矩阵同格)
  for n in 512 2048 4096; do
    for q in 2 4 8; do
      for path in comm r1; do
        has_path "${path}" || continue
        for ((rep = 1; rep <= REPS; rep++)); do
          run_case "${path}" "${n}" "${q}" "${rep}" C2_RING_SIMPLE_CH8
        done
      done
    done
  done
  # q=16 边界 (N=2048)
  n=2048; q=16
  for path in comm gemm rs r1 dc ds d1 d1w; do
    has_path "${path}" || continue
    for ((rep = 1; rep <= REPS; rep++)); do
      run_case "${path}" "${n}" "${q}" "${rep}"
    done
  done
  # 探索性 DX 四件套 (N=4096/q16, comm/r1 × C0/C2; 与海光 K500 DX 格对齐)
  n=4096; q=16
  for path in comm r1; do
    has_path "${path}" || continue
    for cand in C0_DEFAULT C2_RING_SIMPLE_CH8; do
      for ((rep = 1; rep <= REPS; rep++)); do
        run_case "${path}" "${n}" "${q}" "${rep}" "${cand}"
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

echo "== phaseb(bw1000) ${MODE} np=${NP} complete: ${CASE_SEQ} cases, ${TOTAL_FAIL} failed =="
echo "result_root=${RESULT_ROOT}"

python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || \
  echo "analyzer failed; raw data intact under ${RESULT_ROOT}/cases"
