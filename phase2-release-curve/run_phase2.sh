#!/usr/bin/env bash
# Targeted confirmation of the Phase-1 H0 candidate-ranking counterexample.
# Platform fixed by design: K500SM_AI / gfx928 / 4 GPUs / PCIe.
set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULT_ROOT=${1:?usage: ./run_phase2.sh RESULT_ROOT}
BENCH="$SCRIPT_DIR/ag_gemm_rccl"
MASTER="$RESULT_ROOT/phase2_master.log"
CASE_MANIFEST="$RESULT_ROOT/phase2_case_manifest.tsv"

mkdir -p "$RESULT_ROOT"/{cases,summary,platform,source_snapshot}
if [[ ! -f "$CASE_MANIFEST" ]]; then
  printf 'case_id\tphase\trepetition\tranks\tshape_id\tm_local\tn\tk\tq\tpath\tcandidate\tstatus\tstarted_utc\tended_utc\tcase_dir\n' > "$CASE_MANIFEST"
fi

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$MASTER"
}

if [[ ! -x "$BENCH" ]]; then
  log "ERROR missing executable: $BENCH"
  exit 2
fi

{
  echo "platform_id=K500SM_AI"
  echo "gfx_arch=gfx928"
  echo "rank_count=4"
  echo "transport_assumption=PCIe"
  date -u +"captured_utc=%FT%TZ"
  hostname
  rocminfo 2>&1 || true
  rocm-smi 2>&1 || true
  hipconfig --full 2>&1 || true
  ldd "$BENCH" 2>&1 || true
} > "$RESULT_ROOT/platform/platform_facts.txt"

cp -p "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/Makefile" "$SCRIPT_DIR/run_phase2.sh" \
  "$SCRIPT_DIR/analyze_phase2.py" "$SCRIPT_DIR/PHASE2_EXPERIMENT_DESIGN.md" \
  "$RESULT_ROOT/source_snapshot/"
sha256sum "$BENCH" "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/Makefile" \
  "$SCRIPT_DIR/run_phase2.sh" "$SCRIPT_DIR/analyze_phase2.py" \
  "$SCRIPT_DIR/PHASE2_EXPERIMENT_DESIGN.md" > "$RESULT_ROOT/source_snapshot/sha256.txt"

candidate_env() {
  local candidate=$1
  case "$candidate" in
    C0_DEFAULT)
      printf '%s\0' -u NCCL_ALGO -u NCCL_PROTO -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS
      ;;
    C1_RING_SIMPLE_CH4)
      printf '%s\0' NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
      ;;
    C2_RING_SIMPLE_CH8)
      printf '%s\0' NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8
      ;;
    *)
      log "ERROR unknown candidate=$candidate"
      return 1
      ;;
  esac
}

run_case() {
  local phase=$1; shift
  local rep=$1; shift
  local shape=$1; shift
  local m_local=$1; shift
  local n=$1; shift
  local k=$1; shift
  local q=$1; shift
  local path=$1; shift
  local candidate=$1; shift
  local warmup=$1; shift
  local iters=$1; shift
  local case_id="${phase}_r${rep}_${shape}_p4_m${m_local}_n${n}_k${k}_q${q}_${path}_${candidate}"
  local case_dir="$RESULT_ROOT/cases/$case_id"
  local started ended status
  local -a candidate_args=()
  mkdir -p "$case_dir/rccl_logs"
  while IFS= read -r -d '' item; do candidate_args+=("$item"); done < <(candidate_env "$candidate")
  local -a command=(
    env "${candidate_args[@]}" HIP_VISIBLE_DEVICES=0,1,2,3 HSA_FORCE_FINE_GRAIN_PCIE=1
    NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING
    NCCL_DEBUG_FILE="$case_dir/rccl_logs/rccl.%h.%p.log"
    mpirun --allow-run-as-root -np 4 -mca coll ^hcoll
    "$BENCH" --path "$path" --m-local "$m_local" --n "$n" --k "$k" --q "$q"
    --warmup "$warmup" --iters "$iters" --verify-every 1
    --output-dir "$case_dir" --run-id "$case_id" --candidate "$candidate"
  )
  {
    printf '# %q ' "${command[@]}"
    printf '\n'
  } > "$case_dir/command.txt"
  printf 'platform_id=K500SM_AI\ngfx_arch=gfx928\nphase=%s\nshape_id=%s\n' "$phase" "$shape" \
    > "$case_dir/metadata.txt"
  started=$(date -u +%FT%TZ)
  log "START case=$case_id"
  set +e
  timeout 600s "${command[@]}" > "$case_dir/stdout_stderr.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$case_dir/exit_status.txt"
  ended=$(date -u +%FT%TZ)
  printf '%s\t%s\t%s\t4\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$phase" "$rep" "$shape" "$m_local" "$n" "$k" "$q" "$path" "$candidate" \
    "$status" "$started" "$ended" "$case_dir" >> "$CASE_MANIFEST"
  find "$case_dir" -type f -printf '%P\t%s bytes\n' | sort > "$case_dir/file_inventory.txt"
  log "END case=$case_id status=$status"
}

run_targeted_matrix() {
  # The strongest Phase-1 reversal was S03/q=8. q=2 is a control in which
  # C2 remains the H0 winner. Both use M_local=N=K=2048.
  local q candidate rep
  for q in 2 8; do
    for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
      for rep in 1 2 3 4 5; do
        run_case phase2_comm "$rep" S03_CONFIRM 2048 2048 2048 "$q" comm "$candidate" 20 80
      done
    done
    for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
      for rep in 1 2 3 4 5; do
        run_case phase2_h0 "$rep" S03_CONFIRM 2048 2048 2048 "$q" h0 "$candidate" 20 80
      done
    done
    # C0 and C2 are the two endpoints of the Phase-1 reversal. B1 quantifies
    # the cost when the same slice graph forbids comm/compute concurrency.
    for candidate in C0_DEFAULT C2_RING_SIMPLE_CH8; do
      for rep in 1 2 3 4 5; do
        run_case phase2_b1 "$rep" S03_CONFIRM 2048 2048 2048 "$q" b1 "$candidate" 20 80
      done
    done
    for rep in 1 2 3; do
      run_case phase2_gemm "$rep" S03_CONFIRM 2048 2048 2048 "$q" gemm C0_DEFAULT 20 80
    done
  done
  # Full serial baseline has q=1 by construction and is shared by q=2/q=8.
  for rep in 1 2 3 4 5; do
    run_case phase2_b0 "$rep" S03_CONFIRM 2048 2048 2048 1 b0 C0_DEFAULT 20 80
  done
}

log "PHASE2 target: 91 cases; S03 q={2,8}; 5 independent process repetitions for COMM_ONLY and H0."
run_targeted_matrix
python3 "$SCRIPT_DIR/analyze_phase2.py" "$RESULT_ROOT" > "$RESULT_ROOT/summary/analyze_phase2.log" 2>&1
log "COMPLETE raw data retained under $RESULT_ROOT/cases; summary=$RESULT_ROOT/summary/phase2_reversal_analysis.csv"
