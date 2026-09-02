#!/usr/bin/env bash
# Execute the first correctness-first RCCL/HIP release experiment on K500SM_AI/gfx928.
# The script never overwrites a case directory: every invocation has a distinct case_id.
set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULT_ROOT=$1
if [[ -z "$RESULT_ROOT" ]]; then
  echo "usage: run_phase1.sh RESULT_ROOT" >&2
  exit 2
fi
BENCH="$SCRIPT_DIR/ag_gemm_rccl"
MASTER="$RESULT_ROOT/phase1_master.log"
CASE_MANIFEST="$RESULT_ROOT/phase1_case_manifest.tsv"

mkdir -p "$RESULT_ROOT"/{cases,summary,platform,source_snapshot}
if [[ ! -f "$CASE_MANIFEST" ]]; then
  printf 'case_id\tphase\trepetition\tranks\tshape_id\tm_local\tn\tk\tq\tpath\tcandidate\tstatus\tstarted_utc\tended_utc\tcase_dir\n' > "$CASE_MANIFEST"
fi

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$MASTER"
}

if [[ ! -x "$BENCH" ]]; then
  log "ERROR benchmark missing: $BENCH"
  exit 2
fi

sha256sum "$BENCH" "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/run_phase1.sh" \
  "$SCRIPT_DIR/summarize_phase1.py" > "$RESULT_ROOT/source_snapshot/sha256.txt"
cp -p "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/Makefile" "$SCRIPT_DIR/run_phase1.sh" \
  "$SCRIPT_DIR/summarize_phase1.py" "$RESULT_ROOT/source_snapshot/"

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
  local ranks=$1; shift
  local shape_id=$1; shift
  local m_local=$1; shift
  local n=$1; shift
  local k=$1; shift
  local q=$1; shift
  local path=$1; shift
  local candidate=$1; shift
  local warmup=$1; shift
  local iters=$1; shift
  local debug=$1; shift
  local case_id="${phase}_r${rep}_${shape_id}_p${ranks}_m${m_local}_n${n}_k${k}_q${q}_${path}_${candidate}"
  local case_dir="$RESULT_ROOT/cases/$case_id"
  local started ended status
  mkdir -p "$case_dir/rccl_logs"
  started=$(date -u +%FT%TZ)
  log "START case=$case_id"

  local -a candidate_args=()
  while IFS= read -r -d '' item; do candidate_args+=("$item"); done < <(candidate_env "$candidate")
  local debug_level=WARN debug_subsys=INIT,GRAPH,TUNING
  if [[ "$debug" == "info" ]]; then debug_level=INFO; fi
  local -a command=(
    env "${candidate_args[@]}" HIP_VISIBLE_DEVICES=0,1,2,3 HSA_FORCE_FINE_GRAIN_PCIE=1
    NCCL_DEBUG="$debug_level" NCCL_DEBUG_SUBSYS="$debug_subsys"
    NCCL_DEBUG_FILE="$case_dir/rccl_logs/rccl.%h.%p.log"
    mpirun --allow-run-as-root -np "$ranks" -mca coll ^hcoll
    "$BENCH" --path "$path" --m-local "$m_local" --n "$n" --k "$k" --q "$q"
    --warmup "$warmup" --iters "$iters" --verify-every 1
    --output-dir "$case_dir" --run-id "$case_id" --candidate "$candidate"
  )
  {
    printf '# %q ' "${command[@]}"
    printf '\n'
  } > "$case_dir/command.txt"
  printf 'platform_id=K500SM_AI\ngfx_arch=gfx928\nphase=%s\nshape_id=%s\n' "$phase" "$shape_id" \
    > "$case_dir/metadata.txt"

  set +e
  timeout 420s "${command[@]}" > "$case_dir/stdout_stderr.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$case_dir/exit_status.txt"
  ended=$(date -u +%FT%TZ)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$phase" "$rep" "$ranks" "$shape_id" "$m_local" "$n" "$k" "$q" "$path" "$candidate" \
    "$status" "$started" "$ended" "$case_dir" >> "$CASE_MANIFEST"
  find "$case_dir" -type f -printf '%P\t%s bytes\n' | sort > "$case_dir/file_inventory.txt"
  log "END case=$case_id status=$status"
  return 0
}

run_preflight() {
  log "PHASE preflight: 2 ranks, tiny synthetic shape; validates CSV, layout and stream dependency."
  local path q
  for q in 1 2; do
    for path in comm gemm b0 b1 h0; do
      run_case preflight 1 2 P0 128 128 128 "$q" "$path" C0_DEFAULT 3 5 info
    done
  done
}

run_capability() {
  log "PHASE capability: legal 4-rank candidates are probed with RCCL INFO logs."
  local candidate
  for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
    run_case capability 1 4 CAP 512 256 1024 1 comm "$candidate" 5 10 info
  done
}

run_discovery() {
  # Synthetic engineering shapes. Their measured comm/gemm ratio will classify them;
  # they are deliberately not labelled as trace-derived paper workload shapes.
  log "PHASE discovery: 4 ranks, 3 synthetic shapes, q={1,2,4,8}, 3 independent processes."
  local spec shape m n k q candidate rep path
  for spec in S01_COMM_LEAN:2048:128:2048 S02_BALANCE:2048:512:2048 S03_COMPUTE_LEAN:2048:2048:2048; do
    IFS=: read -r shape m n k <<< "$spec"
    for q in 1 2 4 8; do
      for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
        run_case discovery_comm 1 4 "$shape" "$m" "$n" "$k" "$q" comm "$candidate" 20 50 warn
      done
      run_case discovery_gemm 1 4 "$shape" "$m" "$n" "$k" "$q" gemm C0_DEFAULT 20 50 warn
    done
    for rep in 1 2 3; do
      run_case discovery_b0 "$rep" 4 "$shape" "$m" "$n" "$k" 1 b0 C0_DEFAULT 20 50 warn
      for q in 1 2 4 8; do
        for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
          for path in b1 h0; do
            run_case discovery_overlap "$rep" 4 "$shape" "$m" "$n" "$k" "$q" "$path" "$candidate" 20 50 warn
          done
        done
      done
    done
  done
}

run_preflight
run_capability
run_discovery
python3 "$SCRIPT_DIR/summarize_phase1.py" "$RESULT_ROOT" > "$RESULT_ROOT/summary/summarize_command.log" 2>&1
log "COMPLETE raw data retained under $RESULT_ROOT/cases; summary=$RESULT_ROOT/summary/phase1_summary.csv"
