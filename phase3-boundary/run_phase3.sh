#!/usr/bin/env bash
# Phase 3 boundary mapping for K500SM_AI / gfx928 / 4 GPUs / PCIe.
set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULT_ROOT=${1:?usage: ./run_phase3.sh RESULT_ROOT MODE}
MODE=${2:?usage: ./run_phase3.sh RESULT_ROOT MODE}
BENCH="$SCRIPT_DIR/ag_gemm_rccl"
MASTER="$RESULT_ROOT/phase3_master.log"
CASE_MANIFEST="$RESULT_ROOT/phase3_case_manifest.tsv"

if [[ "$MODE" != "preflight" && "$MODE" != "full" ]]; then
  echo "ERROR mode must be preflight or full" >&2
  exit 2
fi
if [[ -e "$RESULT_ROOT/.phase3_initialized" ]]; then
  echo "ERROR refusing to reuse an existing Phase 3 result directory: $RESULT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$BENCH" ]]; then
  echo "ERROR missing executable: $BENCH; run make first" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT"/{cases,summary,platform,source_snapshot}
touch "$RESULT_ROOT/.phase3_initialized"
printf 'case_id\tphase\trepetition\tranks\tshape_id\tm_local\tn\tk\tq\tpath\tcandidate\tstatus\tstarted_utc\tended_utc\tcase_dir\n' > "$CASE_MANIFEST"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$MASTER"
}

{
  echo "platform_id=K500SM_AI"
  echo "gfx_arch=gfx928"
  echo "rank_count=4"
  echo "transport_scope=PCIe"
  echo "experiment_phase=3_boundary_mapping"
  date -u +"captured_utc=%FT%TZ"
  hostname
  env | LC_ALL=C sort
  rocminfo 2>&1 || true
  rocm-smi --showproductname --showuniqueid --showtopo --showmeminfo vram 2>&1 || true
  hipconfig --full 2>&1 || true
  mpirun --version 2>&1 || true
  ldd "$BENCH" 2>&1 || true
} > "$RESULT_ROOT/platform/platform_facts.txt"

cp -p "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/Makefile" "$SCRIPT_DIR/run_phase3.sh" \
  "$SCRIPT_DIR/analyze_phase3.py" "$SCRIPT_DIR/PHASE3_EXPERIMENT_DESIGN.md" \
  "$RESULT_ROOT/source_snapshot/"
sha256sum "$BENCH" "$SCRIPT_DIR/ag_gemm_rccl.cpp" "$SCRIPT_DIR/Makefile" \
  "$SCRIPT_DIR/run_phase3.sh" "$SCRIPT_DIR/analyze_phase3.py" \
  "$SCRIPT_DIR/PHASE3_EXPERIMENT_DESIGN.md" > "$RESULT_ROOT/source_snapshot/sha256.txt"

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

  if (( m_local % q != 0 )); then
    log "ERROR invalid shape: m_local=$m_local is not divisible by q=$q"
    return 2
  fi
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
  printf 'platform_id=K500SM_AI\ngfx_arch=gfx928\nphase=%s\nshape_id=%s\ntransport_scope=PCIe\n' \
    "$phase" "$shape" > "$case_dir/metadata.txt"
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
  find "$case_dir" -type f -printf '%P\t%s bytes\n' | LC_ALL=C sort > "$case_dir/file_inventory.txt"
  log "END case=$case_id status=$status"
}

run_preflight() {
  # Exercise the largest N, the largest q, and both relevant paths before the formal matrix.
  run_case phase3_preflight 1 PF_N4096_Q8 2048 4096 2048 8 h0 C0_DEFAULT 5 10
  run_case phase3_preflight 2 PF_N2048_Q16 2048 2048 2048 16 h0 C2_RING_SIMPLE_CH8 5 10
  run_case phase3_preflight 3 PF_N512_Q2 2048 512 2048 2 comm C2_RING_SIMPLE_CH8 5 10
}

run_full_matrix() {
  local n q candidate rep
  for n in 512 2048 4096; do
    for q in 2 4 8; do
      for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
        for rep in 1 2 3 4 5; do
          run_case phase3_comm "$rep" "N${n}_Q${q}" 2048 "$n" 2048 "$q" comm "$candidate" 20 80
        done
      done
      for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
        for rep in 1 2 3 4 5; do
          run_case phase3_h0 "$rep" "N${n}_Q${q}" 2048 "$n" 2048 "$q" h0 "$candidate" 20 80
        done
      done
    done
  done

  # Extra fine-granularity boundary point around the Phase-2 counterexample.
  for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
    for rep in 1 2 3 4 5; do
      run_case phase3_comm "$rep" N2048_Q16 2048 2048 2048 16 comm "$candidate" 20 80
    done
  done
  for candidate in C0_DEFAULT C1_RING_SIMPLE_CH4 C2_RING_SIMPLE_CH8; do
    for rep in 1 2 3 4 5; do
      run_case phase3_h0 "$rep" N2048_Q16 2048 2048 2048 16 h0 "$candidate" 20 80
    done
  done

  # B1 isolates the cost of forbidding adjacent-slice overlap for the two endpoints.
  for q in 2 4 8 16; do
    for candidate in C0_DEFAULT C2_RING_SIMPLE_CH8; do
      for rep in 1 2 3 4 5; do
        run_case phase3_b1 "$rep" "N2048_Q${q}" 2048 2048 2048 "$q" b1 "$candidate" 20 80
      done
    done
  done

  # The same q/layout/scatter quantifies GEMM fragmentation without concurrent RCCL work.
  for n in 512 2048 4096; do
    for q in 2 4 8; do
      for rep in 1 2 3; do
        run_case phase3_gemm "$rep" "N${n}_Q${q}" 2048 "$n" 2048 "$q" gemm C0_DEFAULT 20 80
      done
    done
  done
  for rep in 1 2 3; do
    run_case phase3_gemm "$rep" N2048_Q16 2048 2048 2048 16 gemm C0_DEFAULT 20 80
  done

  # Full serial uses q=1 by construction; N controls the calculation intensity.
  for n in 512 2048 4096; do
    for rep in 1 2 3 4 5; do
      run_case phase3_b0 "$rep" "N${n}_FULL" 2048 "$n" 2048 1 b0 C0_DEFAULT 20 80
    done
  done
}

if [[ "$MODE" == "preflight" ]]; then
  log "PHASE3 PREFLIGHT: 3 cases, exercising N=4096 and q=16."
  run_preflight
else
  log "PHASE3 FORMAL: 385 cases, 30,800 timed four-rank iterations."
  run_full_matrix
fi

python3 "$SCRIPT_DIR/analyze_phase3.py" "$RESULT_ROOT" > "$RESULT_ROOT/summary/analyze_phase3.log" 2>&1
log "COMPLETE mode=$MODE raw_data=$RESULT_ROOT/cases summary=$RESULT_ROOT/summary"
