#!/usr/bin/env bash
# Phase B formal 收尾一键脚本：跑完整分析链并打印摘要。
# 用法: bash finalize_phaseb.sh <result_root>
#   <result_root> 例: /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_formal_20260902_160115
# 幂等：只写 summary/，不触碰 cases/。
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash finalize_phaseb.sh <result_root>" >&2
  exit 1
fi
RESULT_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -d "${RESULT_ROOT}/cases" ]] || { echo "no cases/ under ${RESULT_ROOT}" >&2; exit 1; }

echo "== [1/5] analyze_phaseb.py (case_summary / cell_matrix / boundary tables / control) =="
python3 "${SCRIPT_DIR}/analyze_phaseb.py" --result-root "${RESULT_ROOT}" || echo "!! analyzer failed"

echo
echo "== [2/5] significance_phaseb.py (Mann-Whitney + rep consistency) =="
python3 "${SCRIPT_DIR}/significance_phaseb.py" --result-root "${RESULT_ROOT}" || echo "!! significance failed"

echo
echo "== [3/5] extract_release_curves.py (per-slice release timing, F6) =="
python3 "${SCRIPT_DIR}/extract_release_curves.py" --result-root "${RESULT_ROOT}" || echo "!! release curves failed"

echo
echo "== [4/5] selector_phaseb.py (LOO evaluation, F5) =="
python3 "${SCRIPT_DIR}/selector_phaseb.py" --summary-dir "${RESULT_ROOT}/summary" || echo "!! selector failed"

echo
echo "== [5/5] refit_b2_thresholds.py (nested-LOO threshold refit, Track B) =="
python3 "${SCRIPT_DIR}/refit_b2_thresholds.py" --summary-dir "${RESULT_ROOT}/summary" || echo "!! refit failed"

# optional [6]: if a phaseb_dsfix_* root exists (fixed-binary ds/d1/d1w batch),
# produce the cross-root merged view into the FORMAL root's summary/.
# Guard: if this script was invoked ON the dsfix root itself, "formal" would
# degenerate to dsfix-on-dsfix — swap in the newest phaseb_formal_* root.
DSFIX_ROOT="$(ls -dt /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_dsfix_* 2>/dev/null | head -1 || true)"
MERGE_FORMAL_ROOT="${RESULT_ROOT}"
if [[ "$(basename "${RESULT_ROOT}")" == phaseb_dsfix_* ]]; then
  MERGE_FORMAL_ROOT="$(ls -dt /root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_formal_* 2>/dev/null | head -1 || true)"
  echo "(invoked on a dsfix root; merging into formal root ${MERGE_FORMAL_ROOT})"
fi
if [[ -n "${DSFIX_ROOT}" && -d "${DSFIX_ROOT}/cases" && -n "${MERGE_FORMAL_ROOT}" ]]; then
  echo
  echo "== [bonus] merge_phaseb_dsfix.py (dsfix root: ${DSFIX_ROOT}) =="
  python3 "${SCRIPT_DIR}/merge_phaseb_dsfix.py" --formal-root "${MERGE_FORMAL_ROOT}" --dsfix-root "${DSFIX_ROOT}" || echo "!! merge failed"
fi

echo
echo "== boundary tables digest (xcand = isolated-vs-e2e winner map) =="
for f in phaseb_boundary.csv phaseb_boundary_xcand.csv phaseb_control.csv; do
  p="${RESULT_ROOT}/summary/${f}"
  [[ -f "${p}" ]] && { echo "--- ${f}"; column -s, -t "${p}" | cut -c1-200; echo; }
done
echo "== done. all outputs under ${RESULT_ROOT}/summary/ =="
