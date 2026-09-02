#!/usr/bin/env bash
# collect_results.sh — 把需要回传的一切打成单个 tar.gz。
# 用法: bash collect_results.sh [备注标签]   # 例如 bash collect_results.sh smoke
# 打包内容:
#   - results/rtx4090_4gpu/ 下所有时间戳结果根（含全部 case 日志/CSV/平台档案）
#   - env_check_report.txt（若存在）
#   - build.log（若存在）
#   - 当前源码快照（跑的到底是哪一版，我要看到）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TAG="${1:-bundle}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="phaseb_nv_${TAG}_${STAMP}.tar.gz"

ITEMS=()
[[ -d results/rtx4090_4gpu ]] && ITEMS+=("results/rtx4090_4gpu")
[[ -f env_check_report.txt ]] && ITEMS+=("env_check_report.txt")
[[ -f build.log ]] && ITEMS+=("build.log")
for f in ag_gemm_phaseb_nv.cpp Makefile run_phaseb_nv.sh env_check.sh collect_results.sh p2p_probe.cu analyze_phaseb.py CHANGES.md; do
  [[ -f "$f" ]] && ITEMS+=("$f")
done
[[ -f README_NVIDIA_PORT.md ]] && ITEMS+=("README_NVIDIA_PORT.md")

if [[ ${#ITEMS[@]} -eq 0 ]]; then
  echo "nothing to collect（既无结果也无源码？确认你在 nvidia_port/ 目录里）" >&2
  exit 1
fi

tar czf "${BUNDLE}" "${ITEMS[@]}"
echo "== bundle: ${BUNDLE} ($(du -sh "${BUNDLE}" | cut -f1)) =="
echo "包含:"
tar tzf "${BUNDLE}" | head -20
echo "..."
echo "把这个文件发回来即可（一个文件包含我需要的全部信息）。"
