#!/usr/bin/env bash
# 把 bw1000 上的全部产物打成一个包 (env 报告 + 所有结果根 + 运行日志)。
# 用法: 在 bw1000_port/ 目录里  bash collect_results_bw.sh
set -uo pipefail
OUT="bw1000_results_$(date +%Y%m%d_%H%M%S).tar.gz"
ITEMS=()

[[ -f env_check_report_bw1000.txt ]] && ITEMS+=(env_check_report_bw1000.txt)
if [[ -d results/bw1000_8gpu ]]; then
  while IFS= read -r root; do ITEMS+=("${root}"); done < <(ls -d results/bw1000_8gpu/phaseb_* 2>/dev/null)
fi
# 手册步骤里 nohup 的运行日志
for f in run_smoke.log run_formal_np4.log run_formal_np8.log \
         run_wm_np8.log run_wm_np4.log run_extrap_np8.log run_extrap_np4.log; do
  [[ -f "${f}" ]] && ITEMS+=("${f}")
done

if [[ ${#ITEMS[@]} -eq 0 ]]; then
  echo "nothing to pack (no env report, no result roots)" >&2; exit 1
fi

tar czf "${OUT}" "${ITEMS[@]}"
sha256sum "${OUT}" > "${OUT}.sha256"
echo "== 打包完成 =="
echo "文件: ${OUT}  ($(du -h "${OUT}" | cut -f1))"
echo "校验: ${OUT}.sha256"
echo "把 ${OUT} 拷回来即可 (summary/ 里的 CSV 和 cases/ 的原始数据都在里面)。"
