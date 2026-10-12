#!/usr/bin/env python3
"""M 轴扫描:按 PHASES.json manifest 把多相位 trace 分到相位子目录(符号链接)。

背景:多相位采集中同一 rank 每相位各落一个 trace 文件(文件名仅时间戳不同),
extract_custom_ar.py 按 pid 键存 rank_custom_ar_<pid>.json,直接跑会互相覆盖
/被断点逻辑跳过。本脚本按 manifest 的 files 列表建 ph<idx>_<kind><val>/
子目录(内为符号链接),之后逐相位跑现有管线即可:
    for d in <dir>/ph*/; do python3 extract_custom_ar.py "$d"; done

校验:manifest 每相位文件数应 = EXPECT_GPU;不在任何相位的散文件列出告警。
用法: python3 split_phases.py <trace_dir> [expect_gpu]
"""
import json
import os
import sys
import glob

D = sys.argv[1] if len(sys.argv) > 1 else ""
if D and not os.path.isabs(D):
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), D)
EXPECT = int(sys.argv[2]) if len(sys.argv) > 2 else 8

man_path = os.path.join(D, "PHASES.json")
if not os.path.isfile(man_path):
    sys.exit(f"ERROR: 未找到 {man_path}(多相位采集才有;先确认 run_vllm_scnet.sh 用了 PHASES)")
with open(man_path) as f:
    phases = json.load(f)

ondisk = {os.path.basename(p) for p in glob.glob(os.path.join(D, "*.pt.trace.json.gz"))}
assigned = set()
print(f"manifest {len(phases)} 个相位,磁盘 trace {len(ondisk)} 个")
for ph in phases:
    sub = os.path.join(D, f"ph{ph['idx']}_{ph['kind']}{ph['ptok'] or ph['conc']}")
    os.makedirs(sub, exist_ok=True)
    miss = [f for f in ph["files"] if f not in ondisk]
    # 幂等:清掉旧链接再建
    for old in glob.glob(os.path.join(sub, "*.pt.trace.json.gz")):
        os.remove(old)
    n = 0
    for fname in ph["files"]:
        src = os.path.join(D, fname)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(sub, fname))
        assigned.add(fname)
        n += 1
    tag = "OK" if n == EXPECT and not miss else "WARN"
    print(f"[{tag}] ph{ph['idx']} {ph['kind']} conc={ph['conc']} ptok={ph['ptok']}"
          f" -> {sub} ({n}/{len(ph['files'])} 链接,期望 {EXPECT})" + (f" 缺:{miss}" if miss else ""))
    if miss:
        assigned.update(miss)

stray = sorted(ondisk - assigned)
if stray:
    print(f"注意: {len(stray)} 个磁盘文件不在任何相位(采集期意外产物?),前 5 个:")
    for s in stray[:5]:
        print("  " + s)
print("done. 逐相位抽取: for d in " + D + "/ph*/; do python3 "
      + os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_custom_ar.py")
      + " \"$d\"; done")
