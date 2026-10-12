#!/usr/bin/env python3
"""A·straggler 可预测性:从 rank_custom_ar_*.json 逐事件到达序列,
评测在线预测器在 horizon h 下对"下一个迟到 rank"的命中率。

问题定位(M2 可行性):若运行时在集合通信 i 时刻决定生产/调度顺序,
能多准地预知第 i+h 个集合的迟到者?命中率 ≈ 可被提前调度的等待占比。

口径:④' 单机同钟,到达 = kernel ts 直读(analyze_arrival_8rank 已双口径验证);
丢弃前 skip_pct% 事件(warmup conc=1 段),全部预测器只用过去信息。
输出: straggler_predict.json + 终端表。
用法: python3 analyze_straggler_predict.py <analysis_dir> [key] [skip_pct] [mode]
  mode=ts(默认,④' 单机同钟,到达=ts) | dur(③ 双机无公共钟,AR 结束同步 → 最晚到达者=最小 dur)
"""
import json, os, sys, glob
from collections import Counter

ADIR = sys.argv[1] if len(sys.argv) > 1 else ""
if ADIR and not os.path.isabs(ADIR):
    ADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ADIR)
KEY = sys.argv[2] if len(sys.argv) > 2 else "1stage"
SKIP_PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
MODE = sys.argv[4] if len(sys.argv) > 4 else "ts"

files = sorted(glob.glob(os.path.join(ADIR, "rank_custom_ar_*.json")))
assert files, f"未找到 rank_custom_ar_*.json: {ADIR}"
ranks = {}
for f in files:
    r = json.load(open(f))
    pid = r["file"].split("_")[-1].split(".")[0]
    ranks[pid] = r
pids = sorted(ranks)

counts = [ranks[p]["n_" + KEY] for p in pids]
N = min(counts)
TS = {p: [e[0] for e in sorted(ranks[p][KEY])[:N]] for p in pids}
DUR = {p: [e[1] for e in sorted(ranks[p][KEY])[:N]] for p in pids}

# ---- 到达序列:每个集合的最晚 rank(straggler) ----
if MODE == "dur":
    # ③ 口径:跨节点无公共钟;AR 结束同步(barrier 性质)→ dur = T_end - t_start,
    # 晚到者 kernel 内等待最短 → 最晚到达 = argmin dur(与"真走线=全局最小时长"自洽)
    seq = [min(pids, key=lambda p: DUR[p][i]) for i in range(N)]
else:
    seq = [max(pids, key=lambda p: TS[p][i]) for i in range(N)]

# AR→AR 中位间隔(换算 horizon 的物理提前量)
p0 = pids[0]
gaps = sorted(TS[p0][i + 1] - TS[p0][i] for i in range(N - 1))
gap_med = gaps[len(gaps) // 2]

skip = int(N * SKIP_PCT / 100)
test = range(skip, N)
n_test = N - skip


def rate(preds):
    hit = sum(1 for i, pr in preds if pr == seq[i])
    return hit / max(len(preds), 1)


# ---- 预测器 1:repeat-last(ŝ_{i}=s_{i-h}) ----
rep = {}
for h in (1, 2, 4, 8, 16, 32, 64):
    pr = [(i, seq[i - h]) for i in test if i - h >= skip]
    rep[h] = rate(pr)

# ---- 预测器 2:滑动窗多数(W 个最近集合的众数) ----
win = {}
for W in (2, 5, 10, 20, 50, 100, 200):
    # 滚动计数易写错,直接切片重算(N=6e4, W<=200, ~1e7 ops 可接受)
    pr = []
    for i in range(skip + W, N):
        c = Counter(seq[i - W:i])
        pr.append((i, c.most_common(1)[0][0]))
    win[W] = rate(pr)

# ---- 预测器 3:静态前缀众数(warmup 段训练,之后固定) ----
static_pred = Counter(seq[:skip]).most_common(1)[0][0]
hit_static = sum(1 for i in test if seq[i] == static_pred) / n_test

# ---- 参照:全序列众数(用了未来信息,静态上界) ----
oracle_pred = Counter(seq).most_common(1)[0][0]
hit_oracle = sum(1 for i in test if seq[i] == oracle_pred) / n_test
base = 1.0 / len(pids)

# ---- run 结构:hazard(连跑 L 次后仍延续的概率) ----
runs, cur = [], 1
for i in range(skip + 1, N):
    if seq[i] == seq[i - 1]:
        cur += 1
    else:
        runs.append(cur)
        cur = 1
runs.append(cur)
hazard = {}
for L in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500):
    reached = sum(1 for r in runs if r >= L)
    ended = sum(1 for r in runs if r == L)
    if reached:
        hazard[L] = {"reached": reached, "stayP": 1 - ended / reached}

# ---- 边界转移矩阵(run 切换时 a→b) ----
trans = Counter()
for i in range(skip + 1, N):
    if seq[i] != seq[i - 1]:
        trans[(seq[i - 1], seq[i])] += 1
top_trans = trans.most_common(8)

print(f"===== A·straggler 可预测性({KEY}, N={N}, skip前{SKIP_PCT:.0f}%={skip}, 测试{n_test}) =====")
print(f"AR→AR 中位间隔: {gap_med}µs(horizon 换算:h=8≈{gap_med*8/1000:.1f}ms 提前量)")
print(f"随机基线 {base:.3f} | 静态前缀众数 {hit_static:.3f} | 全局众数(上界) {hit_oracle:.3f}")
print("\nrepeat-last 命中率 vs horizon h:")
for h, r in rep.items():
    print(f"  h={h:3d} ({gap_med*h/1000:7.1f}ms): {r:.3f}")
print("\n滑动窗多数命中率 vs 窗口 W:")
for W, r in win.items():
    print(f"  W={W:3d}: {r:.3f}")
print("\nrun 延续概率 P(stay | 已连跑 L 次):")
print("  " + "  ".join(f"L{L}:{v['stayP']:.3f}" for L, v in hazard.items()))
print(f"\n边界转移 top8: " + ", ".join(f"{a}→{b}:{c}" for (a, b), c in top_trans))

out = {"key": KEY, "N": N, "skip": skip, "gapMedUs": gap_med, "nRank": len(pids), "mode": MODE,
       "hitStaticPrefix": hit_static, "hitGlobalModeUpper": hit_oracle, "randomBase": base,
       "hitRepeatByH": rep, "hitWinMajority": win,
       "runHazard": hazard, "topTransitions": [[a, b, c] for (a, b), c in top_trans],
       "lastCnt": {p: seq.count(p) for p in pids}}
with open(os.path.join(ADIR, "straggler_predict.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n-> {ADIR}/straggler_predict.json")
