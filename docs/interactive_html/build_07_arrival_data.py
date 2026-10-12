#!/usr/bin/env python3
"""从 phase6 ③ trace 的 per-rank AllReduce 逐事件序列,提取 07 号演示页(实测到达时间线)数据。

输入:analysis/rank_*.json 的 rccl_by_kind.AllReduce.list = [[ts_us, dur_us], ...] × 41,538

口径(2026-10-10 修正版,依据 = 结束时刻同步这一实测事实):
  - 同节点 8 rank 共享主机时钟;跨节点用"AR 结束时刻近似同步"锚定(offset=median,残差 p90≈20µs ≪ δ p50=534µs);
  - 结束同步 ⇒ 时长反序 = 到达反序: dur_r = E − A_r;
  - 真走线 W_i = 全局 16 rank 最小 dur(全局最晚到达者的 kernel≈纯协议+搬运);
  - 真等待_r(i) = dur_r(i) − W_i;
  - REPORT 原方向(w0 早到长等)是错的:实测 w1 整队紧密(节点内 launch 散布 p50=4µs)、
    w0 松散(305µs)且含 75% 的全局最晚者 —— 大家等的是 worker-0 上轮换的迟到者。
输出:07_arrival_data.js(const ARDATA = {...};)
"""
import json, statistics as st

BASE = '/root/private_data/second_hygon/phase6-vllm-deploy/phase6_torch_prof_rdma_20261009_212123/analysis/'
W0 = [927, 929, 931, 933, 934, 935, 936, 939]
W1 = [827, 828, 829, 830, 831, 832, 833, 834]
NODES = [('worker-0', W0), ('worker-1', W1)]
WINDOW_S = 30.3

TS, DUR = {}, {}
for node, pids in NODES:
    for pid in pids:
        ev = json.load(open(f'{BASE}rank_{node}_{pid}.json'))['rccl_by_kind']['AllReduce']['list']
        TS[(node, pid)] = [e[0] for e in ev]
        DUR[(node, pid)] = [e[1] for e in ev]
N = len(TS[('worker-0', 927)])
assert all(len(v) == N for v in TS.values()), '锁步计数不一致!'

ALL = [(n, p) for n, ps in NODES for p in ps]

# ---------- 全局口径:W_i / 真等待 / 锚定 / δ ----------
W = [0.0] * N                      # 每集合全局最小 dur ≈ 真走线
for i in range(N):
    W[i] = min(DUR[k][i] for k in ALL)
WAIT = {k: [DUR[k][i] - W[i] for i in range(N)] for k in ALL}   # 真等待(dur − 全局W)

def mean_end(node, i):
    pids = dict(NODES)[node]
    return sum(TS[(node, p)][i] + DUR[(node, p)][i] for p in pids) / len(pids)
diffs = [mean_end('worker-0', i) - mean_end('worker-1', i) for i in range(N)]
OFF = st.median(diffs)
resid = [abs(d - OFF) for d in diffs]

ADJ = {k: (0.0 if k[0] == 'worker-0' else OFF) for k in ALL}    # 锚定后时间戳
delta, lastNode, lastRank = [], {'worker-0': 0, 'worker-1': 0}, {}
for i in range(N):
    at = {k: TS[k][i] + ADJ[k] for k in ALL}
    lo = min(at.values()); hi = max(at.values())
    delta.append(hi - lo)
    lk = max(at, key=at.get)
    lastNode[lk[0]] += 1; lastRank[lk] = lastRank.get(lk, 0) + 1

# 节点内 launch 散布(= 节点内 dur 极差,结束同步时两者等价)
def node_spread(node):
    pids = dict(NODES)[node]
    return [max(DUR[(node, p)][i] for p in pids) - min(DUR[(node, p)][i] for p in pids) for i in range(N)]
sp0, sp1 = node_spread('worker-0'), node_spread('worker-1')

# 互验(节点内):dur−dur_min_node 应 ≈ node_max_ts − ts(自旋模型一致性)
valD = []
for node, pids in NODES:
    for i in range(2000, 6000):
        tmax = max(TS[(node, p)][i] for p in pids)
        dmin = min(DUR[(node, p)][i] for p in pids)
        for p in pids:
            valD.append(abs((DUR[(node, p)][i] - dmin) - (tmax - TS[(node, p)][i])))

def pct(a, q):
    a = sorted(a); k = (len(a) - 1) * q / 100.0
    f = int(k); c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)

# ---------- 窗口 ----------
gaps = [TS[('worker-0', 927)][i + 1] - TS[('worker-0', 927)][i] for i in range(N - 1)]
blocks = [(b, st.median(gaps[b:b + 1024])) for b in range(0, N - 1024, 1024)]
steady = min((b for b in blocks if N * 0.15 < b[0] < N * 0.85), key=lambda b: abs(b[1] - 730))
WINS = [('窗口A·前段(爬坡)', 50), ('窗口B·稳态decode', steady[0]), ('窗口C·后段', N - 150)]
WLEN = 96

# ---------- 汇总打印 ----------
print(f'锚定 offset={OFF:.1f}µs 残差 p50={pct(resid,50):.1f} p90={pct(resid,90):.1f}µs;互验 p50={pct(valD,50):.1f} p90={pct(valD,90):.1f}µs')
print(f'真走线 W: mean={st.mean(W):.1f} p50={pct(W,50):.1f}µs 总量={st.mean(W)*N/1e6:.2f}s({st.mean(W)*N/1e6/WINDOW_S*100:.1f}%)')
print(f'δ(全局到达散布): p10={pct(delta,10):.0f} p50={pct(delta,50):.0f} p90={pct(delta,90):.0f} p99={pct(delta,99):.0f}µs')
print(f'节点内 launch 散布 p50: w0={pct(sp0,50):.0f}µs w1={pct(sp1,50):.0f}µs;全局最后到:{lastNode}')
for k in sorted(lastRank, key=lastRank.get, reverse=True)[:5]:
    print(f'  全局最常迟到: {k} {lastRank[k]}({lastRank[k]/N*100:.1f}%)')
for k in ALL:
    print(f'  {k[0]} pid{k[1]}: 真等待均值 {st.mean(WAIT[k]):6.1f}µs 占窗口 {sum(WAIT[k])/1e6/WINDOW_S*100:5.1f}%')

# ---------- 生成 JS ----------
ranks = [{'node': n, 'pid': p, 'label': f'r{ni*8+li}'} for ni, (n, ps) in enumerate(NODES) for li, p in enumerate(ps)]
winOut = []
for name, i0 in WINS:
    t0 = TS[('worker-0', 927)][i0]
    ev = []
    for i in range(i0, i0 + WLEN):
        ev.append([[round(TS[k][i] + ADJ[k] - t0), round(DUR[k][i])] for k in ALL])
    winOut.append({'name': name, 'i0': i0, 'n': WLEN,
                   'medianGapUs': round(st.median(gaps[i0:i0 + WLEN])),
                   'spanUs': round(TS[('worker-0', 927)][i0 + WLEN - 1] - t0), 'events': ev})

nb = 40; wmax = min(pct(delta, 99.5), 2000)
cnt = [0] * nb
for d in delta:
    if 0 <= d <= wmax: cnt[min(int(d / wmax * nb), nb - 1)] += 1

perRank = []
for k in ALL:
    perRank.append({'pid': k[1], 'node': k[0],
                    'waitMeanUs': round(st.mean(WAIT[k])),
                    'waitSharePct': round(sum(WAIT[k]) / 1e6 / WINDOW_S * 100, 1),
                    'lastGlobalCnt': lastRank.get(k, 0),
                    'lastGlobalPct': round(lastRank.get(k, 0) / N * 100, 1)})

data = {
    'meta': {'nCollectives': N, 'windowS': WINDOW_S, 'nRank': 16,
             'src': 'phase6_torch_prof_rdma_20261009_212123 · loadgen conc=16 maxtok=128 · TP16 eager · Tree-LL',
             'anchorOffsetUs': round(OFF, 1), 'anchorResidP50': round(pct(resid, 50), 1), 'anchorResidP90': round(pct(resid, 90), 1),
             'validateP50': round(pct(valD, 50), 1), 'validateP90': round(pct(valD, 90), 1),
             'wireMeanUs': round(st.mean(W), 1), 'wireP50Us': round(pct(W, 50), 1),
             'wireTotalS': round(st.mean(W) * N / 1e6, 2), 'wireSharePct': round(st.mean(W) * N / 1e6 / WINDOW_S * 100, 1),
             'deltaP10': round(pct(delta, 10)), 'deltaP50': round(pct(delta, 50)),
             'deltaP90': round(pct(delta, 90)), 'deltaP99': round(pct(delta, 99)),
             'spreadW0P50': round(pct(sp0, 50)), 'spreadW1P50': round(pct(sp1, 50)),
             'lastGlobalW0Pct': round(lastNode['worker-0'] / N * 100, 1), 'lastGlobalW1Pct': round(lastNode['worker-1'] / N * 100, 1)},
    'ranks': ranks, 'windows': winOut,
    'deltaHist': {'bins': cnt, 'edges': [round(wmax * j / nb) for j in range(nb + 1)]},
    'perRank': perRank,
}
with open('/root/private_data/second_hygon/docs/interactive_html/07_arrival_data.js', 'w') as f:
    f.write('// 由 build_07_arrival_data.py 从 phase6 ③ 实测 trace 自动生成(41,538 AR × 16 rank,修正口径)\n')
    f.write('const ARDATA = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('written 07_arrival_data.js')
