#!/usr/bin/env python3
"""统一分解拟合（2026-09-02 深夜）。

三段分解 T(strategy) ≈ R_first + Σ max(comm_i, compute_i | 人工依赖调制) + Tail
的可拟合版本：对 d 族同二进制终判表（11 格）拟合边界定律
    P(N,q) = (T_d1 − T_d0)/T_d0  （正 = d1 慢）
的闭式经验模型，并做：
  1) 控制量模型：C(q)（RCCL allgather）、G(N,q)（分块 GEMM，含分块损失）、FC（fcollect 恒定性）
  2) 惩罚分解：分块损失份额 vs 流水线份额（用每 N 的 min-q GEMM 作整块代理）
  3) 边界轨迹：P=0 等值线在 (q, cols=N/q) 平面的位置 + 外推预测（预注册 P16 素材）
模型选择：留一交叉验证（LOO）RMSE + 符号合理性。

数据来源（只读）：
  formal 根 cell_matrix.csv：COMM_ONLY / GEMM_ONLY / FC_FCOLLECT_ONLY（r 族与 fc 不受 DS 修复影响）
  同二进制终判表（f17aae1d）：d0/d1 干净值（phaseb_d0dc + dsfix + q16fill 根）
  N4096/q16 控制量缺失 → C 用 q 模型、G 用同 slice 尺寸缩放插补（脚本内打印插补依据）。
输出：results/unified_decomposition_fit_20260902.csv + stdout 报告。
"""
import csv, itertools, json, os
import numpy as np

FORMAL = "/root/private_data/lyc/2ndpaper/results/k500sm_ai_gfx928_4gpu/phaseb_formal_20260902_160115/summary/phaseb_cell_matrix.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "unified_decomposition_fit_20260902.csv")

# ---- 同二进制干净值（PhaseB_d0dc_同二进制终判_20260902.md §2 权威表）----
D0 = {(512,2):12575,(512,4):12568,(512,8):12568,(2048,2):14345,(2048,4):14333,
      (2048,8):14348,(4096,2):18347,(4096,4):18336,(4096,8):18347,(2048,16):14339,(4096,16):18357}
D1 = {(512,2):8799,(512,4):9097,(512,8):17598,(2048,2):10212,(2048,4):10307,
      (2048,8):18580,(4096,2):13939,(4096,4):12898,(4096,8):14487,(2048,16):24361,(4096,16):25847}
CELLS = sorted(D0, key=lambda c: (c[1], c[0]))

def load_formal():
    sel = {}
    for r in csv.DictReader(open(FORMAL)):
        if r["candidate"] == "C0_DEFAULT" and r["path"] in ("COMM_ONLY","GEMM_ONLY","FC_FCOLLECT_ONLY"):
            sel[(r["path"], int(r["N"]), int(r["q"]))] = float(r["e2e_p50_us"])
    return sel

SEL = load_formal()

def lstsq(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def r2(y, yhat):
    ss = float(np.sum((y - yhat) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / st

def loo_rmse(X, y):
    n = len(y); err = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        beta = lstsq(X[m], y[m])
        err.append(float(y[i] - X[i] @ beta))
    return float(np.sqrt(np.mean(np.square(err))))

print("=" * 78)
print("1) 控制量模型")
print("=" * 78)
# --- C(q) = c0 + c1*q（N 无关性先检验）---
crows = sorted((n, q, v) for (p, n, q), v in SEL.items() if p == "COMM_ONLY")
Cdat = {}
for n, q, v in crows:
    Cdat.setdefault(q, {})[n] = v
print("COMM_ONLY 按 (q -> {N: e2e})：")
for q in sorted(Cdat):
    vals = Cdat[q]
    spread = (max(vals.values()) - min(vals.values())) / min(vals.values()) * 100
    print(f"  q={q:<3} " + " ".join(f"N{n}={v:.0f}" for n, v in sorted(vals.items())) + f"   N内散布 {spread:.1f}%")
Xc = np.array([[1.0, q] for _, q, _ in crows]); yc = np.array([v for _, _, v in crows])
bc = lstsq(Xc, yc)
print(f"  拟合 C(q) = {bc[0]:.0f} + {bc[1]:.1f}·q µs   R²={r2(yc, Xc@bc):.4f}  LOO-RMSE={loo_rmse(Xc,yc):.1f}µs")
C_of = lambda q: bc[0] + bc[1] * q

# --- FC 恒定性 ---
fc = [v for (p, n, q), v in SEL.items() if p == "FC_FCOLLECT_ONLY"]
print(f"FC_FCOLLECT_ONLY：n={len(fc)} 均值 {np.mean(fc):.0f}µs  极差 {(max(fc)-min(fc))/np.mean(fc)*100:.2f}%  → N,q 无关（纯 payload 函数）")

# --- G(N,q)：候选 (N,q) 与 (N,q,q*N) 与 (N,q,cols) ---
gr = sorted((n, q, v) for (p, n, q), v in SEL.items() if p == "GEMM_ONLY")
G = {(n, q): v for n, q, v in gr}
gm_forms = {
    "G = a + b·N + c·q":            lambda n, q: [1.0, n, q],
    "G = a + b·N + c·q + d·q·N":    lambda n, q: [1.0, n, q, q * n],
    "G = a + b·N + c·q + d·cols":   lambda n, q: [1.0, n, q, n / q],
}
best_g, best_gk = None, None
for k, f in gm_forms.items():
    X = np.array([f(n, q) for n, q, _ in gr]); y = np.array([v for _, _, v in gr])
    b = lstsq(X, y); lo = loo_rmse(X, y)
    print(f"  {k:32s} R²={r2(y,X@b):.4f} LOO-RMSE={lo:7.1f}µs")
    if best_g is None or lo < best_g:
        best_g, best_gk = lo, (k, f, b)
print(f"  → 采用 {best_gk[0]}（LOO 最优）")
gfun, gbeta = best_gk[1], best_gk[2]
G_of = lambda n, q: float(np.dot(gfun(n, q), gbeta))
# 插补 (4096,16)：G 模型值 + 同 slice 尺寸缩放交叉验证
imp_g_model = G_of(4096, 16)
imp_g_scale = 2 * G[(2048, 8)]  # 256 列 × 16 片 ≈ 2 × (256 列 × 8 片)
G[(4096, 16)] = 0.5 * (imp_g_model + imp_g_scale)
print(f"  G(4096,16) 插补：模型 {imp_g_model:.0f} / 同slice缩放 {imp_g_scale:.0f} → 取均值 {G[(4096,16)]:.0f}µs")
C_ctrl = {(n, q): C_of(q) for (n, q) in CELLS}

print()
print("=" * 78)
print("2) 惩罚分解：分块损失份额 vs 流水线份额")
print("=" * 78)
# 整块 GEMM 代理 = 每个 N 取 min_q（最少切片 → 最接近整块）
Gmono = {n: min(v for (nn, q), v in G.items() if nn == n) for n in (512, 2048, 4096)}
dec_rows = []
print(f"{'cell':>10} {'P=%':>7} {'分块损失pt':>9} {'流水线pt':>8}   （pt = 惩罚百分点; G 代理=min-q GEMM）")
for (n, q) in CELLS:
    P = 100 * (D1[(n, q)] - D0[(n, q)]) / D0[(n, q)]
    split = 100 * (G[(n, q)] - Gmono[n]) / D0[(n, q)]
    pipe = P - split
    dec_rows.append((n, q, P, split, pipe))
    print(f" N{n}/q{q:<3} {P:+7.1f} {split:+9.1f} {pipe:+8.1f}")
splits = [r[3] for r in dec_rows]; pipes = [r[4] for r in dec_rows]
print(f"  分块损失份额：均值 {np.mean(splits):+.1f}pt（范围 {min(splits):+.1f}~{max(splits):+.1f}）")
print(f"  流水线份额  ：均值 {np.mean(pipes):+.1f}pt（|max| {max(abs(p) for p in pipes):.1f}）")
print("  → 结论：分块损失只占惩罚的一小部分，q8/q16 惩罚主体是流水线项（驳'小 GEMM 低效'替代解释）")

print()
print("=" * 78)
print("3) 边界定律闭式拟合 P(N,q)（11 格，正=d1 慢）")
print("=" * 78)
yP = np.array([100 * (D1[c] - D0[c]) / D0[c] for c in CELLS])
forms = {
    "M-C  a + b·q + c·cols":                 lambda n, q: [1.0, q, n / q],
    "M-A  a + b·q + c·cols + d·q·cols":      lambda n, q: [1.0, q, n / q, q * n / q],
    "M-B  a + b·q + c·cols + d/cols":        lambda n, q: [1.0, q, n / q, q / n],
    "M-D  a + b·q + c·cols + d·q² + e·cols²":lambda n, q: [1.0, q, n / q, q * q, (n / q) ** 2],
}
fits = {}
for k, f in forms.items():
    X = np.array([f(n, q) for (n, q) in CELLS])
    b = lstsq(X, yP); lo = loo_rmse(X, yP)
    fits[k] = (f, b, r2(yP, X @ b), lo)
    print(f"  {k:40s} R²={r2(yP, X@b):.4f} LOO-RMSE={lo:5.1f}pt")
    print(f"      系数: " + ", ".join(f"{v:+.4g}" for v in b))
best_k = min(fits, key=lambda k: fits[k][3])
fbest, bbest, r2best, lobest = fits[best_k]
print(f"  → 采用 {best_k}（LOO-RMSE={lobest:.1f}pt）")
P_of = lambda n, q: float(np.dot(fbest(n, q), bbest))

print()
print("4) 拟合质量明细（逐格残差）")
print(f"{'cell':>10} {'实测P':>7} {'拟合P':>7} {'残差':>6}")
for i, (n, q) in enumerate(CELLS):
    p, ph = yP[i], P_of(n, q)
    print(f" N{n}/q{q:<3} {p:+7.1f} {ph:+7.1f} {p-ph:+6.1f}")

print()
print("5) 边界轨迹 P=0 与外推预测（预注册 P16 素材，发射前写死）")
print("   实测符号翻转锚点：q8 在 cols 256(+29.5)~512(−21.0) 之间过零；q16 在 cols 256(+40.8) 未过零")
for q in (4, 8, 16, 32):
    # 在 cols 轴上扫描（N = q·cols 连续化，模型各项均为连续函数）
    cols_grid = np.linspace(16, 8192, 20000)
    pv = np.array([P_of(q * c, q) for c in cols_grid])
    sign_change = np.where(np.diff(np.sign(pv)))[0]
    if len(sign_change):
        c0 = cols_grid[sign_change[0]]
        print(f"  q={q:<3} P=0 在 cols≈{c0:6.0f}  → N*≈{q*c0:6.0f}（模型 {best_k}）")
    else:
        print(f"  q={q:<3} 在 cols∈[16,8192] 内无过零（P 均值 {pv.mean():+.1f}pt）")
print()
print("   可立即验证的外推格子（海光机明天可跑，二进制不变）：")
for (n, q) in [(8192, 8), (8192, 16), (4096, 32)]:
    print(f"     N{n}/q{q:<3} 预测 P = {P_of(n, q):+.1f}pt   (d1 {'慢' if P_of(n,q)>0 else '快'} {abs(P_of(n,q)):.0f}pt)")

print()
print("6) 附加：dc 家族轴经验式（不拟合，仅记录）")
print("   dc(q2/q4) ≈ 7.46~7.59ms 与 N 无关（通信主导、credit 流水吸收 GEMM）；")
print("   dc 随 q 抬升（q8≈11.1ms、q16@256cols=14.7ms）——同步点税与 d1 同向但小一个量级")

# ---- CSV 输出 ----
rows = []
for i, (n, q) in enumerate(CELLS):
    rows.append({"N": n, "q": q, "cols": n // q,
                 "d0_us": D0[(n, q)], "d1_us": D1[(n, q)],
                 "P_measured_pt": round(float(yP[i]), 2),
                 "P_fitted_pt": round(P_of(n, q), 2),
                 "residual_pt": round(float(yP[i] - P_of(n, q)), 2),
                 "split_loss_pt": round(dict(((r[0], r[1]), r[3]) for r in dec_rows)[(n, q)], 2),
                 "pipeline_pt": round(dict(((r[0], r[1]), r[4]) for r in dec_rows)[(n, q)], 2),
                 "C_ctrl_us": round(C_ctrl[(n, q)], 0),
                 "G_ctrl_us": round(G[(n, q)], 0),
                 "G_imputed": "Y" if (n, q) == (4096, 16) else ""})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
meta = {"model_chosen": best_k, "coefficients": [float(v) for v in bbest],
        "R2": round(r2best, 4), "LOO_RMSE_pt": round(lobest, 2),
        "C_model_us": [float(v) for v in bc], "FC_mean_us": round(float(np.mean(fc)), 0)}
with open(OUT.replace(".csv", "_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"\n写出 {OUT}\n写出 {OUT.replace('.csv','_meta.json')}")
