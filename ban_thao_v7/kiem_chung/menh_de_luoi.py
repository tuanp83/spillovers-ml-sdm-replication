"""Kiểm chứng số Mệnh đề (lan toả giả SLX cho mọi dự báo) trên lưới rook 40×40 — thay đoạn dùng dữ liệu huyện VN.
x ~ SAR(0,8) (chuẩn hoá). Ba dự báo: (a) ảnh riêng Z=x+ν, ν ~ SAR(0,3), κ=0,7, hệ số Bayes;
(b) ảnh riêng, nhiễu trơn hơn tín hiệu ν ~ SAR(0,95); (c) nhòe thuần x̂ = Px, ô 4×4.
Báo cáo: θ* theo công thức vs hồi quy; γ0, γ1; phân rã γ1 trong/giữa ô và tỷ lệ thỏa C1, C2 (Σ của SAR);
ước lượng từ mẫu kiểm tra rời rạc (1000 mẫu, n_L = 30, 60, 120)."""
import sys, os, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, __import__("os").environ.get("G3_ROOT",".")+"/g3_code")
from g3_v2 import Geo
m = 40; G = Geo(Geo.rook(m)); n = G.n; W = G.W; Wd = W.toarray(); rng = np.random.default_rng(2026)
dm = lambda v: v - v.mean()
x = G.solve(0.8, rng.standard_normal(n)); x = dm(x) / x.std()
def own(phi, kappa=0.7):
    nu = G.solve(phi, rng.standard_normal(n)); nu = nu / nu.std() * np.sqrt((1 - kappa) / kappa); return kappa * (x + nu)
cell = (np.arange(n) // m // 4) * (m // 4) + (np.arange(n) % m) // 4
P = np.zeros(n); 
for c in np.unique(cell): P[cell == c] = x[cell == c].mean()
PRED = {"own_noise_rougher": own(0.3), "own_noise_smoother": own(0.95), "blur_4x4": P}
def parts(xh):
    xh = dm(xh); e = dm(x) - xh; Wh = dm(W @ xh)
    return xh @ xh / n, xh @ Wh / n, Wh @ Wh / n, e @ xh / n, e @ Wh / n
out = {}
for k, xh in PRED.items():
    m11, m12, m22, g0, g1 = parts(xh); f = (m11 * g1 - m12 * g0) / (m11 * m22 - m12 ** 2)
    X = np.column_stack([np.ones(n), xh, W @ xh]); reg = np.linalg.lstsq(X, x, rcond=None)[0][2]
    out[k] = dict(theta_formula=round(float(f), 4), theta_reg=round(float(reg), 4), gamma0=round(float(g0), 4), gamma1=round(float(g1), 4))
# phân rã nhòe + C1, C2 dưới Σ của SAR(0,8)
A = np.linalg.inv(np.eye(n) - 0.8 * Wd); S = A @ A.T; S /= np.mean(np.diag(S))
cells = np.unique(cell); Mb = np.zeros((len(cells), n))
for j, c in enumerate(cells): Mb[j, cell == c] = 1 / np.sum(cell == c)
cib = S @ Mb.T; cbb = Mb @ S @ Mb.T; same = cell[:, None] == cell[None, :]
I, J = np.nonzero(Wd * ~same); c1 = cib[I, cell[J]] - cbb[cell[I], cell[J]]
s_i = (Wd * same).sum(1); cown = cib[np.arange(n), cell]
c2 = [np.sum(cell == c) * np.cov(s_i[cell == c], cown[cell == c], bias=True)[0, 1] for c in cells]
out["blur_conditions"] = dict(C1_share=round(float(np.mean(c1 > 0)), 3), C2_share=round(float(np.mean(np.array(c2) >= -1e-12)), 3),
                              within=round(float(np.sum(c2) / n), 4), between=round(float(np.sum(Wd[I, J] * c1) / n), 4))
# ước lượng từ mẫu kiểm tra
for k in ("own_noise_smoother", "blur_4x4"):
    xh = dm(PRED[k]); Wh = dm(W @ xh); m11, m12, m22 = xh @ xh / n, xh @ Wh / n, Wh @ Wh / n
    true = out[k]["theta_formula"]; res = {}
    for nl in (30, 60, 120):
        th = []
        for _ in range(1000):
            L = rng.choice(n, nl, replace=False); eL = dm(x)[L] - xh[L]
            g0, g1 = np.mean(eL * xh[L]), np.mean(eL * Wh[L]); th.append((m11 * g1 - m12 * g0) / (m11 * m22 - m12 ** 2))
        th = np.array(th); res[nl] = dict(mean=round(float(th.mean()), 3), sd=round(float(th.std()), 3), correct_sign=round(float(np.mean(np.sign(th) == np.sign(true))), 3))
    out[f"audit_{k}"] = res
print(json.dumps(out, indent=1)); json.dump(out, open("menh_de_luoi.json", "w"), indent=1)
