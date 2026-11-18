"""Kiểm chứng Mệnh đề mới (lan toả giả SLX qua γ1 = n^-1 e'W x̂) trên địa lý thật 638 huyện.
(1) Đồng nhất thức: θ*/β = (m11 γ1 − m12 γ0)/(m11 m22 − m12²) khớp hồi quy SLX chính xác (mọi dự báo).
(2) Nhòe theo ô CAMS (x̂ = Px, P = trung bình trong ô): γ0 = 0 chính xác; γ1 > 0; tách phần trong ô / giữa ô.
    Điều kiện đủ: (C1) cặp láng giềng khác ô: cov(x_i, m_b(j)) > cov(m_b(i), m_b(j));
                  (C2) trong ô: tỷ lệ láng giềng cùng ô s_i đồng biến với cov(x_i, m_b(i)).
    Σ không quan sát được -> dùng mô hình SAR(λ) cho x với λ ước lượng (ML trên lưới) để kiểm tra C1, C2.
(3) Ước lượng θ* từ mẫu kiểm tra rời rạc: m.. từ mọi dự báo; γ0, γ1 từ e_i trên mẫu (trung bình mẫu, không trọng số
    vì SRS). Báo cáo trung vị, độ lệch chuẩn, xác suất đúng dấu qua 1000 mẫu, n_L = 30, 60, 120.
"""
import os, json, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import minimize_scalar
HERE = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(HERE, "..", "du_lieu_o_nhiem", "ghep_huyen")
d = pd.read_csv(os.path.join(DL, "vn_districts_G3.csv")); W = sp.load_npz(os.path.join(DL, "W_queen_rowstd.npz")).tocsr()
cams = pd.read_csv(os.path.join(HERE, "cams_huyen_2020.csv")).sort_values("g3_row")
n = W.shape[0]; Wd = W.toarray()
x = d.log_density.values; x = (x - x.mean()) / x.std()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]
dm = lambda v: v - v.mean()
out = {}


def parts(xh, xx=x):
    xh = dm(xh); e = dm(xx) - xh; Wh = dm(W @ xh)
    m11, m12, m22 = xh @ xh / n, xh @ Wh / n, Wh @ Wh / n
    g0, g1 = e @ xh / n, e @ Wh / n
    return m11, m12, m22, g0, g1


def theta_formula(xh):
    m11, m12, m22, g0, g1 = parts(xh); return (m11 * g1 - m12 * g0) / (m11 * m22 - m12 ** 2)


def theta_reg(xh):
    X = np.column_stack([np.ones(n), xh, W @ xh]); return np.linalg.lstsq(X, x, rcond=None)[0][2]


def fit(F):
    X = np.column_stack([np.ones(n), F]); cf, *_ = np.linalg.lstsq(X[tr], x[tr], rcond=None); return X @ cf


V = ["pm2p5", "aod550", "bcaod550", "omaod550", "suaod550", "duaod550", "tcno2", "tcco", "tcso2", "tchcho"]
Z = d.log_built_share_2020.values
cell = cams.cell.values
P = pd.Series(x).groupby(cell).transform("mean").values
PRED = {"GHSL": fit(np.column_stack([Z, Z ** 2])), "CAMS": fit(np.log(cams[V].values)), "blur_Px": P}

# (1) đồng nhất thức
for k, xh in PRED.items():
    m11, m12, m22, g0, g1 = parts(xh)
    out[k] = dict(theta_reg=float(theta_reg(xh)), theta_formula=float(theta_formula(xh)), gamma0=float(g0), gamma1=float(g1),
                  moran_xh=float(m12 / m11), cross_ratio=float((m12 + g1) / (m11 + g0)))
    print(k, {a: round(b, 4) for a, b in out[k].items()})

# (2) nhòe: tách γ1 trong ô / giữa ô
e = x - P; Wh = W @ P
same = (cell[:, None] == cell[None, :])
Wsame, Wdiff = sp.csr_matrix(Wd * same), sp.csr_matrix(Wd * ~same)
g1_in, g1_out = float(e @ (Wsame @ P) / n), float(e @ (Wdiff @ P) / n)
out["blur_split"] = dict(gamma1_trong_o=g1_in, gamma1_giua_o=g1_out, ty_le_lang_gieng_khac_o=float(Wd[~same].sum() / n))
print("tách γ1:", out["blur_split"])

# kiểm tra C1, C2 dưới SAR(λ) ước lượng
eig = np.real(np.linalg.eigvals(Wd))
def nll(l):
    Ax = x - l * (W @ x); s2 = Ax @ Ax / n
    return -(np.sum(np.log(1 - l * eig)) - n / 2 * np.log(s2))
lam = minimize_scalar(nll, bounds=(-0.99, 0.99), method="bounded").x
A = np.linalg.inv(np.eye(n) - lam * Wd); S = A @ A.T; S = S / np.mean(np.diag(S))
cells = np.unique(cell); idx = {c: np.where(cell == c)[0] for c in cells}
Mb = np.zeros((len(cells), n)); ci = {c: k for k, c in enumerate(cells)}
for c in cells: Mb[ci[c], idx[c]] = 1 / len(idx[c])
cov_i_b = S @ Mb.T                     # cov(x_i, m_b)  (n × B)
cov_b_b = Mb @ S @ Mb.T                # cov(m_b, m_b')
I, J = np.nonzero(Wd * ~same)
c1 = cov_i_b[I, [ci[c] for c in cell[J]]] - cov_b_b[[ci[c] for c in cell[I]], [ci[c] for c in cell[J]]]
s_i = (Wd * same).sum(1); cov_own = cov_i_b[np.arange(n), [ci[c] for c in cell]]
from scipy.stats import spearmanr
c2 = [len(idx[c]) * np.cov(s_i[idx[c]], cov_own[idx[c]], bias=True)[0, 1] for c in cells if len(idx[c]) >= 2]
c2_term = float(np.sum(c2)); c1_term = float(np.sum(Wd[I, J] * c1))
g1_pop = float(np.trace(Wd @ (np.eye(n) - np.zeros(n)) @ np.zeros((n, n))) if False else 0)
Pm = np.zeros((n, n))
for c in cells: Pm[np.ix_(idx[c], idx[c])] = 1 / len(idx[c])
g1_pop = float(np.trace(Wd @ Pm @ S @ (np.eye(n) - Pm).T) / n)   # E[e'W x̂]/n với x̂ = Px, e = (I−P)x
out["blur_conditions"] = dict(lambda_SAR=float(lam), C1_ty_le_cap_thoa=float(np.mean(c1 > 0)), C1_trung_binh=float(np.mean(c1)),
                              C2_ty_le_o_cov_khong_am=float(np.mean(np.array(c2) >= -1e-12)), so_hang_trong_o=c2_term/n, so_hang_giua_o=c1_term/n, tong=(c1_term+c2_term)/n,
                              gamma1_tong_the_SAR=g1_pop)
print("điều kiện:", {a: round(b, 4) for a, b in out["blur_conditions"].items()})

# (3) ước lượng θ* từ mẫu kiểm tra
rng = np.random.default_rng(2026)
for k in ("GHSL", "CAMS"):
    xh = dm(PRED[k]); Wh = dm(W @ xh); m11, m12, m22 = xh @ xh / n, xh @ Wh / n, Wh @ Wh / n
    true = theta_formula(PRED[k]); res = {}
    for nl in (30, 60, 120):
        th = []
        for _ in range(1000):
            L = rng.choice(rest, nl, replace=False)
            xbar = x[L].mean() - (PRED[k][L].mean() - PRED[k].mean())   # ước lượng trung bình x (hiệu chỉnh bằng dự báo)
            eL = (x[L] - xbar) - xh[L]
            g0, g1 = np.mean(eL * xh[L]), np.mean(eL * Wh[L])
            th.append((m11 * g1 - m12 * g0) / (m11 * m22 - m12 ** 2))
        th = np.array(th)
        res[nl] = dict(trung_vi=float(np.median(th)), sd=float(th.std()), p_dung_dau=float(np.mean(np.sign(th) == np.sign(true))),
                       ci90=np.percentile(th, [5, 95]).round(3).tolist())
    out[f"audit_{k}"] = dict(theta_true=float(true), **{f"nl{a}": b for a, b in res.items()})
    print("ước lượng từ mẫu kiểm tra", k, json.dumps(out[f"audit_{k}"]))
json.dump(out, open(os.path.join(HERE, "kiem_chung_menh_de.json"), "w"), indent=1, default=float)
