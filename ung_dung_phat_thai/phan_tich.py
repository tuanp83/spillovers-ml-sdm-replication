"""Ứng dụng G3 với y = log phát thải (năng lượng + giao thông), CAMS-GLOB-ANT v6.2, 638 huyện.
x  = log mật độ dân số (chuẩn hoá), Z = log tỷ lệ diện tích xây dựng GHSL 2020, xh = dự báo bậc hai theo Z
     ước lượng trên các huyện huấn luyện (đúng như vn_v2.py). Hằng số được loại bỏ bằng cách lấy độ lệch khỏi trung bình.
Câu hỏi: (1) E có thông tin riêng ngoài mật độ không; (2) A1 (ảnh không phản ứng theo y) có bị bác bỏ không;
(3) SDM oracle có định danh không (rho trong (-1,1), theta+rho*beta xa 0); (4) naive có sinh lan toả giả không;
(5) LC-IV với mẫu kiểm tra ngẫu nhiên có khôi phục oracle và tập tin cậy có thông tin không.
Chạy: ../.venv/bin/python -I phan_tich.py [B]
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import chi2, norm
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "g3_code")
sys.path.insert(0, CODE)
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta, omega
from g3_v2_ident import tid

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
K = 4; C2 = chi2.ppf(0.95, K - 2)
DL = os.path.join(HERE, "..", "du_lieu_o_nhiem")
d = pd.read_csv(os.path.join(DL, "ghep_huyen", "vn_districts_G3.csv"))
Wd = sp.load_npz(os.path.join(DL, "ghep_huyen", "W_queen_rowstd.npz"))
G = Geo(Wd); n = G.n; W = G.W
om = np.real(np.linalg.eigvals(Wd.toarray()))
pt = pd.read_csv(os.path.join(DL, "phat_thai_huyen_G3_dai.csv"))

x = d.log_density.values; xc = (x - x.mean()) / x.std()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]
Z = d.log_built_share_2020.values
X = np.column_stack([np.ones(n), Z, Z ** 2]); cf, *_ = np.linalg.lstsq(X[tr], xc[tr], rcond=None); xh = X @ cf
dm = lambda v: v - v.mean()
xc0, xh0 = dm(xc), dm(xh)
kappa = float(1 - np.var(xc[rest] - xh[rest]) / np.var(xc[rest]))


def E_of(chat, nam, nganh=("nang_luong", "giao_thong")):
    s = pt[(pt.bien == chat) & (pt.nam == nam) & pt.nganh.isin(nganh)]
    return s.groupby("g3_row").t_km2_nam.sum().sort_index().values


def moran(v):
    z = v - v.mean(); return float(z @ (W @ z) / (z @ z))


def impacts(dd):
    r, b, t = dd
    if not -1 < r < 1: return [np.nan] * 3
    adi = float(np.mean((b + t * om) / (1 - r * om))); ati = float((b + t) / (1 - r)); return [adi, ati - adi, ati]


def tsls(v, y):
    """2SLS SDM, H = [v, Wv, W2v, W3v]; SE HC0 (sandwich)."""
    cols = [v]
    for _ in range(K - 1): cols.append(W @ cols[-1])
    H = np.column_stack(cols); Q = np.column_stack([W @ y, v, W @ v])
    Ai = np.linalg.inv(H.T @ H); M = Q.T @ H @ Ai @ H.T @ Q; Mi = np.linalg.inv(M)
    dd = Mi @ Q.T @ H @ Ai @ H.T @ y; u = y - Q @ dd
    Hu = H * u[:, None]; mid = Q.T @ H @ Ai @ (Hu.T @ Hu) @ Ai @ H.T @ Q
    V = Mi @ mid @ Mi
    # SE cho theta + rho*beta (delta method)
    gr = np.array([dd[1], dd[0], 1.0]); s_nonid = float(np.sqrt(gr @ V @ gr))
    return dd, np.sqrt(np.diag(V)), s_nonid


def gamma_test(y, idx):
    Xg = np.column_stack([np.ones(len(idx)), x[idx], y[idx]])
    XtXi = np.linalg.inv(Xg.T @ Xg); b = XtXi @ Xg.T @ Z[idx]; u = Z[idx] - Xg @ b
    V = XtXi @ (Xg.T * u ** 2) @ Xg @ XtXi; t = b[2] / np.sqrt(V[2, 2])
    return dict(gamma=float(b[2]), t=float(t), p=float(2 * norm.sf(abs(t))))


def slx(v, y):
    Xs = np.column_stack([np.ones(n), v, W @ v]); b = np.linalg.lstsq(Xs, y, rcond=None)[0]; u = y - Xs @ b
    XtXi = np.linalg.inv(Xs.T @ Xs); V = XtXi @ (Xs.T * u ** 2) @ Xs @ XtXi
    return dict(beta=float(b[1]), theta=float(b[2]), se_theta=float(np.sqrt(V[2, 2])))


def analyse(y_raw, tag, B, seed=2026, cfrac=0.01):
    c = cfrac * np.median(y_raw[y_raw > 0])   # 9 huyện GT=0 mọi năm (lỗ hổng bản đồ phân bổ)
    y = dm(np.log(y_raw + c))
    o = dict(tag=tag, cfrac=cfrac, n_zero=int((y_raw <= 0).sum()), moran_y=moran(y), r_y_x=float(np.corrcoef(y, xc)[0, 1]),
             r_y_rwi=float(np.corrcoef(y[d.rwi_mean.notna()], d.rwi_mean.dropna())[0, 1]))
    o["gamma_train"] = gamma_test(y, tr); o["gamma_all"] = gamma_test(y, np.arange(n))
    dd, se, snon = tsls(xc0, y)
    o["oracle"] = dict(delta=dd.tolist(), se=se.tolist(), theta_plus_rho_beta=float(dd[2] + dd[0] * dd[1]),
                       se_tprb=snon, impacts=impacts(dd))
    dn, sen, _ = tsls(xh0, y); o["naive"] = dict(delta=dn.tolist(), se=sen.tolist(), impacts=impacts(dn))
    o["slx_oracle"] = slx(xc0, y); o["slx_naive"] = slx(xh0, y)
    if B:
        rng = np.random.default_rng(seed)
        for nl in (60, 120):
            D, rej0, bnd = [], [], []
            for _ in range(B):
                lab = rng.choice(rest, nl, replace=False)
                S = build(G, y, xh0, xc0, lab); de, P = estimate(S); s2 = sigma2_U(S, de); D.append(de)
                rej0.append(T_theta(S, 0.0, s2, de[1], max(abs(de[1]), 0.5)) > C2)
                bnd.append(tid(S["G"][:, 0], S["G"][:, 1:], lambda b, t, S=S: omega(S, b, t, 0.0), (de[1], de[2])) > C2)
            D = np.array(D)
            o[f"lciv_nl{nl}"] = dict(median=np.median(D, 0).tolist(),
                                     iqr=(np.percentile(D, 75, 0) - np.percentile(D, 25, 0)).tolist(),
                                     med_abs_dev_oracle=np.median(np.abs(D - dd), 0).tolist(),
                                     rej_theta0=float(np.mean(rej0)), rho_cs_bounded=float(np.mean(bnd)))
    print(json.dumps(o, default=float), flush=True)
    return o


out = dict(n=n, kappa=kappa, moran_x=moran(xc), B=B)
# phân tích chính: NOx 2020
out["chinh_NOx_2020"] = analyse(E_of("nitrogen_oxides", 2020), "NOx 2020 NL+GT", B)
# độ vững (không lặp kiểm tra để nhanh, trừ SO2)
for chat in ("sulphur_dioxide", "black_carbon", "organic_carbon"):
    out[f"{chat}_2020"] = analyse(E_of(chat, 2020), f"{chat} 2020 NL+GT", 0)
out["NOx_2010"] = analyse(E_of("nitrogen_oxides", 2010), "NOx 2010 NL+GT", 0)
out["NOx_2020_giao_thong"] = analyse(E_of("nitrogen_oxides", 2020, ("giao_thong",)), "NOx 2020 GT", 0)
e_nl = E_of("nitrogen_oxides", 2020, ("nang_luong",))
out["NOx_2020_nang_luong"] = analyse(e_nl, "NOx 2020 NL", 0)
out["NOx_2020_c10"] = analyse(E_of("nitrogen_oxides", 2020), "NOx 2020 NL+GT c=10%", 0, cfrac=0.10)
out["ty_le_nang_luong_bang_0"] = float(np.mean(e_nl <= 0))
json.dump(out, open(os.path.join(HERE, "ket_qua.json"), "w"), indent=1, default=float)
print("đã ghi ket_qua.json")
