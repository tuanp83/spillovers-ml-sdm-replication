"""Lan toả giả với một "ảnh" vệ tinh thô thật: CAMS EAC4 0,75° (74 ô cho 638 huyện).
So sánh hai dự báo của x = log mật độ (chuẩn hoá), cùng địa lý thật và W thật:
  GHSL : bậc hai theo log tỷ lệ xây dựng GHSL 2020 (dự báo của bài, κ≈0,95)
  CAMS : tuyến tính theo log của 10 biến CAMS 2020 (PM2.5, AOD và thành phần, cột NO2/CO/SO2/HCHO)
Cả hai đều ước lượng trên cùng 20% huyện huấn luyện.
(1) Giá trị giả-thật CHÍNH XÁC của θ SLX ngây thơ khi θ=0: β × hệ số của W x̂ trong phép chiếu x lên (1, x̂, W x̂)
    — xác định hoàn toàn từ dữ liệu thật, không cần mô phỏng.
(2) Mô phỏng bán tổng hợp y = S(ρ)(βx + θWx + ε): sai lệch naive vs LC-IV, size T_θ, tỷ lệ CS_ρ bị chặn.
Chạy: ../.venv/bin/python -I vn_cams.py [R]
"""
import os, sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import chi2, ncx2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "g3_code"))
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta, omega
from g3_v2_ident import tid, pop_strength

R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
K = 4; C = chi2.ppf(0.95, K - 2)
DL = os.path.join(HERE, "..", "du_lieu_o_nhiem", "ghep_huyen")
d = pd.read_csv(os.path.join(DL, "vn_districts_G3.csv")); Wd = sp.load_npz(os.path.join(DL, "W_queen_rowstd.npz"))
cams = pd.read_csv(os.path.join(HERE, "cams_huyen_2020.csv")).sort_values("g3_row")
G = Geo(Wd); n = G.n; W = G.W; om = np.real(np.linalg.eigvals(Wd.toarray()))
x = d.log_density.values; xc = (x - x.mean()) / x.std()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]


def fit(F):
    X = np.column_stack([np.ones(n), F]); cf, *_ = np.linalg.lstsq(X[tr], xc[tr], rcond=None); return X @ cf


Z = d.log_built_share_2020.values
V = ["pm2p5", "aod550", "bcaod550", "omaod550", "suaod550", "duaod550", "tcno2", "tcco", "tcso2", "tchcho"]
PRED = {"GHSL": fit(np.column_stack([Z, Z ** 2])), "CAMS": fit(np.log(cams[V].values))}
dm = lambda v: v - v.mean()
moran = lambda v: float(dm(v) @ (W @ dm(v)) / (dm(v) @ dm(v)))


def impacts(dd):
    r, b, t = dd
    if not -1 < r < 1: return np.array([np.nan] * 3)
    adi = np.mean((b + t * om) / (1 - r * om)); ati = (b + t) / (1 - r); return np.array([adi, ati - adi, ati])


def tsls(v, y):
    cols = [v]
    for _ in range(K - 1): cols.append(W @ cols[-1])
    H = np.column_stack(cols); Q = np.column_stack([W @ y, v, W @ v]); A = H.T @ H; Gm = H.T @ Q
    return np.linalg.solve(Gm.T @ np.linalg.solve(A, Gm), Gm.T @ np.linalg.solve(A, H.T @ y))


def slx(v, y):
    X = np.column_stack([np.ones(n), v, W @ v]); return np.linalg.lstsq(X, y, rcond=None)[0][1:]


out = {}
for nm, xh in PRED.items():
    e = xc - xh
    # (1) θ SLX giả-thật khi θ=0, β=1 (chính xác, không mô phỏng)
    b_slx = slx(xh, xc)
    out[nm] = dict(kappa=float(1 - np.var(e[rest]) / np.var(xc[rest])), corr_e_xh=float(np.corrcoef(e[rest], xh[rest])[0, 1]),
                   moran_x=moran(xc), moran_xh=moran(xh), moran_e=moran(e), n_distinct=int(len(np.unique(xh.round(8)))),
                   slx_pseudo_true_beta=float(b_slx[0]), slx_pseudo_true_theta=float(b_slx[1]))
    print(nm, json.dumps(out[nm]), flush=True)

rng0 = 2026
for nm, xh0 in PRED.items():
    xh = dm(xh0); xcc = dm(xc)
    for (rho, beta, theta, tag) in [(0.0, 1.0, 0.0, "null"), (0.4, 1.0, 0.5, "sdm")]:
        true = np.array([rho, beta, theta]); ti = impacts(true)
        ymean = G.solve(rho, beta * xcc + theta * (W @ xcc))
        for nl in (60, 120, 250):
            rng = np.random.default_rng(rng0)
            mu2 = pop_strength(G, xcc, xh, ymean, nl)
            rec = {k: [] for k in ("naive", "lciv", "slx_naive", "slx_oracle", "rej_th", "bnd")}; t0 = time.time()
            for _ in range(R):
                y = ymean + G.solve(rho, rng.standard_normal(n)); y = y - y.mean()
                lab = rng.choice(rest, nl, replace=False)
                rec["naive"].append(tsls(xh, y)); rec["slx_naive"].append(slx(xh, y)); rec["slx_oracle"].append(slx(xcc, y))
                S = build(G, y, xh, xcc, lab); dd, P = estimate(S); s2 = sigma2_U(S, dd); rec["lciv"].append(dd)
                rec["rej_th"].append(T_theta(S, theta, s2, dd[1], max(abs(dd[1]), .5)) > C)
                rec["bnd"].append(tid(S["G"][:, 0], S["G"][:, 1:], lambda b, t, S=S: omega(S, b, t, 0.0), (dd[1], dd[2])) > C)
            o = dict(nl=nl, R=R, mu2=float(mu2), bounded_pred=float(ncx2.sf(C, K - 2, max(mu2, 1e-12))),
                     bounded_obs=float(np.mean(rec["bnd"])), size_theta=float(np.mean(rec["rej_th"])), secs=round(time.time() - t0))
            for k in ("naive", "lciv"):
                D = np.array(rec[k]); I = np.array([impacts(v) for v in D])
                o[k] = dict(medbias=(np.median(D, 0) - true).round(3).tolist(),
                            imp_medbias=(np.nanmedian(I, 0) - ti).round(3).tolist() if tag != "null" else None,
                            mad=np.median(np.abs(D - true), 0).round(3).tolist())
            for k in ("slx_naive", "slx_oracle"):
                D = np.array(rec[k]); o[k] = dict(theta_med=float(np.median(D[:, 1])), theta_rej0=None)
            out[f"{nm}_{tag}_nl{nl}"] = o; print(nm, tag, json.dumps(o), flush=True)
            json.dump(out, open(os.path.join(HERE, "ket_qua_cams.json"), "w"), indent=1, default=float)
print("xong")
