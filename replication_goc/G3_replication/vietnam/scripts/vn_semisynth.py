"""Semi-synthetic experiment on real Vietnamese geography:
real W (queen, 638 districts), real x (official log density), real satellite predictor xh (GHSL 2020),
synthetic y = S(rho)(beta x + theta W x + eps). Audit = random subsample of non-training districts.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp
from types import SimpleNamespace
from scipy.stats import chi2, norm
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from g3_core import build, estimate, wald_se, sigma2_U, ar_rho, ols_fit
from g3_sim import moran_test

d = pd.read_csv("vn_districts_G3.csv"); W = sp.load_npz("W_queen_rowstd.npz").toarray(); n = len(d); K = 4
L = SimpleNamespace(W=W, n=n, K=K, Wp=[np.linalg.matrix_power(W, k) for k in range(K)])
om = np.real(np.linalg.eigvals(W))
x = d.log_density.values; xc = (x - x.mean()) / x.std()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]
Z = d.log_built_share_2020.values
xh = ols_fit([Z, Z ** 2], xc, tr)
xt = ols_fit([xh, W @ xh, L.Wp[2] @ xh], xc, tr)
rng = np.random.default_rng(99); RG = np.linspace(-0.99, 0.99, 199)
def impacts(dd):
    r, b, t = dd
    if not -1 < r < 1: return np.array([np.nan] * 3)
    adi = np.mean((b + t * om) / (1 - r * om)); ati = (b + t) / (1 - r); return np.array([adi, ati - adi, ati])
def tsls(v, y):
    H = np.column_stack([L.Wp[k] @ v for k in range(K)]); Q = np.column_stack([W @ y, v, W @ v]); Ai = np.linalg.inv(H.T @ H)
    return np.linalg.solve(Q.T @ H @ Ai @ H.T @ Q, Q.T @ H @ Ai @ H.T @ y)
res = {"kappa": float(1 - np.var(xc[rest] - xh[rest]) / np.var(xc[rest])), "n": n}
for (rho, beta, theta, tag) in [(0.0, 1.0, 0.0, "null"), (0.4, 1.0, 0.5, "sdm")]:
    S_inv = np.linalg.inv(np.eye(n) - rho * W); true = np.array([rho, beta, theta]); ti = impacts(true)
    for nl in (30, 60, 120):
        rec = {k: [] for k in ("naive", "LCIV", "CVSAP", "oracle", "slx_naive", "slx_oracle", "mor_naive", "mor_oracle", "cov_LCIV", "cov_CVSAP", "bnd_LCIV", "bnd_CVSAP")}
        for r in range(300):
            y = S_inv @ (beta * xc + theta * (W @ xc) + rng.standard_normal(n))
            lab = rest[rng.permutation(len(rest))[:nl]]
            rec["oracle"].append(tsls(xc, y)); rec["naive"].append(tsls(xh, y))
            for v, k in ((xh, "slx_naive"), (xc, "slx_oracle")):
                X = np.column_stack([np.ones(n), v, W @ v]); rec[k].append(np.linalg.lstsq(X, y, rcond=None)[0][2])
            for v, k in ((xh, "mor_naive"), (xc, "mor_oracle")):
                Xm = (v - v.mean())[:, None]; yc = y - y.mean(); u = yc - Xm @ np.linalg.lstsq(Xm, yc, rcond=None)[0]
                rec[k].append(moran_test(u, Xm, W) > norm.ppf(0.95))
            for phi, k in ((xh, "LCIV"), (xt, "CVSAP")):
                S = build(L, y, xh, phi, xc, lab); S["mode"] = "srs"; dd, P = estimate(S)
                s2 = sigma2_U(L, y, xc, phi, lab, dd); V = wald_se(S, dd, P, s2)
                rec[k].append(dd); rec["cov_" + k].append(np.abs(dd - true) <= 1.96 * np.sqrt(np.diag(V)))
                acc = ar_rho(S, RG, s2, dd[1], dd[2]) <= chi2.ppf(0.95, 2); rec["bnd_" + k].append(not (acc[0] or acc[-1]))
        o = {}
        for k in ("oracle", "naive", "LCIV", "CVSAP"):
            D = np.array(rec[k]); I = np.array([impacts(v) for v in D])
            o[k] = dict(medbias=(np.median(D, 0) - true).round(3).tolist(),
                        ati_medbias=float(np.round(np.nanmedian(I[:, 2]) - ti[2], 3)) if tag == "sdm" else None,
                        aii_medbias=float(np.round(np.nanmedian(I[:, 1]) - ti[1], 3)) if tag == "sdm" else None)
        for k in ("LCIV", "CVSAP"):
            o[k]["wald_cov"] = np.mean(rec["cov_" + k], 0).round(3).tolist(); o[k]["rho_bounded"] = float(np.mean(rec["bnd_" + k]))
        o["slx_theta_naive"] = float(np.mean(rec["slx_naive"])); o["slx_theta_oracle"] = float(np.mean(rec["slx_oracle"]))
        o["moran_rej_naive"] = float(np.mean(rec["mor_naive"])); o["moran_rej_oracle"] = float(np.mean(rec["mor_oracle"]))
        res[f"{tag}_nl{nl}"] = o
        print(tag, nl, json.dumps(o), flush=True)
json.dump(res, open("vn_semisynth.json", "w"), indent=1)
