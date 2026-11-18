"""Semi-synthetic SDM on real Vietnamese geography with the revised inference (g3_v2).
Real W, real x (log density, standardised), real predictor xh (quadratic in log built-up share, fitted on training
districts). y = S(rho)(beta x + theta W x + eps). Audit = SRS of non-training districts.
Reports: median bias (naive, LC-IV) incl. AII/ATI; size of T_theta, T_rho, T_ati; bounded share of CS_rho (T_id > c)
and its prediction from the population identification strength mu^2 (Proposition 6).
"""
import os, sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import chi2, ncx2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta, T_rho, T_ati, omega
from g3_v2_ident import tid, pop_strength

K = 4; C = chi2.ppf(0.95, K - 2)
d = pd.read_csv(os.path.join(HERE, "..", "data", "vn_districts_G3.csv")); Wd = sp.load_npz(os.path.join(HERE, "..", "data", "W_queen_rowstd.npz"))
G = Geo(Wd); n = G.n; W = G.W
x = d.log_density.values; xc = (x - x.mean()) / x.std()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]
Z = d.log_built_share_2020.values
X = np.column_stack([np.ones(n), Z, Z ** 2]); cf, *_ = np.linalg.lstsq(X[tr], xc[tr], rcond=None); xh = X @ cf
om = np.real(np.linalg.eigvals(Wd.toarray()))


def impacts(dd):
    r, b, t = dd
    if not -1 < r < 1:
        return np.array([np.nan] * 3)
    adi = np.mean((b + t * om) / (1 - r * om)); ati = (b + t) / (1 - r); return np.array([adi, ati - adi, ati])


def tsls(v, y):
    cols = [v]
    for _ in range(K - 1):
        cols.append(W @ cols[-1])
    H = np.column_stack(cols); Q = np.column_stack([W @ y, v, W @ v]); A = H.T @ H; Gm = H.T @ Q
    return np.linalg.solve(Gm.T @ np.linalg.solve(A, Gm), Gm.T @ np.linalg.solve(A, H.T @ y))


def run(R):
    rng = np.random.default_rng(2026); out = {"kappa": float(1 - np.var(xc[rest] - xh[rest]) / np.var(xc[rest]))}
    for (rho, beta, theta, tag) in [(0.4, 1.0, 0.5, "sdm"), (0.5, 1.0, -0.5, "nonid"), (0.0, 1.0, 0.0, "null")]:
        true = np.array([rho, beta, theta]); ti = impacts(true); ati = ti[2]
        ymean = G.solve(rho, beta * xc + theta * (W @ xc))
        for nl in (30, 60, 120):
            mu2 = pop_strength(G, xc, xh, ymean, nl)
            rec = {k: [] for k in ("naive", "lciv", "rej_th", "rej_rh", "rej_at", "bnd")}; t0 = time.time()
            for r in range(R):
                y = ymean + G.solve(rho, rng.standard_normal(n))
                lab = rng.choice(rest, nl, replace=False)
                rec["naive"].append(tsls(xh, y))
                S = build(G, y, xh, xc, lab); dd, P = estimate(S); s2 = sigma2_U(S, dd); rec["lciv"].append(dd)
                sb = max(abs(dd[1]), 0.5); st = max(abs(dd[2]), 0.5); sr = max(abs(dd[0]), 0.5)
                rec["rej_th"].append(T_theta(S, theta, s2, dd[1], sb) > C)
                rec["rej_rh"].append(T_rho(S, rho, s2, dd, (sb, st)) > C)
                if tag != "null":
                    rec["rej_at"].append(T_ati(S, ati, s2, dd, (sr, sb)) > C)
                rec["bnd"].append(tid(S["G"][:, 0], S["G"][:, 1:], lambda b, t, S=S: omega(S, b, t, 0.0), (dd[1], dd[2])) > C)
            o = dict(nl=nl, R=R, mu2=mu2, bounded_pred=float(ncx2.sf(C, K - 2, max(mu2, 1e-12))),
                     bounded_obs=float(np.mean(rec["bnd"])), secs=round(time.time() - t0))
            for k in ("naive", "lciv"):
                D = np.array(rec[k]); I = np.array([impacts(v) for v in D])
                o[k] = dict(medbias=(np.median(D, 0) - true).round(3).tolist(),
                            imp_medbias=(np.nanmedian(I, 0) - ti).round(3).tolist() if tag != "null" else None)
            for k in ("rej_th", "rej_rh", "rej_at"):
                if rec[k]:
                    o[k] = float(np.mean(rec[k]))
            out[f"{tag}_nl{nl}"] = o; print(tag, json.dumps(o), flush=True)
            json.dump(out, open(os.path.join(HERE, "..", "data", "vn_v2.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
