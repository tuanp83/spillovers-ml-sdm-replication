"""Item 4: violation of A1 (image reflects the outcome).
DGP: Z = x + gamma*y + nu. Then xh is correlated with eps and LC-IV is inconsistent:
   q_k = n^-1 E[(W^k xh)' eps] = c*gamma*sigma^2 * n^-1 tr(W^k' S'),  xh = a + c Z.
Remedies compared:
   (i)  purged predictor: gamma_hat from OLS Z ~ 1 + x + y on the training labels, Z~ = Z - gamma_hat*y;
   (ii) pre-period image: Z_prev = x + gamma*y_prev + nu, y_prev with independent shocks.
Diagnostic: HC t-test of gamma = 0 on the training labels.
Run: python3 g3_a1.py [R]
"""
import numpy as np, json, sys, time
from g3_core import Lattice, ols_fit, build, estimate, wald_se, sigma2_U

RHO, BETA, THETA = 0.4, 1.0, 0.5
TRUE = np.array([RHO, BETA, THETA])


def hc_t(X, yv):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ yv; u = yv - X @ b
    V = XtXi @ (X.T * u ** 2) @ X @ XtXi
    return b, b / np.sqrt(np.diag(V))


def lciv(L, y, xh, x, lab):
    S = build(L, y, xh, xh, x, lab); d, P = estimate(S)
    s2 = sigma2_U(L, y, x, xh, lab, d); V = wald_se(S, d, P, s2)
    return d, np.sqrt(np.diag(V))


def run(gamma, R, rng, m=30, kappa=0.7, nl=90, pi_tr=0.10):
    L = Lattice(m, 0.8, RHO, 4); n = L.n; W = L.W
    tau = np.sqrt(L.s2 * (1 - kappa) / kappa)
    S_inv = np.linalg.inv(np.eye(n) - RHO * W)
    names = ["oracle", "naive", "LCIV", "LCIV_purged", "LCIV_preperiod"]
    est = {k: [] for k in names}; cov = {k: [] for k in names}
    rej = []; q_mc = []; q_th = []
    for r in range(R):
        x = L.draw_x(rng)
        y, eps = L.draw_y(x, BETA, THETA, rng)
        y_prev, _ = L.draw_y(x, BETA, THETA, rng)
        nu = tau * rng.standard_normal(n)
        Z = x + gamma * y + nu
        Zp = x + gamma * y_prev + tau * rng.standard_normal(n)
        perm = rng.permutation(n); ntr = int(pi_tr * n)
        tr, lab = perm[:ntr], perm[ntr:ntr + nl]
        xh = ols_fit([Z], x, tr)
        # diagnostic + purging (training labels only)
        X = np.column_stack([np.ones(ntr), x[tr], y[tr]])
        b, tstat = hc_t(X, Z[tr])
        rej.append(abs(tstat[2]) > 1.96)
        Zt = Z - b[2] * y
        xh_pg = ols_fit([Zt], x, tr)
        xh_pp = ols_fit([Zp], x, tr)
        # moment contamination check: n^-1 H'eps vs formula
        c = np.polyfit(Z[tr], x[tr], 1)[0]
        H = np.column_stack([L.Wp[k] @ xh for k in range(4)])
        q_mc.append(H.T @ eps / n)
        q_th.append([c * gamma * np.trace(L.Wp[k].T @ S_inv.T) / n for k in range(4)])
        # estimators
        Ho = np.column_stack([L.Wp[k] @ x for k in range(4)])
        Q = np.column_stack([W @ y, x, W @ x]); Ai = np.linalg.inv(Ho.T @ Ho)
        do = np.linalg.solve(Q.T @ Ho @ Ai @ Ho.T @ Q, Q.T @ Ho @ Ai @ Ho.T @ y)
        Qn = np.column_stack([W @ y, xh, W @ xh]); Ai = np.linalg.inv(H.T @ H)
        dn = np.linalg.solve(Qn.T @ H @ Ai @ H.T @ Qn, Qn.T @ H @ Ai @ H.T @ y)
        est["oracle"].append(do); est["naive"].append(dn)
        cov["oracle"].append(np.nan); cov["naive"].append(np.nan)
        for k, xx in (("LCIV", xh), ("LCIV_purged", xh_pg), ("LCIV_preperiod", xh_pp)):
            d, se = lciv(L, y, xx, x, lab)
            est[k].append(d); cov[k].append(abs(d[2] - THETA) <= 1.96 * se[2])
    out = dict(gamma=gamma, R=R, diag_reject=float(np.mean(rej)),
               q_mc=np.mean(q_mc, 0).tolist(), q_theory=np.mean(q_th, 0).tolist())
    ati_true = (BETA + THETA) / (1 - RHO)
    for k in names:
        E = np.array(est[k])
        ati = np.where(E[:, 0] < 1, (E[:, 1] + E[:, 2]) / (1 - np.minimum(E[:, 0], 0.999)), np.inf)
        out[k] = dict(medbias=(np.median(E, 0) - TRUE).tolist(),
                      ati_medbias=float(np.median(ati) - ati_true),
                      cov_theta=float(np.nanmean(np.array(cov[k], float))) if k.startswith("LCIV") else None)
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(404); res = []; t0 = time.time()
    for g in (0.0, 0.1, 0.25, 0.5):
        o = run(g, R, rng); res.append(o)
        print(json.dumps(o, default=float), f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open("g3_a1.json", "w"), indent=1)
