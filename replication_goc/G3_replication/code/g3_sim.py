"""
G3 — Machine-learned regressors in Spatial Durbin Models.
Validates (i) exact pseudo-true bias of naive 2SLS (Theorem 1),
(ii) closed-form spurious spillover in SLX (Proposition 2),
and compares oracle / naive / spatially-augmented prediction (SAP) /
label-corrected IV (LC-IV) by Monte Carlo.

Run: python3 g3_sim.py  (numpy, scipy)
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.stats import norm
import json, sys, time

# ---------------------------------------------------------------- utilities
def rook_W(m):
    n = m * m
    W = np.zeros((n, n))
    for r in range(m):
        for c in range(m):
            i = r * m + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < m:
                    W[i, rr * m + cc] = 1.0
    return W / W.sum(1, keepdims=True)

def sar_cov(W, psi):
    n = W.shape[0]
    A = np.linalg.inv(np.eye(n) - psi * W)
    return A @ A.T

_EIG = {}
def impacts(rho, beta, theta, W):
    key = id(W)
    if key not in _EIG:
        _EIG[key] = np.real(np.linalg.eigvals(W))
    om = _EIG[key]
    adi = np.mean((beta + theta * om) / (1 - rho * om))
    ati = (beta + theta) / (1 - rho)
    return np.array([adi, ati - adi, ati])

def tsls(y, Q, H):
    A = H.T @ H
    G = H.T @ Q
    g = H.T @ y
    Ai = np.linalg.inv(A)
    return np.linalg.solve(G.T @ Ai @ G, G.T @ Ai @ g)

def gmm_from_moments(A, G, g):
    Ai = np.linalg.inv(A)
    return np.linalg.solve(G.T @ Ai @ G, G.T @ Ai @ g)

# ------------------------------------------------- Theorem 1: exact plim
def exact_naive_plim(W, Sx, Snu, c, rho, beta, theta, K=4):
    """Pseudo-true value of naive 2SLS, regressors [Wy, xh, Wxh],
    instruments H=[W^k xh]_{k<K}, xh = c (x + nu), all mean zero Gaussian."""
    n = W.shape[0]
    I = np.eye(n)
    Sxh = c ** 2 * (Sx + Snu)          # Var(xh)
    Cxxh = c * Sx                       # E[x xh']
    S = np.linalg.inv(I - rho * W)
    B = beta * I + theta * W
    Wp = [np.linalg.matrix_power(W, k) for k in range(K)]
    A = np.array([[np.trace(Wp[j].T @ Wp[k] @ Sxh) for k in range(K)]
                  for j in range(K)]) / n
    G = np.zeros((K, 3)); g = np.zeros(K)
    for j in range(K):
        G[j, 0] = np.trace(Wp[j].T @ W @ S @ B @ Cxxh)
        G[j, 1] = np.trace(Wp[j].T @ Sxh)
        G[j, 2] = np.trace(Wp[j].T @ W @ Sxh)
        g[j] = np.trace(Wp[j].T @ S @ B @ Cxxh)
    G /= n; g /= n
    return gmm_from_moments(A, G, g)

# --------------------------------------------- Proposition 2: closed form
def prop2_slx(W, Sx, kappa):
    n = W.shape[0]
    s2 = np.trace(Sx) / n
    r1 = np.trace(W @ Sx) / (n * s2)
    r2 = np.trace(W.T @ W @ Sx) / (n * s2)
    w2 = np.trace(W.T @ W) / n
    den = kappa * r2 + w2 * (1 - kappa) - kappa ** 2 * r1 ** 2
    beta_hat = (kappa * (r2 - r1 ** 2) + w2 * (1 - kappa)) / den
    theta_hat = r1 * (1 - kappa) / den
    return beta_hat, theta_hat, dict(r1=r1, r2=r2, w2=w2)

# ------------------------------------------------------------ Moran test
_TR = {}
def moran_test(res, Xreg, W):
    """Cliff-Ord normal-theory Moran z for OLS residuals, one regressor."""
    n, k = Xreg.shape
    key = id(W)
    if key not in _TR:
        _TR[key] = (np.trace(W @ W.T), np.trace(W @ W))
    tWWt, tWW = _TR[key]
    u = Xreg[:, 0] / np.linalg.norm(Xreg[:, 0])
    Wu = W @ u; Wtu = W.T @ u
    uWu = u @ Wu
    trMW = -uWu
    trMWMWt = tWWt - Wu @ Wu - Wtu @ Wtu + uWu ** 2
    trMWMW = tWW - 2 * (Wtu @ Wu) + uWu ** 2
    I_obs = (res @ W @ res) / (res @ res)
    EI = trMW / (n - k)
    VI = (trMWMWt + trMWMW + trMW ** 2) / ((n - k) * (n - k + 2)) - EI ** 2
    return (I_obs - EI) / np.sqrt(VI)

# ------------------------------------------------------------- one design
def run_design(W, Wp, psi, kappa, phi, rho, beta, theta, pi_lab, pi_train,
               R, rng, K=4):
    n = W.shape[0]
    I = np.eye(n)
    Lx = lu_factor(I - psi * W)
    Ly = lu_factor(I - rho * W)
    Sx = sar_cov(W, psi)
    s2 = np.trace(Sx) / n
    tau2 = s2 * (1 - kappa) / kappa
    if phi > 0:
        Lnu = lu_factor(I - phi * W)
        Snu_shape = sar_cov(W, phi)
        nu_scale = np.sqrt(tau2 / (np.trace(Snu_shape) / n))
    else:
        nu_scale = np.sqrt(tau2)
    true = np.array([rho, beta, theta])
    true_imp = impacts(rho, beta, theta, W)
    names = ["oracle", "naive", "SAP", "LCIV", "LCIV_SAP"]
    est = {k: [] for k in names}
    imp = {k: [] for k in names}
    for r in range(R):
        x = lu_solve(Lx, rng.standard_normal(n))
        if phi > 0:
            nu = nu_scale * lu_solve(Lnu, rng.standard_normal(n))
        else:
            nu = nu_scale * rng.standard_normal(n)
        Z = x + nu                                   # "image signal" of own tile
        eps = rng.standard_normal(n)
        y = lu_solve(Ly, beta * x + theta * (W @ x) + eps)
        perm = rng.permutation(n)
        ntr = int(pi_train * n); nlab = int(pi_lab * n)
        tr, lab = perm[:ntr], perm[ntr:ntr + nlab]
        # predictor trained on separate training labels (own tile only)
        Xtr = np.column_stack([np.ones(ntr), Z[tr]])
        a, b = np.linalg.lstsq(Xtr, x[tr], rcond=None)[0]
        xh = a + b * Z
        Wy = W @ y
        # oracle
        Ho = np.column_stack([Wp[k] @ x for k in range(K)])
        d_o = tsls(y, np.column_stack([Wy, x, W @ x]), Ho)
        # naive
        H = np.column_stack([Wp[k] @ xh for k in range(K)])
        d_n = tsls(y, np.column_stack([Wy, xh, W @ xh]), H)
        # SAP: x on [1, xh, W xh, W^2 xh] fitted on labeled set
        F = np.column_stack([np.ones(n), xh, W @ xh, Wp[2] @ xh])
        gam = np.linalg.lstsq(F[tr], x[tr], rcond=None)[0]
        xt = F @ gam
        Ht = np.column_stack([Wp[k] @ xt for k in range(K)])
        d_s = tsls(y, np.column_stack([Wy, xt, W @ xt]), Ht)
        # LC-IV: instruments from xh, cross-moments corrected with labels
        e = np.zeros(n); e[lab] = x[lab] - xh[lab]
        pih = nlab / n
        m0 = (H.T @ e) / pih            # estimates E[H'e]
        m1 = ((W.T @ H).T @ e) / pih     # estimates E[H'We]
        A = H.T @ H
        G = np.column_stack([H.T @ Wy, H.T @ xh + m0, H.T @ (W @ xh) + m1])
        g = H.T @ y
        d_l = gmm_from_moments(A, G, g)
        # LC-IV with SAP instruments/regressors (smaller correction -> less noise)
        et = np.zeros(n); et[lab] = x[lab] - xt[lab]
        m0t = (Ht.T @ et) / pih
        m1t = ((W.T @ Ht).T @ et) / pih
        Gt = np.column_stack([Ht.T @ Wy, Ht.T @ xt + m0t, Ht.T @ (W @ xt) + m1t])
        d_ls = gmm_from_moments(Ht.T @ Ht, Gt, Ht.T @ y)
        for k, d in zip(names, [d_o, d_n, d_s, d_l, d_ls]):
            est[k].append(d)
            imp[k].append(impacts(*d, W))
    out = {}
    for k in names:
        E = np.array(est[k]); P = np.array(imp[k])
        out[k] = dict(
            bias=(E.mean(0) - true).tolist(), sd=E.std(0).tolist(),
            medbias=(np.median(E, 0) - true).tolist(),
            rmse=np.sqrt(((E - true) ** 2).mean(0)).tolist(),
            imp_bias=(P.mean(0) - true_imp).tolist(),
            imp_medbias=(np.median(P, 0) - true_imp).tolist(),
            imp_rmse=np.sqrt(((P - true_imp) ** 2).mean(0)).tolist())
    out["true_imp"] = true_imp.tolist()
    return out

# ------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(20261116)
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    m = 30
    W = rook_W(m); n = m * m
    Wp = [np.linalg.matrix_power(W, k) for k in range(4)]
    results = {"n": n, "R": R}
    t0 = time.time()

    # --- Table 1: theory validation, null of no spillover (rho=0, theta=0)
    tab1 = []
    for psi in (0.5, 0.8):
        Sx = sar_cov(W, psi)
        for kappa in (0.5, 0.7, 0.9):
            bh, th, aux = prop2_slx(W, Sx, kappa)
            # MC for SLX OLS with xh = kappa * Z (oracle-calibrated, own tile)
            s2 = np.trace(Sx) / n; tau = np.sqrt(s2 * (1 - kappa) / kappa)
            Lx = lu_factor(np.eye(n) - psi * W)
            num = np.zeros(2); denm = np.zeros((2, 2)); th_list = []
            z_or, z_nv = [], []
            for r in range(R):
                x = lu_solve(Lx, rng.standard_normal(n))
                xh = kappa * (x + tau * rng.standard_normal(n))
                y = x + rng.standard_normal(n)
                X = np.column_stack([xh, W @ xh])
                th_list.append(np.linalg.lstsq(X, y, rcond=None)[0])
                # Moran test on residuals of y ~ xh (non-spatial OLS)
                Xo = x[:, None]; Xn = xh[:, None]
                ro = y - Xo @ np.linalg.lstsq(Xo, y, rcond=None)[0]
                rn = y - Xn @ np.linalg.lstsq(Xn, y, rcond=None)[0]
                z_or.append(moran_test(ro, Xo, W)); z_nv.append(moran_test(rn, Xn, W))
            th_mc = np.array(th_list).mean(0)
            # Theorem 1 exact for SDM-2SLS under same null
            d_star = exact_naive_plim(W, Sx, (s2 * (1 - kappa) / kappa) * np.eye(n),
                                      kappa, 0.0, 1.0, 0.0)
            tab1.append(dict(psi=psi, kappa=kappa, r1=aux["r1"], r2=aux["r2"], w2=aux["w2"],
                             slx_beta_prop2=bh, slx_theta_prop2=th,
                             slx_beta_mc=th_mc[0], slx_theta_mc=th_mc[1],
                             sdm_plim_rho=d_star[0], sdm_plim_beta=d_star[1],
                             sdm_plim_theta=d_star[2],
                             moran_rej_oracle=float(np.mean(np.array(z_or) > norm.ppf(0.95))),
                             moran_rej_naive=float(np.mean(np.array(z_nv) > norm.ppf(0.95)))))
            print("T1", psi, kappa, round(th, 4), round(th_mc[1], 4),
                  np.round(d_star, 4), f"{time.time()-t0:.0f}s", flush=True)
    results["table1"] = tab1

    # --- Table 2: SDM with true spillovers, compare estimators
    tab2 = []
    designs = []
    for kappa in (0.5, 0.7, 0.9):
        for phi in (0.0, 0.5):
            designs.append(dict(psi=0.8, kappa=kappa, phi=phi, pi_lab=0.10))
    designs.append(dict(psi=0.8, kappa=0.7, phi=0.0, pi_lab=0.05))
    designs.append(dict(psi=0.5, kappa=0.7, phi=0.0, pi_lab=0.10))
    for d in designs:
        out = run_design(W, Wp, d["psi"], d["kappa"], d["phi"], 0.4, 1.0, 0.5,
                         d["pi_lab"], 0.10, R, rng)
        # exact naive plim (fixed calibration c=kappa approximates trained predictor)
        Sx = sar_cov(W, d["psi"]); s2 = np.trace(Sx) / n
        tau2 = s2 * (1 - d["kappa"]) / d["kappa"]
        if d["phi"] > 0:
            Sn = sar_cov(W, d["phi"]); Sn *= tau2 / (np.trace(Sn) / n)
        else:
            Sn = tau2 * np.eye(n)
        c = np.trace(Sx) / np.trace(Sx + Sn)
        dstar = exact_naive_plim(W, Sx, Sn, c, 0.4, 1.0, 0.5)
        out["exact_naive_bias"] = (dstar - np.array([0.4, 1.0, 0.5])).tolist()
        out.update(d); tab2.append(out)
        print("T2", d, f"{time.time()-t0:.0f}s", flush=True)
        for k in ["oracle", "naive", "SAP", "LCIV", "LCIV_SAP"]:
            print("   ", k.ljust(9), "bias", np.round(out[k]["bias"], 3),
                  "rmse", np.round(out[k]["rmse"], 3),
                  "ATI medbias", round(out[k]["imp_medbias"][2], 3), flush=True)
    results["table2"] = tab2

    with open("g3_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
