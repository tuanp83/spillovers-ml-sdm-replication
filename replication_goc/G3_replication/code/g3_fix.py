"""
G3 — fixes for the two weak points of LC-IV.

Fix 1 (feasible sigma^2). At an audited unit i,
    r_i = y_i - rho (Wy)_i - x_i beta - (W xh)_i theta = eps_i + theta (We)_i ,
so  E mean_L r_i^2 = sigma^2 + theta^2 n^-1 E[e'W'We]  >= sigma^2.
  sigma2_U   = mean_L r_i^2                       (conservative upper bound)
  sigma2_IPW = sigma2_U - theta^2 * S_hat,  S_hat = n^-1 [ sum_{j in L} (W'W)_jj e_j^2 / p1
                                                   + sum_{j!=k in L} (W'W)_jk e_j e_k / p2 ]
  (p1, p2 = single / pair inclusion probabilities of the audit sample).

Fix 2 (heavy tails when rho is weakly identified).
  (a) Subvector Anderson-Rubin sets (chi2(K-2)) for rho, theta and ATI,
      with rho restricted to (-1,1) for ATI; ATI=a0 <=> theta = a0(1-rho) - beta.
  (b) RF-LC: impacts from the truncated reduced form
      y ~ sum_{k=0}^{Kr} pi_k W^k x  (no endogenous Wy, no 1/(1-rho)),
      ATI = sum pi_k, ADI = sum pi_k tr(W^k)/n, with label-corrected cross moments.

Run: python3 g3_fix.py [R]
"""
import numpy as np, json, sys, time
from scipy.linalg import lu_factor, lu_solve
from g3_sim import rook_W
from scipy.stats import f as f_dist

CRIT_AR = 5.991464547107979   # chi2(2) 0.95 (K=4 moments, 2 nuisance params)
Z95 = 1.959963984540054
RHO_GRID = np.linspace(-0.99, 0.99, 397)
A0_GRID = np.linspace(-5.0, 15.0, 801)

def lc_moments(y, Wy, xh, W, Wp, K, lab, e, n):
    H = np.column_stack([Wp[k] @ xh for k in range(K)])
    pih = len(lab) / n
    WtH = W.T @ H
    m0 = (H.T @ e) / pih
    m1 = (WtH.T @ e) / pih
    G = np.column_stack([H.T @ Wy, H.T @ xh + m0, H.T @ (W @ xh) + m1])   # sums
    return H, WtH, G, H.T @ y, pih

def omega(H, WtH, lab, e, beta, theta, sigma2, pih):
    b = beta * H + theta * WtH
    return sigma2 * (H.T @ H) + (1 - pih) / pih ** 2 * (b[lab].T * e[lab] ** 2) @ b[lab]

def sigma2_feasible(y, Wy, x, xh, W, WtW, lab, e, d, n):
    rho, beta, theta = d
    rho = np.clip(rho, -0.99, 0.99)
    r = y[lab] - rho * Wy[lab] - beta * x[lab] - theta * (W @ xh)[lab]
    s2U = np.mean(r ** 2)
    nl = len(lab)
    p1 = nl / n; p2 = nl * (nl - 1) / (n * (n - 1))
    el = e[lab]
    Wsub = WtW[np.ix_(lab, lab)]
    diag = np.sum(np.diag(Wsub) * el ** 2) / p1
    off = (el @ Wsub @ el - np.sum(np.diag(Wsub) * el ** 2)) / p2
    S = (diag + off) / n
    s2I = max(s2U - theta ** 2 * S, 0.1 * s2U)
    return s2U, s2I

def ar_quadratic(hy, Gr, Gn, Oi_chol):
    """AR(v0) for hypothesis on the coefficient of column Gr, nuisances Gn (closed form,
    quadratic in v0): returns coefficients (c0, c1, c2): AR = c0 - 2 c1 v + c2 v^2."""
    L = Oi_chol
    h = L.T @ hy; g = L.T @ Gr; N = L.T @ Gn
    M = np.eye(len(h)) - N @ np.linalg.pinv(N)
    mh = M @ h; mg = M @ g
    return mh @ mh, mh @ mg, mg @ mg

def ar_ati(hy, G, Oi_chol, a_grid, rho_grid):
    """AR_ATI(a0) = min_{rho in grid, beta} || L'(hy - G_r rho - G_t a0(1-rho) - (G_b - G_t)beta) ||^2."""
    L = Oi_chol
    h = L.T @ hy; gr = L.T @ G[:, 0]; gb = L.T @ G[:, 1]; gt = L.T @ G[:, 2]
    nb = gb - gt
    M = np.eye(len(h)) - np.outer(nb, nb) / (nb @ nb)
    u0, u1, u2 = M @ h, M @ gr, M @ gt
    # residual(rho, a) = u0 - rho*u1 - a*(1-rho)*u2
    R = rho_grid[:, None, None]; A = a_grid[None, :, None]
    res = u0[None, None, :] - R * u1[None, None, :] - A * (1 - R) * u2[None, None, :]
    return (res ** 2).sum(-1).min(0)

def rf_lc(y, xh, W, Wp_rf, lab, e, n):
    Kr = len(Wp_rf)
    Rh = np.column_stack([Wp_rf[k] @ xh for k in range(Kr)])
    pih = len(lab) / n
    corr = np.column_stack([(Wp_rf[k].T @ Rh).T @ e for k in range(Kr)]) / pih
    M = Rh.T @ Rh + corr
    pi_hat = np.linalg.solve(M, Rh.T @ y)
    trk = np.array([np.trace(Wp_rf[k]) / n for k in range(Kr)])
    ati = pi_hat.sum(); adi = pi_hat @ trk
    return np.array([adi, ati - adi, ati])

def run(m, psi, kappa, pi_lab, R, rng, rho=0.4, beta=1.0, theta=0.5, K=4, Kr=6, pi_tr=0.10):
    W = rook_W(m); n = m * m; I = np.eye(n)
    Wp = [np.linalg.matrix_power(W, k) for k in range(max(K, Kr))]
    Wp_rf = Wp[:Kr]
    WtW = W.T @ W
    Lx = lu_factor(I - psi * W); Ly = lu_factor(I - rho * W)
    Ainv = np.linalg.inv(I - psi * W); s2 = np.mean(np.sum(Ainv ** 2, 1))
    tau = np.sqrt(s2 * (1 - kappa) / kappa)
    om = np.real(np.linalg.eigvals(W))
    true = np.array([rho, beta, theta])
    ati_true = (beta + theta) / (1 - rho)
    adi_true = np.mean((beta + theta * om) / (1 - rho * om))
    true_imp = np.array([adi_true, ati_true - adi_true, ati_true])
    rec = {k: [] for k in ["cov_known", "cov_U", "cov_IPW", "s2U", "s2I",
                           "ar_rho", "ar_theta", "ar_ati", "ati_len", "ati_bounded", "rho_bounded",
                           "imp_lciv", "imp_rf", "wald_ati_IPW", "ati_ar_hat", "ati_ar_cov_or_unb", "arF_rho", "arF_theta", "arF_ati"]}
    for r in range(R):
        x = lu_solve(Lx, rng.standard_normal(n))
        Z = x + tau * rng.standard_normal(n)
        y = lu_solve(Ly, beta * x + theta * (W @ x) + rng.standard_normal(n))
        perm = rng.permutation(n); ntr, nl = int(pi_tr * n), int(pi_lab * n)
        tr, lab = perm[:ntr], perm[ntr:ntr + nl]
        a, b = np.linalg.lstsq(np.column_stack([np.ones(ntr), Z[tr]]), x[tr], rcond=None)[0]
        xh = a + b * Z
        e = np.zeros(n); e[lab] = x[lab] - xh[lab]
        Wy = W @ y
        H, WtH, G, hy, pih = lc_moments(y, Wy, xh, W, Wp, K, lab, e, n)
        A = H.T @ H; Ai = np.linalg.inv(A)
        Bm = np.linalg.inv(G.T @ Ai @ G)
        d = Bm @ G.T @ Ai @ hy
        # ---- Fix 1: feasible sigma^2 and Wald coverage
        s2U, s2I = sigma2_feasible(y, Wy, x, xh, W, WtW, lab, e, d, n)
        rec["s2U"].append(s2U); rec["s2I"].append(s2I)
        covs = {}
        for tag, s2v in (("known", 1.0), ("U", s2U), ("IPW", s2I)):
            Om = omega(H, WtH, lab, e, d[1], d[2], s2v, pih)
            V = Bm @ G.T @ Ai @ Om @ Ai @ G @ Bm
            se = np.sqrt(np.diag(V))
            covs[tag] = (np.abs(d - true) <= Z95 * se).astype(float)
            if tag == "IPW":
                rr = min(d[0], 0.99)
                grad = np.array([(d[1] + d[2]) / (1 - rr) ** 2, 1 / (1 - rr), 1 / (1 - rr)])
                se_ati = np.sqrt(grad @ V @ grad)
                rec["wald_ati_IPW"].append(float(abs((d[1] + d[2]) / (1 - rr) - ati_true) <= Z95 * se_ati))
        rec["cov_known"].append(covs["known"]); rec["cov_U"].append(covs["U"]); rec["cov_IPW"].append(covs["IPW"])
        # ---- Fix 2a: AR sets (Omega at plug-in beta, theta; sigma2_IPW)
        Om = omega(H, WtH, lab, e, d[1], d[2], s2I, pih)
        Lc = np.linalg.cholesky(np.linalg.inv(Om))
        c0, c1, c2 = ar_quadratic(hy, G[:, 0], G[:, 1:], Lc)
        ar_r = c0 - 2 * c1 * rho + c2 * rho ** 2
        rec["ar_rho"].append(float(ar_r <= CRIT_AR))
        critF = 2 * f_dist.ppf(0.95, 2, len(lab) - K)
        rec["arF_rho"].append(float(ar_r <= critF))
        acc_rho = (c0 - 2 * c1 * RHO_GRID + c2 * RHO_GRID ** 2) <= CRIT_AR
        rec["rho_bounded"].append(float(not (acc_rho[0] or acc_rho[-1])))
        c0, c1, c2 = ar_quadratic(hy, G[:, 2], G[:, :2], Lc)
        rec["ar_theta"].append(float(c0 - 2 * c1 * theta + c2 * theta ** 2 <= CRIT_AR))
        rec["arF_theta"].append(float(c0 - 2 * c1 * theta + c2 * theta ** 2 <= critF))
        ar_true = ar_ati(hy, G, Lc, np.array([ati_true]), np.linspace(-0.99, 0.99, 1981))[0]
        rec["ar_ati"].append(float(ar_true <= CRIT_AR))
        rec["arF_ati"].append(float(ar_true <= critF))
        arv = ar_ati(hy, G, Lc, A0_GRID, RHO_GRID)
        acc = arv <= CRIT_AR
        rec["ati_ar_hat"].append(float(A0_GRID[np.argmin(arv)]))
        rec["ati_bounded"].append(float(not (acc[0] or acc[-1])))
        rec["ati_len"].append(float(acc.sum() * (A0_GRID[1] - A0_GRID[0])))
        # ---- point estimates of impacts: LC-IV vs RF-LC
        rr = d[0]
        adi = np.mean((d[1] + d[2] * om) / (1 - rr * om)) if abs(rr) < 1 else np.nan
        ati = (d[1] + d[2]) / (1 - rr) if rr < 1 else np.nan
        rec["imp_lciv"].append([adi, ati - adi, ati])
        rec["imp_rf"].append(rf_lc(y, xh, W, Wp_rf, lab, e, n))
    out = dict(n=n, psi=psi, kappa=kappa, pi=pi_lab, R=R, true_imp=true_imp.tolist())
    for k in ["cov_known", "cov_U", "cov_IPW"]:
        out[k] = np.mean(rec[k], 0).tolist()
    for k in ["ar_rho", "ar_theta", "ar_ati", "arF_rho", "arF_theta", "arF_ati", "ati_bounded", "rho_bounded", "wald_ati_IPW"]:
        out[k] = float(np.mean(rec[k]))
    out["ati_len_median_bounded"] = float(np.median(np.array(rec["ati_len"])[np.array(rec["ati_bounded"]) == 1])) \
        if np.sum(rec["ati_bounded"]) > 0 else None
    # tails conditional on identification flag (AR set for rho bounded inside (-1,1))
    flag = np.array(rec["rho_bounded"]) == 1
    P = np.array(rec["imp_lciv"], float); ae = np.abs(P[:, 2] - true_imp[2])
    ae_ar = np.abs(np.array(rec["ati_ar_hat"]) - true_imp[2])
    def tail(v):
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), med=float(np.median(v)) if len(v) else None,
                    q90=float(np.quantile(v, .9)) if len(v) else None,
                    max=float(v.max()) if len(v) else None, rmse=float(np.sqrt(np.mean(v**2))) if len(v) else None)
    out["lciv_ati_ae_strong"] = tail(ae[flag]); out["lciv_ati_ae_weak"] = tail(ae[~flag])
    out["lciv_nonfinite_weak"] = float(np.mean(~np.isfinite(ae[~flag]))) if (~flag).sum() else None
    out["lciv_nonfinite_strong"] = float(np.mean(~np.isfinite(ae[flag]))) if flag.sum() else None
    out["arhat_ati_ae_all"] = tail(ae_ar); out["arhat_ati_ae_strong"] = tail(ae_ar[flag])
    wald = np.array(rec["wald_ati_IPW"]); arc = np.array(rec["ar_ati"])
    out["wald_ati_cov_given_strong"] = float(wald[flag].mean()) if flag.sum() else None
    out["ar_ati_cov_given_weak"] = float(arc[~flag].mean()) if (~flag).sum() else None
    out["hybrid_ati_cov"] = float(np.where(flag, wald, arc).mean())
    out["s2U_mean"] = float(np.mean(rec["s2U"])); out["s2I_mean"] = float(np.mean(rec["s2I"]))
    for k in ["imp_lciv", "imp_rf"]:
        P = np.array(rec[k], float)
        err = P - true_imp
        fin = np.all(np.isfinite(err), 1)
        ae = np.abs(err[:, 2])
        out[k] = dict(medbias=np.nanmedian(err, 0).tolist(),
                      meanbias=np.nanmean(err, 0).tolist(),
                      rmse=np.sqrt(np.nanmean(err ** 2, 0)).tolist(),
                      ati_mae_med=float(np.nanmedian(ae)),
                      ati_ae_q90=float(np.nanquantile(ae, 0.9)),
                      ati_ae_max=float(np.nanmax(ae)),
                      frac_ati_err_gt1=float(np.nanmean(ae > 1.0)),
                      frac_nonfinite=float(1 - fin.mean()))
    return out

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(2026)
    designs = [(30, 0.8, 0.5, 0.10), (30, 0.8, 0.7, 0.10), (30, 0.8, 0.9, 0.10),
               (30, 0.8, 0.7, 0.05), (50, 0.8, 0.7, 0.10), (50, 0.8, 0.5, 0.10)]
    res = []; t0 = time.time()
    for (m, psi, kappa, pl) in designs:
        o = run(m, psi, kappa, pl, R, rng); res.append(o)
        print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in o.items()
                          if k not in ("imp_lciv", "imp_rf")}), flush=True)
        for k in ("imp_lciv", "imp_rf"):
            print("  ", k, {kk: np.round(vv, 3).tolist() for kk, vv in o[k].items()}, flush=True)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open("g3_fix.json", "w"), indent=1)
