"""(1) Hajek/SRS centred audit variance vs Horvitz-Thompson (uncentred) form;
(2) McCloskey-type size-corrected Bonferroni (SC-Bonf) via a plug-in limit experiment.

Limit experiment (whitened by Omega^-1/2, sum scale): z ~ N(0, I_K),
  g_rho(s) = J_s a_hat + s u   (u: unit direction of M_{J_s} g_rho; s = unknown strength of rho)
  AR1(rho0)        = || M_{J_s} (z + (rho - rho0) g_rho(s)) ||^2
  AR2_theta(rho0)  = || M_{g_beta} (z + (rho - rho0) g_rho(s)) ||^2
  AR2_ATI(rho0)    = || M_{g_beta - g_theta} (z + (rho - rho0)(g_rho(s) - a0 g_theta)) ||^2
  T = min_{rho0 in grid: AR1 <= c1} AR2 ;  c2* = sup_{s, rho} q_{1-alpha}(T).
Run: python3 g3_srs_mcc.py [R]
"""
import numpy as np, json, sys, time
from scipy.stats import chi2
from g3_core import (Lattice, ols_fit, build, estimate, omega, wald_se, sigma2_U,
                     ar_rho, ar_theta_given_rho, ar_ati_given_rho)

RHO, BETA, THETA = 0.4, 1.0, 0.5
TRUE = np.array([RHO, BETA, THETA]); ATI = (BETA + THETA) / (1 - RHO)
K = 4
RG = np.linspace(-0.99, 0.99, 199)
TG = np.linspace(-2.0, 3.0, 101)
S_GRID = np.array([0, 1, 2, 3, 5, 8, 12, 20, 35, 60.0])
RHO_GRID_EXP = np.array([-0.6, -0.2, 0.2, 0.4, 0.6, 0.8])
C1 = chi2.ppf(0.99, K - 2)          # alpha1 = 0.01
C2_BONF = chi2.ppf(0.96, K - 1)     # alpha2 = 0.04


def draw(L, kappa, nl, rng, scheme, pi_tr=0.10):
    n = L.n
    x = L.draw_x(rng)
    tau = np.sqrt(L.s2 * (1 - kappa) / kappa)
    Z = x + tau * rng.standard_normal(n)
    y, _ = L.draw_y(x, BETA, THETA, rng)
    perm = rng.permutation(n); ntr = int(pi_tr * n)
    tr, rest = perm[:ntr], perm[ntr:]
    if scheme == "srs":
        lab = rest[:nl]
    else:   # Bernoulli audit with expected size nl
        lab = rest[rng.random(len(rest)) < nl / len(rest)]
    xh = ols_fit([Z], x, tr)
    return x, y, xh, lab


def ati_grad(d):
    r = min(d[0], 0.999)
    return np.array([(d[1] + d[2]) / (1 - r) ** 2, 1 / (1 - r), 1 / (1 - r)])


def sc_crit(S, d, s2, rng, a0=None, M=400, alpha=0.05):
    Om = omega(S, d[1], d[2], s2)
    Lw = np.linalg.cholesky(np.linalg.inv(Om))
    g = Lw.T @ S["G"]
    Js = g[:, 1:3]
    Ms = np.eye(K) - Js @ np.linalg.pinv(Js)
    a_hat = np.linalg.pinv(Js) @ g[:, 0]
    r = Ms @ g[:, 0]; u = r / np.linalg.norm(r)
    gb, gt = g[:, 1], g[:, 2]
    if a0 is None:
        v = gb
    else:
        v = gb - gt
    Mb = np.eye(K) - np.outer(v, v) / (v @ v)
    z = rng.standard_normal((M, K))
    Msz = z @ Ms.T; Mbz = z @ Mb.T
    best = 0.0
    for s in S_GRID:
        grho = Js @ a_hat + s * u
        geff = grho if a0 is None else grho - a0 * gt
        Mbg = Mb @ geff
        su = s * u                           # M_s g_rho = s u
        for rt in RHO_GRID_EXP:
            D = rt - RG                      # (G,)
            ar1 = ((Msz[:, None, :] + D[None, :, None] * su[None, None, :]) ** 2).sum(-1)
            ar2 = ((Mbz[:, None, :] + D[None, :, None] * Mbg[None, None, :]) ** 2).sum(-1)
            T = np.where(ar1 <= C1, ar2, np.inf).min(1)
            q = np.quantile(T, 1 - alpha)
            best = max(best, q)
    return best


def run(m, kappa, nl, scheme, R, rng):
    L = Lattice(m, 0.8, RHO, K)
    rec = {k: [] for k in ["w_bern", "w_srs", "len_bern", "len_srs", "nl",
                           "th_bonf", "th_sc", "ati_bonf", "ati_sc", "thlen_bonf", "thlen_sc",
                           "th_bnd_bonf", "th_bnd_sc", "c2sc_theta", "c2sc_ati", "th_hat", "se_th_srs"]}
    for r in range(R):
        x, y, xh, lab = draw(L, kappa, nl, rng, scheme)
        S = build(L, y, xh, xh, x, lab)
        d, P = estimate(S)
        s2 = sigma2_U(L, y, x, xh, lab, d)
        rec["nl"].append(len(lab))
        ev = np.append(d, (d[1] + d[2]) / (1 - d[0]) if d[0] < 1 else np.inf)
        tv = np.append(TRUE, ATI)
        for mode in ("bern", "srs"):
            S["mode"] = mode
            V = wald_se(S, d, P, s2); gr = ati_grad(d)
            se = np.append(np.sqrt(np.diag(V)), np.sqrt(gr @ V @ gr))
            rec["w_" + mode].append(np.abs(ev - tv) <= 1.96 * se)
            rec["len_" + mode].append(2 * 1.96 * se)
            if mode == "srs":
                rec["th_hat"].append(d[2]); rec["se_th_srs"].append(se[2])
        S["mode"] = "srs"
        # Bonferroni with H0-Omega (SRS form)
        ar_r = ar_rho(S, RG, s2, d[1], d[2])
        inCS = ar_r <= C1
        th_true = ar_theta_given_rho(S, RG, np.full(len(RG), THETA), s2, d[1])
        at_true = ar_ati_given_rho(S, RG, np.full(len(RG), ATI), s2, d[1])
        c2t = sc_crit(S, d, s2, rng); c2a = sc_crit(S, d, s2, rng, a0=ATI)
        rec["c2sc_theta"].append(c2t); rec["c2sc_ati"].append(c2a)
        for tag, c2 in (("bonf", C2_BONF), ("sc", c2t)):
            rec["th_" + tag].append(bool(np.any(th_true[inCS] <= c2)) if inCS.any() else False)
        for tag, c2 in (("bonf", C2_BONF), ("sc", c2a)):
            rec["ati_" + tag].append(bool(np.any(at_true[inCS] <= c2)) if inCS.any() else False)
        RR, TT = np.meshgrid(RG[inCS], TG, indexing="ij")
        if inCS.any():
            arTH = ar_theta_given_rho(S, RR.ravel(), TT.ravel(), s2, d[1]).reshape(RR.shape)
            for tag, c2 in (("bonf", C2_BONF), ("sc", c2t)):
                acc = (arTH <= c2).any(0)
                rec["thlen_" + tag].append(acc.sum() * (TG[1] - TG[0]))
                rec["th_bnd_" + tag].append(not (acc[0] or acc[-1]))
        else:
            for tag in ("bonf", "sc"):
                rec["thlen_" + tag].append(0.0); rec["th_bnd_" + tag].append(True)
    out = dict(n=L.n, kappa=kappa, nl=nl, scheme=scheme, R=R, nl_mean=float(np.mean(rec["nl"])))
    for k in ("w_bern", "w_srs"):
        out[k] = np.mean(np.array(rec[k], float), 0).tolist()
    for k in ("len_bern", "len_srs"):
        A = np.array(rec[k], float)
        out[k] = np.nanmedian(np.where(np.isfinite(A), A, np.nan), 0).tolist()
    for k in ("th_bonf", "th_sc", "ati_bonf", "ati_sc", "th_bnd_bonf", "th_bnd_sc"):
        out[k] = float(np.mean(rec[k]))
    for k in ("thlen_bonf", "thlen_sc", "c2sc_theta", "c2sc_ati"):
        out[k] = float(np.median(rec[k]))
    out["c2_bonf"] = float(C2_BONF)
    th = np.array(rec["th_hat"])
    q75, q25 = np.percentile(th, [75, 25])
    out["theta_robust_sd"] = float((q75 - q25) / 1.349)
    out["theta_mc_sd"] = float(th.std())
    out["theta_median_se"] = float(np.median(rec["se_th_srs"]))
    from scipy.stats import kurtosis
    out["theta_kurtosis"] = float(kurtosis(th))
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(777); res = []; t0 = time.time()
    for (k, nl, sch) in [(0.5, 90, "srs"), (0.7, 45, "srs"), (0.7, 90, "srs"), (0.9, 90, "srs"), (0.7, 90, "bern")]:
        o = run(30, k, nl, sch, R, rng); res.append(o)
        print(json.dumps(o, default=float), f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open("g3_srs_mcc.json", "w"), indent=1)
