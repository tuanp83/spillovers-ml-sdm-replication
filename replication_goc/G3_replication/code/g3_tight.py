"""Tightening of Theorem 4'(c) / Section 10.
(1) Universal limit experiment for the Bonferroni statistic (Lemma 10.1'):
      T_lam = min_{D: q + (t + D*lam)^2 <= c1} (z2 + D)^2 + q + (t + D*lam)^2,
    z2, t ~ N(0,1), q ~ chi2_{K-3}, all independent; lam in [0, inf] is the only nuisance.
    c*(K, a1, a) = sup_lam q_{1-a}(T_lam) is a universal size-corrected critical value.
(2) Monte Carlo coverage of SC-Bonferroni with c* and sigma^2 evaluated under H0.
Run: python3 g3_tight.py universal | python3 g3_tight.py mc [R]
"""
import numpy as np, json, sys, time
from scipy.stats import chi2, f as fdist

LAMS = np.concatenate([[0.0], np.logspace(-2, 3, 61), [np.inf]])


def T_draws(lam, K, c1, M, rng):
    z2 = rng.standard_normal(M); t = rng.standard_normal(M)
    q = rng.chisquare(K - 3, M) if K > 3 else np.zeros(M)
    T = np.full(M, np.inf)
    if lam == 0:
        ok = q + t ** 2 <= c1
        T[ok] = (q + t ** 2)[ok]
    elif np.isinf(lam):
        ok = q <= c1
        T[ok] = (z2 ** 2 + q)[ok]
    else:
        ok = q <= c1
        r = np.sqrt(np.maximum(c1 - q, 0))
        lo, hi = (-t - r) / lam, (-t + r) / lam
        Dst = -(z2 + lam * t) / (1 + lam ** 2)
        D = np.clip(Dst, lo, hi)
        f = (z2 + D) ** 2 + q + (t + D * lam) ** 2
        T[ok] = f[ok]
    return T


def universal(M=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for K in (3, 4, 5, 6):
        for a in (0.05, 0.10):
            for a1 in (0.005, 0.01, 0.02, 0.025, 0.03):
                c1 = chi2.ppf(1 - a1, K - 2)
                qs = []; rej_bonf = []
                c2_bonf = chi2.ppf(1 - (a - a1), K - 1)
                for lam in LAMS:
                    T = T_draws(lam, K, c1, M, rng)
                    qs.append(float(np.quantile(T, 1 - a)))
                    rej_bonf.append(float(np.mean(T > c2_bonf)))
                qs = np.array(qs)
                rec = dict(K=K, alpha=a, alpha1=a1, c1=float(c1), c2_bonf=float(c2_bonf),
                           c_star=float(qs.max()), lam_argmax=float(LAMS[int(np.argmax(qs))]),
                           c_inf=float(qs.min()), q_lam0=float(qs[0]), q_laminf=float(qs[-1]),
                           size_bonf_sup=float(max(rej_bonf)), size_bonf_lam0=float(rej_bonf[0]),
                           size_bonf_laminf=float(rej_bonf[-1]),
                           chi2_Km2=float(chi2.ppf(1 - a, K - 2)))
                out.append(rec)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    # profile over lambda for the headline case
    rng = np.random.default_rng(seed + 1)
    prof = []
    c1 = chi2.ppf(0.99, 2)
    for lam in LAMS:
        T = T_draws(lam, 4, c1, M, rng)
        prof.append(dict(lam=float(lam), q95=float(np.quantile(T, 0.95)), p_inf=float(np.mean(np.isinf(T)))))
    json.dump(dict(table=out, profile_K4_a1_01=prof), open("g3_universal.json", "w"), indent=1)


# ------------------------------------------------------------------ Monte Carlo with H0-sigma^2
def mc(R, small=False):
    from g3_core import (Lattice, ols_fit, build, estimate, omega, wald_se, sigma2_U, _gls_batch,
                         ar_rho, ar_theta_given_rho)
    from g3_srs_mcc import draw, sc_crit
    RHO, BETA, THETA = 0.4, 1.0, 0.5
    K = 4
    RG = np.linspace(-0.99, 0.99, 199); TG = np.linspace(-2.0, 3.0, 101)
    cstar = chi2.ppf(0.95, K - 2)            # Theorem 4'': exact size-alpha critical value
    C1 = chi2.ppf(0.99, K - 2); C1b = chi2.ppf(0.95, K - 2); cstar_b = cstar
    C2B = chi2.ppf(0.96, K - 1)

    def s2_h0(S, rho0, beta, theta):
        r = S["yl"][None, :] - rho0[:, None] * S["Wyl"][None, :] - beta[:, None] * S["xl"][None, :] \
            - theta[:, None] * S["Wphil"][None, :]
        return np.mean(r ** 2, 1)

    def ar_rho_h0(S, rhos, b0, t0, iters=3):
        G, hy = S["G"], S["hy"]
        Uu = hy[None, :] - rhos[:, None] * G[None, :, 0]; V = G[:, 1:]
        beta = np.full(len(rhos), b0); theta = np.full(len(rhos), t0)
        for _ in range(iters):
            s2 = s2_h0(S, rhos, beta, theta)
            Oi = np.linalg.inv(omega(S, beta, theta, s2, DF_[0]))
            coef, g, _ = _gls_batch(Uu, V, Oi); beta, theta = coef[:, 0], coef[:, 1]
        s2 = s2_h0(S, rhos, beta, theta); Oi = np.linalg.inv(omega(S, beta, theta, s2, DF_[0]))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)

    def ar_theta_h0(S, rhos, thetas, b0, iters=3):
        G, hy = S["G"], S["hy"]
        Uu = hy[None, :] - rhos[:, None] * G[None, :, 0] - thetas[:, None] * G[None, :, 2]; V = G[:, 1:2]
        beta = np.full(len(rhos), b0)
        for _ in range(iters):
            s2 = s2_h0(S, rhos, beta, thetas)
            Oi = np.linalg.inv(omega(S, beta, thetas, s2, DF_[0]))
            coef, g, _ = _gls_batch(Uu, V, Oi); beta = coef[:, 0]
        s2 = s2_h0(S, rhos, beta, thetas); Oi = np.linalg.inv(omega(S, beta, thetas, s2, DF_[0]))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)

    DF_ = [1.0]
    res = []; rng = np.random.default_rng(4242); t0 = time.time()
    for (kappa, nl) in [(0.5, 90), (0.7, 45), (0.7, 90), (0.9, 90)]:
        L = Lattice(30, 0.8, RHO, K)
        rec = {k: [] for k in ["cov_plug_bonf", "cov_plug_sc", "cov_h0_bonf", "cov_h0_univ", "cov_h0_univ_best",
                               "len_plug_sc", "len_h0_bonf", "len_h0_univ", "len_h0_univ_best",
                               "bnd_h0_bonf", "bnd_h0_univ", "bnd_h0_univ_best", "rho_h0", "rho_plug"]}
        for r in range(R):
            x, y, xh, lab = draw(L, kappa, nl, rng, "srs")
            S = build(L, y, xh, xh, x, lab); S["mode"] = "srs"
            DF = nl / (nl - 3) if small else 1.0
            cK2 = (K - 2) * fdist.ppf(0.95, K - 2, nl - K) if small else chi2.ppf(0.95, K - 2)
            DF_[0] = DF
            d, P = estimate(S); s2 = sigma2_U(L, y, x, xh, lab, d)
            # plug-in (previous implementation)
            arp = ar_rho(S, RG, s2, d[1], d[2])
            rec["rho_plug"].append(ar_rho(S, np.array([RHO]), s2, d[1], d[2])[0] <= chi2.ppf(0.95, 2))
            inP = arp <= C1
            c2p = sc_crit(S, d, s2, rng)
            thp = ar_theta_given_rho(S, RG, np.full(len(RG), THETA), s2, d[1])
            rec["cov_plug_bonf"].append(bool((thp[inP] <= C2B).any()) if inP.any() else False)
            rec["cov_plug_sc"].append(bool((thp[inP] <= c2p).any()) if inP.any() else False)
            # sigma^2 under H0
            arh = ar_rho_h0(S, RG, d[1], d[2])
            rec["rho_h0"].append(ar_rho_h0(S, np.array([RHO]), d[1], d[2])[0] <= cK2)
            inH = arh <= C1; inHb = arh <= C1b
            thh = ar_theta_h0(S, RG, np.full(len(RG), THETA), d[1])
            rec["cov_h0_bonf"].append(bool((thh[inH] <= C2B).any()) if inH.any() else False)
            rec["cov_h0_univ"].append(bool((thh[inH] <= cK2).any()) if inH.any() else False)
            rec["cov_h0_univ_best"].append(bool(thh.min() <= cK2))          # direct subvector AR (no first step)
            # lengths on the theta grid
            RR, TT = np.meshgrid(RG, TG, indexing="ij")
            AH = ar_theta_h0(S, RR.ravel(), TT.ravel(), d[1]).reshape(RR.shape)
            AP = ar_theta_given_rho(S, RR.ravel(), TT.ravel(), s2, d[1]).reshape(RR.shape)
            dT = TG[1] - TG[0]
            for tag, A_, inC, c2 in (("plug_sc", AP, inP, c2p), ("h0_bonf", AH, inH, C2B), ("h0_univ", AH, inH, cK2),
                                     ("h0_univ_best", AH, inHb, cK2)):
                acc = (A_[inC] <= c2).any(0) if inC.any() else np.zeros(len(TG), bool)
                rec["len_" + tag].append(acc.sum() * dT)
                if tag == "h0_univ":
                    acc_direct = AH.min(0) <= cK2
                    rec.setdefault("identity_bonf_eq_direct", []).append(bool(np.array_equal(acc, acc_direct)))
                if "bnd_" + tag in rec:
                    rec["bnd_" + tag].append(bool(acc.any() and not (acc[0] or acc[-1])))
        o = dict(kappa=kappa, nl=nl, R=R, c_star=float(cstar))
        for k, v in rec.items():
            o[k] = float(np.median(v)) if k.startswith("len_") else float(np.mean(v))
        res.append(o)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}), f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open("g3_tight_mc%s.json" % ("_small" if small else ""), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "universal":
        universal()
    else:
        mc(int(sys.argv[2]) if len(sys.argv) > 2 else 300, small=(len(sys.argv) > 3 and sys.argv[3] == "small"))
