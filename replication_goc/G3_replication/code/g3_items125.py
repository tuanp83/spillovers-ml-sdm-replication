"""Items 1, 2, 5 of the G3 follow-up.
 item1: size-guaranteed AR inference: step-1 subvector AR for rho (nuisance beta,theta strongly
        identified), Bonferroni step-2 for theta and ATI given rho (nuisance beta only),
        Omega evaluated under H0 (iterated), projection (chi2_K) fallback.
 item2: small audit samples: Wald, HC1 + t, audit bootstrap-t (audit resampling + Gaussian eps part).
 item5: PPI++ power tuning (lambda*) and SAP control variates; calibrated vs mis-calibrated predictors.
Run: python3 g3_items125.py item1|item2|item5 [R]
"""
import numpy as np, json, sys, time
from scipy.stats import chi2, norm, t as tdist
from g3_core import (Lattice, ols_fit, build, estimate, omega, wald_se, sigma2_U,
                     ar_rho, ar_theta_given_rho, ar_ati_given_rho, ar_beta_given_rho)
from scipy.stats import f as fdist

RHO, BETA, THETA = 0.4, 1.0, 0.5
TRUE = np.array([RHO, BETA, THETA])
RG = np.linspace(-0.99, 0.99, 199)
TG = np.linspace(-2.0, 3.0, 101)
AG = np.linspace(-2.0, 12.0, 141)
K = 4


def one_draw(L, kappa, nl, rng, pi_tr=0.10):
    n = L.n
    x = L.draw_x(rng)
    tau = np.sqrt(L.s2 * (1 - kappa) / kappa)
    Z = x + tau * rng.standard_normal(n)
    y, _ = L.draw_y(x, BETA, THETA, rng)
    perm = rng.permutation(n); ntr = int(pi_tr * n)
    tr, lab = perm[:ntr], perm[ntr:ntr + nl]
    xh = ols_fit([Z], x, tr)
    return x, y, Z, xh, tr, lab


def ati_of(d):
    return (d[1] + d[2]) / (1 - d[0]) if d[0] < 1 else np.inf


def ati_grad(d):
    r = min(d[0], 0.999)
    return np.array([(d[1] + d[2]) / (1 - r) ** 2, 1 / (1 - r), 1 / (1 - r)])


# ============================================================== item 1
def item1(m, kappa, nl, R, rng, a1=0.01):
    L = Lattice(m, 0.8, RHO, K)
    ati_true = (BETA + THETA) / (1 - RHO)
    c2_95, c4_95 = chi2.ppf(0.95, K - 2), chi2.ppf(0.95, K)
    c2_1a1 = chi2.ppf(1 - a1, K - 2)
    c3_bonf = chi2.ppf(1 - (0.05 - a1), K - 1)
    keys = ["rho_sub", "rho_wald", "th_bonf", "th_direct", "th_proj", "th_wald",
            "ati_bonf", "ati_direct", "ati_proj", "ati_wald",
            "th_len_bonf", "th_len_direct", "th_bnd_bonf", "ati_len_bonf", "ati_len_direct",
            "ati_bnd_bonf", "ati_bnd_direct", "rho_bnd"]
    rec = {k: [] for k in keys}
    for r in range(R):
        x, y, Z, xh, tr, lab = one_draw(L, kappa, nl, rng)
        S = build(L, y, xh, xh, x, lab)
        d, P = estimate(S)
        s2 = sigma2_U(L, y, x, xh, lab, d)
        b0, t0 = d[1], d[2]
        V = wald_se(S, d, P, s2); se = np.sqrt(np.diag(V))
        rec["rho_wald"].append(abs(d[0] - RHO) <= 1.96 * se[0])
        rec["th_wald"].append(abs(d[2] - THETA) <= 1.96 * se[2])
        g = ati_grad(d); se_a = np.sqrt(g @ V @ g)
        rec["ati_wald"].append(abs(ati_of(d) - ati_true) <= 1.96 * se_a)
        # step 1: rho
        ar_r = ar_rho(S, RG, s2, b0, t0)
        ar_r_true = ar_rho(S, np.array([RHO]), s2, b0, t0)[0]
        rec["rho_sub"].append(ar_r_true <= c2_95)
        acc95 = ar_r <= c2_95
        rec["rho_bnd"].append(not (acc95[0] or acc95[-1]))
        CS1 = RG[ar_r <= c2_1a1]
        # theta / ATI at the truth
        th_full = ar_theta_given_rho(S, RG, np.full(len(RG), THETA), s2, b0)
        at_full = ar_ati_given_rho(S, RG, np.full(len(RG), ati_true), s2, b0)
        inCS = ar_r <= c2_1a1
        rec["th_bonf"].append(bool(np.any(th_full[inCS] <= c3_bonf)) if inCS.any() else False)
        rec["th_direct"].append(th_full.min() <= c2_95)
        rec["th_proj"].append(th_full.min() <= c4_95)
        rec["ati_bonf"].append(bool(np.any(at_full[inCS] <= c3_bonf)) if inCS.any() else False)
        rec["ati_direct"].append(at_full.min() <= c2_95)
        rec["ati_proj"].append(at_full.min() <= c4_95)
        # set lengths on grids
        RR, TT = np.meshgrid(RG, TG, indexing="ij")
        arTH = ar_theta_given_rho(S, RR.ravel(), TT.ravel(), s2, b0).reshape(RR.shape)
        RR2, AA = np.meshgrid(RG, AG, indexing="ij")
        arAT = ar_ati_given_rho(S, RR2.ravel(), AA.ravel(), s2, b0).reshape(RR2.shape)
        accTb = (arTH[inCS] <= c3_bonf).any(0) if inCS.any() else np.zeros(len(TG), bool)
        accTd = arTH.min(0) <= c2_95
        accAb = (arAT[inCS] <= c3_bonf).any(0) if inCS.any() else np.zeros(len(AG), bool)
        accAd = arAT.min(0) <= c2_95
        dT, dA = TG[1] - TG[0], AG[1] - AG[0]
        rec["th_len_bonf"].append(accTb.sum() * dT); rec["th_len_direct"].append(accTd.sum() * dT)
        rec["th_bnd_bonf"].append(not (accTb[0] or accTb[-1]))
        rec["ati_len_bonf"].append(accAb.sum() * dA); rec["ati_len_direct"].append(accAd.sum() * dA)
        rec["ati_bnd_bonf"].append(not (accAb[0] or accAb[-1]))
        rec["ati_bnd_direct"].append(not (accAd[0] or accAd[-1]))
    out = dict(n=L.n, kappa=kappa, nl=nl, R=R, alpha1=a1)
    for k in keys:
        v = np.array(rec[k], float)
        out[k] = float(np.median(v)) if "len" in k else float(v.mean())
    return out


# ============================================================== item 2
def item2(m, kappa, nl, R, rng, B=199):
    L = Lattice(m, 0.8, RHO, K)
    ati_true = (BETA + THETA) / (1 - RHO)
    p = 3
    names = ["wald", "hc1t", "boot", "perc"]
    cov = {k: [] for k in names}; ests = []
    extra = {k: [] for k in ["rho_ar", "rho_ar_hc1F", "beta_bonf", "beta_bonf_hc1", "theta_bonf", "ati_bonf"]}
    for r in range(R):
        x, y, Z, xh, tr, lab = one_draw(L, kappa, nl, rng)
        S = build(L, y, xh, xh, x, lab)
        d, P = estimate(S); ests.append(d)
        s2 = sigma2_U(L, y, x, xh, lab, d)
        a_hat = ati_of(d)
        tv = np.append(TRUE, ati_true); ev = np.append(d, a_hat)
        # Wald and HC1-t
        for tag, df, crit in (("wald", 1.0, 1.959964), ("hc1t", nl / (nl - p), tdist.ppf(0.975, nl - p))):
            V = wald_se(S, d, P, s2, df); g = ati_grad(d)
            se = np.append(np.sqrt(np.diag(V)), np.sqrt(g @ V @ g))
            cov[tag].append(np.abs(ev - tv) <= crit * se)
        # audit bootstrap-t (+ Gaussian eps component)
        V0 = wald_se(S, d, P, s2); g0 = ati_grad(d)
        se0 = np.append(np.sqrt(np.diag(V0)), np.sqrt(g0 @ V0 @ g0))
        counts = rng.multinomial(nl, np.ones(nl) / nl, size=B).astype(float)
        H, WtH = S["H"], S["WtH"]; Hl, Wl = H[lab], WtH[lab]; dl = S["dl"]
        base0 = H.T @ xh; base1 = WtH.T @ xh
        c0 = base0[None, :] + (counts * dl) @ Hl / S["pih"]
        c1 = base1[None, :] + (counts * dl) @ Wl / S["pih"]
        Gr = np.broadcast_to(S["G"][:, 0], c0.shape)
        Gs = np.stack([Gr, c0, c1], axis=2)                       # (B,K,3)
        Ai = np.linalg.inv(S["A"])
        GtAi = np.einsum("bkp,kl->bpl", Gs, Ai)
        Mb = GtAi @ Gs
        Bmb = np.linalg.inv(Mb)
        Pb = Bmb @ GtAi                                             # (B,3,K)
        db = np.einsum("bpk,k->bp", Pb, S["hy"])
        q = counts * dl ** 2
        Mhh = np.einsum("bi,ik,il->bkl", q, Hl, Hl); Mww = np.einsum("bi,ik,il->bkl", q, Wl, Wl)
        Mhw = np.einsum("bi,ik,il->bkl", q, Hl, Wl); Mhw = Mhw + np.transpose(Mhw, (0, 2, 1))
        bb, tt = db[:, 1], db[:, 2]
        Om = s2 * S["A"][None] + S["kfac"] * (bb[:, None, None] ** 2 * Mhh + (bb * tt)[:, None, None] * Mhw
                                              + tt[:, None, None] ** 2 * Mww)
        Vb = Pb @ Om @ np.transpose(Pb, (0, 2, 1))
        # Gaussian eps component added to the numerator
        La = np.linalg.cholesky(s2 * S["A"])
        xi = rng.standard_normal((B, K)) @ La.T
        dbh = db + xi @ P.T
        rb = np.minimum(dbh[:, 0], 0.999)
        atib = np.where(dbh[:, 0] < 1, (dbh[:, 1] + dbh[:, 2]) / (1 - rb), np.inf)
        gb = np.stack([(dbh[:, 1] + dbh[:, 2]) / (1 - rb) ** 2, 1 / (1 - rb), 1 / (1 - rb)], 1)
        se_b = np.sqrt(np.einsum("bii->bi", Vb))
        se_ab = np.sqrt(np.einsum("bp,bpq,bq->b", gb, Vb, gb))
        tstar = np.column_stack([(dbh - d[None, :]) / se_b, (atib - a_hat) / se_ab])
        tstar = np.where(np.isfinite(tstar), tstar, 1e6)
        qlo, qhi = np.quantile(tstar, 0.025, 0), np.quantile(tstar, 0.975, 0)
        lo, hi = ev - qhi * se0, ev - qlo * se0
        cov["boot"].append((tv >= lo) & (tv <= hi))
        # percentile bootstrap
        allb = np.column_stack([dbh, np.where(np.isfinite(atib), atib, 1e6)])
        plo, phi_ = np.quantile(allb, 0.025, 0), np.quantile(allb, 0.975, 0)
        cov["perc"].append((tv >= plo) & (tv <= phi_))
        # AR-based: rho subvector (plain / HC1 + F critical), Bonferroni for beta, theta, ATI
        dfh = nl / (nl - p)
        a_r = ar_rho(S, np.array([RHO]), s2, d[1], d[2])[0]
        a_rh = ar_rho(S, np.array([RHO]), s2, d[1], d[2], df=dfh)[0]
        extra["rho_ar"].append(a_r <= chi2.ppf(0.95, K - 2))
        extra["rho_ar_hc1F"].append(a_rh <= (K - 2) * fdist.ppf(0.95, K - 2, nl - K))
        for dfv, tag in ((1.0, "beta_bonf"), (dfh, "beta_bonf_hc1")):
            inCS = ar_rho(S, RG, s2, d[1], d[2], df=dfv) <= chi2.ppf(0.99, K - 2)
            if inCS.any():
                ab = ar_beta_given_rho(S, RG[inCS], np.full(inCS.sum(), BETA), s2, d[2], df=dfv)
                extra[tag].append(bool(np.any(ab <= chi2.ppf(0.96, K - 1))))
            else:
                extra[tag].append(False)
            if tag == "beta_bonf":
                if inCS.any():
                    at = ar_theta_given_rho(S, RG[inCS], np.full(inCS.sum(), THETA), s2, d[1])
                    aa = ar_ati_given_rho(S, RG[inCS], np.full(inCS.sum(), ati_true), s2, d[1])
                    extra["theta_bonf"].append(bool(np.any(at <= chi2.ppf(0.96, K - 1))))
                    extra["ati_bonf"].append(bool(np.any(aa <= chi2.ppf(0.96, K - 1))))
                else:
                    extra["theta_bonf"].append(False); extra["ati_bonf"].append(False)
    E = np.array(ests)
    out = dict(n=L.n, kappa=kappa, nl=nl, R=R, bias=(np.median(E, 0) - TRUE).tolist())
    for k in names:
        out[k] = np.mean(np.array(cov[k], float), 0).tolist()   # rho, beta, theta, ATI
    for k, v in extra.items():
        out[k] = float(np.mean(v))
    return out


# ============================================================== item 5
def lam_star(S, P, d, x, phi_base, mu, lab, target):
    """lambda minimising the audit-variance of target'delta for phi = mu + lam*(phi_base - mu)."""
    b = d[1] * S["H"][lab] + d[2] * S["WtH"][lab]
    w = (b @ (P.T @ target)) ** 2
    u = phi_base[lab] - mu; v = x[lab] - mu
    return float(np.sum(w * u * v) / np.sum(w * u ** 2))


def item5(m, kappa, nl, scale, R, rng):
    L = Lattice(m, 0.8, RHO, K)
    ati_true = (BETA + THETA) / (1 - RHO)
    tgt = np.array([0.0, 0.0, 1.0])
    names = ["LCIV", "PPIpp", "CVSAP", "CVSAPpp"]
    rec = {k: dict(d=[], ati=[], cov_th=[], bnd=[]) for k in names}
    lams = {"PPIpp": [], "CVSAPpp": []}
    c2_95 = chi2.ppf(0.95, K - 2)
    for r in range(R):
        x, y, Z, xh0, tr, lab = one_draw(L, kappa, nl, rng)
        mu0 = xh0.mean()
        xh = mu0 + scale * (xh0 - mu0)                 # (mis-)calibrated predictor
        W = L.W
        xt = ols_fit([xh, W @ xh, L.Wp[2] @ xh], x, tr)  # SAP control variate (training set)
        mu, mut = xh.mean(), xt.mean()
        S0 = build(L, y, xh, xh, x, lab); d0, P0 = estimate(S0)
        lam1 = lam_star(S0, P0, d0, x, xh, mu, lab, tgt)
        Sx = build(L, y, xh, xt, x, lab); dx, Px = estimate(Sx)
        lam2 = lam_star(Sx, Px, dx, x, xt, mut, lab, tgt)
        lams["PPIpp"].append(lam1); lams["CVSAPpp"].append(lam2)
        phis = {"LCIV": xh, "PPIpp": mu + lam1 * (xh - mu), "CVSAP": xt, "CVSAPpp": mut + lam2 * (xt - mut)}
        for k, phi in phis.items():
            S = build(L, y, xh, phi, x, lab); d, P = estimate(S)
            s2 = sigma2_U(L, y, x, phi, lab, d)
            V = wald_se(S, d, P, s2)
            rec[k]["d"].append(d); rec[k]["ati"].append(ati_of(d))
            rec[k]["cov_th"].append(abs(d[2] - THETA) <= 1.96 * np.sqrt(V[2, 2]))
            ar = ar_rho(S, RG, s2, d[1], d[2]) <= c2_95
            rec[k]["bnd"].append(not (ar[0] or ar[-1]))
    out = dict(n=L.n, kappa=kappa, nl=nl, scale=scale, R=R,
               lam_mean={k: float(np.mean(v)) for k, v in lams.items()})
    for k in names:
        E = np.array(rec[k]["d"]); a = np.array(rec[k]["ati"])
        ae = np.abs(a - ati_true); ae = np.where(np.isfinite(ae), ae, 1e6)
        out[k] = dict(theta_bias=float(np.median(E[:, 2]) - THETA),
                      theta_rmse=float(np.sqrt(np.mean((E[:, 2] - THETA) ** 2))),
                      theta_iqr=float(np.subtract(*np.percentile(E[:, 2], [75, 25]))),
                      ati_mae_med=float(np.median(ae)), ati_ae_q90=float(np.quantile(ae, 0.9)),
                      cov_theta=float(np.mean(rec[k]["cov_th"])), rho_bounded=float(np.mean(rec[k]["bnd"])))
    return out


if __name__ == "__main__":
    which = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = np.random.default_rng({"item1": 101, "item2": 202, "item5": 505}[which])
    res = []; t0 = time.time()
    if which == "item1":
        designs = [(30, 0.5, 90), (30, 0.7, 45), (30, 0.7, 90), (30, 0.9, 90), (50, 0.5, 250), (50, 0.7, 250)]
        for (m, k, nl) in designs:
            o = item1(m, k, nl, R if m == 30 else max(R // 2, 50), rng); res.append(o)
            print(json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in o.items()}),
                  f"[{time.time()-t0:.0f}s]", flush=True)
    elif which == "item2":
        designs = [(30, 0.7, 27), (30, 0.7, 45), (30, 0.7, 90), (30, 0.9, 45)]
        for (m, k, nl) in designs:
            o = item2(m, k, nl, R, rng); res.append(o)
            print(json.dumps({a: (np.round(b, 3).tolist() if isinstance(b, list) else b) for a, b in o.items()}),
                  f"[{time.time()-t0:.0f}s]", flush=True)
    else:
        for nl in (45, 90):
            for sc in (1.0, 1.5, 0.6):
                o = item5(30, 0.7, nl, sc, R, rng); res.append(o)
                print(json.dumps(o, default=float), f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(f"g3_{which}.json", "w"), indent=1)
