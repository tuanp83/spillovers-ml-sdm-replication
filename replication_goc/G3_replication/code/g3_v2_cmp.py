"""(1) Length / boundedness of the revised confidence sets; (2) comparison with competing estimators.
Usage: python g3_v2_cmp.py len R  |  python g3_v2_cmp.py cmp R
"""
import sys, json, time
import numpy as np
from scipy.stats import chi2
import scipy.sparse as sp
from g3_v2 import Geo, draw, build, estimate, sigma2_U, T_theta, T_rho, omega
from g3_v2_mc import DESIGNS, geo

K = 4; C2 = chi2.ppf(0.95, K - 2); C1 = chi2.ppf(0.95, K - 1)


def impacts(Gg, d, om):
    rho, beta, theta = d
    if not (-1 < rho < 1):
        return np.array([np.nan] * 3)
    adi = np.mean((beta + theta * om) / (1 - rho * om)); ati = (beta + theta) / (1 - rho)
    return np.array([adi, ati - adi, ati])


def lengths(R):
    TG = np.linspace(-3, 4, 141); RG = np.linspace(-3, 3, 121); out = []
    rng = np.random.default_rng(31)
    for nm in ["base_fixedpi", "base_audit", "smooth_k9_audit", "vn_base", "nonid_audit"]:
        g, psi, kap, rho, beta, theta, nl, phn, eps = DESIGNS[nm]; Gg = geo(g); t0 = time.time()
        rec = dict(lt2=[], lt1=[], bt2=[], bt1=[], lr=[], br=[], br_in=[], wald_len=[], wald_cov=[])
        for r in range(R):
            x, y, xh, lab = draw(Gg, psi, kap, rho, beta, theta, nl, rng, phn, eps)
            S = build(Gg, y, xh, x, lab); d, P = estimate(S); s2 = sigma2_U(S, d)
            sb = max(abs(d[1]), 0.5); st = max(abs(d[2]), 0.5)
            Tt = np.array([T_theta(S, t0, s2, d[1], sb, N=81) for t0 in TG])
            Tr = np.array([T_rho(S, r0, s2, d, (sb, st)) for r0 in RG])
            step_t = TG[1] - TG[0]; step_r = RG[1] - RG[0]
            for crit, kk in ((C2, "2"), (C1, "1")):
                acc = Tt <= crit
                rec["lt" + kk].append(acc.sum() * step_t)
                rec["bt" + kk].append(bool(acc.any() and not acc[0] and not acc[-1]))
            accr = Tr <= C2
            rec["lr"].append(accr.sum() * step_r); rec["br"].append(bool(not accr[0] and not accr[-1]))
            inside = accr[(RG > -1) & (RG < 1)]
            rec["br_in"].append(bool(inside.any() and not inside.all()))
            V = P @ omega(S, d[1], d[2], s2) @ P.T; se = np.sqrt(V[2, 2])
            rec["wald_len"].append(2 * 1.96 * se); rec["wald_cov"].append(abs(d[2] - theta) <= 1.96 * se)
        o = dict(name=nm, R=R, secs=round(time.time() - t0),
                 theta_len_chi2K2=float(np.median(rec["lt2"])), theta_bounded_chi2K2=float(np.mean(rec["bt2"])),
                 theta_len_chi2K1=float(np.median(rec["lt1"])), theta_bounded_chi2K1=float(np.mean(rec["bt1"])),
                 rho_len=float(np.median(rec["lr"])), rho_bounded=float(np.mean(rec["br"])),
                 wald_theta_len=float(np.median(rec["wald_len"])), wald_theta_cov=float(np.mean(rec["wald_cov"])))
        out.append(o); print(json.dumps(o), flush=True)
        json.dump(out, open("g3_v2_len.json", "w"), indent=1)


def tsls(Gg, y, xv, K=4):
    """2SLS of SDM with regressor vector xv (instruments: W^k xv)."""
    W = Gg.W; cols = [xv]
    for _ in range(K - 1):
        cols.append(W @ cols[-1])
    H = np.column_stack(cols); Q = np.column_stack([W @ y, xv, W @ xv])
    A = H.T @ H; Gm = H.T @ Q
    return np.linalg.solve(Gm.T @ np.linalg.solve(A, Gm), Gm.T @ np.linalg.solve(A, H.T @ y))


def compare(R):
    rng = np.random.default_rng(41); out = []
    for nm in ["base_fixedpi", "base_audit", "corrnoise_audit", "vn_base"]:
        g, psi, kap, rho, beta, theta, nl, phn, eps = DESIGNS[nm]; Gg = geo(g)
        om = np.real(np.linalg.eigvals(Gg.W.toarray())) if Gg.n <= 2500 else None
        true_imp = impacts(Gg, (rho, beta, theta), om)
        est = {k: [] for k in ["oracle", "naive", "MI", "RC_SAP", "LCIV", "CVSAP"]}; t0 = time.time()
        for r in range(R):
            x, y, xh, lab = draw(Gg, psi, kap, rho, beta, theta, nl, rng, phn, eps)
            est["oracle"].append(tsls(Gg, y, x)); est["naive"].append(tsls(Gg, y, xh))
            # multiple imputation (Proctor-Carleton-Sum style): x | xh ~ N(a + b xh, s^2), independent errors
            Xl = np.column_stack([np.ones(nl), xh[lab]]); cf, *_ = np.linalg.lstsq(Xl, x[lab], rcond=None)
            su = np.std(x[lab] - Xl @ cf, ddof=2); ds = []
            for m in range(10):
                xs = cf[0] + cf[1] * xh + su * rng.standard_normal(Gg.n); xs[lab] = x[lab]
                ds.append(tsls(Gg, y, xs))
            est["MI"].append(np.mean(ds, 0))
            # regression calibration with spatial lags (SAP), fitted on the audit sample
            Xs = np.column_stack([np.ones(Gg.n), xh, Gg.W @ xh, Gg.W @ (Gg.W @ xh)])
            cs, *_ = np.linalg.lstsq(Xs[lab], x[lab], rcond=None); xsap = Xs @ cs
            est["RC_SAP"].append(tsls(Gg, y, xsap))
            S = build(Gg, y, xh, x, lab); d, _ = estimate(S); est["LCIV"].append(d)
            # CV-SAP: SAP control variate fitted on an independent split of the audit (cross-fitting)
            half = lab[: nl // 2]; other = lab[nl // 2:]
            c1, *_ = np.linalg.lstsq(Xs[half], x[half], rcond=None); c2, *_ = np.linalg.lstsq(Xs[other], x[other], rcond=None)
            phi = Xs @ ((c1 + c2) / 2); phi[half] = Xs[half] @ c2; phi[other] = Xs[other] @ c1
            S2 = build(Gg, y, xh, x, lab, phi=phi); d2, _ = estimate(S2); est["CVSAP"].append(d2)
        o = dict(name=nm, R=R, secs=round(time.time() - t0), true=[rho, beta, theta], true_imp=list(true_imp))
        for k, v in est.items():
            v = np.array(v); imp = np.array([impacts(Gg, dd, om) for dd in v])
            o[k] = dict(medbias=list(np.round(np.median(v, 0) - [rho, beta, theta], 3)),
                        imp_medbias=list(np.round(np.nanmedian(imp, 0) - true_imp, 3)),
                        imp_mad=list(np.round(np.nanmedian(np.abs(imp - true_imp), 0), 3)),
                        theta_mad=float(np.round(np.median(np.abs(v[:, 2] - theta)), 3)))
        out.append(o); print(json.dumps(o), flush=True)
        json.dump(out, open("g3_v2_cmp.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "len":
        lengths(int(sys.argv[2]))
    else:
        compare(int(sys.argv[2]))
