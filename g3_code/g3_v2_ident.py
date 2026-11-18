"""Proposition (identification strength of rho) and Proposition 2' (correlated image noise): numerical checks.
ident:  P(AR set for rho bounded) vs predicted P(chi2_{K-2}(mu^2) > c), mu^2 from population quantities.
slx:    closed form for spurious SLX theta with spatially correlated image noise vs Monte Carlo.
"""
import sys, json, time
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.optimize import minimize
import scipy.sparse as sp
from g3_v2 import Geo, draw, build, estimate, omega, _tan_grid

K = 4
C = chi2.ppf(0.95, K - 2)


def tid(Gr, Gs, Om_fun, b0):
    """min_b (Gr - Gs b)' Om_a(b)^-1 (Gr - Gs b), global over R^2."""
    def f(b, t):
        g = Gr[None, :] - b[:, None] * Gs[None, :, 0] - t[:, None] * Gs[None, :, 1]
        Oi = np.linalg.inv(Om_fun(b, t))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)
    g0 = _tan_grid(b0[0], max(abs(b0[0]), 0.5), 41); g1 = _tan_grid(b0[1], max(abs(b0[1]), 0.5), 41)
    P0, P1 = np.meshgrid(g0, g1, indexing="ij"); v = f(P0.ravel(), P1.ravel())
    best = np.inf
    for j in np.argsort(v)[:4]:
        res = minimize(lambda z: f(np.array([z[0]]), np.array([z[1]]))[0], [P0.ravel()[j], P1.ravel()[j]],
                       method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=3000))
        best = min(best, res.fun, v[j])
    return float(best)


def pop_strength(G, x, xh, y_mean, nl):
    """population mu^2: replace audit sums by their design expectation using all units."""
    W, n = G.W, G.n
    cols = [xh]
    for _ in range(K - 1):
        cols.append(W @ cols[-1])
    H = np.column_stack(cols); WtH = W.T @ H
    e = x - xh; pi = nl / n
    Gr = H.T @ (W @ y_mean)                      # E[H'Wy | F]
    Gs = np.column_stack([H.T @ x, H.T @ (W @ x)])
    u = e[:, None] * H; v = e[:, None] * WtH
    uc = u - u.mean(0); vc = v - v.mean(0)
    f = (1 - pi) / pi
    Chh, Chw, Cww = f * uc.T @ uc, f * (uc.T @ vc + vc.T @ uc), f * vc.T @ vc

    def Om(b, t):
        return (b ** 2)[:, None, None] * Chh + (b * t)[:, None, None] * Chw + (t ** 2)[:, None, None] * Cww
    a = np.linalg.lstsq(Gs, Gr, rcond=None)[0]
    return tid(Gr, Gs[:, :, None].reshape(K, 2), Om, a)


def ident(R, which):
    designs = {
        # name: (geo, psi, kappa, rho, beta, theta, nl)
        "L50_k7_n50": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 50),
        "L50_k7_n100": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 100),
        "L50_k7_n200": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 200),
        "L50_k9_n100": ("L50", 0.8, 0.9, 0.4, 1.0, 0.5, 100),
        "L50_k5_n200": ("L50", 0.8, 0.5, 0.4, 1.0, 0.5, 200),
        "L50_smooth_k9_n200": ("L50", 0.95, 0.9, 0.4, 1.0, 0.5, 200),
        "L50_weak_tpb_n200": ("L50", 0.8, 0.9, 0.4, 1.0, -0.2, 200),   # theta+rho*beta = 0.2
        "L50_nonid_n200": ("L50", 0.8, 0.9, 0.5, 1.0, -0.5, 200),       # theta+rho*beta = 0
        "L30_k7_n90": ("L30", 0.8, 0.7, 0.4, 1.0, 0.5, 90),
        "VN_k7_n40": ("VN", 0.8, 0.7, 0.4, 1.0, 0.5, 40),
        "VN_k95_n30": ("VN", 0.8, 0.95, 0.4, 1.0, 0.5, 30),
        "VN_smooth_k95_n60": ("VN", 0.95, 0.95, 0.4, 1.0, 0.5, 60),
        "VN_smooth_k95_n120": ("VN", 0.95, 0.95, 0.4, 1.0, 0.5, 120),
    }
    geos = {}; rng = np.random.default_rng(99); out = []
    for nm, (g, psi, kap, rho, beta, theta, nl) in designs.items():
        if g not in geos:
            geos[g] = Geo(Geo.rook(int(g[1:]))) if g.startswith("L") else Geo(sp.load_npz(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "vietnam", "data", "W_queen_rowstd.npz")))
        Gg = geos[g]; mus, tids = [], []; t0 = time.time()
        for r in range(R):
            x, y, xh, lab = draw(Gg, psi, kap, rho, beta, theta, nl, rng)
            ymean = Gg.solve(rho, beta * x + theta * (Gg.W @ x))
            mus.append(pop_strength(Gg, x, xh, ymean, nl))
            S = build(Gg, y, xh, x, lab)
            d, P = estimate(S)

            def Oma(b, t, S=S):
                return omega(S, b, t, 0.0)
            tids.append(tid(S["G"][:, 0], S["G"][:, 1:], Oma, (d[1], d[2])))
        mus = np.array(mus); tids = np.array(tids)
        o = dict(name=nm, n=Gg.n, psi=psi, kappa=kap, rho=rho, beta=beta, theta=theta, nl=nl, R=R,
                 tpb=theta + rho * beta, index=nl * kap / (1 - kap),
                 mu2_median=float(np.median(mus)), mu2_mean=float(np.mean(mus)),
                 mu2_per_label=float(np.median(mus) / nl),
                 bounded_obs=float(np.mean(tids > C)),
                 bounded_pred=float(np.mean(ncx2.sf(C, K - 2, np.maximum(mus, 1e-12)))),
                 secs=round(time.time() - t0))
        out.append(o); print(json.dumps(o), flush=True)
        json.dump(out, open("g3_v2_ident.json", "w"), indent=1)


def slx(R):
    """Prop 2': theta_SLX = (kappa/c) beta (1-kappa)(r1 - r1nu)/Delta_nu."""
    Gg = Geo(Geo.rook(40)); n = Gg.n; W = Gg.W.toarray(); I = np.eye(n)
    rng = np.random.default_rng(7); out = []
    for psi in (0.3, 0.6, 0.8):
        Sx = np.linalg.inv(I - psi * W); Sx = Sx @ Sx.T
        for phn in (0.0, 0.3, 0.6, 0.8):
            Sn = np.linalg.inv(I - phn * W); Sn = Sn @ Sn.T
            for kap in (0.5, 0.7):
                s2 = np.trace(Sx) / n; t2c = np.trace(Sn) / n
                c_n = np.sqrt(s2 * (1 - kap) / kap / t2c); Snu = c_n ** 2 * Sn; t2 = np.trace(Snu) / n
                r1 = np.trace(W @ Sx) / (n * s2); r2 = np.trace(W.T @ W @ Sx) / (n * s2)
                r1n = np.trace(W @ Snu) / (n * t2); r2n = np.trace(W.T @ W @ Snu) / (n * t2)
                k = s2 / (s2 + t2)
                Dn = k * r2 + (1 - k) * r2n - (k * r1 + (1 - k) * r1n) ** 2
                th_formula = (1 - k) * (r1 - r1n) / Dn          # beta=1, c=kappa so kappa/c = 1
                Lx = np.linalg.cholesky(Sx); Ln = np.linalg.cholesky(Snu)
                ths = []
                for r in range(R):
                    x = Lx @ rng.standard_normal(n); Z = x + Ln @ rng.standard_normal(n)
                    xh = k * Z; y = x + rng.standard_normal(n)
                    X = np.column_stack([np.ones(n), xh, W @ xh])
                    ths.append(np.linalg.lstsq(X, y, rcond=None)[0][2])
                o = dict(psi=psi, phi_nu=phn, kappa=kap, r1=r1, r1nu=r1n, formula=th_formula,
                         mc=float(np.mean(ths)), mc_se=float(np.std(ths) / np.sqrt(R)))
                out.append(o); print(json.dumps(o), flush=True)
    json.dump(out, open("g3_v2_slx.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "ident":
        ident(int(sys.argv[2]), "all")
    else:
        slx(int(sys.argv[2]))
