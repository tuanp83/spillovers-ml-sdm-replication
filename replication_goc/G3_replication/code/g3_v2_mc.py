"""Monte Carlo for revised Theorem 5: size of subvector AR tests (theta, rho, ATI) at the true value.
Usage: python g3_v2_mc.py <design_set> <R>
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2, f as fdist
from g3_v2 import Geo, draw, build, estimate, sigma2_U, T_theta, T_rho, T_ati

K = 4


def crit(nl, alpha=0.05):
    p = K - 2
    return dict(chi2=chi2.ppf(1 - alpha, p),
                hot=p * (nl - 1) / (nl - p) * fdist.ppf(1 - alpha, p, nl - p))


def run(name, G, psi, kappa, rho, beta, theta, nl, R, rng, phi_nu=0.0, eps="normal", do_ati=True):
    cv = crit(nl); out = {k: [] for k in ["th", "rh", "at"]}
    ests = []; t0 = time.time()
    ati = (beta + theta) / (1 - rho) if rho != 1 else np.nan
    for r in range(R):
        x, y, xh, lab = draw(G, psi, kappa, rho, beta, theta, nl, rng, phi_nu, eps)
        S = build(G, y, xh, x, lab)
        d, P = estimate(S); s2 = sigma2_U(S, d); ests.append(d)
        sc = (max(abs(d[0]), 0.5), max(abs(d[1]), 0.5), max(abs(d[2]), 0.5))
        out["th"].append(T_theta(S, theta, s2, d[1], sc[1]))
        out["rh"].append(T_rho(S, rho, s2, d, (sc[1], sc[2])))
        if do_ati:
            out["at"].append(T_ati(S, ati, s2, d, (sc[0], sc[1])))
    res = dict(name=name, n=G.n, psi=psi, kappa=kappa, rho=rho, beta=beta, theta=theta, nl=nl, R=R,
               phi_nu=phi_nu, eps=eps, index=nl * kappa / (1 - kappa), secs=round(time.time() - t0))
    for k, v in out.items():
        if not v:
            continue
        v = np.array(v)
        res[f"rej_{k}_chi2"] = float(np.mean(v > cv["chi2"]))
        res[f"rej_{k}_hot"] = float(np.mean(v > cv["hot"]))
    E = np.array(ests); res["medbias"] = list(np.round(np.median(E, 0) - [rho, beta, theta], 3))
    return res


DESIGNS = {
    # name: (geo, psi, kappa, rho, beta, theta, nl, phi_nu, eps)
    "base_fixedpi": ("L30", 0.8, 0.7, 0.4, 1.0, 0.5, 90, 0.0, "normal"),
    "base_audit": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 100, 0.0, "normal"),
    "nonid_audit": ("L50", 0.8, 0.7, 0.5, 1.0, -0.5, 100, 0.0, "normal"),     # theta + rho*beta = 0
    "beta0_audit": ("L50", 0.8, 0.7, 0.4, 0.0, 0.5, 100, 0.0, "normal"),      # U4 fails
    "rhobnd_audit": ("L50", 0.8, 0.7, 0.9, 1.0, 0.5, 100, 0.0, "normal"),     # rho near boundary
    "rhoneg_audit": ("L50", 0.8, 0.7, -0.4, 1.0, 0.5, 100, 0.0, "normal"),
    "smooth_audit": ("L50", 0.95, 0.7, 0.4, 1.0, 0.5, 100, 0.0, "normal"),    # very smooth x
    "nullx_audit": ("L50", 0.8, 0.7, 0.4, 0.0, 0.0, 100, 0.0, "normal"),      # outside theorem: beta=theta=0
    "hetero_audit": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 100, 0.0, "hetero_t"),
    "corrnoise_audit": ("L50", 0.8, 0.7, 0.4, 1.0, 0.5, 100, 0.5, "normal"),
    "nonid_fixedpi": ("L30", 0.8, 0.7, 0.5, 1.0, -0.5, 90, 0.0, "normal"),
    "beta0_fixedpi": ("L30", 0.8, 0.7, 0.4, 0.0, 0.5, 90, 0.0, "normal"),
    "vn_base": ("VN", 0.8, 0.7, 0.4, 1.0, 0.5, 40, 0.0, "normal"),
    "vn_nonid": ("VN", 0.8, 0.7, 0.5, 1.0, -0.5, 40, 0.0, "normal"),
    "small_k5_fixedpi": ("L30", 0.8, 0.5, 0.4, 1.0, 0.5, 90, 0.0, "normal"),
    "small_k7n45_fixedpi": ("L30", 0.8, 0.7, 0.4, 1.0, 0.5, 45, 0.0, "normal"),
    "small_k5_audit": ("L50", 0.8, 0.5, 0.4, 1.0, 0.5, 60, 0.0, "normal"),
    "smooth_fixedpi": ("L30", 0.95, 0.7, 0.4, 1.0, 0.5, 90, 0.0, "normal"),
    "smooth_k9_audit": ("L50", 0.95, 0.9, 0.4, 1.0, 0.5, 200, 0.0, "normal"),
    "vn_smooth": ("VN", 0.95, 0.9, 0.4, 1.0, 0.5, 60, 0.0, "normal"),
}

SETS = {
    "quick": ["base_audit", "nonid_audit"],
    "A": ["base_fixedpi", "base_audit", "nonid_audit", "beta0_audit", "rhobnd_audit", "rhoneg_audit", "smooth_audit"],
    "C": ["small_k5_fixedpi", "small_k7n45_fixedpi", "small_k5_audit", "smooth_fixedpi", "smooth_k9_audit", "vn_smooth"],
    "B": ["nullx_audit", "hetero_audit", "corrnoise_audit", "nonid_fixedpi", "beta0_fixedpi", "vn_base", "vn_nonid"],
}


def geo(tag):
    if tag == "L30":
        return Geo(Geo.rook(30))
    if tag == "L50":
        return Geo(Geo.rook(50))
    if tag == "VN":
        import os; W = sp.load_npz(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vietnam", "data", "W_queen_rowstd.npz"))
        return Geo(W)


if __name__ == "__main__":
    import scipy.sparse as sp
    which, R = sys.argv[1], int(sys.argv[2])
    rng = np.random.default_rng({"quick": 1, "A": 2026, "B": 2027, "C": 2028}.get(which, 3))
    res = []; geos = {}
    for nm in SETS.get(which, [which]):
        g, psi, kap, rho, beta, theta, nl, phn, eps = DESIGNS[nm]
        if g not in geos:
            geos[g] = geo(g)
        o = run(nm, geos[g], psi, kap, rho, beta, theta, nl, R, rng, phn, eps)
        res.append(o); print(json.dumps(o), flush=True)
        json.dump(res, open(f"g3_v2_mc_{which}.json", "w"), indent=1)
