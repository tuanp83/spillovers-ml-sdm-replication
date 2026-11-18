"""G3 v2: exact subvector Anderson-Rubin inference for LC-IV (revised Theorem 5).

Statistic definitions (all minimisations are global and unrestricted: nuisance in R):
  AR(d0) = g(d0)' Omega(b0,t0)^-1 g(d0),  g(d0) = H'y - G d0,
  Omega(b0,t0) = s2*A + (1-pi)/pi^2 * nL/(nL-1) * sum_L (a_i - abar)(a_i - abar)',  a_i = (b0 H_i + t0 (W'H)_i) e_i.
  T_theta(t0) = min_{r0 in R, b0 in R} AR(r0,b0,t0)       (closed form in r0, 1-D global search in b0)
  T_rho(r0)   = min_{b0, t0 in R} AR(r0,b0,t0)            (2-D global search)
  T_ati(a0)   = min_{r0, b0 in R} AR(r0,b0, a0(1-r0)-b0)    (2-D global search)
s2 is held fixed (any O_p(1) estimate); under audit dominance it is asymptotically irrelevant.
"""
import numpy as np
from scipy.optimize import minimize
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class Geo:
    """Spatial structure with sparse W (lattice or arbitrary row-standardised W)."""
    def __init__(self, W, K=4):
        self.W = sp.csr_matrix(W); self.n = self.W.shape[0]; self.K = K
        self.I = sp.identity(self.n, format="csc")
        self._lu = {}

    @staticmethod
    def rook(m):
        n = m * m; rows, cols = [], []
        for i in range(m):
            for j in range(m):
                k = i * m + j
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < m and 0 <= b < m:
                        rows.append(k); cols.append(a * m + b)
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        d = np.asarray(W.sum(1)).ravel()
        return sp.diags(1 / d) @ W

    def solve(self, c, v):
        """(I - c W)^-1 v"""
        if c == 0:
            return v.copy()
        if c not in self._lu:
            self._lu[c] = splu((self.I - c * self.W).tocsc())
        return self._lu[c].solve(v)

    def sar(self, c, rng, scale_to_unit=True):
        v = self.solve(c, rng.standard_normal(self.n))
        return v


def draw(G, psi, kappa, rho, beta, theta, nl, rng, phi_nu=0.0, eps="normal"):
    """x ~ SAR(psi); image Z = x + nu, nu ~ SAR(phi_nu) scaled so that reliability = kappa.
    Predictor xh = a + c Z calibrated on an INDEPENDENT training map. Audit = SRS of size nl."""
    n = G.n

    def one_map():
        x = G.solve(psi, rng.standard_normal(n))
        nu = G.solve(phi_nu, rng.standard_normal(n))
        return x, nu
    x, nu = one_map(); xt, nut = one_map()
    sx2 = x.var(); snu = np.sqrt(sx2 * (1 - kappa) / kappa)
    k_nu = snu / nu.std(); Z = x + k_nu * nu
    Zt = xt + (snu / nut.std()) * nut
    c = np.cov(Zt, xt)[0, 1] / Zt.var(); a = xt.mean() - c * Zt.mean()
    xh = a + c * Z
    if eps == "normal":
        e = rng.standard_normal(n)
    elif eps == "hetero_t":          # heteroskedastic, heavy tailed, spatially correlated
        e = rng.standard_t(5, n) / np.sqrt(5 / 3) * np.sqrt(0.5 + np.abs(x) / np.abs(x).mean() * 0.5)
        e = G.solve(0.5, e); e = e / e.std()
    y = G.solve(rho, beta * x + theta * (G.W @ x) + e)
    lab = rng.choice(n, nl, replace=False)
    return x, y, xh, lab


def build(G, y, xh, x, lab, phi=None):
    if phi is None:
        phi = xh
    W, n, K = G.W, G.n, G.K
    cols = [xh]
    for _ in range(K - 1):
        cols.append(W @ cols[-1])
    H = np.column_stack(cols); WtH = W.T @ H
    nl = len(lab); pih = nl / n
    d = x[lab] - phi[lab]
    Hl, Wl = H[lab], WtH[lab]
    c0 = H.T @ phi + Hl.T @ d / pih
    c1 = WtH.T @ phi + Wl.T @ d / pih
    Wy = W @ y
    Gm = np.column_stack([H.T @ Wy, c0, c1])
    u = d[:, None] * Hl; v = d[:, None] * Wl
    uc = u - u.mean(0); vc = v - v.mean(0)
    fac = (1 - pih) / pih ** 2 * nl / (nl - 1)
    S = dict(G=Gm, hy=H.T @ y, A=H.T @ H, Chh=fac * uc.T @ uc, Chw=fac * (uc.T @ vc + vc.T @ uc),
             Cww=fac * vc.T @ vc, nl=nl, n=n, K=K, pih=pih,
             yl=y[lab], Wyl=Wy[lab], xl=x[lab], Wphil=(W @ phi)[lab])
    return S


def estimate(S):
    Ai = np.linalg.inv(S["A"]); Gm = S["G"]
    P = np.linalg.solve(Gm.T @ Ai @ Gm, Gm.T @ Ai)
    return P @ S["hy"], P


def omega(S, b, t, s2):
    b = np.asarray(b, float); t = np.asarray(t, float)
    return s2 * S["A"] + (b ** 2)[..., None, None] * S["Chh"] + (b * t)[..., None, None] * S["Chw"] \
        + (t ** 2)[..., None, None] * S["Cww"]


def sigma2_U(S, d):
    r = S["yl"] - d[0] * S["Wyl"] - d[1] * S["xl"] - d[2] * S["Wphil"]
    return float(np.mean(r ** 2))


def _tan_grid(center, scale, N):
    u = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, N)
    return center + scale * np.tan(u)


# ---------------------------------------------------------------- theta: closed form in rho0
def _arth(S, b, t0, s2):
    """min over rho0 in R of AR(rho0, b, t0); b array."""
    Gm, hy = S["G"], S["hy"]
    b = np.atleast_1d(b)
    r = hy[None, :] - b[:, None] * Gm[None, :, 1] - t0 * Gm[None, :, 2]
    a = Gm[:, 0]
    Oi = np.linalg.inv(omega(S, b, np.full_like(b, t0), s2))
    rOr = np.einsum("nk,nkl,nl->n", r, Oi, r)
    aOr = np.einsum("k,nkl,nl->n", a, Oi, r)
    aOa = np.einsum("k,nkl,l->n", a, Oi, a)
    return rOr - aOr ** 2 / aOa


def T_theta(S, t0, s2, bhat, bscale, N=161):
    bs = _tan_grid(bhat, bscale, N)
    v = _arth(S, bs, t0, s2)
    best = np.inf
    for j in np.argsort(v)[:3]:
        lo = bs[max(j - 1, 0)]; hi = bs[min(j + 1, N - 1)]
        res = minimize(lambda z: _arth(S, z, t0, s2)[0], x0=[bs[j]], bounds=[(lo, hi)], method="L-BFGS-B")
        best = min(best, float(res.fun), float(v[j]))
    # asymptote |b0| -> inf
    lim = _arth(S, np.array([bhat + 1e8 * bscale]), t0, s2)[0]
    return min(best, lim)


# ---------------------------------------------------------------- generic 2-D global search
def _ar_lin(S, U, V, b, t, s2):
    """U (N,K) intercepts, V (N,K,2) nuisance columns, coefficients c (N,2) given; returns AR."""
    raise NotImplementedError


def _min2(fun, x0, scales, N=41):
    """global min of fun(p0,p1) (vectorised) over R^2 using tan grid + local refine."""
    g0 = _tan_grid(x0[0], scales[0], N); g1 = _tan_grid(x0[1], scales[1], N)
    P0, P1 = np.meshgrid(g0, g1, indexing="ij")
    v = fun(P0.ravel(), P1.ravel())
    best = np.inf
    for j in np.argsort(v)[:4]:
        st = np.array([P0.ravel()[j], P1.ravel()[j]])
        res = minimize(lambda z: fun(np.array([z[0]]), np.array([z[1]]))[0], st, method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-9, maxiter=2000))
        best = min(best, float(res.fun), float(v[j]))
    return best


def T_rho(S, r0, s2, d, scales):
    Gm, hy = S["G"], S["hy"]
    U = hy - r0 * Gm[:, 0]

    def f(b, t):
        g = U[None, :] - b[:, None] * Gm[None, :, 1] - t[:, None] * Gm[None, :, 2]
        Oi = np.linalg.inv(omega(S, b, t, s2))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)
    return _min2(f, (d[1], d[2]), scales)


def T_ati(S, a0, s2, d, scales):
    Gm, hy = S["G"], S["hy"]

    def f(r, b):
        t = a0 * (1 - r) - b
        g = hy[None, :] - r[:, None] * Gm[None, :, 0] - b[:, None] * Gm[None, :, 1] - t[:, None] * Gm[None, :, 2]
        Oi = np.linalg.inv(omega(S, b, t, s2))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)
    return _min2(f, (d[0], d[1]), scales)


def T_ati_closed(S, a0, s2, d, scales):
    """alternative parametrisation: nuisance (rho0, theta0) with beta0 = a0(1-rho0) - theta0"""
    Gm, hy = S["G"], S["hy"]

    def f(r, t):
        b = a0 * (1 - r) - t
        g = hy[None, :] - r[:, None] * Gm[None, :, 0] - b[:, None] * Gm[None, :, 1] - t[:, None] * Gm[None, :, 2]
        Oi = np.linalg.inv(omega(S, b, t, s2))
        return np.einsum("nk,nkl,nl->n", g, Oi, g)
    return _min2(f, (d[0], d[2]), scales)
