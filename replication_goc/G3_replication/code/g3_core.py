"""Core machinery for G3 extensions (items 1, 2, 4, 5).

Generalised LC-IV with control variate phi (any predictor trained independently of the audit):
    c0 = H'phi + (1/pi) sum_{i in L} H_i (x_i - phi_i)        (estimates E[H'x])
    c1 = (W'H)'phi + (1/pi) sum_{i in L} (W'H)_i (x_i - phi_i) (estimates E[H'Wx])
    G  = [H'Wy, c0, c1],  delta = (G'A^-1 G)^-1 G'A^-1 H'y.
phi = xh gives LC-IV; phi = lambda*xh gives PPI++; phi = SAP prediction gives CV-SAP.
Omega(beta,theta) = sigma2*A + df*(1-pi)/pi^2 * sum_L b_i b_i' d_i^2,  b = beta H + theta W'H, d = x - phi.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from g3_sim import rook_W


class Lattice:
    def __init__(self, m, psi, rho, K=4):
        W = rook_W(m); n = m * m; I = np.eye(n)
        self.W, self.n, self.K, self.m = W, n, K, m
        self.Wp = [np.linalg.matrix_power(W, k) for k in range(max(K, 3))]
        self.Lx = lu_factor(I - psi * W); self.Ly = lu_factor(I - rho * W)
        Ainv = np.linalg.inv(I - psi * W)
        self.s2 = float(np.mean(np.sum(Ainv ** 2, 1)))
        self.om = np.real(np.linalg.eigvals(W))
        self.rho = rho

    def draw_x(self, rng):
        return lu_solve(self.Lx, rng.standard_normal(self.n))

    def draw_y(self, x, beta, theta, rng, eps=None):
        if eps is None:
            eps = rng.standard_normal(self.n)
        return lu_solve(self.Ly, beta * x + theta * (self.W @ x) + eps), eps

    def impacts(self, d):
        rho, beta, theta = d
        if not (-1 < rho < 1):
            return np.array([np.nan, np.nan, np.nan])
        adi = np.mean((beta + theta * self.om) / (1 - rho * self.om))
        ati = (beta + theta) / (1 - rho)
        return np.array([adi, ati - adi, ati])


def ols_fit(Xcols, target, idx):
    X = np.column_stack([np.ones(len(target))] + list(Xcols))
    coef = np.linalg.lstsq(X[idx], target[idx], rcond=None)[0]
    return X @ coef


def build(L, y, xh, phi, x, lab, w=None):
    """Sufficient statistics of the LC moment system. w: audit weights (bootstrap counts)."""
    W, n, K = L.W, L.n, L.K
    H = np.column_stack([L.Wp[k] @ xh for k in range(K)])
    WtH = W.T @ H
    nl = len(lab); pih = nl / n
    if w is None:
        w = np.ones(nl)
    dl = x[lab] - phi[lab]
    Hl, Wl = H[lab], WtH[lab]
    c0 = H.T @ phi + (Hl.T @ (w * dl)) / pih
    c1 = WtH.T @ phi + (Wl.T @ (w * dl)) / pih
    G = np.column_stack([H.T @ (W @ y), c0, c1])
    q = w * dl ** 2
    u = dl[:, None] * Hl; v = dl[:, None] * Wl
    uc = u - u.mean(0); vc = v - v.mean(0)
    S = dict(Chh=uc.T @ uc, Chw=uc.T @ vc, Cww=vc.T @ vc, cfac=nl / max(nl - 1, 1), mode="bern",
             H=H, WtH=WtH, G=G, hy=H.T @ y, A=H.T @ H, pih=pih, nl=nl,
             kfac=(1 - pih) / pih ** 2,
             Mhh=(Hl.T * q) @ Hl, Mhw=(Hl.T * q) @ Wl, Mww=(Wl.T * q) @ Wl,
             lab=lab, dl=dl, yl=y[lab], Wyl=(W @ y)[lab], xl=x[lab], Wphil=(W @ phi)[lab])
    return S


def estimate(S):
    Ai = np.linalg.inv(S["A"]); G = S["G"]
    Bm = np.linalg.inv(G.T @ Ai @ G)
    P = Bm @ G.T @ Ai
    return P @ S["hy"], P


def omega(S, beta, theta, sigma2, df=1.0):
    """beta, theta scalars or arrays (N,) -> (K,K) or (N,K,K)."""
    beta = np.asarray(beta, float); theta = np.asarray(theta, float)
    if S.get("mode", "bern") == "srs":      # Hajek / SRS: centred audit covariance, FPC (1-f)
        Mhh, Mhw, Mww, fac = S["Chh"], S["Chw"] + S["Chw"].T, S["Cww"], S["kfac"] * S["cfac"]
    else:                                  # Horvitz-Thompson form (uncentred)
        Mhh, Mhw, Mww, fac = S["Mhh"], S["Mhw"] + S["Mhw"].T, S["Mww"], S["kfac"]
    aud = (beta[..., None, None] ** 2) * Mhh + (beta * theta)[..., None, None] * Mhw \
        + (theta[..., None, None] ** 2) * Mww
    sigma2 = np.asarray(sigma2, float)
    return sigma2[..., None, None] * S["A"] + df * fac * aud


def wald_se(S, d, P, sigma2, df=1.0):
    Om = omega(S, d[1], d[2], sigma2, df)
    V = P @ Om @ P.T
    return V


def sigma2_U(L, y, x, phi, lab, d):
    rho, beta, theta = d
    rho = float(np.clip(rho, -0.99, 0.99))
    r = y[lab] - rho * (L.W @ y)[lab] - beta * x[lab] - theta * (L.W @ phi)[lab]
    return float(np.mean(r ** 2))


# ----------------------------------------------------- batched AR statistics
def _gls_batch(U, V, Oi):
    """U (N,K), V (K,p) or (N,K,p), Oi (N,K,K). Returns coef (N,p), resid (N,K), AR (N,)."""
    if V.ndim == 2:
        V = np.broadcast_to(V, (U.shape[0],) + V.shape)
    VtOi = np.einsum("nkp,nkl->npl", V, Oi)
    M = VtOi @ V
    rhs = np.einsum("npk,nk->np", VtOi, U)
    coef = np.linalg.solve(M, rhs[..., None])[..., 0]
    g = U - np.einsum("nkp,np->nk", V, coef)
    ar = np.einsum("nk,nkl,nl->n", g, Oi, g)
    return coef, g, ar


def ar_rho(S, rhos, sigma2, b0, t0, df=1.0, iters=2):
    """AR(rho0) = min_{beta,theta} g'Omega(beta,theta)^-1 g, Omega iterated under H0."""
    G, hy = S["G"], S["hy"]
    U = hy[None, :] - rhos[:, None] * G[None, :, 0]
    V = G[:, 1:]
    beta = np.full(len(rhos), b0); theta = np.full(len(rhos), t0)
    for _ in range(iters):
        Oi = np.linalg.inv(omega(S, beta, theta, sigma2, df))
        coef, g, ar = _gls_batch(U, V, Oi)
        beta, theta = coef[:, 0], coef[:, 1]
    Oi = np.linalg.inv(omega(S, beta, theta, sigma2, df))
    return np.einsum("nk,nkl,nl->n", g, Oi, g)


def ar_theta_given_rho(S, rhos, thetas, sigma2, b0, df=1.0, iters=2):
    """theta and rho fixed (arrays of same length N); beta free."""
    G, hy = S["G"], S["hy"]
    U = hy[None, :] - rhos[:, None] * G[None, :, 0] - thetas[:, None] * G[None, :, 2]
    V = G[:, 1:2]
    beta = np.full(len(rhos), b0)
    for _ in range(iters):
        Oi = np.linalg.inv(omega(S, beta, thetas, sigma2, df))
        coef, g, ar = _gls_batch(U, V, Oi)
        beta = coef[:, 0]
    Oi = np.linalg.inv(omega(S, beta, thetas, sigma2, df))
    return np.einsum("nk,nkl,nl->n", g, Oi, g)


def ar_ati_given_rho(S, rhos, atis, sigma2, b0, df=1.0, iters=2):
    """ATI=a and rho fixed: theta = a(1-rho) - beta, beta free."""
    G, hy = S["G"], S["hy"]
    U = hy[None, :] - rhos[:, None] * G[None, :, 0] - (atis * (1 - rhos))[:, None] * G[None, :, 2]
    V = (G[:, 1] - G[:, 2])[:, None]
    beta = np.full(len(rhos), b0)
    for _ in range(iters):
        theta = atis * (1 - rhos) - beta
        Oi = np.linalg.inv(omega(S, beta, theta, sigma2, df))
        coef, g, ar = _gls_batch(U, V, Oi)
        beta = coef[:, 0]
    theta = atis * (1 - rhos) - beta
    Oi = np.linalg.inv(omega(S, beta, theta, sigma2, df))
    return np.einsum("nk,nkl,nl->n", g, Oi, g)


def ar_beta_given_rho(S, rhos, betas, sigma2, t0, df=1.0, iters=2):
    """beta and rho fixed; theta free (strongly identified via E[H'Wx])."""
    G, hy = S["G"], S["hy"]
    U = hy[None, :] - rhos[:, None] * G[None, :, 0] - betas[:, None] * G[None, :, 1]
    V = G[:, 2:3]
    theta = np.full(len(rhos), t0)
    for _ in range(iters):
        Oi = np.linalg.inv(omega(S, betas, theta, sigma2, df))
        coef, g, ar = _gls_batch(U, V, Oi)
        theta = coef[:, 0]
    Oi = np.linalg.inv(omega(S, betas, theta, sigma2, df))
    return np.einsum("nk,nkl,nl->n", g, Oi, g)
