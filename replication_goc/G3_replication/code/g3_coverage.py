"""Coverage of LC-IV sandwich variance (Theorem 4) — 'audit whitening':
Omega = sigma^2 H'H + (1-pi)/pi^2 * sum_{i in L} b_i b_i' e_i^2,  b = beta H + theta W'H.
sigma^2 treated as known (=1) here; feasible sigma^2 estimation is left for the paper."""
import numpy as np, sys, json
from scipy.linalg import lu_factor, lu_solve
from g3_sim import rook_W

def run(m, psi, kappa, R, rng, pi=0.10, pi_tr=0.10, rho=0.4, beta=1.0, theta=0.5, K=4):
    W = rook_W(m); n = m * m; I = np.eye(n)
    Wp = [np.linalg.matrix_power(W, k) for k in range(K)]
    Lx = lu_factor(I - psi * W); Ly = lu_factor(I - rho * W)
    Ainv = np.linalg.inv(I - psi * W); s2 = np.mean(np.sum(Ainv ** 2, 1))
    tau = np.sqrt(s2 * (1 - kappa) / kappa)
    true = np.array([rho, beta, theta]); z = 1.959964
    cover = np.zeros(3); est = []
    for r in range(R):
        x = lu_solve(Lx, rng.standard_normal(n))
        Z = x + tau * rng.standard_normal(n)
        y = lu_solve(Ly, beta * x + theta * (W @ x) + rng.standard_normal(n))
        perm = rng.permutation(n); ntr, nl = int(pi_tr * n), int(pi * n)
        tr, lab = perm[:ntr], perm[ntr:ntr + nl]
        a, b = np.linalg.lstsq(np.column_stack([np.ones(ntr), Z[tr]]), x[tr], rcond=None)[0]
        xh = a + b * Z
        H = np.column_stack([Wp[k] @ xh for k in range(K)])
        e = np.zeros(n); e[lab] = x[lab] - xh[lab]; pih = nl / n
        Wy = W @ y
        G = np.column_stack([H.T @ Wy, H.T @ xh + (H.T @ e) / pih,
                             H.T @ (W @ xh) + ((W.T @ H).T @ e) / pih])
        A = H.T @ H; Ai = np.linalg.inv(A)
        Bm = np.linalg.inv(G.T @ Ai @ G)
        d = Bm @ G.T @ Ai @ (H.T @ y)
        bvec = d[1] * H + d[2] * (W.T @ H)
        Om = 1.0 * A + (1 - pih) / pih ** 2 * (bvec[lab].T * e[lab] ** 2) @ bvec[lab]
        V = Bm @ G.T @ Ai @ Om @ Ai @ G @ Bm
        se = np.sqrt(np.diag(V))
        cover += (np.abs(d - true) <= z * se)
        est.append(d)
    E = np.array(est)
    return dict(n=n, psi=psi, kappa=kappa, coverage=(cover / R).tolist(),
                mc_sd=E.std(0).tolist())

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(11)
    out = []
    for (m, kappa) in [(30, 0.9), (30, 0.7), (50, 0.9), (50, 0.7)]:
        o = run(m, 0.8, kappa, R, rng); out.append(o)
        print(o, flush=True)
    json.dump(out, open("g3_coverage.json", "w"), indent=1)
