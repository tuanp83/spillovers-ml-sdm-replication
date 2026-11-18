"""Kiểm chứng độc lập 3 phản ví dụ của bản thẩm định (18/11/2026).
A: T_rho tại (β,θ)=0, sốc ε ~ SAR(0,9) (A6b), chế độ mẫu kiểm tra chi phối, σ̂²=1 đúng.
B: T_θ khi x trơn, trên đa tạp β+b̄(θ+ρβ)=0 (độ yếu thứ hai), so với β=1.
C: ρ̂ LC-IV trên đa tạp θ+ρβ=0: IQR theo n.
Chạy: ../../.venv/bin/python -I phan_vi_du.py [R]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "g3_code"))
from scipy.stats import chi2
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta, T_rho
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
K = 4; C2 = chi2.ppf(.95, 2); out = {}


def calib(G, x, kappa, rng):
    nu = rng.standard_normal(G.n); nu = nu / nu.std() * np.sqrt(x.var() * (1 - kappa) / kappa)
    return kappa * (x + nu) + (1 - kappa) * x.mean()


def one(G, psi, kappa, rho, beta, theta, nl, rng, eps_sar=0.0, test="theta"):
    x = G.solve(psi, rng.standard_normal(G.n)); x = (x - x.mean()) / x.std()
    xh = calib(G, x, kappa, rng)
    e = G.solve(eps_sar, rng.standard_normal(G.n)); e = e / e.std()
    y = G.solve(rho, beta * x + theta * (G.W @ x) + e)
    lab = rng.choice(G.n, nl, replace=False)
    S = build(G, y, xh, x, lab); dd, _ = estimate(S)
    if test == "theta":
        s2 = sigma2_U(S, dd); return T_theta(S, theta, s2, dd[1], max(abs(dd[1]), .5)) > C2, dd
    return T_rho(S, rho, 1.0, dd, (max(abs(dd[1]), .5), max(abs(dd[2]), .5))) > C2, dd


# A
G = Geo(Geo.rook(40)); rng = np.random.default_rng(11)
for es in (0.0, 0.9):
    rej = [one(G, 0.8, 0.7, 0.4, 0.0, 0.0, 40, rng, es, "rho")[0] for _ in range(R)]
    out[f"A_Trho_b0_epsSAR{es}"] = float(np.mean(rej)); print("A", es, np.mean(rej), flush=True)


# B: b̄ = hệ số hồi quy của M H'WSWx lên M H'Wx (M chiếu bỏ H'x), tính trên một bản đồ lớn
def bbar(G, psi, kappa, rho, rng, reps=20):
    v = []
    for _ in range(reps):
        x = G.solve(psi, rng.standard_normal(G.n)); x = (x - x.mean()) / x.std(); xh = calib(G, x, kappa, rng)
        H = np.column_stack([xh, G.W @ xh, G.W @ (G.W @ xh), G.W @ (G.W @ (G.W @ xh))])
        a = H.T @ x; c1 = H.T @ (G.W @ x); c2 = H.T @ (G.W @ G.solve(rho, G.W @ x))
        M = np.eye(K) - np.outer(a, a) / (a @ a); u = M @ c1; w = M @ c2; v.append(u @ w / (u @ u))
    return float(np.median(v))
for psi in (0.95, 0.8):
    b = bbar(G, psi, 0.7, 0.4, rng); bstar = -0.5 * b / (1 + 0.4 * b)
    for beta, tag in ((bstar, "beta_star"), (1.0, "beta1")):
        for nl in (20, 40, 160):
            rej = [one(G, psi, 0.7, 0.4, beta, 0.5, nl, rng)[0] for _ in range(R)]
            out[f"B_psi{psi}_{tag}_nl{nl}"] = dict(bbar=b, beta=beta, size=float(np.mean(rej)))
            print("B", psi, tag, round(beta, 3), nl, np.mean(rej), flush=True)

# C
for m in (20, 40, 80):
    Gm = Geo(Geo.rook(m)); nl = int(0.2 * m * m)
    for theta, tag in ((-0.4, "manifold"), (0.5, "identified")):
        r = [one(Gm, 0.8, 0.7, 0.4, 1.0, theta, nl, rng)[1][0] for _ in range(min(R, 300))]
        q = np.percentile(r, [25, 75]); out[f"C_n{m*m}_{tag}"] = float(q[1] - q[0]); print("C", m * m, tag, q[1] - q[0], flush=True)
json.dump(out, open("phan_vi_du.json", "w"), indent=1)
