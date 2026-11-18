"""Kiểm tra trần L*: T̃_θ(θ0) ≤ L* với mọi θ0 -> tập tin cậy có thể = R khi √n‖b‖ nhỏ. So với T_θ cũ."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3_code"))
from scipy.stats import chi2
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta
from g3_sigma_h import T_theta_h, cap
C = chi2.ppf(.95, 2); rng = np.random.default_rng(5)
for m, nl, beta in ((20, 40, 0.1), (20, 40, 0.3), (20, 40, 1.0), (30, 90, 0.3)):
    G = Geo(Geo.rook(m)); n = G.n; Wn = np.linalg.norm(G.W.toarray(), 2); res = {k: [] for k in ("far_old", "far_new", "far2_new", "size_new", "far_cap", "far2_cap", "size_cap", "size_old")}
    for _ in range(1000):
        x = G.solve(0.8, rng.standard_normal(n)); x = (x - x.mean()) / x.std(); xh = 0.7 * (x + rng.standard_normal(n) * np.sqrt(.3 / .7))
        y = G.solve(0.4, beta * x + rng.standard_normal(n)); L = rng.choice(n, nl, replace=False)
        S = build(G, y, xh, x, L); d, _ = estimate(S); s2 = sigma2_U(S, d); sc = (max(abs(d[0]), .5), max(abs(d[1]), .5))
        res["far_old"].append(T_theta(S, 50.0, s2, d[1], max(abs(d[1]), .5)) > C)
        res["far_new"].append(T_theta_h(S, 50.0, d, sc) > C); res["far2_new"].append(T_theta_h(S, 5.0, d, sc) > C)
        res["size_new"].append(T_theta_h(S, 0.0, d, sc) > C); cb = cap(S, Wn)
        res["far_cap"].append(T_theta_h(S, 50.0, d, sc, cb) > C); res["far2_cap"].append(T_theta_h(S, 5.0, d, sc, cb) > C)
        res["size_cap"].append(T_theta_h(S, 0.0, d, sc, cb) > C); res["size_old"].append(T_theta(S, 0.0, s2, d[1], max(abs(d[1]), .5)) > C)
    print(f"n={n} nL={nl} beta={beta} sqrt(n)|b|={np.sqrt(n)*beta:.0f}", {k: round(float(np.mean(v)), 3) for k, v in res.items()}, flush=True)
