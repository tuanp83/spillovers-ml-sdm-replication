"""Kiểm chứng Định lý thm:feasible (bản có chặn C̄ và giới hạn theo tia; kết quả bản không chặn: ket_qua_sigma.json): σ̂²_U tại δ̂_LC (cũ) vs σ̃² tại giả thuyết (mới).
Lưới rook, x ~ SAR(ψ=0,8), κ=0,7, ε iid N(0,1). Size của T_θ tại θ thật và của T_ρ tại ρ thật; lực của T_θ tại θ±0,5.
Chạy: ../.venv/bin/python -I mc_sigma.py [R]"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3_code"))
from scipy.stats import chi2
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta, T_rho
from g3_sigma_h import T_theta_h, T_rho_h, cap
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
C = chi2.ppf(.95, 2)
DES = [("30x30 nL90", 30, 90, (0.0, 1.0, 0.0)), ("30x30 nL90", 30, 90, (0.4, 1.0, 0.0)),
       ("30x30 nL90", 30, 90, (0.5, 1.0, -0.5)), ("30x30 nL90", 30, 90, (0.4, 1.0, 0.5)),
       ("30x30 nL90", 30, 90, (0.4, 0.3, 0.0)),
       ("50x50 nL100", 50, 100, (0.0, 1.0, 0.0)), ("50x50 nL100", 50, 100, (0.4, 1.0, 0.5))]
out = {}
for lab_, m, nl, (rho, beta, theta) in DES:
    G = Geo(Geo.rook(m)); n = G.n; Wn = np.linalg.norm(G.W.toarray(), 2); rng = np.random.default_rng(7); t0 = time.time()
    rec = {k: [] for k in ("th_old", "th_new", "rho_old", "rho_new", "pw_old+", "pw_new+", "pw_old-", "pw_new-", "s2_old", "s2_new", "far_old", "far_new")}
    for _ in range(R):
        x = G.solve(0.8, rng.standard_normal(n)); x = (x - x.mean()) / x.std()
        nu = rng.standard_normal(n) * np.sqrt(0.3 / 0.7); xh = 0.7 * (x + nu)
        y = G.solve(rho, beta * x + theta * (G.W @ x) + rng.standard_normal(n))
        L = rng.choice(n, nl, replace=False); S = build(G, y, xh, x, L); d, _ = estimate(S)
        s2 = sigma2_U(S, d); cb = cap(S, Wn); sc = (max(abs(d[0]), .5), max(abs(d[1]), .5)); sb = max(abs(d[1]), .5)
        rec["s2_old"].append(s2)
        rec["s2_new"].append(float(np.mean((S["yl"] - rho * S["Wyl"] - beta * S["xl"] - theta * S["Wphil"]) ** 2)))
        rec["th_old"].append(T_theta(S, theta, s2, d[1], sb) > C); rec["th_new"].append(T_theta_h(S, theta, d, sc, cb) > C)
        rec["rho_old"].append(T_rho(S, rho, s2, d, (sb, max(abs(d[2]), .5))) > C)
        rec["rho_new"].append(T_rho_h(S, rho, d, (sb, max(abs(d[2]), .5)), cb) > C)
        for sg, k in ((0.5, "+"), (-0.5, "-")):
            rec["pw_old" + k].append(T_theta(S, theta + sg, s2, d[1], sb) > C)
            rec["pw_new" + k].append(T_theta_h(S, theta + sg, d, sc, cb) > C)
        rec["far_old"].append(T_theta(S, theta + 50, s2, d[1], sb) > C); rec["far_new"].append(T_theta_h(S, theta + 50, d, sc, cb) > C)
    o = {k: float(np.mean(v)) if not k.startswith("s2") else float(np.median(v)) for k, v in rec.items()}
    o.update(R=R, secs=round(time.time() - t0), manifold=bool(abs(theta + rho * beta) < 1e-12))
    key = f"{lab_} (rho,beta,theta)=({rho},{beta},{theta})"; out[key] = o; print(key, json.dumps(o), flush=True)
    json.dump(out, open("ket_qua_sigma_cap.json", "w"), indent=1)
print("xong")
