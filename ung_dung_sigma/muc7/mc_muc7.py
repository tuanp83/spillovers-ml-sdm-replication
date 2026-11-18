"""Chạy lại Bảng tab:size và tab:len (Mục 7) với kiểm định mới (σ̃² tại giả thuyết, có chặn; thm:feasible),
so với kiểm định cũ (σ̂²_U tại δ̂_LC) TRÊN CÙNG CÁC MẪU. Thiết kế lấy nguyên từ G3_replication/code/g3_v2_mc.py.
T_ATI giữ nguyên bản cũ (chưa có bản σ̃² tại giả thuyết cho ATI).
Chạy: ../../.venv/bin/python -I mc_muc7.py size 2000   |   ... len 300
Song song: multiprocessing, 11 tiến trình, BLAS 1 luồng."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
REP = os.path.join(ROOT, "replication_goc", "G3_replication", "code")
sys.path.insert(0, os.path.join(ROOT, "g3_code")); sys.path.insert(1, REP)
from scipy.stats import chi2, f as fdist
from g3_v2 import draw, build, estimate, sigma2_U, T_theta, T_rho, T_ati
from g3_sigma_h import T_theta_h, T_rho_h, cap
from g3_v2_mc import DESIGNS, SETS, geo
K = 4; p = K - 2; C2 = chi2.ppf(.95, p); C1 = chi2.ppf(.95, K - 1)
_G = {}


def G_of(tag):
    if tag not in _G:
        g = geo(tag); _G[tag] = (g, float(np.linalg.norm(g.W.toarray(), 2)))
    return _G[tag]


def hot(nl):
    return p * (nl - 1) / (nl - p) * fdist.ppf(.95, p, nl - p)


def size_task(args):
    nm, R, seed = args
    g, psi, kap, rho, beta, theta, nl, phn, eps = DESIGNS[nm]; G, Wn = G_of(g); rng = np.random.default_rng(seed)
    ati = (beta + theta) / (1 - rho); out = {k: [] for k in ("th_old", "th_new", "rh_old", "rh_new", "at_old")}; t0 = time.time()
    for _ in range(R):
        x, y, xh, lab = draw(G, psi, kap, rho, beta, theta, nl, rng, phn, eps)
        S = build(G, y, xh, x, lab); d, _ = estimate(S); s2 = sigma2_U(S, d); cb = cap(S, Wn)
        sc = (max(abs(d[0]), .5), max(abs(d[1]), .5), max(abs(d[2]), .5))
        out["th_old"].append(T_theta(S, theta, s2, d[1], sc[1])); out["rh_old"].append(T_rho(S, rho, s2, d, (sc[1], sc[2])))
        out["at_old"].append(T_ati(S, ati, s2, d, (sc[0], sc[1])))
        out["th_new"].append(T_theta_h(S, theta, d, (sc[0], sc[1]), cb)); out["rh_new"].append(T_rho_h(S, rho, d, (sc[1], sc[2]), cb))
    res = dict(name=nm, R=R, nl=nl, rho=rho, beta=beta, theta=theta, psi=psi, kappa=kap, eps=eps, secs=round(time.time() - t0))
    for k, v in out.items():
        v = np.array(v); res[f"{k}_chi2"] = float(np.mean(v > C2)); res[f"{k}_hot"] = float(np.mean(v > hot(nl)))
    return res


TG = np.linspace(-3, 4, 141); RG = np.linspace(-3, 3, 121)


def len_task(args):
    nm, R, seed = args
    g, psi, kap, rho, beta, theta, nl, phn, eps = DESIGNS[nm]; G, Wn = G_of(g); rng = np.random.default_rng(seed)
    rec = {k: [] for k in ("lt2_old", "lt2_new", "bt2_old", "bt2_new", "lt1_old", "lt1_new", "bt1_old", "bt1_new",
                           "lr_old", "lr_new", "br_old", "br_new")}
    st, sr = TG[1] - TG[0], RG[1] - RG[0]
    for _ in range(R):
        x, y, xh, lab = draw(G, psi, kap, rho, beta, theta, nl, rng, phn, eps)
        S = build(G, y, xh, x, lab); d, _ = estimate(S); s2 = sigma2_U(S, d); cb = cap(S, Wn)
        sb, stt = max(abs(d[1]), .5), max(abs(d[2]), .5); sc = (max(abs(d[0]), .5), sb)
        T = {"old": np.array([T_theta(S, t, s2, d[1], sb, N=81) for t in TG]),
             "new": np.array([T_theta_h(S, t, d, sc, cb) for t in TG])}
        Tr = {"old": np.array([T_rho(S, r, s2, d, (sb, stt)) for r in RG]),
              "new": np.array([T_rho_h(S, r, d, (sb, stt), cb) for r in RG])}
        for v in ("old", "new"):
            for c, kk in ((C2, "2"), (C1, "1")):
                a = T[v] <= c; rec[f"lt{kk}_{v}"].append(a.sum() * st); rec[f"bt{kk}_{v}"].append(bool(a.any() and not a[0] and not a[-1]))
            a = Tr[v] <= C2; rec[f"lr_{v}"].append(a.sum() * sr); rec[f"br_{v}"].append(bool(not a[0] and not a[-1]))
    return nm, rec


if __name__ == "__main__":
    kind, R = sys.argv[1], int(sys.argv[2])
    with Pool(11) as pool:
        if kind == "size":
            names = SETS["A"] + SETS["B"] + SETS["C"]
            tasks = [(nm, R, 5000 + i) for i, nm in enumerate(names)]
            res = []
            for o in pool.imap_unordered(size_task, tasks):
                res.append(o); print(json.dumps(o), flush=True)
                json.dump(res, open(os.path.join(HERE, "size_muc7.json"), "w"), indent=1)
        else:
            names = ["base_fixedpi", "base_audit", "smooth_k9_audit", "vn_base", "nonid_audit"]; nch = 4
            tasks = [(nm, R // nch, 7000 + 10 * i + c) for i, nm in enumerate(names) for c in range(nch)]
            agg = {nm: None for nm in names}
            for nm, rec in pool.imap_unordered(len_task, tasks):
                agg[nm] = rec if agg[nm] is None else {k: agg[nm][k] + rec[k] for k in rec}
                print(nm, len(agg[nm]["lt2_old"]), flush=True)
            out = []
            for nm in names:
                r = agg[nm]; o = dict(name=nm, R=len(r["lt2_old"]))
                for k, v in r.items():
                    o[k] = float(np.median(v)) if k.startswith("l") else float(np.mean(v))
                out.append(o)
            json.dump(out, open(os.path.join(HERE, "len_muc7.json"), "w"), indent=1); print(json.dumps(out, indent=1))
