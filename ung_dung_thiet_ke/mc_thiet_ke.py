"""Mô phỏng LC-IV với mẫu kiểm tra có thiết kế phức tạp, trên địa lý thật 638 huyện (W queen).
x = log mật độ dân số (chuẩn hoá, thật), x̂ = dự báo CAMS (κ≈0,41) hoặc GHSL (κ≈0,95), y = S(ρ)(βx+θWx+ε), ε iid.
Khung mẫu = 511 huyện ngoài tập huấn luyện; 127 huyện huấn luyện đã biết x -> đơn vị chắc chắn (π=1).
Thiết kế (n_L ≈ 120):
  D0 SRS 120 huyện.
  D1 Phân tầng theo tam phân vị x̂, chọn lệch về đô thị (25/35/60) -> π không bằng nhau.
  D2 Hai giai đoạn: 30/63 tỉnh (SRS), 4 huyện/tỉnh -> π_i = (30/63)(4/N_c), cụm theo tỉnh.
  D3 = D2 + nhãn có sai số khảo sát: x̃ = x + η, sd(η)=0,3.
Biến thể: naive (coi như SRS, trọng số bằng nhau), ht_unit (trọng số HT, phương sai bỏ qua cụm),
          ht_design (trọng số HT + phương sai thiết kế không chệch), ht_ultimate (cụm tối hậu, bảo thủ).
Báo cáo: trung vị sai lệch θ̂, MAD, tỷ lệ bác bỏ T_θ tại θ thật (χ²_2 và tới hạn Hotelling theo bậc tự do thiết kế),
         tỷ lệ CS_ρ bị chặn, và kiểm tra trực tiếp E[V̂] / Var(ζ) (Bổ đề 3 tổng quát).
Chạy: ../.venv/bin/python -I mc_thiet_ke.py [R] [sd_eps] [CAMS,GHSL]
"""
import os, sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import chi2, f as fdist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "g3_code"))
from g3_v2 import Geo, estimate, T_theta, omega
from g3_v2_ident import tid
from g3_design import build_design, sigma2_w

R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SIG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0          # sd của ε; nhỏ = chế độ mẫu kiểm tra chi phối
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
OUT = "ket_qua_thiet_ke.json" if SIG == 1.0 else f"ket_qua_thiet_ke_sig{SIG}.json"
K = 4; p = K - 2; C = chi2.ppf(0.95, p)
DL = os.path.join(HERE, "..", "du_lieu_o_nhiem", "ghep_huyen")
d = pd.read_csv(os.path.join(DL, "vn_districts_G3.csv")); Wd = sp.load_npz(os.path.join(DL, "W_queen_rowstd.npz"))
cams = pd.read_csv(os.path.join(HERE, "..", "ung_dung_cams", "cams_huyen_2020.csv")).sort_values("g3_row")
G = Geo(Wd); n = G.n; W = G.W
x = d.log_density.values; xc = (x - x.mean()) / x.std(); xc = xc - xc.mean()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]; NF = len(rest)
prov = d.province.values


def fit(F):
    X = np.column_stack([np.ones(n), F]); cf, *_ = np.linalg.lstsq(X[tr], xc[tr], rcond=None); return X @ cf


Z = d.log_built_share_2020.values
V = ["pm2p5", "aod550", "bcaod550", "omaod550", "suaod550", "duaod550", "tcno2", "tcco", "tcso2", "tchcho"]
PRED = {"CAMS": fit(np.log(cams[V].values)), "GHSL": fit(np.column_stack([Z, Z ** 2]))}
if ONLY: PRED = {k: PRED[k] for k in ONLY}

# khung tỉnh
P_ids = np.unique(prov); Nc = {c: int(np.sum(prov[rest] == c)) for c in P_ids}; rest_by = {c: rest[prov[rest] == c] for c in P_ids}


def crit_hot(df):
    nu = max(df, p + 1); return p * nu / (nu - p + 1) * fdist.ppf(0.95, p, nu - p + 1)


def draw_design(D, xh, rng):
    """trả về lab_audit, pi, strata, psu, M, Nc_dict, nhãn quan sát"""
    if D == "D0":
        L = rng.choice(rest, 120, replace=False); pi = np.full(120, 120 / NF)
        return L, pi, np.zeros(120), L.copy(), {0: NF}, None, xc[L]
    if D == "D1":
        q = np.quantile(xh[rest], [1 / 3, 2 / 3]); st = np.digitize(xh[rest], q)
        L, pi, S = [], [], []
        for h, nh in zip((0, 1, 2), (25, 35, 60)):
            fr = rest[st == h]; Lh = rng.choice(fr, nh, replace=False)
            L += list(Lh); pi += [nh / len(fr)] * nh; S += [h] * nh
        L = np.array(L); return L, np.array(pi), np.array(S), L.copy(), {h: int(np.sum(st == h)) for h in (0, 1, 2)}, None, xc[L]
    if D in ("D2", "D3"):
        ps = rng.choice(P_ids, 30, replace=False); L, pi, U = [], [], []
        for c in ps:
            Lc = rng.choice(rest_by[c], 4, replace=False); L += list(Lc); pi += [30 / len(P_ids) * 4 / Nc[c]] * 4; U += [c] * 4
        L = np.array(L); xo = xc[L] + (rng.normal(0, 0.3, len(L)) if D == "D3" else 0)
        return L, np.array(pi), np.zeros(len(L)), np.array(U), {0: len(P_ids)}, Nc, xo


VARIANTS = {"D0": ["ht_design"], "D1": ["naive", "ht_design"], "D2": ["naive", "ht_unit", "ht_design", "ht_ultimate"],
            "D3": ["ht_unit", "ht_design", "ht_ultimate"]}


def make_S(var, y, xh, L, pi, st, psu, M, Ncd, xo):
    lab = np.concatenate([tr, L]); xl = np.concatenate([xc[tr], xo]); k = len(tr)
    cert_s = np.full(k, -1); cert_p = np.arange(-k, 0)
    if var == "naive":
        pi2 = np.full(len(L), len(L) / NF)
        return build_design(G, y, xh, lab, xl, np.r_[np.ones(k), pi2], np.r_[cert_s, np.zeros(len(L))], np.r_[cert_p, L], "two_stage", M={0: NF})
    if var == "ht_unit":
        return build_design(G, y, xh, lab, xl, np.r_[np.ones(k), pi], np.r_[cert_s, st], np.r_[cert_p, L], "ultimate")
    if var == "ht_design":
        return build_design(G, y, xh, lab, xl, np.r_[np.ones(k), pi], np.r_[cert_s, st], np.r_[cert_p, psu], "two_stage", M=M, Nc=Ncd)
    if var == "ht_ultimate":
        return build_design(G, y, xh, lab, xl, np.r_[np.ones(k), pi], np.r_[cert_s, st], np.r_[cert_p, psu], "ultimate")


out = {}; true = np.array([0.4, 1.0, 0.5]); rho, beta, theta = true
for pn, xh0 in PRED.items():
    xh = xh0 - xh0.mean()
    ymean = G.solve(rho, beta * xc + theta * (W @ xc))
    # a_i tại giá trị thật cho kiểm tra E[V̂] vs Var(ζ)
    cols = [xh]
    for _ in range(K - 1): cols.append(W @ cols[-1])
    H = np.column_stack(cols); WtH = W.T @ H
    for D, vars_ in VARIANTS.items():
        rng = np.random.default_rng(2026); t0 = time.time()
        rec = {v: dict(th=[], rej=[], rejh=[], bnd=[], zeta=[], Vd=[]) for v in vars_}
        for r in range(R):
            y = ymean + G.solve(rho, SIG * rng.standard_normal(n)); y = y - y.mean()
            L, pi, st, psu, M, Ncd, xo = draw_design(D, xh, rng)
            for v in vars_:
                S = make_S(v, y, xh, L, pi, st, psu, M, Ncd, xo)
                dd, _ = estimate(S); s2 = sigma2_w(S, dd)
                T = T_theta(S, theta, s2, dd[1], max(abs(dd[1]), .5))
                o = rec[v]; o["th"].append(dd[2]); o["rej"].append(T > C); o["rejh"].append(T > crit_hot(S["df"]))
                o["bnd"].append(tid(S["G"][:, 0], S["G"][:, 1:], lambda b, t, S=S: omega(S, b, t, 0.0), (dd[1], dd[2])) > C)
                # ζ(β,θ) thật = (sai lệch phần hiệu chỉnh) — so với V̂ của S tại (β,θ)
                corr = S["G"][:, 1:] @ np.array([beta, theta]) - (H.T @ xc * beta + WtH.T @ xc * theta)
                o["zeta"].append(corr); o["Vd"].append(np.diag(omega(S, beta, theta, 0.0)))
        res = {}
        for v, o in rec.items():
            th = np.array(o["th"]); Z_ = np.array(o["zeta"]); Vd = np.array(o["Vd"])
            res[v] = dict(theta_medbias=float(np.median(th) - theta), theta_meanbias=float(np.mean(th) - theta),
                          theta_mad=float(np.median(np.abs(th - theta))), theta_sd=float(np.std(th)),
                          size_chi2=float(np.mean(o["rej"])), size_hotelling=float(np.mean(o["rejh"])),
                          bounded=float(np.mean(o["bnd"])),
                          zeta_meanbias_rel=(np.abs(Z_.mean(0)) / Z_.std(0)).round(3).tolist(),
                          EV_over_Var=(Vd.mean(0) / Z_.var(0)).round(3).tolist())
        out[f"{pn}_{D}"] = dict(R=R, secs=round(time.time() - t0), **res)
        print(pn, D, json.dumps(out[f"{pn}_{D}"]), flush=True)
        json.dump(out, open(os.path.join(HERE, OUT), "w"), indent=1, default=float)
print("xong")
