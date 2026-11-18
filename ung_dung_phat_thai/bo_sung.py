"""Bổ sung cho Mục 8: (i) tập AR 95% cho theta và rho từ một mẫu kiểm tra; (ii) T_id; (iii) tự tương quan bậc 1
của tín hiệu và nhiễu ảnh (cho dự báo của luật dấu); (iv) chênh lệch theta SLX ngây thơ - oracle với SE bootstrap."""
import os, sys, json, numpy as np
sys.argv = [sys.argv[0], "0"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phan_tich.py")).read().split("out = dict(")[0])
from g3_v2 import T_rho
E = E_of("nitrogen_oxides", 2020); c = 0.01 * np.median(E[E > 0]); y = dm(np.log(E + c))
res = {}
# (iii) luật dấu: r1 tín hiệu vs nhiễu (e = x - xh), theo định nghĩa Moran-type trên W
e = xc - xh
res["r1_signal"] = moran(xc); res["r1_noise"] = moran(e); res["corr_e_xh"] = float(np.corrcoef(e, xh)[0, 1])
# (iv) bootstrap chênh lệch SLX
rng = np.random.default_rng(7); diffs = []
def slx_th(v, yy, idx):
    Xs = np.column_stack([np.ones(n), v, W @ v])[idx]; return np.linalg.lstsq(Xs, yy[idx], rcond=None)[0][2]
for _ in range(999):
    idx = rng.integers(0, n, n); diffs.append(slx_th(xh0, y, idx) - slx_th(xc0, y, idx))
full = slx_th(xh0, y, np.arange(n)) - slx_th(xc0, y, np.arange(n))
res["slx_diff"] = float(full); res["slx_diff_boot_se"] = float(np.std(diffs)); res["slx_diff_ci"] = np.percentile(diffs, [2.5, 97.5]).tolist()
# (i)-(ii) một mẫu kiểm tra
for nl in (60, 120):
    lab = np.random.default_rng(1).choice(rest, nl, replace=False)
    S = build(G, y, xh0, xc0, lab); de, P = estimate(S); s2 = sigma2_U(S, de)
    Tid = tid(S["G"][:, 0], S["G"][:, 1:], lambda b, t: omega(S, b, t, 0.0), (de[1], de[2]))
    TG = np.linspace(-6, 6, 241); th_acc = [t for t in TG if T_theta(S, t, s2, de[1], max(abs(de[1]), .5)) <= C2]
    RG = np.linspace(-3, 3, 121); sc = (max(abs(de[1]), .5), max(abs(de[2]), .5))
    rh_acc = [r for r in RG if T_rho(S, r, s2, de, sc) <= C2]
    seg = lambda a, g: None if not a else [float(min(a)), float(max(a)), bool(min(a) > g[0] and max(a) < g[-1]), len(a) == (round((max(a)-min(a))/(g[1]-g[0]))+1)]
    res[f"nl{nl}"] = dict(lciv=de.tolist(), T_id=float(Tid), c=float(C2), cs_theta=seg(th_acc, TG), cs_rho=seg(rh_acc, RG))
    print(nl, json.dumps(res[f"nl{nl}"]), flush=True)
json.dump(res, open(os.path.join(HERE, "ket_qua_bo_sung.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("nl")}, default=float))
