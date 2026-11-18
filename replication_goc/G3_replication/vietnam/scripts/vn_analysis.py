"""Vietnam application of G3 (district level, ~650 units).
y  = Meta Relative Wealth Index (district mean of 2.4 km grid; ML prediction from satellite + connectivity data)
x  = log population density, official (government commune-merger dossiers via vi.wikipedia; pop / area)
Z  = satellite image signal: log GHSL built-up surface share (2020; 2010 as pre-period image)
xh = predictor of x trained on a random training split (own-district receptive field)
The audit is a random subsample of the non-training districts with x revealed; because x is known
everywhere, the full-information (oracle) SDM is available as a benchmark.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd, geopandas as gpd
from types import SimpleNamespace
from scipy.stats import chi2, norm
from libpysal.weights import Queen
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from g3_core import build, estimate, wald_se, sigma2_U, ar_rho, ar_theta_given_rho, ar_ati_given_rho, ols_fit
from g3_sim import moran_test
from g3_srs_mcc import sc_crit

rng = np.random.default_rng(2026)
g = pd.read_pickle("districts_full.pkl"); g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
ISLANDS = {"Hoàng Sa", "Trường Sa", "Bạch Long Vĩ", "Cồn Cỏ", "Phú Quý", "Côn Đảo", "Lý Sơn", "Kiên Hải", "Cô Tô", "Cát Hải", "Vân Đồn", "Phú Quốc"}
keep = (g.area_ratio.between(0.8, 1.25)) & (g.popn > 0) & g.rwi_mean.notna() & (~g.name.isin(ISLANDS))
d = g[keep].reset_index(drop=True)
wq = Queen.from_dataframe(d, use_index=False, silence_warnings=True)
iso = [i for i, nb in wq.neighbors.items() if len(nb) == 0]
d = d.drop(index=iso).reset_index(drop=True)
wq = Queen.from_dataframe(d, use_index=False, silence_warnings=True); wq.transform = "r"
W = wq.full()[0]; n = len(d)
K = 4
L = SimpleNamespace(W=W, n=n, K=K, Wp=[np.linalg.matrix_power(W, k) for k in range(K)])
om = np.real(np.linalg.eigvals(W))

x = np.log(d.popn / d.area_km2).values
y = d.rwi_mean.values
Z20 = np.log(d.built_share_2020.values + 1e-4); Z10 = np.log(d.built_share_2010.values + 1e-4)
out = {"n": int(n), "dropped_isolated": len(iso)}

def moran_I(v):
    v = v - v.mean(); return float(v @ W @ v / (v @ v))
out["moran_x"], out["moran_y"], out["moran_Z20"] = moran_I(x), moran_I(y), moran_I(Z20)

# ---------- training split and predictors
perm = rng.permutation(n); ntr = int(0.2 * n)
tr, rest = perm[:ntr], perm[ntr:]
def predictor(Z):
    return ols_fit([Z, Z ** 2], x, tr)
xh = predictor(Z20); xh10 = predictor(Z10)
xt = ols_fit([xh, W @ xh, L.Wp[2] @ xh], x, tr)          # SAP control variate
def r2(a): return float(1 - np.var(x[rest] - a[rest]) / np.var(x[rest]))
e = x - xh
out["kappa_xh2020"], out["kappa_xh2010"], out["kappa_sap"] = r2(xh), r2(xh10), r2(xt)
out["corr_e_xh"] = float(np.corrcoef(e[rest], xh[rest])[0, 1])
out["corr_e_Wxh"] = float(np.corrcoef(e[rest], (W @ xh)[rest])[0, 1])
kap = out["kappa_xh2020"]
nL_req = int(np.ceil(500 * (1 - kap) / kap))
nL = int(min(max(nL_req, 60), len(rest)))
out["audit_rule_nL_required"] = nL_req; out["audit_nL_used"] = nL; out["audit_pool"] = int(len(rest))

# ---------- gamma test (A1): Z ~ 1 + x + y on training labels, HC SE
def gamma_test(Z, idx):
    X = np.column_stack([np.ones(len(idx)), x[idx], y[idx]])
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Z[idx]; u = Z[idx] - X @ b
    V = XtXi @ (X.T * u ** 2) @ X @ XtXi
    return float(b[2]), float(b[2] / np.sqrt(V[2, 2]))
for nm, Z in (("2020", Z20), ("2010", Z10)):
    gm, t = gamma_test(Z, tr); out[f"gamma_{nm}_train"] = dict(gamma=gm, t=t, p=float(2 * norm.sf(abs(t))))
    gm, t = gamma_test(Z, np.arange(n)); out[f"gamma_{nm}_alllabels"] = dict(gamma=gm, t=t, p=float(2 * norm.sf(abs(t))))

# ---------- oracle / naive 2SLS
def tsls(v):
    H = np.column_stack([L.Wp[k] @ v for k in range(K)])
    Q = np.column_stack([W @ y, v, W @ v]); Ai = np.linalg.inv(H.T @ H)
    Bm = np.linalg.inv(Q.T @ H @ Ai @ H.T @ Q); dd = Bm @ Q.T @ H @ Ai @ H.T @ y
    u = y - Q @ dd; s2 = u @ u / (n - 3)
    return dd, np.sqrt(np.diag(s2 * Bm))
def impacts(dd):
    r, b, t = dd
    if not -1 < r < 1: return [np.nan] * 3
    adi = float(np.mean((b + t * om) / (1 - r * om))); ati = float((b + t) / (1 - r)); return [adi, ati - adi, ati]
for nm, v in (("oracle", x), ("naive_2020", xh), ("naive_2010", xh10)):
    dd, se = tsls(v); out[nm] = dict(delta=dd.tolist(), se=se.tolist(), impacts=impacts(dd))
# SLX and Moran (non-spatial OLS residual) comparisons
for nm, v in (("oracle", x), ("naive_2020", xh)):
    X = np.column_stack([np.ones(n), v, W @ v]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b; s2 = u @ u / (n - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    out[f"slx_{nm}"] = dict(beta=float(b[1]), theta=float(b[2]), se_theta=float(se[2]))
    Xm = (v - v.mean())[:, None]; yc = y - y.mean()
    res = yc - Xm @ np.linalg.lstsq(Xm, yc, rcond=None)[0]
    out[f"moran_resid_{nm}"] = float(moran_test(res, Xm, W))

# ---------- LC-IV and CV-SAP over repeated random audits
RG = np.linspace(-0.99, 0.99, 199); TG = np.linspace(-1.5, 1.5, 121)
def lc_run(xpred, phi, lab, full=False):
    S = build(L, y, xpred, phi, x, lab); S["mode"] = "srs"
    dd, P = estimate(S); s2 = sigma2_U(L, y, x, phi, lab, dd)
    V = wald_se(S, dd, P, s2); se = np.sqrt(np.diag(V))
    res = dict(delta=dd, se=se, impacts=impacts(dd))
    acc = ar_rho(S, RG, s2, dd[1], dd[2]) <= chi2.ppf(0.95, 2)
    res["rho_bounded"] = bool(not (acc[0] or acc[-1]))
    if full:
        res["rho_AR95"] = [float(RG[acc].min()), float(RG[acc].max())] if acc.any() else None
        inCS = ar_rho(S, RG, s2, dd[1], dd[2]) <= chi2.ppf(0.99, 2)
        c2 = sc_crit(S, dd, s2, rng)
        RR, TT = np.meshgrid(RG[inCS], TG, indexing="ij")
        arT = ar_theta_given_rho(S, RR.ravel(), TT.ravel(), s2, dd[1]).reshape(RR.shape)
        accT = (arT <= c2).any(0)
        res["theta_SCBonf95"] = [float(TG[accT].min()), float(TG[accT].max())] if accT.any() else None
        res["theta_SCBonf_bounded"] = bool(accT.any() and not (accT[0] or accT[-1]))
        res["c2_star"] = float(c2)
    return res
B = 300
store = {k: [] for k in ("LCIV", "CVSAP")}
for b in range(B):
    lab = rest[rng.permutation(len(rest))[:nL]]
    store["LCIV"].append(lc_run(xh, xh, lab)); store["CVSAP"].append(lc_run(xh, xt, lab))
orc = np.array(out["oracle"]["delta"]); orc_imp = np.array(out["oracle"]["impacts"])
for k, lst in store.items():
    D = np.array([r["delta"] for r in lst]); I = np.array([r["impacts"] for r in lst], float)
    SE = np.array([r["se"] for r in lst])
    out[k] = dict(delta_median=np.median(D, 0).tolist(), delta_iqr=(np.percentile(D, 75, 0) - np.percentile(D, 25, 0)).tolist(),
                  impacts_median=np.nanmedian(I, 0).tolist(),
                  median_abs_dev_from_oracle=np.median(np.abs(D - orc), 0).tolist(),
                  ati_median_abs_dev_from_oracle=float(np.nanmedian(np.abs(I[:, 2] - orc_imp[2]))),
                  wald_covers_oracle=np.mean(np.abs(D - orc) <= 1.96 * SE, 0).tolist(),
                  rho_bounded_share=float(np.mean([r["rho_bounded"] for r in lst])))
lab0 = rest[np.random.default_rng(1).permutation(len(rest))[:nL]]
for k, phi in (("LCIV_draw1", xh), ("CVSAP_draw1", xt)):
    r = lc_run(xh, phi, lab0, full=True)
    out[k] = {a: (v.tolist() if isinstance(v, np.ndarray) else v) for a, v in r.items()}

# ---------- save
d_out = d.drop(columns=["geometry"]).copy()
d_out["log_density"] = x; d_out["xhat_2020"] = xh; d_out["xhat_2010"] = xh10; d_out["xhat_sap"] = xt
d_out["log_built_share_2020"] = Z20; d_out["log_built_share_2010"] = Z10
d_out["train"] = np.isin(np.arange(n), tr)
cols = ["name", "province", "kind", "popn", "area_km2", "area_geo", "area_ratio", "n_communes", "built_m2_2010", "built_m2_2020",
        "built_share_2010", "built_share_2020", "rwi_mean", "rwi_n", "rwi_err_mean", "log_density", "log_built_share_2010",
        "log_built_share_2020", "xhat_2010", "xhat_2020", "xhat_sap", "train"]
d_out[cols].to_csv("vn_districts_G3.csv", index=False)
gpd.GeoDataFrame(d_out[cols], geometry=d.geometry.values, crs=4326).to_file("vn_districts_G3.gpkg", driver="GPKG")
import scipy.sparse as sp; sp.save_npz("W_queen_rowstd.npz", sp.csr_matrix(W))
json.dump(out, open("vn_results.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
