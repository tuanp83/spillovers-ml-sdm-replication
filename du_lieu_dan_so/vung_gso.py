"""Kiểm tra độ vững: chuẩn hoá dân số huyện theo tổng tỉnh chính thức (TCTK, 01/4/2021),
x_b = log(dân số_huyện × hệ số_tỉnh / diện tích), hệ số_tỉnh = GSO_tỉnh / tổng_huyện_tỉnh.
Tính lại: tương quan x–x_b, κ và θ* SLX giả-thật (công thức chính xác, Mệnh đề 3) cho dự báo GHSL, CAMS, nhòe thuần."""
import json, re, numpy as np, pandas as pd, scipy.sparse as sp
from unidecode import unidecode
D = "../du_lieu_o_nhiem/ghep_huyen/"
d = pd.read_csv(D + "vn_districts_G3.csv"); W = sp.load_npz(D + "W_queen_rowstd.npz").tocsr(); n = W.shape[0]
cams = pd.read_csv("../ung_dung_cams/cams_huyen_2020.csv").sort_values("g3_row")
m = pd.read_csv("doi_chieu_tinh.csv", index_col=0)
key = lambda s: re.sub(r"^(tinh|thanh pho|tp\.?)\s+", "", unidecode(str(s)).lower().replace("-", " ").strip()).replace(" ", "")
ALIAS = {"hue": "thuathienhue"}
w = pd.read_csv("districts_wiki_raw.csv"); w["k"] = w.province.map(key).map(lambda k: ALIAS.get(k, k))
fac = (m.total / m.wiki).to_dict()
d["k"] = d.province.map(key).map(lambda k: ALIAS.get(k, k)); assert d.k.map(fac).notna().all()
xb = np.log(d.popn * d.k.map(fac) / d.area_km2)
dm = lambda v: v - v.mean()
tr = np.where(d.train.values)[0]; rest = np.where(~d.train.values)[0]
def theta(xc, xh):
    xh = dm(xh); e = dm(xc) - xh; Wh = dm(W @ xh)
    m11, m12, m22, g0, g1 = xh @ xh / n, xh @ Wh / n, Wh @ Wh / n, e @ xh / n, e @ Wh / n
    return float((m11 * g1 - m12 * g0) / (m11 * m22 - m12 ** 2)), float(1 - np.var(e[rest]) / np.var(dm(xc)[rest]))
out = {}
for tag, xr in (("goc", d.log_density.values), ("chuan_hoa_GSO", xb.values)):
    xc = (xr - xr.mean()) / xr.std()
    def fit(F):
        X = np.column_stack([np.ones(n), F]); cf, *_ = np.linalg.lstsq(X[tr], xc[tr], rcond=None); return X @ cf
    Z = d.log_built_share_2020.values
    V = ["pm2p5", "aod550", "bcaod550", "omaod550", "suaod550", "duaod550", "tcno2", "tcco", "tcso2", "tchcho"]
    P = pd.Series(xc).groupby(cams.cell.values).transform("mean").values
    r = {}
    for nm, xh in (("GHSL", fit(np.column_stack([Z, Z ** 2]))), ("CAMS", fit(np.log(cams[V].values))), ("nhoe_Px", P)):
        th, ka = theta(xc, xh); r[nm] = dict(theta_star=round(th, 3), kappa=round(ka, 3))
    out[tag] = r
out["tuong_quan_x"] = float(np.corrcoef(d.log_density, xb)[0, 1])
out["he_so_tinh"] = dict(min=float(min(fac.values())), max=float(max(fac.values())))
print(json.dumps(out, indent=1, ensure_ascii=False)); json.dump(out, open("vung_gso.json", "w"), indent=1, ensure_ascii=False)
