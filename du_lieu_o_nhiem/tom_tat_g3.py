# Tóm tắt phát thải CAMS-GLOB-ANT cấp huyện cho G3: tổng cả nước, chẩn đoán (Moran, tương quan), bảng rộng cho G3.
# Chạy từ du_lieu_o_nhiem/: ../.venv/bin/python -I tom_tat_g3.py
import os, pandas as pd, numpy as np, scipy.sparse as sp
os.chdir(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv("phat_thai_huyen_G3_dai.csv"); G = pd.read_csv("ghep_huyen/vn_districts_G3.csv"); W = sp.load_npz("ghep_huyen/W_queen_rowstd.npz")
VI = {"nitrogen_oxides": "NOx", "sulphur_dioxide": "SO2", "black_carbon": "BC", "organic_carbon": "OC"}
d["chat"] = d.bien.map(VI)
moran = lambda x: float((x - x.mean()) @ (W @ (x - x.mean())) / ((x - x.mean()) @ (x - x.mean())))
out = []
t = d[d.nganh == "tong"].copy(); t["trieu_tan"] = t.t_km2_nam * G.area_geo.values[t.g3_row.values] / 1e6
tong = t.pivot_table(index="nam", columns="chat", values="trieu_tan", aggfunc="sum").round(3)
tong.to_csv("tong_ca_nuoc.csv"); out.append(("Tổng cả nước (triệu tấn/năm)", tong))
rows = []
for (c, k), s in d[d.nam == 2020].groupby(["chat", "nganh"]):
    x = np.log(s.sort_values("g3_row").t_km2_nam.values + 1e-6)
    if np.std(x) == 0: continue
    m = G.rwi_mean.notna().values
    rows.append(dict(chat=c, nganh=k, moran=moran(x), r_log_mat_do=np.corrcoef(x, G.log_density)[0, 1],
                     r_rwi=np.corrcoef(x[m], G.rwi_mean[m])[0, 1], r_xhat2020=np.corrcoef(x, G.xhat_2020)[0, 1]))
cd = pd.DataFrame(rows).round(3); cd.to_csv("chan_doan_2020.csv", index=False); out.append(("Chẩn đoán 2020 (log)", cd))
t["lx"] = np.log(t.t_km2_nam)
fe = {c: (s.lx - s.groupby("g3_row").lx.transform("mean") - s.groupby("nam").lx.transform("mean") + s.lx.mean()).var() / s.lx.var()
      for c, s in t.groupby("chat")}
out.append(("Tỷ lệ phương sai log tổng còn lại sau FE huyện + năm", pd.Series(fe).round(4)))
# bảng rộng: mỗi huyện một dòng, log(t/km2/năm) cho tổng, năng lượng, giao thông; 2010 và 2020
w = d[d.nam.isin([2010, 2020]) & d.nganh.isin(["tong", "nang_luong", "giao_thong"])].copy()
w["col"] = "log_" + w.chat + "_" + w.nganh + "_" + w.nam.astype(str); w["v"] = np.log(w.t_km2_nam + 1e-6)
wide = w.pivot(index="g3_row", columns="col", values="v").sort_index()
wide.insert(0, "province", G.province.values); wide.insert(0, "name", G.name.values)
wide.to_csv("phat_thai_huyen_G3_rong.csv", index_label="g3_row")
with open("tom_tat.txt", "w") as f:
    for h, x in out: f.write(f"== {h}\n{x.to_string()}\n\n")
print(open("tom_tat.txt").read())
