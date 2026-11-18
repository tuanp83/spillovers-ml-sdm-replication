# Khớp hình học (districts_geo.pkl từ match3.py) với 638 huyện của G3 theo thứ tự hàng của vn_districts_G3.csv; kiểm tra W queen.
import pandas as pd, geopandas as gpd, numpy as np, scipy.sparse as sp
g = gpd.GeoDataFrame(pd.read_pickle("districts_geo.pkl"), geometry="geometry", crs=4326)
d = pd.read_csv("vn_districts_G3.csv")
d["kind"] = d["kind"].fillna(""); g["kind"] = g["kind"].fillna("")
m = d[["name", "province", "kind"]].reset_index().merge(g[["name", "province", "kind", "geometry"]], on=["name", "province", "kind"], how="left")
print("hàng G3:", len(d), "| khớp:", m.geometry.notna().sum(), "| trùng khóa:", m["index"].duplicated().sum())
m = gpd.GeoDataFrame(m.sort_values("index"), geometry="geometry", crs=4326)
# W queen
sidx = m.sindex; n = len(m); rows, cols = [], []
geom = m.geometry.buffer(0).values
for i in range(n):
    for j in sidx.query(geom[i], predicate="intersects"):
        if j != i: rows.append(i); cols.append(j)
A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); A.data[:] = 1
W = sp.diags(1 / np.maximum(A.sum(1).A1, 1)) @ A
W0 = sp.load_npz("W_queen_rowstd.npz")
same = ((W0 > 0) != (W > 0)).nnz
print("số cặp láng giềng: gốc", (W0 > 0).nnz, "| dựng lại", (W > 0).nnz, "| khác biệt", same)
m[["index", "name", "province", "geometry"]].rename(columns={"index": "g3_row"}).to_file("huyen_G3_638.gpkg", driver="GPKG")
print("đã lưu huyen_G3_638.gpkg")
