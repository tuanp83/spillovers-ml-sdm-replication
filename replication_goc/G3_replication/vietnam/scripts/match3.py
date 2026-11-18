import pandas as pd, geopandas as gpd, re, numpy as np
from unidecode import unidecode
def norm(s):
    s = unidecode(str(s)).lower().replace("\t", "").replace(" district", "")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"^(thanh pho|thi xa|quan|huyen)\s+(?=[a-z])", "", s.strip())
    return re.sub(r"[^a-z0-9]", "", s)
def kind_of_geo(name):
    n = unidecode(name).lower()
    if n.startswith("huyen "): return ""
    if n.startswith("thi xa "): return "thị xã"
    if n.startswith("thanh pho "): return "thành phố"
    return None
w = pd.read_csv("districts_wiki_raw.csv").fillna({"kind": ""})
w["pn"] = w.province.map(norm); w["dn"] = w.name.map(norm)
a2 = gpd.read_file("adm2.geojson").to_crs(4326); a1 = gpd.read_file("adm1.geojson").to_crs(4326)
a1["pn"] = a1.shapeName.map(norm).replace({"thuathienhue": "hue", "condao": "bariavungtau"})
a1d = a1.dissolve("pn").reset_index()
cent = a2.copy(); cent["geometry"] = a2.representative_point()
j = gpd.sjoin(cent, a1[["pn", "geometry"]], how="left", predicate="within")
a2["pn"] = j["pn"].values; a2["dn"] = a2.shapeName.map(norm)
MERGE = {"Quan 2": "Thủ Đức", "Quan 9": "Thủ Đức", "Thu Duc": "Thủ Đức", "Dat Do": "Long Đất", "Long Dien": "Long Đất",
         "Hoanh Bo": "Hạ Long", "Ha Long": "Hạ Long", "Phuc Hoa": "Quảng Hòa", "Quang Uyen": "Quảng Hòa",
         "Thong Nong": "Hà Quảng", "Ha Quang": "Hà Quảng", "Tra Linh": "Trùng Khánh", "Trung Khanh": "Trùng Khánh",
         "Tay Tra": "Trà Bồng", "Tra Bong": "Trà Bồng", "Tinh Gia": "Nghi Sơn", "Cua Lo": "Vinh", "Vinh": "Vinh",
         "Dong Son": "Thanh Hóa", "Thanh Hoa": "Thanh Hóa", "My Loc": "Nam Định", "Nam Dinh": "Nam Định",
         "Ninh Bình": "Hoa Lư", "Hoa Lu": "Hoa Lư", "Yen Dung": "Bắc Giang", "Bac Giang": "Bắc Giang",
         "Nam Dong": "Phú Lộc", "Phu Loc": "Phú Lộc", "Nong Son": "Quế Sơn", "Que Son": "Quế Sơn",
         "Cat Tien": "Đạ Huoai", "Da Teh": "Đạ Huoai", "Da Huoai": "Đạ Huoai"}
cand = {k: g.index.tolist() for k, g in w.groupby("dn")}
match = []
for i, r in a2.iterrows():
    key = norm(MERGE[r.shapeName]) if r.shapeName in MERGE else r.dn
    c = cand.get(key, [])
    if len(c) > 1:
        kg = kind_of_geo(r.shapeName)
        if kg is not None:
            c2 = [k for k in c if w.loc[k, "kind"] == kg]
            if c2: c = c2
    if len(c) > 1:
        c2 = [k for k in c if w.loc[k, "pn"] == r.pn]
        if len(c2) == 1: c = c2
    if len(c) > 1:   # nearest province polygon
        pt = r.geometry.representative_point()
        dists = [a1d.loc[a1d.pn == w.loc[k, "pn"], "geometry"].distance(pt).min() for k in c]
        c = [c[int(np.argmin(dists))]]
    match.append(c[0] if c else None)
a2["wiki_idx"] = match
g = a2[a2.wiki_idx.notna()].dissolve("wiki_idx").reset_index()
g["wiki_idx"] = g.wiki_idx.astype(int)
g = g.merge(w, left_on="wiki_idx", right_index=True)
g["area_geo"] = g.to_crs(6933).area / 1e6
g["area_ratio"] = g.area_geo / g.area_km2
bad = g[(g.area_ratio < 0.8) | (g.area_ratio > 1.25)]
print("units:", len(g), "| area ratio median", round(g.area_ratio.median(), 3), "| outside [0.8,1.25]:", len(bad))
print(bad[["name", "province", "area_km2", "area_geo", "area_ratio"]].round(2).to_string())
print("unmatched geo:", a2[a2.wiki_idx.isna()].shapeName.tolist())
print("wiki rows not used:", len(set(w.index) - set(g.wiki_idx)))
g.to_pickle("districts_geo.pkl")
