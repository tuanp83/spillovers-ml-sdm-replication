import json, re, pandas as pd, geopandas as gpd
t = json.load(open("huyen_wiki.json"))["parse"]["wikitext"]["*"]
rows = t.split("\n|-")
recs = []
def clean(s):
    s = s.strip().lstrip("|").strip()
    m = re.search(r"\[\[([^\]|]+)\|([^\]]+)\]\]", s)
    if m: return m.group(2).strip()
    m = re.search(r"\[\[([^\]]+)\]\]", s)
    if m: return m.group(1).strip()
    return s
def num(s):
    s = re.sub(r"<ref.*", "", s.strip().lstrip("|").strip()).strip().replace(".", "").replace(",", ".")
    try: return float(s)
    except: return None
for r in rows:
    cells = [c for c in r.split("\n") if c.startswith("|") and not c.startswith("|}") and not c.startswith("|-")]
    if len(cells) >= 9 and cells[0].strip("| ").isdigit():
        recs.append(dict(stt=int(cells[0].strip("| ")), name=clean(cells[1]), province=clean(cells[2]),
                         kind=cells[3].strip("| ").strip(), year=cells[4].strip("| "), popn=num(cells[5]),
                         area_km2=num(cells[6]), density=num(cells[7]), n_communes=num(cells[8])))
w = pd.DataFrame(recs)
print(w.shape, w.popn.isna().sum(), w.area_km2.isna().sum(), w.province.nunique())
print(w.head(3).to_string())
w.to_csv("districts_wiki_raw.csv", index=False)
a2 = gpd.read_file("adm2.geojson"); a1 = gpd.read_file("adm1.geojson")
print("geoBoundaries ADM2", len(a2), "ADM1", len(a1)); print(sorted(a1.shapeName.tolist())[:70])
print(a2.columns.tolist())
