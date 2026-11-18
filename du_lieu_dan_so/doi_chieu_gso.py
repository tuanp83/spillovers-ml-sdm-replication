"""Đối chiếu dân số huyện (đề án sắp xếp ĐVHC, tổng hợp trên Wikipedia tiếng Việt) với số liệu chính thức cấp tỉnh
của Tổng cục Thống kê: Biểu 1, 'Kết quả chủ yếu Điều tra biến động dân số và KHHGĐ thời điểm 01/4/2021' (NXB Thống kê, 2022).
Chạy: ../.venv/bin/python -I doi_chieu_gso.py"""
import re, json, numpy as np, pandas as pd
from pypdf import PdfReader
from unidecode import unidecode
r = PdfReader("nguon_gso/dskhhgd2021.pdf")
txt = "\n".join((r.pages[i].extract_text() or "") for i in range(80, 86))
start = txt.index("Biểu 1"); txt = txt[start:txt.index("Biểu 2")] if "Biểu 2" in txt else txt[start:]
rows = {}
for line in txt.splitlines():
    m = re.match(r"\s*(\d{2})\s+([^\d]+?)\s+([\d][\d.\s]*)$", line.strip())
    if not m: continue
    code, name, rest = m.group(1), m.group(2).strip(), re.sub(r"\s+", "", m.group(3))
    nums = [int(x.replace(".", "")) for x in re.findall(r"\d{1,3}(?:\.\d{3})+", rest)]
    if len(nums) >= 3 and code not in rows:
        rows[code] = dict(name=name, total=nums[0], male=nums[1], female=nums[2])
g = pd.DataFrame.from_dict(rows, orient="index")
g["ok"] = g.total == g.male + g.female
print("số tỉnh trích được:", len(g), "| dòng thỏa Chung=Nam+Nữ:", int(g.ok.sum()), "| tổng:", g.total.sum())
key = lambda s: re.sub(r"^(tinh|thanh pho|tp\.?)\s+", "", unidecode(str(s)).lower().replace("-", " ").strip()).replace(" ", "")
ALIAS = {"hue": "thuathienhue", "hochiminh": "hochiminh", "bariavungtau": "bariavungtau", "hoabinh": "hoabinh", "thanhhoa": "thanhhoa", "khanhhoa": "khanhhoa"}
g["k"] = g.name.map(key).map(lambda k: ALIAS.get(k, k))
w = pd.read_csv("districts_wiki_raw.csv"); w["k"] = w.province.map(key).map(lambda k: ALIAS.get(k, k))
pw = w.groupby("k").popn.sum()
m = g.set_index("k").join(pw.rename("wiki"), how="outer")
miss = m[m.total.isna() | m.wiki.isna()]
m = m.dropna(subset=["total", "wiki"]); m["chenh_pct"] = 100 * (m.wiki / m.total - 1)
out = dict(nguon="GSO, Điều tra biến động DS-KHHGĐ 01/4/2021, Biểu 1", n_tinh_khop=len(m), khong_khop=miss.index.tolist(),
           tong_gso=int(m.total.sum()), tong_wiki=int(m.wiki.sum()), chenh_tong_pct=float(100 * (m.wiki.sum() / m.total.sum() - 1)),
           chenh_tuyet_doi_trung_vi_pct=float(m.chenh_pct.abs().median()), chenh_max_pct=float(m.chenh_pct.abs().max()),
           tinh_chenh_max=m.chenh_pct.abs().idxmax(), tuong_quan_log=float(np.corrcoef(np.log(m.total), np.log(m.wiki))[0, 1]),
           so_tinh_chenh_tren_5pct=int((m.chenh_pct.abs() > 5).sum()))
print(json.dumps(out, indent=1, ensure_ascii=False))
m.sort_values("chenh_pct").to_csv("doi_chieu_tinh.csv"); json.dump(out, open("doi_chieu_gso.json", "w"), indent=1, ensure_ascii=False)
print(m.sort_values("chenh_pct")[["name", "total", "wiki", "chenh_pct"]].iloc[[0, 1, 2, -3, -2, -1]].to_string())
