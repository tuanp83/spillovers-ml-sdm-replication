"""Tách hai cơ chế trên địa lý thật 638 huyện (θ SLX giả-thật chính xác, β=1, θ thật=0):
 (a) NHIỄU TRƠN: Z = x + ν, ν là SAR(φ) chuẩn hoá theo κ; x̂ = E[x|Z] tuyến tính (ước lượng trên toàn mẫu)  -> Mệnh đề 2′
 (b) NHÒE TÍN HIỆU: Z = B x (+ nhiễu iid), B = trung bình trong ô lưới CAMS 0,75° (74 ô)                    -> ảnh lưới thô
 (c) NHÒE THEO LÁNG GIỀNG: Z = (I + aW) x / (1+a) + nhiễu iid
Báo cáo θ*, β*, Moran(e), Moran(ν) và dự báo dấu của luật dấu."""
import os, numpy as np, pandas as pd, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(HERE, "..", "du_lieu_o_nhiem", "ghep_huyen")
d = pd.read_csv(os.path.join(DL, "vn_districts_G3.csv")); W = sp.load_npz(os.path.join(DL, "W_queen_rowstd.npz")).tocsr()
cell = pd.read_csv(os.path.join(HERE, "cams_huyen_2020.csv")).sort_values("g3_row").cell.values
n = W.shape[0]; x = d.log_density.values; x = (x - x.mean()) / x.std()
dm = lambda v: v - v.mean(); moran = lambda v: float(dm(v) @ (W @ dm(v)) / (dm(v) @ dm(v)))
I = sp.identity(n, format="csc")
def proj(xh):
    X = np.column_stack([np.ones(n), xh, W @ xh]); b = np.linalg.lstsq(X, x, rcond=None)[0]; return b[1], b[2]
def calib(Z):  # dự báo tuyến tính tốt nhất E[x|Z] trong lớp a+cZ
    c = np.cov(x, Z)[0, 1] / np.var(Z); return x.mean() + c * (Z - Z.mean())
rng = np.random.default_rng(1)
print("Moran(x) =", round(moran(x), 3))
print("\n(a) Nhiễu trơn: Z = x + ν, ν ~ SAR(φ), Var(ν) chọn sao cho κ≈0,6")
for phi in (0.0, 0.5, 0.8, 0.9, 0.95):
    th = []
    for r in range(50):
        nu = sp.linalg.spsolve(I - phi * W, rng.standard_normal(n)); nu = nu / nu.std() * np.sqrt(1 / 0.6 - 1)
        b, t = proj(calib(x + nu)); th.append((b, t, moran(nu)))
    th = np.array(th).mean(0)
    print(f"  φ={phi:4}: Moran(ν)={th[2]:.2f}  β*={th[0]:.3f}  θ*={th[1]:+.3f}  luật dấu dự báo: {'+' if moran(x) > th[2] else '−'}")
B = pd.Series(x).groupby(cell).transform("mean").values
print("\n(b) Nhòe tín hiệu theo ô CAMS: Z = Bx + s·ε")
for s in (0.0, 0.3, 0.6):
    th = np.array([proj(calib(B + s * rng.standard_normal(n))) for _ in range(50)]).mean(0)
    xh = calib(B + s * rng.standard_normal(n)); e = x - xh
    print(f"  s={s}: κ={1-np.var(e)/np.var(x):.2f}  Moran(e)={moran(e):.2f}  β*={th[0]:.3f}  θ*={th[1]:+.3f}")
print("\n(c) Nhòe theo láng giềng: Z = (x + aWx)/(1+a) + 0,5ε")
for a in (0.0, 0.5, 1.0, 2.0):
    th = np.array([proj(calib((x + a * (W @ x)) / (1 + a) + 0.5 * rng.standard_normal(n))) for _ in range(50)]).mean(0)
    print(f"  a={a}: β*={th[0]:.3f}  θ*={th[1]:+.3f}")
