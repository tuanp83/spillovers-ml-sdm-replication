"""LC-IV cho mẫu kiểm tra có thiết kế phức tạp (phân tầng, nhiều giai đoạn, xác suất chọn không bằng nhau).

Mở rộng `g3_v2.build`:
  c0 = H'phi + sum_{i in L} H_i d_i / pi_i,   c1 = (W'H)'phi + sum_{i in L} (W'H)_i d_i / pi_i   (Horvitz–Thompson)
  Omega_a(b0,t0) = Vhat( b0 u + t0 v ),  u_i = d_i H_i,  v_i = d_i (W'H)_i,
với Vhat là ước lượng phương sai thiết kế của tổng HT. Vhat là dạng toàn phương theo các giá trị mẫu, nên
  Vhat(b0 u + t0 v) = b0^2 Vhat(u) + b0 t0 [Vhat(u,v)+Vhat(v,u)] + t0^2 Vhat(v)
=> khớp giả thuyết tại MỌI (b0,t0) (Bổ đề 3 tổng quát) — chỉ cần lưu ba khối Chh, Chw, Cww như g3_v2.

Ba kiểu phương sai (tham số `var`):
  "two_stage": không chệch cho phân tầng + 2 giai đoạn SRSWOR (giai đoạn 1 chọn m_h trong M_h PSU,
               giai đoạn 2 chọn k_c trong N_c đơn vị). PSU = đơn vị và N_c = k_c = 1 cho ra SRS phân tầng.
  "ultimate":  cụm tối hậu (with-replacement, không fpc) — bảo thủ, dùng được cho mọi thiết kế nhiều giai đoạn
               và cho nhãn có sai số khảo sát.
  "unit":      coi mỗi đơn vị là một PSU trong tầng của nó (bỏ qua cụm) — dùng để minh hoạ sai lầm.
Đơn vị có pi_i = 1 (ví dụ huyện huấn luyện đã biết x) là đơn vị chắc chắn: đóng góp đúng d_i, phương sai 0.
"""
import warnings
import numpy as np


def _design_var(Z, strata, psu, w, var, M=None, Nc=None):
    """Vhat của tổng HT sum_i w_i Z_i (Z: n_L x p). strata/psu: nhãn cho từng đơn vị mẫu.
    M: dict tầng -> số PSU tổng thể; Nc: dict psu -> số đơn vị cấp 2 trong khung."""
    p = Z.shape[1]; V = np.zeros((p, p)); df = 0
    for h in np.unique(strata):
        ih = strata == h
        ps = np.unique(psu[ih]); m = len(ps)
        if m < 2:
            warnings.warn(f"tầng {h} chỉ có {m} PSU: bỏ khỏi phương sai -> phương sai bị đánh giá thấp; hãy gộp tầng")
            continue
        T = np.array([(w[ih & (psu == c)][:, None] * Z[ih & (psu == c)]).sum(0) for c in ps])   # tổng HT theo PSU
        Tc = T - T.mean(0)
        f1 = m / M[h] if (var == "two_stage" and M is not None) else 0.0
        V += (1 - f1) * m / (m - 1) * Tc.T @ Tc
        df += m - 1
        if var == "two_stage" and f1 > 0 and Nc is not None:               # số hạng giai đoạn 2 (Rao)
            for c in ps:
                jc = ih & (psu == c); k = jc.sum(); N = Nc[c]
                if k >= 2 and N > k:
                    zc = Z[jc] - Z[jc].mean(0)
                    V += (M[h] / m) * N ** 2 * (1 - k / N) / k * (zc.T @ zc) / (k - 1)
    return V, df


def build_design(G, y, xh, lab, xlab, pi, strata, psu, var="two_stage", M=None, Nc=None, phi=None):
    """lab: chỉ số đơn vị có nhãn (kể cả đơn vị chắc chắn); xlab: nhãn quan sát (có thể có sai số);
    pi: xác suất chọn của từng đơn vị trong lab."""
    if phi is None:
        phi = xh
    W, n, K = G.W, G.n, G.K
    cols = [xh]
    for _ in range(K - 1):
        cols.append(W @ cols[-1])
    H = np.column_stack(cols); WtH = W.T @ H
    w = 1.0 / np.asarray(pi, float)
    d = xlab - phi[lab]
    Hl, Wl = H[lab], WtH[lab]
    c0 = H.T @ phi + Hl.T @ (w * d)
    c1 = WtH.T @ phi + Wl.T @ (w * d)
    Wy = W @ y
    Gm = np.column_stack([H.T @ Wy, c0, c1])
    rnd = w > 1 + 1e-12                                        # bỏ đơn vị chắc chắn khỏi phương sai
    Z = np.column_stack([d[:, None] * Hl, d[:, None] * Wl])[rnd]
    V, df = _design_var(Z, np.asarray(strata)[rnd], np.asarray(psu)[rnd], w[rnd], var, M, Nc)
    Chh, Cuv, Cww = V[:K, :K], V[:K, K:], V[K:, K:]
    # sigma^2_U: trung bình Hájek có trọng số của phần dư trên mẫu (xem Mệnh đề sigma)
    S = dict(G=Gm, hy=H.T @ y, A=H.T @ H, Chh=Chh, Chw=Cuv + Cuv.T, Cww=Cww, nl=len(lab), n=n, K=K,
             df=df, w=w, yl=y[lab], Wyl=Wy[lab], xl=xlab, Wphil=(W @ phi)[lab])
    return S


def sigma2_w(S, dd):
    r = S["yl"] - dd[0] * S["Wyl"] - dd[1] * S["xl"] - dd[2] * S["Wphil"]
    return float(np.sum(S["w"] * r ** 2) / np.sum(S["w"]))
