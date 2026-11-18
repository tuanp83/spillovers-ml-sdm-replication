"""Kiểm định AR con với σ̃² tính TẠI GIẢ THUYẾT, có chặn trên (Định lý thm:feasible).
σ̃²(δ0) = n_L^-1 Σ_L (y_i − ρ0 (Wy)_i − β0 x_i − θ0 (Wφ)_i)²;  σ̃²_c = min{σ̃², C̄},
C̄ = 2(1+ρ̄‖W‖₂)² · mean_L y_i²  (≥ 2σ² w.p.a.1);  Ω̃ = σ̃²_c Â + Ω̂_a(β0,θ0).
T̃_θ(θ0) = inf_{ρ0,β0∈R} AR ;  T̃_ρ(ρ0) = inf_{β0,θ0∈R} AR.
Infimum trên R² = min(lưới tan + Nelder–Mead từ hai tâm, giới hạn theo tia ở vô cùng). Giới hạn theo tia: dạng đóng L*_c
cho T̃_θ có chặn; lưới góc + tinh chỉnh 1-D cho T̃_ρ và cho bản không chặn.
cbar=np.inf cho bản không chặn (chỉ để so sánh; bản này có trần lực, xem Mệnh đề prop:far)."""
import numpy as np
from g3_v2 import _min2


def _w(S):
    w = S.get("w"); return np.ones(len(S["yl"])) if w is None else w


def cap(S, Wnorm, rho_bar=0.99):
    w = _w(S); return 2 * (1 + rho_bar * Wnorm) ** 2 * float(np.sum(w * S["yl"] ** 2) / np.sum(w))


def _s2(S, r, b, t, c0=1.0):
    """trung bình có trọng số của (c0*y − r Wy − b x − t Wφ)² trên mẫu kiểm tra; c0=0 cho giới hạn theo tia."""
    w = _w(S); w = w / w.sum()
    R = c0 * S["yl"][None, :] - r[:, None] * S["Wyl"][None, :] - b[:, None] * S["xl"][None, :] - t[:, None] * S["Wphil"][None, :]
    return (R ** 2) @ w


def _quad(S, g, s2, b, t):
    Om = s2[:, None, None] * S["A"][None] + (b ** 2)[:, None, None] * S["Chh"] + (b * t)[:, None, None] * S["Chw"] \
        + (t ** 2)[:, None, None] * S["Cww"]
    return np.einsum("nk,nk->n", g, np.linalg.solve(Om, g[..., None])[..., 0])


def _ar(S, r, b, t, cbar=np.inf):
    Gm, hy = S["G"], S["hy"]
    g = hy[None, :] - r[:, None] * Gm[None, :, 0] - b[:, None] * Gm[None, :, 1] - t[:, None] * Gm[None, :, 2]
    return _quad(S, g, np.minimum(_s2(S, r, b, t), cbar), b, t)


def _ray(S, which, fixed, cbar, N=1441):
    """giới hạn của AR khi nuisance = s·(cos φ, sin φ), s→∞. Với chặn hữu hạn, s^-2 σ̃²_c → 0;
    không chặn, s^-2 σ̃² → trung bình bình phương của tổ hợp tuyến tính (c0=0)."""
    Gm = S["G"]; ph = np.linspace(0, np.pi, N, endpoint=False); c, s = np.cos(ph), np.sin(ph); z = np.zeros(N)
    if which == "theta":     # nuisance (ρ0, β0)
        g = -(c[:, None] * Gm[None, :, 0] + s[:, None] * Gm[None, :, 1]); bb, tt = s, z
        s2 = z if np.isfinite(cbar) else _s2(S, c, s, z, 0.0)
    else:                    # nuisance (β0, θ0)
        g = -(c[:, None] * Gm[None, :, 1] + s[:, None] * Gm[None, :, 2]); bb, tt = c, s
        s2 = z if np.isfinite(cbar) else _s2(S, z, c, s, 0.0)
    s2 = s2 + 1e-300
    ok = (np.abs(bb) + np.abs(tt) > 1e-9) | (s2 > 1e-200)
    v = np.full(N, np.inf)
    with np.errstate(all="ignore"):
        v[ok] = _quad(S, g[ok], s2[ok], bb[ok], tt[ok])
    v[~np.isfinite(v)] = np.inf
    return float(np.min(v))


def ray_theta_capped(S):
    """Dạng đóng của giới hạn theo tia cho T̃_θ có chặn (và cho T_θ cũ):
    L*_c = Gβ'C⁻¹Gβ − (Gρ'C⁻¹Gβ)²/(Gρ'C⁻¹Gρ), C = khối β0² của Ω̂_a (Mệnh đề prop:far)."""
    Gr, Gb = S["G"][:, 0], S["G"][:, 1]; Ci = np.linalg.inv(S["Chh"])
    return float(Gb @ Ci @ Gb - (Gr @ Ci @ Gb) ** 2 / (Gr @ Ci @ Gr))


def _ray_refined(S, which, fixed, cbar):
    """lưới góc + tinh chỉnh 1-D quanh các điểm tốt nhất (dùng cho T̃_ρ và cho bản không chặn)."""
    from scipy.optimize import minimize_scalar
    v0 = _ray(S, which, fixed, cbar)
    Gm = S["G"]

    def f(ph):
        c, s = np.array([np.cos(ph)]), np.array([np.sin(ph)]); z = np.zeros(1)
        if which == "theta":
            g = -(c[:, None] * Gm[None, :, 0] + s[:, None] * Gm[None, :, 1]); bb, tt = s, z
            s2 = z if np.isfinite(cbar) else _s2(S, c, s, z, 0.0)
        else:
            g = -(c[:, None] * Gm[None, :, 1] + s[:, None] * Gm[None, :, 2]); bb, tt = c, s
            s2 = z if np.isfinite(cbar) else _s2(S, z, c, s, 0.0)
        with np.errstate(all="ignore"):
            val = _quad(S, g, s2 + 1e-300, bb, tt)[0]
        return val if np.isfinite(val) else 1e300
    grid = np.linspace(0, np.pi, 181, endpoint=False); vals = np.array([f(p) for p in grid])
    best = v0
    for j in np.argsort(vals)[:3]:
        r = minimize_scalar(f, bounds=(grid[j] - np.pi / 180, grid[j] + np.pi / 180), method="bounded")
        best = min(best, float(r.fun))
    return best


def _inner(fun, d, scales, t0):
    """tìm kiếm toàn cục: tâm tại ước lượng LC-IV và tâm (0,0) với thang 1+|θ0| (thẩm định vòng 2)."""
    w = 1 + abs(t0)
    return min(_min2(fun, d, scales), _min2(fun, (0.0, 0.0), (w, w)))


def T_theta_h(S, t0, d, scales, cbar=np.inf):
    inner = _inner(lambda r, b: _ar(S, r, b, np.full_like(r, t0), cbar), (d[0], d[1]), scales, t0)
    ray = ray_theta_capped(S) if np.isfinite(cbar) else _ray_refined(S, "theta", t0, cbar)
    return min(inner, ray)


def T_rho_h(S, r0, d, scales, cbar=np.inf):
    inner = _inner(lambda b, t: _ar(S, np.full_like(b, r0), b, t, cbar), (d[1], d[2]), scales, r0)
    return min(inner, _ray_refined(S, "rho", r0, cbar))
