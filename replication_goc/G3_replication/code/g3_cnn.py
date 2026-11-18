"""Item 3: real CNNs on simulated 16x16-pixel tiles.
Raster: pixel = tanh(0.5*(x_cell + s_u*U)) + s_w*eta, U smooth Gaussian field (crosses tile borders),
eta white noise. Two CNNs, trained on independent maps (A3):
  own : input = own 16x16 tile           (receptive field = own cell)
  ctx : input = 48x48 window (3x3 tiles)  (receptive field covers rook/queen neighbours)
Evaluation maps: spurious spillover (SLX, null DGP), Moran size, naive SDM vs LC-IV (true SDM).
Run: python3 g3_cnn.py [R_eval]
"""
import numpy as np, json, sys, time
import torch, torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from g3_core import Lattice, ols_fit, build, estimate, wald_se, sigma2_U
from g3_sim import moran_test

torch.set_num_threads(2)
TILE, M = 16, 30
S_U, S_W, SMOOTH = 1.6, 0.35, 4.0
RHO, BETA, THETA = 0.4, 1.0, 0.5


def make_raster(x, rng):
    X = np.kron(x.reshape(M, M), np.ones((TILE, TILE)))
    U = gaussian_filter(rng.standard_normal(X.shape), SMOOTH, mode="wrap")
    U /= U.std()
    return np.tanh(0.5 * (X + S_U * U)) + S_W * rng.standard_normal(X.shape)


def tiles(img, ctx):
    if not ctx:
        return img.reshape(M, TILE, M, TILE).transpose(0, 2, 1, 3).reshape(M * M, 1, TILE, TILE)
    P = np.pad(img, TILE, mode="reflect")
    out = np.empty((M * M, 1, 3 * TILE, 3 * TILE), np.float32)
    for r in range(M):
        for c in range(M):
            out[r * M + c, 0] = P[r * TILE:(r + 3) * TILE, c * TILE:(c + 3) * TILE]
    return out


class OwnCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(), nn.Linear(32, 1))
    def forward(s, z): return s.f(z).squeeze(-1)


class CtxCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Flatten(), nn.Linear(32 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, z): return s.f(z).squeeze(-1)


def train(model, Xtr, ytr, Xva, yva, epochs=25, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); loss = ((model(Xt[idx]) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - yv) ** 2).mean().item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return 1 - best / yva.var()


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy().astype(float)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = np.random.default_rng(303); t0 = time.time()
    L0 = Lattice(M, 0.8, 0.0, 4); L1 = Lattice(M, 0.8, RHO, 4); W = L0.W; n = L0.n
    # ---------------- training data: independent maps
    ntrain = 7
    Xo, Xc, Y = [], [], []
    for _ in range(ntrain + 1):
        x = L0.draw_x(rng); img = make_raster(x, rng).astype(np.float32)
        Xo.append(tiles(img, False).astype(np.float32)); Xc.append(tiles(img, True)); Y.append(x)
    mu, sd = np.mean(Y[:-1]), np.std(Y[:-1])
    ys = [(y - mu) / sd for y in Y]
    own, ctx = OwnCNN(), CtxCNN()
    r2o = train(own, np.concatenate(Xo[:-1]), np.concatenate(ys[:-1]), Xo[-1], ys[-1])
    r2c = train(ctx, np.concatenate(Xc[:-1]), np.concatenate(ys[:-1]), Xc[-1], ys[-1])
    print(f"trained: val R2 own={r2o:.3f} ctx={r2c:.3f} [{time.time()-t0:.0f}s]", flush=True)
    # ---------------- evaluation maps
    rec = {k: dict(r2=[], ce=[], cwe=[], th_slx=[], moran=[], naive=[], lciv=[], cov=[], bnd=[])
           for k in ("own", "ctx", "oracle")}
    from scipy.stats import chi2
    from g3_core import ar_rho
    RG = np.linspace(-0.99, 0.99, 199)
    for r in range(R):
        x = L0.draw_x(rng); img = make_raster(x, rng).astype(np.float32)
        preds = {"own": mu + sd * predict(own, tiles(img, False).astype(np.float32)),
                 "ctx": mu + sd * predict(ctx, tiles(img, True)),
                 "oracle": x.copy()}
        eps = rng.standard_normal(n)
        y0 = BETA * x + eps                                    # null: no spillover, no rho
        y1, _ = L1.draw_y(x, BETA, THETA, rng, eps=eps)        # true SDM
        lab = rng.permutation(n)[:90]
        for k, xh in preds.items():
            e = x - xh
            rec[k]["r2"].append(1 - np.var(e) / np.var(x))
            rec[k]["ce"].append(np.corrcoef(e, xh)[0, 1] if k != "oracle" else 0.0)
            rec[k]["cwe"].append(np.corrcoef(e, W @ xh)[0, 1] if k != "oracle" else 0.0)
            Xs = np.column_stack([np.ones(n), xh, W @ xh])
            rec[k]["th_slx"].append(np.linalg.lstsq(Xs, y0, rcond=None)[0][2])
            Xm = (xh - xh.mean())[:, None]; yc = y0 - y0.mean()
            res = yc - Xm @ np.linalg.lstsq(Xm, yc, rcond=None)[0]
            rec[k]["moran"].append(moran_test(res, Xm, W) > norm.ppf(0.95))
            H = np.column_stack([L1.Wp[j] @ xh for j in range(4)])
            Q = np.column_stack([W @ y1, xh, W @ xh]); Ai = np.linalg.inv(H.T @ H)
            rec[k]["naive"].append(np.linalg.solve(Q.T @ H @ Ai @ H.T @ Q, Q.T @ H @ Ai @ H.T @ y1))
            if k != "oracle":
                S = build(L1, y1, xh, xh, x, lab); d, P = estimate(S)
                s2 = sigma2_U(L1, y1, x, xh, lab, d); V = wald_se(S, d, P, s2)
                rec[k]["lciv"].append(d); rec[k]["cov"].append(abs(d[2] - THETA) <= 1.96 * np.sqrt(V[2, 2]))
                acc = ar_rho(S, RG, s2, d[1], d[2]) <= chi2.ppf(0.95, 2)
                rec[k]["bnd"].append(not (acc[0] or acc[-1]))
        if (r + 1) % 25 == 0:
            print(f"  eval {r+1}/{R} [{time.time()-t0:.0f}s]", flush=True)
    true = np.array([RHO, BETA, THETA]); ati_true = (BETA + THETA) / (1 - RHO)
    def ati(E):
        return np.where(E[:, 0] < 1, (E[:, 1] + E[:, 2]) / (1 - np.minimum(E[:, 0], .999)), np.inf)
    out = dict(R=R, val_r2_own=float(r2o), val_r2_ctx=float(r2c))
    for k in rec:
        o = dict(r2=float(np.mean(rec[k]["r2"])), corr_e_xh=float(np.mean(rec[k]["ce"])),
                 corr_e_Wxh=float(np.mean(rec[k]["cwe"])),
                 slx_theta_null=float(np.mean(rec[k]["th_slx"])),
                 moran_reject_null=float(np.mean(rec[k]["moran"])))
        En = np.array(rec[k]["naive"])
        o["naive_medbias"] = (np.median(En, 0) - true).tolist()
        o["naive_ati_medbias"] = float(np.median(ati(En)) - ati_true)
        if rec[k]["lciv"]:
            El = np.array(rec[k]["lciv"])
            o["lciv_medbias"] = (np.median(El, 0) - true).tolist()
            o["lciv_ati_medbias"] = float(np.median(ati(El)) - ati_true)
            o["lciv_cov_theta"] = float(np.mean(rec[k]["cov"]))
            o["lciv_rho_bounded"] = float(np.mean(rec[k]["bnd"]))
        out[k] = o
    print(json.dumps(out, indent=1, default=float))
    json.dump(out, open("g3_cnn.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
