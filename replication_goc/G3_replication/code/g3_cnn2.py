"""Receptive field -> spurious spillover curve (links G3 with G1).
For psi in {0.3, 0.6, 0.8, 0.9}: predictors with receptive field r in {own, 3x3, 5x5} tiles,
both CNN (PyTorch) and linear-window (OLS on window tile means), trained on independent maps.
Metrics on evaluation maps: R2, corr(e, W xh), spurious theta in SLX (null DGP), Moran rejection
(null), naive SDM ATI bias and LC-IV (true SDM), share of bounded AR sets for rho.
Run: python3 g3_cnn2.py [R_eval] [epochs]
"""
import numpy as np, json, sys, time
import torch, torch.nn as nn
from scipy.stats import norm, chi2
from g3_core import Lattice, build, estimate, wald_se, sigma2_U, ar_rho
from g3_sim import moran_test
from g3_cnn import make_raster, TILE, M

torch.set_num_threads(1)
RHO, BETA, THETA = 0.4, 1.0, 0.5
RG = np.linspace(-0.99, 0.99, 199)


def windows(img, r):
    """r=0: own tile (16x16); r=1: 3x3 tiles (48x48); r=2: 5x5 tiles avg-pooled to 40x40."""
    if r == 0:
        return img.reshape(M, TILE, M, TILE).transpose(0, 2, 1, 3).reshape(M * M, 1, TILE, TILE).astype(np.float32)
    w = 2 * r + 1
    P = np.pad(img, r * TILE, mode="reflect")
    out = []
    for a in range(M):
        for c in range(M):
            out.append(P[a * TILE:(a + w) * TILE, c * TILE:(c + w) * TILE])
    X = np.array(out, np.float32)[:, None]
    if r == 2:
        X = X.reshape(len(X), 1, 40, 2, 40, 2).mean((3, 5))
    return X


def tile_means(img):
    return img.reshape(M, TILE, M, TILE).mean((1, 3))


def window_features(img, r):
    T = tile_means(img); P = np.pad(T, r, mode="reflect")
    feats = [P[r + da:r + da + M, r + dc:r + dc + M].ravel() for da in range(-r, r + 1) for dc in range(-r, r + 1)]
    return np.column_stack([np.ones(M * M)] + feats)


class Net(nn.Module):
    def __init__(self, size, pos):
        super().__init__()
        layers = [nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()]
        if pos:
            layers += [nn.MaxPool2d(2), nn.Flatten()]
            fl = 32 * (size // 8) ** 2
            layers += [nn.Linear(fl, 64), nn.ReLU(), nn.Linear(64, 1)]
        else:
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1)]
        self.f = nn.Sequential(*layers)
    def forward(self, z): return self.f(z).squeeze(-1)


def train(model, Xtr, ytr, Xva, yva, epochs, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)
    best, state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); loss = ((model(Xt[idx]) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - yv) ** 2).mean().item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return 1 - best / float(np.var(yva))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(X[i:i + 300])).numpy() for i in range(0, len(X), 300)]).astype(float)


def evaluate(L0, L1, preds, x, rng, rec):
    W = L0.W; n = L0.n
    eps = rng.standard_normal(n)
    y0 = BETA * x + eps
    y1, _ = L1.draw_y(x, BETA, THETA, rng, eps=eps)
    lab = rng.permutation(n)[:90]
    for k, xh in preds.items():
        e = x - xh
        R = rec.setdefault(k, {m: [] for m in ["r2", "cwe", "th", "mor", "nai", "lc", "bnd"]})
        R["r2"].append(1 - np.var(e) / np.var(x))
        R["cwe"].append(np.corrcoef(e, W @ xh)[0, 1] if k != "oracle" else 0.0)
        Xs = np.column_stack([np.ones(n), xh, W @ xh])
        R["th"].append(np.linalg.lstsq(Xs, y0, rcond=None)[0][2])
        Xm = (xh - xh.mean())[:, None]; yc = y0 - y0.mean()
        res = yc - Xm @ np.linalg.lstsq(Xm, yc, rcond=None)[0]
        R["mor"].append(moran_test(res, Xm, W) > norm.ppf(0.95))
        H = np.column_stack([L1.Wp[j] @ xh for j in range(4)])
        Q = np.column_stack([W @ y1, xh, W @ xh]); Ai = np.linalg.inv(H.T @ H)
        dn = np.linalg.solve(Q.T @ H @ Ai @ H.T @ Q, Q.T @ H @ Ai @ H.T @ y1)
        R["nai"].append((dn[1] + dn[2]) / (1 - dn[0]) if dn[0] < 1 else np.inf)
        if k != "oracle":
            S = build(L1, y1, xh, xh, x, lab); S["mode"] = "srs"; d, P = estimate(S)
            s2 = sigma2_U(L1, y1, x, xh, lab, d)
            R["lc"].append((d[1] + d[2]) / (1 - d[0]) if d[0] < 1 else np.inf)
            acc = ar_rho(S, RG, s2, d[1], d[2]) <= chi2.ppf(0.95, 2)
            R["bnd"].append(not (acc[0] or acc[-1]))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    rng = np.random.default_rng(909); t0 = time.time()
    ATI = (BETA + THETA) / (1 - RHO)
    results = []
    for psi in (0.3, 0.6, 0.8, 0.9):
        L0 = Lattice(M, psi, 0.0, 4); L1 = Lattice(M, psi, RHO, 4)
        ntrain = 5
        imgs, xs = [], []
        for _ in range(ntrain + 1):
            x = L0.draw_x(rng); imgs.append(make_raster(x, rng)); xs.append(x)
        mu, sd = np.mean(xs[:-1]), np.std(xs[:-1])
        ys = [(v - mu) / sd for v in xs]
        models, val = {}, {}
        for r in (0, 1, 2):
            Xs = [windows(im, r) for im in imgs]
            size = Xs[0].shape[-1]
            net = Net(size, pos=(r > 0))
            val[f"cnn{2*r+1}"] = train(net, np.concatenate(Xs[:-1]), np.concatenate(ys[:-1]), Xs[-1], ys[-1], epochs)
            models[r] = net
        lin = {}
        for r in (0, 1, 2):
            F = np.concatenate([window_features(im, r) for im in imgs[:-1]])
            lin[r] = np.linalg.lstsq(F, np.concatenate(xs[:-1]), rcond=None)[0]
        print(f"psi={psi} trained {val} [{time.time()-t0:.0f}s]", flush=True)
        rec = {}
        for it in range(R):
            x = L0.draw_x(rng); img = make_raster(x, rng)
            preds = {"oracle": x.copy()}
            for r in (0, 1, 2):
                preds[f"lin{2*r+1}"] = window_features(img, r) @ lin[r]
                preds[f"cnn{2*r+1}"] = mu + sd * predict(models[r], windows(img, r))
            evaluate(L0, L1, preds, x, rng, rec)
        out = dict(psi=psi, R=R, val_r2=val)
        for k, Rk in rec.items():
            nai = np.array(Rk["nai"]); lc = np.array(Rk["lc"]) if Rk["lc"] else None
            out[k] = dict(r2=float(np.mean(Rk["r2"])), corr_e_Wxh=float(np.mean(Rk["cwe"])),
                          theta_spur=float(np.mean(Rk["th"])), theta_spur_se=float(np.std(Rk["th"]) / np.sqrt(R)),
                          moran=float(np.mean(Rk["mor"])),
                          naive_ati_medbias=float(np.median(nai) - ATI),
                          lciv_ati_medbias=None if lc is None else float(np.median(lc) - ATI),
                          rho_bounded=None if not Rk["bnd"] else float(np.mean(Rk["bnd"])))
        results.append(out)
        print(json.dumps(out, default=float), f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(results, open("g3_cnn2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
