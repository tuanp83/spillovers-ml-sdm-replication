"""CNN recalibration via SAP (spatially-augmented projection) + larger training set.
Per psi: 7 training maps (CNN), 1 validation map (early stopping), 2 calibration maps (SAP fit),
then 120 evaluation maps. SAP: x ~ 1 + p + W p + W^2 p fitted on the calibration maps (p = CNN output),
which restores unit-level calibration and absorbs neighbour information linearly.
Run: python3 g3_cnn3.py [R_eval] [epochs]
"""
import numpy as np, json, sys, time
from g3_core import Lattice
from g3_cnn import make_raster, M
from g3_cnn2 import windows, window_features, Net, train, predict, evaluate
import torch
torch.set_num_threads(2)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rng = np.random.default_rng(1313); t0 = time.time()
    ATI = 1.5 / 0.6; results = []
    for psi in (0.3, 0.6, 0.8, 0.9):
        L0 = Lattice(M, psi, 0.0, 4); L1 = Lattice(M, psi, 0.4, 4); W = L0.W; W2 = L0.Wp[2]
        maps = []
        for _ in range(10):
            x = L0.draw_x(rng); maps.append((x, make_raster(x, rng)))
        tr, va, ca = maps[:7], maps[7], maps[8:]
        mu, sd = np.mean([m[0] for m in tr]), np.std([m[0] for m in tr])
        models, val, lin, sap = {}, {}, {}, {}
        for r in (0, 1, 2):
            Xs = np.concatenate([windows(im, r) for _, im in tr]); ys = np.concatenate([(x - mu) / sd for x, _ in tr])
            net = Net(Xs.shape[-1], pos=(r > 0))
            val[f"cnn{2*r+1}"] = train(net, Xs, ys, windows(va[1], r), (va[0] - mu) / sd, epochs)
            models[r] = net
            F = np.concatenate([window_features(im, r) for _, im in tr])
            lin[r] = np.linalg.lstsq(F, np.concatenate([x for x, _ in tr]), rcond=None)[0]
            # SAP recalibration on the calibration maps
            P = [mu + sd * predict(net, windows(im, r)) for _, im in ca]
            Fc = np.concatenate([np.column_stack([np.ones(len(p)), p, W @ p, W2 @ p]) for p in P])
            sap[r] = np.linalg.lstsq(Fc, np.concatenate([x for x, _ in ca]), rcond=None)[0]
            Fl = np.concatenate([np.column_stack([np.ones(M * M), window_features(im, 0) @ lin[0]] ) for _, im in ca])
        # linear own-tile + SAP (as linear benchmark of recalibration)
        P0 = [window_features(im, 0) @ lin[0] for _, im in ca]
        Fc0 = np.concatenate([np.column_stack([np.ones(len(p)), p, W @ p, W2 @ p]) for p in P0])
        sap_lin = np.linalg.lstsq(Fc0, np.concatenate([x for x, _ in ca]), rcond=None)[0]
        print(f"psi={psi} trained {val} sap_coef_own={np.round(sap[0],3)} [{time.time()-t0:.0f}s]", flush=True)
        rec = {}
        for it in range(R):
            x = L0.draw_x(rng); img = make_raster(x, rng)
            preds = {"oracle": x.copy()}
            p_lin1 = window_features(img, 0) @ lin[0]
            preds["lin1"] = p_lin1
            preds["lin1_sap"] = np.column_stack([np.ones(M * M), p_lin1, W @ p_lin1, W2 @ p_lin1]) @ sap_lin
            for r in (0, 1, 2):
                if r > 0:
                    preds[f"lin{2*r+1}"] = window_features(img, r) @ lin[r]
                p = mu + sd * predict(models[r], windows(img, r))
                preds[f"cnn{2*r+1}"] = p
                preds[f"cnn{2*r+1}_sap"] = np.column_stack([np.ones(M * M), p, W @ p, W2 @ p]) @ sap[r]
            evaluate(L0, L1, preds, x, rng, rec)
        out = dict(psi=psi, R=R, val_r2=val, sap_coef={f"cnn{2*r+1}": sap[r].tolist() for r in sap})
        for k, Rk in rec.items():
            nai = np.array(Rk["nai"]); lc = np.array(Rk["lc"]) if Rk["lc"] else None
            out[k] = dict(r2=float(np.mean(Rk["r2"])), corr_e_Wxh=float(np.mean(Rk["cwe"])),
                          theta_spur=float(np.mean(Rk["th"])), theta_spur_se=float(np.std(Rk["th"]) / np.sqrt(R)),
                          moran=float(np.mean(Rk["mor"])),
                          naive_ati_medbias=float(np.median(nai) - ATI),
                          lciv_ati_medbias=None if lc is None else float(np.median(lc) - ATI),
                          rho_bounded=None if not Rk["bnd"] else float(np.mean(Rk["bnd"])))
        results.append(out)
        print(json.dumps({k: v for k, v in out.items() if k not in ("sap_coef",)}, default=float), f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(results, open("g3_cnn3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
