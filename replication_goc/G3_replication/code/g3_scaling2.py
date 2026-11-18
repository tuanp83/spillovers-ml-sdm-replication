import numpy as np, json, time
from g3_sim import rook_W, run_design
rng = np.random.default_rng(99); res=[]; t0=time.time()
for m, R in ((30, 800), (50, 800), (70, 300)):
    W = rook_W(m); Wp = [np.eye(m*m), W, W@W, W@W@W]
    out = run_design(W, Wp, 0.8, 0.7, 0.0, 0.4, 1.0, 0.5, 0.10, 0.10, R, rng)
    out["n"] = m*m; out["R"] = R; res.append(out)
    print("n", m*m, "R", R, f"{time.time()-t0:.0f}s", flush=True)
    for k in ["naive","SAP","LCIV","LCIV_SAP"]:
        print("   ", k.ljust(9), "bias", np.round(out[k]["bias"],3), "medbias", np.round(out[k]["medbias"],3),
              "rmse", np.round(out[k]["rmse"],3), "ATI medbias", round(out[k]["imp_medbias"][2],3), flush=True)
json.dump(res, open("g3_scaling2.json","w"), indent=1)
