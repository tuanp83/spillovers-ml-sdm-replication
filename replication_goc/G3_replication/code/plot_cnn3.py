import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = json.load(open("g3_cnn3.json"))
cols = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
xs = [1, 3, 5]
for ax, suf, title in ((axs[0], "", "CNN, raw predictions"), (axs[1], "_sap", "CNN after spatial recalibration (SAP)")):
    ax.axhline(0, color="#888888", lw=1, zorder=0)
    for o, c in zip(r, cols):
        m = [o[f"cnn{k}{suf}"]["theta_spur"] for k in xs]; se = [o[f"cnn{k}{suf}"]["theta_spur_se"] for k in xs]
        ax.errorbar(xs, m, yerr=1.96 * np.array(se), color=c, lw=2, marker="o", ms=8, capsize=3,
                    mec="white", mew=1.5, label=f"ψ = {o['psi']}")
        if not suf: ax.annotate(f"ψ={o['psi']}", (5.1, m[-1]), color="#333333", fontsize=8, va="center")
    ax.set_xticks(xs); ax.set_xticklabels(["own tile (1×1)", "3×3 tiles", "5×5 tiles"])
    ax.set_xlim(0.6, 5.8); ax.set_title(title, fontsize=11, color="#222222")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa"); ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(colors="#444444"); ax.grid(axis="y", color="#eeeeee")
axs[0].set_ylabel("Spurious SLX spillover (true value 0)", color="#333333")
axs[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig("g3_receptive_field_sap.png", dpi=200)
