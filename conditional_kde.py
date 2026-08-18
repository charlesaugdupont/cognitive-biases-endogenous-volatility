"""Generate Figure I.1: macroscopic distributions conditioned on alpha and r.

Run from the repository root, after compute_key_metrics.py has been run for
eut/pt/cpt at beta=0.95. Writes figures/kde_conditional_alpha_rate.pdf and
prints the descriptive statistics quoted in Section 4.2.
"""
import pickle, numpy as np, matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.size": 15, "figure.dpi": 100, "grid.alpha": 0.3,
                     "axes.grid": False, "axes.axisbelow": True,
                     "mathtext.fontset": "cm", "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "axes.labelsize": 16,
                     "legend.fontsize": 12})
plt.rc("text", usetex=False)
plt.rc("font", family="serif")

BASE = "data"
dirs = [f"{BASE}/eut/eut_95", f"{BASE}/pt/pt_95", f"{BASE}/cpt/cpt_kt_95"]
labels = ["EUT", "PT", "CPT"]
colors = ["dodgerblue", "crimson", "green"]


_trapz = getattr(np, "trapezoid", None) or np.trapz


def load(d, name):
    with open(f"{d}/{name}", "rb") as f:
        return np.asarray(pickle.load(f), dtype=float)


D = {}
for d, lab in zip(dirs, labels):
    P = np.asarray([list(r) for r in pickle.load(open(f"{d}/params", "rb"))], dtype=float)
    D[lab] = {"alpha": P[:, 0], "rate": P[:, 1], "A": P[:, 2],
              "mean_util": load(d, "mean_util"), "sen": load(d, "sen_welfare"),
              "gini": load(d, "gini")}
    print(lab, "n =", len(P), "alpha", P[:, 0].min().round(3), P[:, 0].max().round(3),
          "rate", P[:, 1].min().round(3), P[:, 1].max().round(3))

# --- descriptive numbers for the text -------------------------------------
print("\n--- tercile means (low / high) ---")
for lab in labels:
    d = D[lab]
    for pname in ["alpha", "rate"]:
        v = d[pname]
        lo_t, hi_t = np.quantile(v, 1 / 3), np.quantile(v, 2 / 3)
        lo, hi = v <= lo_t, v >= hi_t
        for metric in ["mean_util", "sen"]:
            m = d[metric]
            print(f"{lab:4s} {pname:5s} {metric:9s} low={m[lo].mean():7.2f} high={m[hi].mean():7.2f} "
                  f"delta={m[hi].mean()-m[lo].mean():+7.2f}")

print("\n--- Spearman correlations with Sen welfare ---")
from scipy.stats import spearmanr
for lab in labels:
    d = D[lab]
    for pname in ["alpha", "rate", "A"]:
        rho, p = spearmanr(d[pname], d["sen"])
        print(f"{lab:4s} {pname:5s} rho={rho:+.3f} p={p:.2e}")

# --- share of mass below utility 100 (the low mode) ------------------------
print("\n--- share of calibrations with mean final utility < 100 ---")
for lab in labels:
    d = D[lab]
    for pname in ["alpha", "rate"]:
        v = d[pname]
        lo_t, hi_t = np.quantile(v, 1 / 3), np.quantile(v, 2 / 3)
        lo, hi = v <= lo_t, v >= hi_t
        m = d["mean_util"]
        print(f"{lab:4s} {pname:5s} low={100*(m[lo]<100).mean():5.1f}%  high={100*(m[hi]<100).mean():5.1f}%")

# --- figure ----------------------------------------------------------------
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def lighten(c, f=0.45):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


fig, axs = plt.subplots(2, 2, figsize=(13, 7.2))
cond = [("alpha", r"$\alpha$"), ("rate", r"$r$")]
metrics = [("mean_util", "Mean Final Utility"), ("sen", "Sen Welfare")]

styles = [("lower", "-", 2.0, lambda c: c),
          ("upper", (0, (5, 2)), 1.9, lighten)]

for row, (pname, psym) in enumerate(cond):
    for col, (metric, mlabel) in enumerate(metrics):
        ax = axs[row][col]
        for i, lab in enumerate(labels):
            d = D[lab]
            v = d[pname]
            lo_t, hi_t = np.quantile(v, 1 / 3), np.quantile(v, 2 / 3)
            masks = {"lower": v <= lo_t, "upper": v >= hi_t}
            for tag, ls, lw, cf in styles:
                vals = d[metric][masks[tag]]
                kde = gaussian_kde(vals, bw_method=0.2)
                x = np.linspace(1, 200, 500)
                y = kde(x)
                y /= _trapz(y, x)
                ax.plot(x, y, color=cf(colors[i]), lw=lw, ls=ls,
                        solid_capstyle="round", zorder=3 if tag == "lower" else 2)
        ax.set_xlim(1, 200)
        ax.set_ylim(0,)
        ax.set_xticks([1, 50, 100, 150, 200])
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlabel(mlabel)
        ax.set_title(f"Conditioned on {psym}", fontsize=15, pad=8)
        if col == 0:
            ax.set_ylabel("Density")

handles = []
for i, lab in enumerate(labels):
    for tag, ls, lw, cf in styles:
        handles.append(Line2D([], [], color=cf(colors[i]), lw=lw, ls=ls,
                              label=f"{lab} ({tag} tercile)"))

fig.subplots_adjust(wspace=0.22, hspace=0.55)
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
           bbox_to_anchor=(0.5, -0.10), handlelength=2.6, columnspacing=1.8,
           fontsize=13)

fig.savefig("figures/kde_conditional_alpha_rate.pdf", bbox_inches="tight")
print("\nsaved")
