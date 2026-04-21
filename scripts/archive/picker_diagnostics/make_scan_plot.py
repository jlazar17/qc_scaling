"""
Plot efficiency η vs nstate/nstate0 for all (nqubit, H) combinations from the scaling scan.
Reads η_max from new-format logs; falls back to nothing if a block is incomplete.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from parse_scaling_logs import parse_log, merge_data

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                     "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150})

TURBO = plt.colormaps["rainbow"]

def qualitative_colorbar(fig, cax, values, labels, title=""):
    """Draw a qualitative colorbar as discrete color swatches with labels."""
    n = len(values)
    listed = mcolors.ListedColormap([TURBO(v) for v in values])
    bounds = np.arange(n + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, listed.N)
    sm = cm.ScalarMappable(cmap=listed, norm=norm)
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(n))
    cb.set_ticklabels(labels)
    cb.set_label(title)
    return cb

# nstate0 = base_nstate from the log header
nstate0 = {4: 11, 6: 23, 8: 52}

# Parse all available logs and merge (later logs overwrite earlier for same n/H)
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../scripts/7_scaling/data")
log_files = [
    os.path.join(LOG_DIR, "scaling_study_full.log"),
]
data = merge_data(*[parse_log(f) for f in log_files if os.path.exists(f)])

print("Loaded data:")
for n in sorted(data):
    hs = sorted(data[n].keys())
    print(f"  n={n}: H = {[f'{h:.3f}' for h in hs]}")

ALL_N     = [4, 6, 8]
N_vals    = [n for n in ALL_N if n in data]  # only plot n values with data
N_MARKERS = {4: "o", 6: "s", 8: "^"}

# All unique H values across all n, sorted — used for colormap sampling
all_H_vals = sorted({H for n_data in data.values() for H in n_data.keys()})

def h_color(H):
    return TURBO(H)  # H in [0,1] maps directly to rainbow

N_COLORS = {n: TURBO(i / (len(ALL_N) - 1)) for i, n in enumerate(ALL_N)}

# ---------------------------------------------------------------------------
# Version 1: panels by nqubit, color by H
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, n in zip(axes, N_vals):
    n0 = nstate0[n]
    for H in sorted(data[n].keys()):
        pts = sorted(data[n][H], key=lambda x: x[0])
        xs = [ns / n0 for ns, _ in pts]
        ys = [eta for _, eta in pts]
        ax.plot(xs, ys, "o-", color=h_color(H), lw=2, ms=5)
        best = max(pts, key=lambda x: x[1])
        ax.vlines(best[0]/n0, 0, best[1], color=h_color(H), lw=1.5, ls="--", alpha=0.7)

    ax.set_xlabel("nstate / nstate₀")
    ax.set_ylabel("Efficiency η" if n == 4 else "")
    ax.set_title(f"n = {n}  (nstate₀ = {n0})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.tight_layout()
fig.subplots_adjust(right=0.88)
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
qualitative_colorbar(fig, cax,
                     values=all_H_vals,
                     labels=[f"H = {H:.3f}" for H in all_H_vals],
                     title="Goal entropy H")
fig.suptitle("Efficiency η vs nstate/nstate₀  —  panels by n", fontsize=13, y=1.01)
fig.savefig("scripts/9_picker_diagnostics/fig_scan_by_n.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig_scan_by_n.png", bbox_inches="tight")
print("Saved fig_scan_by_n")

# ---------------------------------------------------------------------------
# Version 2: panels by entropy H, color/marker by nqubit
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

def nearest_H(n, H_target, tol=0.05):
    """Return the closest available H key for n within tol, or None."""
    keys = list(data[n].keys())
    if not keys:
        return None
    closest = min(keys, key=lambda h: abs(h - H_target))
    return closest if abs(closest - H_target) < tol else None

for ax, H_target in zip(axes, [0.0, 0.5, 1.0]):
    any_plotted = False
    for n in N_vals:
        n0 = nstate0[n]
        H = nearest_H(n, H_target)
        if H is None:
            continue
        pts = sorted(data[n][H], key=lambda x: x[0])
        xs = [ns / n0 for ns, _ in pts]
        ys = [eta for _, eta in pts]
        ax.plot(xs, ys, N_MARKERS[n] + "-", color=N_COLORS[n], lw=2, ms=5)
        best = max(pts, key=lambda x: x[1])
        ax.vlines(best[0]/n0, 0, best[1], color=N_COLORS[n], lw=1.5, ls="--", alpha=0.7)
        any_plotted = True

    ax.set_xlabel("nstate / nstate₀")
    ax.set_ylabel("Efficiency η" if H_target == 0.0 else "")
    ax.set_title(f"H ≈ {H_target:.1f}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.tight_layout()
fig.subplots_adjust(right=0.88)
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
n_count = len(N_vals)
n_cbar_vals = [i / max(n_count - 1, 1) for i in range(n_count)]
qualitative_colorbar(fig, cax,
                     values=n_cbar_vals,
                     labels=[f"n = {n}" for n in N_vals],
                     title="System size n")
fig.suptitle("Efficiency η vs nstate/nstate₀  —  panels by H", fontsize=13, y=1.01)
fig.savefig("scripts/9_picker_diagnostics/fig_scan_by_H.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig_scan_by_H.png", bbox_inches="tight")
print("Saved fig_scan_by_H")

# ---------------------------------------------------------------------------
# Figure: peak nstate* / nstate0 vs H for each nqubit
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for n in N_vals:
    n0 = nstate0[n]
    H_sorted = sorted(data[n].keys())
    peak_nstates     = [max(data[n][H], key=lambda x: x[1])[0] for H in H_sorted]
    peak_nstates_norm = [ns / n0 for ns in peak_nstates]
    peak_etas        = [max(data[n][H], key=lambda x: x[1])[1] for H in H_sorted]

    axes[0].plot(H_sorted, peak_nstates_norm, N_MARKERS[n] + "-",
                 color=N_COLORS[n], lw=2, ms=7, label=f"n = {n}")
    axes[1].plot(H_sorted, peak_etas, N_MARKERS[n] + "-",
                 color=N_COLORS[n], lw=2, ms=7, label=f"n = {n}")

axes[0].set_xlabel("Goal entropy H")
axes[0].set_ylabel("Optimal nstate* / nstate₀")
axes[0].set_title("A  Optimal ensemble size vs H")
axes[0].legend()
axes[0].set_xlim(0, 1)
axes[0].set_ylim(bottom=0)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Goal entropy H")
axes[1].set_ylabel("Peak efficiency η")
axes[1].set_title("B  Peak efficiency vs H")
axes[1].legend()
axes[1].set_xlim(0, 1)
axes[1].set_ylim(bottom=0)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Scaling summary vs goal entropy H", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("scripts/9_picker_diagnostics/fig_peak_vs_H.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig_peak_vs_H.png", bbox_inches="tight")
print("Saved fig_peak_vs_H")
