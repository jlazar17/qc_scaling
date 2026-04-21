"""
Summary plots for the H=0 vs H=1 efficiency gap investigation.
Generates figures suitable for sharing with theory colleagues.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {"H0": "#2166ac", "H05": "#4dac26", "H1": "#d6604d"}
NS = [4, 6, 8]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Scaling study: peak η at optimal nstate  (from scaling_study_adaptive.log)
eta = {
    0.0: [0.50, 1.068, 1.733],
    0.5: [0.455, 0.692, 0.915],
    1.0: [0.438, 0.495, 0.364],
}
nstate_opt = {
    0.0: [20,  40,  147],
    1.0: [19,  97,  599],
}

# GF(2) capacity vs H  (gf2_capacity_vs_H.log)
gf2_H = {
    4: {
        "H_act": [0.0, 0.1687, 0.1687, 0.2864, 0.3843, 0.4690, 0.6098, 0.7219, 0.8113, 0.9097, 1.0],
        "frac_zero": [0.000, 0.000, 0.000, 0.059, 0.114, 0.080, 0.081, 0.119, 0.113, 0.167, 0.156],
    },
    6: {
        "H_act": [0.0, 0.1047, 0.1955, 0.2957, 0.4010, 0.4996, 0.5987, 0.7006, 0.7980, 0.8999, 1.0],
        "frac_zero": [0.000, 0.043, 0.098, 0.179, 0.323, 0.345, 0.442, 0.566, 0.657, 0.751, 0.811],
    },
    8: {
        "H_act": [0.0, 0.1008, 0.1999, 0.3005, 0.3996, 0.5001, 0.5998, 0.7001, 0.8000, 0.9002, 1.0],
        "frac_zero": [0.000, 0.175, 0.384, 0.601, 0.724, 0.810, 0.891, 0.934, 0.956, 0.976, 0.981],
    },
}

# GF(2) capacity vs n at H=1  (gf2_capacity_vs_n.log)
gf2_n = {
    "n": [4, 6, 8],
    "frac_with_pairs": [0.178, 0.218, 0.234],
    "avg_pairs": [1.35, 4.16, 15.81],
    "frac_zero_H1": [0.2223, 0.8440, 0.9880],
}

# Per-state frustration  (per_state_frustration.log)
frustration = {
    6: {
        "H_act": [0.0, 0.2478, 0.4996, 0.7496, 1.0],
        "mean_max": [0.762, 0.753, 0.752, 0.744, 0.746],
        "med_max":  [0.750, 0.750, 0.750, 0.739, 0.739],
    },
    8: {
        "H_act": [0.0, 0.2503, 0.5001, 0.7502, 1.0],
        "mean_max": [0.758, 0.741, 0.723, 0.646, 0.643],
        "med_max":  [0.748, 0.731, 0.713, 0.643, 0.641],
    },
}

# GF(2)-constrained ablation  (gf2_constrained_ablation.log)
ablation = {
    6: {
        "H_act":     [0.000, 0.105, 0.196, 0.296, 0.401, 0.500, 0.599, 0.701, 0.798, 0.900, 1.000],
        "acc_smart": [0.9245, 0.9011, 0.9025, 0.8791, 0.8654, 0.8448, 0.8173, 0.7830, 0.7651, 0.7610, 0.7596],
        "acc_gf2":   [0.9148, 0.9052, 0.8956, 0.8819, 0.8723, 0.8503, 0.8283, 0.7967, 0.7665, 0.7637, 0.7527],
    },
    8: {
        "H_act":     [0.000, 0.250, 0.500, 0.750, 1.000],
        "acc_smart": [0.9213, 0.8845, 0.8131, 0.7375, 0.6085],
        "acc_gf2":   [0.9116, 0.8723, 0.8274, 0.7378, 0.6140],
    },
}

# Optimal nstate experiment  (optimal_nstate_h1.log)
optimal_nstate_exp = {
    6: {"H0_opt": (40, 0.9121, 0.865), "H1_subopt": (40, 0.7555, 0.300), "H1_opt": (97, 0.9698, 0.503)},
    8: {"H0_opt": (147, 0.9204, 1.672), "H1_subopt": (147, 0.6177, 0.113), "H1_opt": (599, 0.8942, 0.351)},
}


# ---------------------------------------------------------------------------
# Figure 1: Main result — η vs n, and nstate* vs n
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: η vs n
ax = axes[0]
ax.plot(NS, eta[0.0], "o-", color=COLORS["H0"], lw=2, ms=7, label="H=0.0")
ax.plot(NS, eta[0.5], "s-", color=COLORS["H05"], lw=2, ms=7, label="H=0.5")
ax.plot(NS, eta[1.0], "^-", color=COLORS["H1"], lw=2, ms=7, label="H=1.0")
ax.set_xlabel("n (number of qubits)")
ax.set_ylabel("Peak efficiency η")
ax.set_xticks(NS)
ax.set_title("A  Peak efficiency vs system size")
ax.legend()
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# Panel B: optimal nstate* vs n
ax = axes[1]
ax.plot(NS, nstate_opt[0.0], "o-", color=COLORS["H0"], lw=2, ms=7, label="H=0")
ax.plot(NS, nstate_opt[1.0], "^-", color=COLORS["H1"], lw=2, ms=7, label="H=1")
ratios = [h1/h0 for h0, h1 in zip(nstate_opt[0.0], nstate_opt[1.0])]
ax2 = ax.twinx()
ax2.plot(NS, ratios, "D--", color="gray", lw=1.5, ms=6, label="ratio H1/H0")
ax2.set_ylabel("nstate*(H=1) / nstate*(H=0)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax2.set_ylim(bottom=0)
ax.set_xlabel("n (number of qubits)")
ax.set_ylabel("Optimal nstate*")
ax.set_xticks(NS)
ax.set_title("B  Optimal ensemble size vs system size")
ax.set_yscale("log")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("scripts/9_picker_diagnostics/fig1_main_result.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig1_main_result.png", bbox_inches="tight")
print("Saved fig1_main_result")


# ---------------------------------------------------------------------------
# Figure 2: GF(2) wall — frac_zero_valid vs H for n=4,6,8
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

n_colors = {4: "#abdda4", 6: "#2b83ba", 8: "#d7191c"}
n_markers = {4: "o", 6: "s", 8: "^"}

ax = axes[0]
for n in [4, 6, 8]:
    ax.plot(gf2_H[n]["H_act"], gf2_H[n]["frac_zero"],
            n_markers[n] + "-", color=n_colors[n], lw=2, ms=6, label=f"n={n}")
ax.set_xlabel("Goal entropy H")
ax.set_ylabel("Fraction of generators with zero valid configs")
ax.set_title("A  GF(2) infeasibility vs entropy H")
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.axvline(0.5, color="gray", lw=1, ls=":")

ax = axes[1]
eff_frac = [gf2_n["frac_with_pairs"][i] * gf2_n["frac_zero_H1"][i] for i in range(3)]
ax.bar([4, 6, 8], gf2_n["frac_zero_H1"], width=0.8, color=[n_colors[n] for n in [4,6,8]],
       alpha=0.7, label="among generators with both-covered pairs")
ax.bar([4, 6, 8], eff_frac, width=0.8, color=[n_colors[n] for n in [4,6,8]],
       alpha=1.0, label="effective (all generators)")
ax.set_xlabel("n (number of qubits)")
ax.set_ylabel("Fraction with zero valid H=1 configs")
ax.set_title("B  GF(2) infeasibility at H=1 vs n")
ax.set_xticks([4, 6, 8])
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("scripts/9_picker_diagnostics/fig2_gf2_wall.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig2_gf2_wall.png", bbox_inches="tight")
print("Saved fig2_gf2_wall")


# ---------------------------------------------------------------------------
# Figure 3: What explains the gap? Ablation summary
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: GF(2)-constrained picker ablation at n=8
ax = axes[0]
d = ablation[8]
ax.plot(d["H_act"], d["acc_smart"], "o-", color="#2166ac", lw=2, ms=6, label="Standard picker")
ax.plot(d["H_act"], d["acc_gf2"],  "s--", color="#d6604d", lw=2, ms=6, label="GF(2)-constrained")
ax.set_xlabel("Goal entropy H")
ax.set_ylabel("Median accuracy")
ax.set_title("A  GF(2) picker ablation (n=8, nstate=147)")
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.0)
ax.grid(True, alpha=0.3)

delta8 = [g - s for g, s in zip(d["acc_gf2"], d["acc_smart"])]
ax_ins = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
ax_ins.bar(d["H_act"], delta8, width=0.08,
           color=["#d6604d" if v >= 0 else "#2166ac" for v in delta8])
ax_ins.axhline(0, color="black", lw=0.8)
ax_ins.set_xlabel("H", fontsize=9)
ax_ins.set_ylabel("Δacc", fontsize=9)
ax_ins.tick_params(labelsize=8)

# Panel B: Per-state frustration
ax = axes[1]
for n, ls in [(6, "-"), (8, "--")]:
    f = frustration[n]
    ax.plot(f["H_act"], f["mean_max"], "o" + ls, color=n_colors[n],
            lw=2, ms=6, label=f"n={n} mean")
    ax.plot(f["H_act"], f["med_max"],  "s" + ls, color=n_colors[n],
            lw=1.5, ms=5, alpha=0.5, label=f"n={n} median")
ax.axhline(0.5, color="gray", lw=1, ls=":", label="chance")
ax.set_xlabel("Goal entropy H")
ax.set_ylabel("Max achievable companion-goal fraction")
ax.set_title("B  Per-state frustration (max over all configs)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0.4, 0.85)
ax.grid(True, alpha=0.3)

# Panel C: η at optimal nstate — H=0 vs H=1
ax = axes[2]
labels = ["H=0\n(opt nstate)", "H=1\n(sub-opt)", "H=1\n(opt nstate)"]
x = np.array([0, 1, 2])
w = 0.3
for i, (nn, col) in enumerate([(6, "#2b83ba"), (8, "#d7191c")]):
    exp = optimal_nstate_exp[nn]
    etas = [exp["H0_opt"][2], exp["H1_subopt"][2], exp["H1_opt"][2]]
    ax.bar(x + (i - 0.5) * w, etas, width=w, color=col, alpha=0.8, label=f"n={nn}")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Efficiency η")
ax.set_title("C  η at optimal nstate for each H")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("scripts/9_picker_diagnostics/fig3_ablations.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig3_ablations.png", bbox_inches="tight")
print("Saved fig3_ablations")


# ---------------------------------------------------------------------------
# Figure 4: Coverage scaling and summary of ruled-out mechanisms
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Coverage statistics (pair_cov_frac ~ (2/3)^n)
ax = axes[0]
ns_cov = [4, 6, 8, 10]
pair_cov = [0.216245, 0.088023, 0.038193, 0.016879]
states_to_cover = [1/p for p in pair_cov]
pred = [(2/3)**n for n in ns_cov]
ax.semilogy(ns_cov, pair_cov, "o-", color="#2166ac", lw=2, ms=7, label="pair_cov_frac (measured)")
ax.semilogy(ns_cov, pred, "--", color="#2166ac", lw=1.5, alpha=0.6, label="(2/3)^n (prediction)")
ax2 = ax.twinx()
ax2.semilogy(ns_cov, states_to_cover, "s-", color="#d6604d", lw=2, ms=7, label="states_to_cover ≈ (3/2)^n")
pred2 = [(3/2)**n for n in ns_cov]
ax2.semilogy(ns_cov, pred2, "--", color="#d6604d", lw=1.5, alpha=0.6)
ax.set_xlabel("n (number of qubits)")
ax.set_ylabel("Pair coverage fraction per state", color="#2166ac")
ax2.set_ylabel("States needed for full coverage", color="#d6604d")
ax.tick_params(axis="y", labelcolor="#2166ac")
ax2.tick_params(axis="y", labelcolor="#d6604d")
ax.set_title("A  Coverage scaling: pair_cov_frac ~ (2/3)^n")
ax.set_xticks(ns_cov)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Summary table — what we ruled out
ax = axes[1]
ax.axis("off")
table_data = [
    ["Mechanism", "Test", "Result"],
    ["SA gets stuck in bad basins", "Basin variance (100 seeds)", "RULED OUT\nH=1 has lower variance"],
    ["Per-step landscape is harder", "Acceptance trajectory", "RULED OUT\nIdentical across all H"],
    ["H=0 solution warms up H=1", "Warmstart H=0→H=1", "RULED OUT\nWarm start hurts (−0.018)"],
    ["Per-state frustration", "Max CG fraction vs H", "PARTIAL\nn=8: 0.76→0.64, no zeros"],
    ["GF(2) wall (infeasible proposals)", "Constrained picker ablation", "PARTIAL\n~5% of gap explained"],
    ["H=1 just needs more states", "Optimal nstate experiment", "CONFIRMED\n4.1× more at n=8"],
    ["nstate* ratio grows with n", "n=4,6,8 comparison", "CONFIRMED\n1×→2.4×→4.1×"],
]
col_widths = [0.28, 0.30, 0.30]
row_colors = [["#d0d0d0"]*3] + [["#f5f5f5" if i%2==0 else "white"]*3 for i in range(len(table_data)-1)]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="left", loc="center",
                 colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.2, 2.2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#404040")
        cell.set_text_props(color="white", fontweight="bold")
    elif "RULED OUT" in cell.get_text().get_text():
        cell.set_facecolor("#fde0d0")
    elif "CONFIRMED" in cell.get_text().get_text():
        cell.set_facecolor("#d0f0d0")
    elif "PARTIAL" in cell.get_text().get_text():
        cell.set_facecolor("#fff0c0")
ax.set_title("B  Summary of diagnostic experiments", pad=15)

fig.tight_layout()
fig.savefig("scripts/9_picker_diagnostics/fig4_coverage_summary.pdf", bbox_inches="tight")
fig.savefig("scripts/9_picker_diagnostics/fig4_coverage_summary.png", bbox_inches="tight")
print("Saved fig4_coverage_summary")

print("\nAll figures saved to scripts/9_picker_diagnostics/")
