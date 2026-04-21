using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

c_H0 = Makie.wong_colors()[1]   # blue
c_H1 = Makie.wong_colors()[2]   # orange

# ---------------------------------------------------------------------------
# Data (hardcoded from run outputs)
# ---------------------------------------------------------------------------

# Script 03: positive proposal rate vs accuracy
# nqubit=6, shuffled pairing, n_trials=6
acc_targets  = [0.70, 0.80, 0.88, 0.93]
pos_rate_H0  = [0.2828, 0.2248, 0.1303, 0.0474]
pos_rate_H1  = [0.1882, 0.0408, 0.0056, 0.0004]

# Script 07: acceptance rate by accuracy bucket
# nqubit=6, shuffled pairing, nstate=100, nsteps=60000, n_trials=8
bucket_centers = [0.60, 0.775, 0.90, 0.975]
bucket_labels  = ["[0.50,0.70)", "[0.70,0.85)", "[0.85,0.95)", "[0.95,1.00)"]
accept_rate_H0 = [0.8803, 0.7808, 0.3951, 0.0543]
accept_rate_H1 = [0.7729, 0.3146, 0.0070, NaN]

# Scripts 04 & 05: saturation curves (acc vs nstate)
# nqubit=6, 10 seeds, 3 restarts, nsteps=80000
sat_nstates_orig = [23, 33, 48, 70, 84, 121, 176]
sat_H0_orig      = [0.6731, 0.8214, 0.9368, 0.9945, 1.0000, 1.0000, 1.0000]
sat_H1_orig      = [0.5632, 0.6676, 0.7775, 0.8846, 0.9066, 0.9808, 1.0000]

sat_nstates_shuf = [23, 33, 48, 70, 84, 121, 146, 176]
sat_H0_shuf      = [0.6538, 0.8162, 0.9307, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
sat_H1_shuf      = [0.5612, 0.6613, 0.7712, 0.8819, 0.9107, 0.9890, 1.0000, 1.0000]

# ---------------------------------------------------------------------------
# Figure 1: Saturation curves — original vs shuffled pairing
# Tests whether both-covered pairs explain the 2x ratio (they don't)
# ---------------------------------------------------------------------------
fig1 = Figure(size=(700, 480))
ax = Axis(fig1[1, 1],
    xlabel = "Number of states",
    ylabel = "Best accuracy achieved",
    title  = "Saturation test: H=0 vs H=1 (nqubit=6)",
    xticks = [23, 70, 84, 121, 146, 176],
    yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Original pairing
lines!(ax, sat_nstates_orig, sat_H0_orig,
       color=c_H0, linewidth=2.5, label=L"H=0\ \mathrm{(original)}")
scatter!(ax, sat_nstates_orig, sat_H0_orig, color=c_H0, markersize=9)

lines!(ax, sat_nstates_orig, sat_H1_orig,
       color=c_H1, linewidth=2.5, label=L"H=1\ \mathrm{(original)}")
scatter!(ax, sat_nstates_orig, sat_H1_orig, color=c_H1, markersize=9)

# Shuffled pairing (dashed)
lines!(ax, sat_nstates_shuf, sat_H0_shuf,
       color=c_H0, linewidth=2, linestyle=:dash, label=L"H=0\ \mathrm{(shuffled)}")
scatter!(ax, sat_nstates_shuf, sat_H0_shuf, color=c_H0, markersize=7, marker=:diamond)

lines!(ax, sat_nstates_shuf, sat_H1_shuf,
       color=c_H1, linewidth=2, linestyle=:dash, label=L"H=1\ \mathrm{(shuffled)}")
scatter!(ax, sat_nstates_shuf, sat_H1_shuf, color=c_H1, markersize=7, marker=:diamond)

# Convergence markers
vlines!(ax, [84],  color=c_H0, linestyle=:dot, linewidth=1.2, alpha=0.7)
vlines!(ax, [176], color=c_H1, linestyle=:dot, linewidth=1.2, alpha=0.7)
text!(ax, 86,  0.505, text=L"n^*_{H=0}=84",  color=c_H0, fontsize=14)
text!(ax, 131, 0.505, text=L"n^*_{H=1}=176", color=c_H1, fontsize=14)

text!(ax, 155, 0.675,
      text="Shuffled pairing:\nratio = 2.09×\n(same as original 2.10×)",
      fontsize=12, color=:gray30, align=(:center, :center))

axislegend(ax, position=:rb, framevisible=false, fontsize=13)
ylims!(ax, 0.48, 1.04)
xlims!(ax, 15, 195)

save(joinpath(outdir, "fig1_saturation.pdf"), fig1)
save(joinpath(outdir, "fig1_saturation.png"), fig1, px_per_unit=2)
println("Saved fig1_saturation")

# ---------------------------------------------------------------------------
# Figure 2: Acceptance rate by accuracy bucket
# Shows the collapse in H=1 acceptance rate at high accuracy
# ---------------------------------------------------------------------------
fig2 = Figure(size=(700, 480))
ax2 = Axis(fig2[1, 1],
    xlabel  = "Accuracy bucket (midpoint)",
    ylabel  = "Step acceptance rate",
    title   = "SA acceptance rate by accuracy phase (nqubit=6, shuffled pairing)",
    xticks  = (bucket_centers, bucket_labels),
    xticklabelrotation = 0.3,
    yticks  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

# Bar width
w = 0.04
barplot!(ax2, bucket_centers .- w/2, accept_rate_H0,
         width=w, color=(c_H0, 0.85), label=L"H=0")
barplot!(ax2, bucket_centers .+ w/2,
         [isnan(v) ? 0.0 : v for v in accept_rate_H1],
         width=w, color=(c_H1, 0.85), label=L"H=1")

# Annotate H=1 value in the critical bucket
text!(ax2, bucket_centers[3] + w/2, 0.0075 + 0.015,
      text="0.7%", color=c_H1, fontsize=13, align=(:center, :bottom))
text!(ax2, bucket_centers[3] - w/2, 0.3951 + 0.015,
      text="39.5%", color=c_H0, fontsize=13, align=(:center, :bottom))

# Note that H=1 never reached the last bucket
text!(ax2, bucket_centers[4], 0.03,
      text="H=1 never\nreached here", color=c_H1, fontsize=12,
      align=(:center, :bottom))

axislegend(ax2, position=:rt, framevisible=false)
ylims!(ax2, 0.0, 1.08)

save(joinpath(outdir, "fig2_acceptance_rate.pdf"), fig2)
save(joinpath(outdir, "fig2_acceptance_rate.png"), fig2, px_per_unit=2)
println("Saved fig2_acceptance_rate")

# ---------------------------------------------------------------------------
# Figure 3a: Positive proposal rate vs accuracy (log scale)
# ---------------------------------------------------------------------------
fig3a = Figure(size=(640, 480))
ax3a = Axis(fig3a[1, 1],
    xlabel = "Current accuracy",
    ylabel = "Positive proposal rate",
    title  = "Proposal availability vs accuracy (nqubit=6, shuffled pairing)",
    yscale = log10,
    yticks = ([1e-4, 1e-3, 0.01, 0.1, 0.5], ["0.0001", "0.001", "0.01", "0.1", "0.5"]),
    xticks = acc_targets,
)

lines!(ax3a, acc_targets, pos_rate_H0, color=c_H0, linewidth=2.5, label=L"H=0")
scatter!(ax3a, acc_targets, pos_rate_H0, color=c_H0, markersize=11)

pr_H1_plot = copy(pos_rate_H1)
pr_H1_plot[end] = 4e-4
lines!(ax3a, acc_targets, pr_H1_plot, color=c_H1, linewidth=2.5, label=L"H=1")
scatter!(ax3a, acc_targets, pr_H1_plot, color=c_H1, markersize=11)

# Ratio annotations
for (i, acc) in enumerate(acc_targets)
    ratio = pos_rate_H0[i] / max(pos_rate_H1[i], 1e-5)
    label = ratio > 500 ? L"\infty" : @sprintf("%.0f×", ratio)
    ypos  = sqrt(pos_rate_H0[i] * max(pr_H1_plot[i], 4e-4))
    text!(ax3a, acc + 0.003, ypos, text=label, fontsize=13, color=:gray35,
          align=(:left, :center))
end

axislegend(ax3a, position=:lb, framevisible=false)
ylims!(ax3a, 5e-5, 1.0)
xlims!(ax3a, 0.67, 0.96)

save(joinpath(outdir, "fig3a_proposal_rate.pdf"), fig3a)
save(joinpath(outdir, "fig3a_proposal_rate.png"), fig3a, px_per_unit=2)
println("Saved fig3a_proposal_rate")
