using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using Statistics
using Printf
using LaTeXStrings

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

function load_data(datafile)
    results = Dict()
    h5open(datafile, "r") do h5f
        log_every = read(HDF5.attributes(h5f)["log_every"])
        for nq_key in keys(h5f)
            gp_nq  = h5f[nq_key]
            nqubit = read(HDF5.attributes(gp_nq)["nqubit"])
            nsteps = read(HDF5.attributes(gp_nq)["nsteps"])
            nb     = read(HDF5.attributes(gp_nq)["nsteps_warmup"])

            results[nqubit] = Dict("nsteps" => nsteps, "warmup_boundary" => nb,
                                   "log_every" => log_every, "conditions" => [])

            for key in keys(gp_nq)
                gp = gp_nq[key]
                H  = read(HDF5.attributes(gp)["H_target"])

                hybrid_trajs = read(gp["hybrid_trajs"])  # [n_log_pts × nseeds]
                gsa_trajs    = read(gp["gsa_trajs"])
                hybrid_steps = read(gp["hybrid_steps"])  # reference x-axis (seed 1)
                gsa_steps    = read(gp["gsa_steps"])
                hybrid_final = read(gp["hybrid_final"])
                gsa_final    = read(gp["gsa_final"])

                push!(results[nqubit]["conditions"], Dict(
                    "H"            => H,
                    "hybrid_trajs" => hybrid_trajs,
                    "gsa_trajs"    => gsa_trajs,
                    "hybrid_steps" => hybrid_steps,
                    "gsa_steps"    => gsa_steps,
                    "hybrid_final" => hybrid_final,
                    "gsa_final"    => gsa_final,
                ))
            end
            sort!(results[nqubit]["conditions"], by = r -> r["H"])
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# Trajectory helpers
#
# The gsa trajectory has 3 * (nsteps / log_every) points because 3 restarts
# each log steps 1..nsteps into the same array. Reindex them to a continuous
# compute axis [log_every, 2*log_every, ..., 3*nsteps].
#
# The hybrid trajectory has warmup points (steps 1..nsteps_warmup) followed
# by 3 copies of annealing points (all offset to nsteps_warmup). Deduplicate
# the annealing portion by keeping the last (highest) value at each step.
# ---------------------------------------------------------------------------

function gsa_compute_axis(gsa_steps, nsteps, log_every)
    # gsa_steps repeats [log_every..nsteps] three times; reindex sequentially
    n_per_restart = nsteps ÷ log_every
    total         = 3 * n_per_restart
    return [i * log_every for i in 1:total]
end

function hybrid_compute_axis(hybrid_steps, nsteps_warmup, nsteps, log_every)
    # warmup portion: steps are already correct (log_every..nsteps_warmup)
    # annealing portion: 3 chains all mapped to same step range; deduplicate
    warmup_mask = hybrid_steps .<= nsteps_warmup
    w_steps = hybrid_steps[warmup_mask]

    ann_steps_raw = hybrid_steps[.!warmup_mask]
    ann_trajs_raw = nothing  # handled per-seed in the caller

    # x-axis for the annealing portion: after dedup, n_refine_pts unique steps
    nsteps_refine = nsteps - nsteps_warmup
    n_refine_pts  = nsteps_refine ÷ log_every
    ann_steps_unique = [nsteps_warmup + i * log_every for i in 1:n_refine_pts]

    return vcat(w_steps, ann_steps_unique)
end

function dedup_hybrid_traj(traj, hybrid_steps, nsteps_warmup, log_every)
    # warmup portion: take as-is
    warmup_mask = hybrid_steps .<= nsteps_warmup
    w_accs = traj[warmup_mask]

    # annealing portion: 3 chains share the same step values; max-pool per unique step
    ann_steps_raw = hybrid_steps[.!warmup_mask]
    ann_accs_raw  = traj[.!warmup_mask]
    unique_steps  = sort(unique(ann_steps_raw))
    ann_accs = [maximum(ann_accs_raw[ann_steps_raw .== s]) for s in unique_steps]

    return vcat(w_accs, ann_accs)
end

# ---------------------------------------------------------------------------
# Main plot: 2 rows (H=0, H=1) × 3 cols (n=4,6,8)
# Each panel: hybrid vs goal_sa trajectories + warmup boundary line
# ---------------------------------------------------------------------------

function main(datafile)
    outdir  = dirname(datafile)
    results = load_data(datafile)
    nqubits = sort(collect(keys(results)))

    H_targets   = [0.0, 1.0]
    colors      = Makie.wong_colors()
    col_hybrid  = colors[1]
    col_gsa     = colors[2]
    col_boundary = (:black, 0.4)

    n_rows = length(H_targets)
    n_cols = length(nqubits)
    fig    = Figure(size=(420 * n_cols, 340 * n_rows))

    for (row, H_t) in enumerate(H_targets)
        for (col, nq) in enumerate(nqubits)
            d      = results[nq]
            nsteps = d["nsteps"]
            nb     = d["warmup_boundary"]
            le     = d["log_every"]

            cond = findfirst(c -> abs(c["H"] - H_t) < 1e-9, d["conditions"])
            isnothing(cond) && continue
            c = d["conditions"][cond]

            ax = Axis(fig[row, col],
                title  = L"n = %$(nq),\; H = %$(H_t)",
                xlabel = row == n_rows ? "Compute steps" : "",
                ylabel = col == 1 ? "Best accuracy so far" : "",
                limits = (nothing, nothing, 0.2, 1.05),
            )

            # Build compute axes
            gsa_x    = gsa_compute_axis(c["gsa_steps"], nsteps, le)
            hybrid_x = hybrid_compute_axis(c["hybrid_steps"], nb, nsteps, le)

            nseeds = size(c["hybrid_trajs"], 2)

            # Plot per-seed trajectories (faded)
            for si in 1:nseeds
                h_traj = dedup_hybrid_traj(c["hybrid_trajs"][:, si],
                                           c["hybrid_steps"], nb, le)
                lines!(ax, hybrid_x, h_traj;
                       color=(col_hybrid, 0.2), linewidth=1)

                g_traj = c["gsa_trajs"][:, si]
                lines!(ax, gsa_x, g_traj;
                       color=(col_gsa, 0.2), linewidth=1)
            end

            # Median trajectories
            hybrid_mat = hcat([dedup_hybrid_traj(c["hybrid_trajs"][:, si],
                                                  c["hybrid_steps"], nb, le)
                               for si in 1:nseeds]...)
            gsa_mat    = c["gsa_trajs"]

            h_med = median(hybrid_mat, dims=2)[:]
            g_med = median(gsa_mat,    dims=2)[:]

            lines!(ax, hybrid_x, h_med;
                   color=col_hybrid, linewidth=2.5,
                   label="Hybrid (PT→goal SA)")
            lines!(ax, gsa_x, g_med;
                   color=col_gsa, linewidth=2.5,
                   label="Goal SA (3 restarts)")

            # Warmup boundary line
            vlines!(ax, [nb]; color=col_boundary, linestyle=:dash, linewidth=1.5)

            # Annotate final median values
            text!(ax, 0.97, 0.08;
                  text=@sprintf("hybrid %.3f\ngsa    %.3f",
                                median(c["hybrid_final"]), median(c["gsa_final"])),
                  align=(:right, :bottom), space=:relative, fontsize=11,
                  color=:black)

            row == 1 && col == 1 && axislegend(ax; position=:rb, labelsize=11)
        end
    end

    # Shared x-label annotation for warmup boundary
    Label(fig[0, :], "Dashed line marks end of PT warm-up phase";
          fontsize=12, color=:gray)

    save(joinpath(outdir, "hybrid_convergence.pdf"), fig)
    println("Saved hybrid_convergence.pdf")

    # ---------------------------------------------------------------------------
    # Second plot: final accuracy comparison bar chart, hybrid vs goal_sa
    # ---------------------------------------------------------------------------
    fig2 = Figure(size=(500, 380))
    ax2  = Axis(fig2[1, 1],
        xlabel = "n (nqubit)",
        ylabel = "Best accuracy (median)",
        title  = "Hybrid PT→goal SA vs goal SA alone",
        xticks = (1:length(nqubits)*length(H_targets),
                  [@sprintf("n=%d H=%.0f", nq, H)
                   for H in H_targets for nq in nqubits]),
        xticklabelrotation = π/4,
    )

    xs_hybrid = Float64[]
    xs_gsa    = Float64[]
    ys_hybrid = Float64[]
    ys_gsa    = Float64[]
    xtick_pos = Float64[]
    xtick_lbl = String[]

    for (i, H_t) in enumerate(H_targets)
        for (j, nq) in enumerate(nqubits)
            x = (i-1)*length(nqubits) + j
            d = results[nq]
            cond = findfirst(c -> abs(c["H"] - H_t) < 1e-9, d["conditions"])
            isnothing(cond) && continue
            c = d["conditions"][cond]
            push!(ys_hybrid, median(c["hybrid_final"]))
            push!(ys_gsa,    median(c["gsa_final"]))
            push!(xtick_pos, x)
            push!(xtick_lbl, @sprintf("n=%d\nH=%.0f", nq, H_t))
        end
    end

    xs = 1:length(ys_hybrid)
    barplot!(ax2, xs .- 0.2, ys_hybrid; width=0.35, color=col_hybrid, label="Hybrid")
    barplot!(ax2, xs .+ 0.2, ys_gsa;    width=0.35, color=col_gsa,    label="Goal SA")
    ax2.xticks = (collect(xs), xtick_lbl)
    axislegend(ax2; position=:lb)

    save(joinpath(outdir, "hybrid_vs_gsa_final.pdf"), fig2)
    println("Saved hybrid_vs_gsa_final.pdf")
end

length(ARGS) < 1 && error("Usage: julia plot_hybrid_convergence.jl <hybrid_convergence.h5>")
main(ARGS[1])
