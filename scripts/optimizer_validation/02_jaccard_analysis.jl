# Script 02: Pairwise Jaccard analysis across seeds.
#
# Loads solutions from 01_solutions.h5.  For each (nqubit, H):
#   1. Generator Jaccard: treats each PseudoGHZState as a unique element
#      identified by (theta_s, theta_z, alphas, gen_index).
#      J(A,B) = |A ∩ B| / |A ∪ B|  where A,B are sets of state keys.
#
#   2. Rep parity agreement: compares majority-vote sign at each position
#      across the two rep vectors, maximised over global parity flip.
#
# All C(n_seeds, 2) pairwise values are collected into a distribution.
# Results are saved to data/02_jaccard.h5 and plotted.
#
# Skip logic: if data/02_jaccard.h5 already exists, Jaccard computation is
# skipped and only the plots are regenerated.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie, HDF5, Printf, Statistics

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

# ---------------------------------------------------------------------------
# State key for generator Jaccard.
# Uses gen_index + alphas + theta_s/z.
# ---------------------------------------------------------------------------
function state_key(gen_index, theta_s, theta_z, alphas_row)
    return (theta_s, theta_z, Tuple(alphas_row), gen_index)
end

# ---------------------------------------------------------------------------
# Generator Jaccard for two ensembles loaded from HDF5 datasets.
# ---------------------------------------------------------------------------
function gen_jaccard(gen_a, ts_a, tz_a, al_a, gen_b, ts_b, tz_b, al_b)
    nstate = length(gen_a)
    sa = Set(state_key(gen_a[i], ts_a[i], tz_a[i], al_a[i, :]) for i in 1:nstate)
    sb = Set(state_key(gen_b[i], ts_b[i], tz_b[i], al_b[i, :]) for i in 1:nstate)
    isempty(sa) && isempty(sb) && return 1.0
    return length(intersect(sa, sb)) / length(union(sa, sb))
end

# ---------------------------------------------------------------------------
# Majority parity at each position: +1, -1, or 0 (tied / uncovered).
# ---------------------------------------------------------------------------
function majority_parity(rep_sum, rep_ctr)
    n   = length(rep_sum)
    out = zeros(Int, n)
    for k in 1:n
        rep_ctr[k] == 0 && continue
        v = 2 * rep_sum[k] - rep_ctr[k]
        out[k] = v > 0 ? 1 : (v < 0 ? -1 : 0)
    end
    return out
end

# ---------------------------------------------------------------------------
# Rep agreement: fraction of jointly-covered positions with matching parity,
# maximised over global flip.
# ---------------------------------------------------------------------------
function rep_agreement(rs_a, rc_a, rs_b, rc_b)
    p1   = majority_parity(rs_a, rc_a)
    p2   = majority_parity(rs_b, rc_b)
    both = [i for i in eachindex(p1) if p1[i] != 0 && p2[i] != 0]
    isempty(both) && return NaN
    agree      = count(i -> p1[i] ==  p2[i], both) / length(both)
    agree_flip = count(i -> p1[i] == -p2[i], both) / length(both)
    return max(agree, agree_flip)
end

# ---------------------------------------------------------------------------
# Load one HDF5 group into named arrays.
# ---------------------------------------------------------------------------
function load_seed(grp)
    return (
        gen_indices = read(grp["gen_indices"]),
        theta_s_vec = read(grp["theta_s_vec"]),
        theta_z_vec = read(grp["theta_z_vec"]),
        alphas_mat  = read(grp["alphas_mat"]),
        rep_sum     = read(grp["rep_sum"]),
        rep_ctr     = read(grp["rep_ctr"]),
        final_acc   = read(HDF5.attributes(grp)["final_acc"]),
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    infile  = joinpath(@__DIR__, "data/01_solutions.h5")
    outfile = joinpath(@__DIR__, "data/02_jaccard.h5")
    outdir  = joinpath(@__DIR__, "data")

    isfile(infile) || error("Run 01_collect_solutions.jl first.")

    # Storage: jac_gen[nqubit][H] = Float64[] of pairwise values
    jac_gen_data = Dict{Int, Dict{Float64, Vector{Float64}}}()
    rep_agr_data = Dict{Int, Dict{Float64, Vector{Float64}}}()
    acc_data     = Dict{Int, Dict{Float64, Vector{Float64}}}()
    nqubits      = Int[]
    all_H        = Float64[]

    if isfile(outfile)
        println("Loading existing Jaccard data from $outfile (skipping recomputation).")
        h5open(outfile, "r") do h5f
            for nq_key in sort(collect(keys(h5f)))
                g      = h5f[nq_key]
                nqubit = read(HDF5.attributes(g)["nqubit"])
                push!(nqubits, nqubit)
                jac_gen_data[nqubit] = Dict{Float64, Vector{Float64}}()
                rep_agr_data[nqubit] = Dict{Float64, Vector{Float64}}()
                acc_data[nqubit]     = Dict{Float64, Vector{Float64}}()
                for H_key in keys(g)
                    gp = g[H_key]
                    H  = read(HDF5.attributes(gp)["H"])
                    H in all_H || push!(all_H, H)
                    jac_gen_data[nqubit][H] = read(gp["jac_gen"])
                    rep_agr_data[nqubit][H] = read(gp["rep_agr"])
                    acc_data[nqubit][H]     = read(gp["accs"])
                end
            end
        end
    else
        h5open(infile, "r") do h5f
            for nq_key in sort(collect(keys(h5f)))
                g      = h5f[nq_key]
                nqubit = read(HDF5.attributes(g)["nqubit"])
                push!(nqubits, nqubit)
                jac_gen_data[nqubit] = Dict{Float64, Vector{Float64}}()
                rep_agr_data[nqubit] = Dict{Float64, Vector{Float64}}()
                acc_data[nqubit]     = Dict{Float64, Vector{Float64}}()

                for H_key in sort(collect(keys(g)))
                    H_grp = g[H_key]
                    H     = read(HDF5.attributes(H_grp)["H_target"])
                    H in all_H || push!(all_H, H)

                    seed_keys = sort(collect(keys(H_grp)))
                    seeds     = [load_seed(H_grp[sk]) for sk in seed_keys]
                    nseed     = length(seeds)

                    jac_vals = Float64[]
                    rep_vals = Float64[]

                    for i in 1:nseed, j in (i+1):nseed
                        a, b = seeds[i], seeds[j]
                        jv = gen_jaccard(a.gen_indices, a.theta_s_vec, a.theta_z_vec, a.alphas_mat,
                                         b.gen_indices, b.theta_s_vec, b.theta_z_vec, b.alphas_mat)
                        rv = rep_agreement(a.rep_sum, a.rep_ctr, b.rep_sum, b.rep_ctr)
                        push!(jac_vals, jv)
                        isnan(rv) || push!(rep_vals, rv)
                    end

                    jac_gen_data[nqubit][H] = jac_vals
                    rep_agr_data[nqubit][H] = rep_vals
                    acc_data[nqubit][H]     = [s.final_acc for s in seeds]

                    @printf("nqubit=%d  H=%.3f  seeds=%d  jac=%.3f±%.3f  rep=%.3f±%.3f\n",
                        nqubit, H, nseed,
                        isempty(jac_vals) ? NaN : mean(jac_vals),
                        isempty(jac_vals) ? NaN : std(jac_vals),
                        isempty(rep_vals) ? NaN : mean(rep_vals),
                        isempty(rep_vals) ? NaN : std(rep_vals),
                    )
                end
            end
        end

        sort!(nqubits); sort!(all_H)

        # Save distributions to HDF5
        h5open(outfile, "w") do h5f
            for nqubit in nqubits
                g = create_group(h5f, "nq$(nqubit)")
                attributes(g)["nqubit"] = nqubit
                for H in all_H
                    haskey(jac_gen_data[nqubit], H) || continue
                    gp = create_group(g, @sprintf("H%.3f", H))
                    attributes(gp)["H"] = H
                    gp["jac_gen"] = jac_gen_data[nqubit][H]
                    gp["rep_agr"] = rep_agr_data[nqubit][H]
                    gp["accs"]    = acc_data[nqubit][H]
                end
            end
        end
        println("Saved $outfile")
    end

    # -----------------------------------------------------------------------
    # Figure 1: pairwise distributions as violin plots, one subplot per nqubit.
    # -----------------------------------------------------------------------
    nq_colors = Makie.wong_colors()[1:length(nqubits)]

    for (metric_key, metric_name, metric_data) in [
            ("jac_gen", "Generator Jaccard",  jac_gen_data),
            ("rep_agr", "Rep parity agreement", rep_agr_data),
        ]

        fig = Figure(size=(350 * length(nqubits), 420))
        for (ci, nqubit) in enumerate(nqubits)
            ax = Axis(fig[1, ci],
                title  = "nqubit = $nqubit",
                xlabel = "H",
                ylabel = ci == 1 ? metric_name : "",
            )
            ylims!(ax, 0, 1)

            Hs_present = sort([H for H in all_H if haskey(metric_data[nqubit], H)])
            for (pos, H) in enumerate(Hs_present)
                vals = metric_data[nqubit][H]
                isempty(vals) && continue
                violin!(ax, fill(Float64(pos), length(vals)), vals;
                        color=(nq_colors[ci], 0.6))
                scatter!(ax, [Float64(pos)], [mean(vals)];
                         color=nq_colors[ci], markersize=8)
            end

            ax.xticks = (1:length(Hs_present), [@sprintf("%.3f", H) for H in Hs_present])
            ax.xticklabelrotation = pi/4

            ci > 1 && hideydecorations!(ax, grid=false)
        end

        outpng = joinpath(outdir, "02_$(metric_key)_by_nqubit.pdf")
        save(outpng, fig)
        println("Saved $outpng")
    end

    # -----------------------------------------------------------------------
    # Figure 2: mean pairwise Jaccard vs H, one line per nqubit.
    # -----------------------------------------------------------------------
    for (metric_key, metric_name, metric_data) in [
            ("jac_gen", "Mean generator Jaccard",   jac_gen_data),
            ("rep_agr", "Mean rep parity agreement", rep_agr_data),
        ]

        fig = Figure(size=(600, 420))
        ax  = Axis(fig[1, 1],
            xlabel = "H",
            ylabel = metric_name,
            title  = "$metric_name vs entropy",
        )
        ylims!(ax, 0, 1)

        for (ci, nqubit) in enumerate(nqubits)
            Hs_present = sort([H for H in all_H if haskey(metric_data[nqubit], H)])
            means = [mean(metric_data[nqubit][H]) for H in Hs_present]
            lines!(ax, Hs_present, means; color=nq_colors[ci], linewidth=2, label="n=$nqubit")
            scatter!(ax, Hs_present, means; color=nq_colors[ci], markersize=8)
        end

        axislegend(ax; position=:lt)
        out = joinpath(outdir, "02_$(metric_key)_vs_H.pdf")
        save(out, fig)
        println("Saved $out")
    end
end

main()
