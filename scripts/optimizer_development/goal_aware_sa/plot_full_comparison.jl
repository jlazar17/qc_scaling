using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using Statistics
using LaTeXStrings

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function efficiency(acc::Float64, nqubit::Int, nstate::Int)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    return (1 - binary_entropy(acc)) * n_classical / n_quantum
end

function load_data(datafile)
    results = Dict()
    h5open(datafile, "r") do h5f
        for nq_key in keys(h5f)
            gp_nq   = h5f[nq_key]
            nqubit  = read(HDF5.attributes(gp_nq)["nqubit"])
            nstate  = read(HDF5.attributes(gp_nq)["nstate"])
            results[nqubit] = Dict("nstate" => nstate, "hamming" => [], "bernoulli" => [])

            for gkey in keys(gp_nq)
                gp   = gp_nq[gkey]
                mode = read(HDF5.attributes(gp)["mode"])
                row = Dict{String, Any}(
                    "base"     => read(gp["base"]),
                    "improved" => read(gp["improved"]),
                    "rand_sa"  => read(gp["rand_sa"]),
                    "goal_sa"  => read(gp["goal_sa"]),
                )
                if mode == "hamming"
                    row["H"] = read(HDF5.attributes(gp)["H_target"])
                    row["k"] = read(HDF5.attributes(gp)["k"])
                    push!(results[nqubit]["hamming"], row)
                else
                    row["pzero"] = read(HDF5.attributes(gp)["pzero"])
                    row["H_emp"] = read(HDF5.attributes(gp)["H_empirical"])
                    push!(results[nqubit]["bernoulli"], row)
                end
            end

            # sort by entropy
            sort!(results[nqubit]["hamming"],   by = r -> r["H"])
            sort!(results[nqubit]["bernoulli"], by = r -> r["H_emp"])
        end
    end
    return results
end

function main(datafile)
    outdir  = dirname(datafile)
    results = load_data(datafile)
    nqubits = sort(collect(keys(results)))

    methods      = ["base", "improved", "rand_sa", "goal_sa"]
    method_labels = ["Base", "Improved", "Rand SA", "Goal SA"]
    colors        = Makie.wong_colors()[1:4]
    markers       = [:circle, :rect, :diamond, :utriangle]

    # -----------------------------------------------------------------------
    # Plot 1: accuracy vs entropy, one panel per nqubit
    # Hamming = solid lines, Bernoulli = dashed markers overlaid
    # -----------------------------------------------------------------------
    fig1 = Figure(size=(400 * length(nqubits), 380))

    for (col, nq) in enumerate(nqubits)
        d      = results[nq]
        nstate = d["nstate"]
        hrows  = d["hamming"]
        brows  = d["bernoulli"]
        H_vals = [r["H"] for r in hrows]

        ax = Axis(fig1[1, col],
            title  = L"n = %$(nq),\; N_s = %$(nstate)",
            xlabel = "Goal entropy H",
            ylabel = col == 1 ? "Accuracy (median)" : "",
            limits = (nothing, nothing, 0.4, 1.05),
        )

        for (mi, (meth, label)) in enumerate(zip(methods, method_labels))
            ys = [median(r[meth]) for r in hrows]
            lines!(ax,  H_vals, ys; color=colors[mi], linewidth=2, label=label)
            scatter!(ax, H_vals, ys; color=colors[mi], marker=markers[mi], markersize=10)

            # Bernoulli overlay as open markers
            for brow in brows
                scatter!(ax, [brow["H_emp"]], [median(brow[meth])];
                         color=(:white, 0.0), marker=markers[mi], markersize=12,
                         strokecolor=colors[mi], strokewidth=2)
            end
        end

        col == 1 && axislegend(ax; position=:lb, labelsize=11)
    end

    save(joinpath(outdir, "comparison_vs_entropy.pdf"), fig1)
    println("Saved comparison_vs_entropy.pdf")

    # -----------------------------------------------------------------------
    # Plot 2: goal_sa accuracy vs nqubit for each entropy, Hamming only
    # -----------------------------------------------------------------------
    fig2 = Figure(size=(480, 380))
    ax2  = Axis(fig2[1, 1],
        xlabel = "Number of qubits n",
        ylabel = "Accuracy (median, goal SA)",
        title  = "Goal-aware SA scaling with n",
    )

    entropy_colors  = Makie.wong_colors()[[1,3,5]]
    entropy_markers = [:circle, :rect, :utriangle]
    H_targets       = [0.0, 0.5, 1.0]

    for (ei, H_t) in enumerate(H_targets)
        xs = Int[]; ys = Float64[]
        for nq in nqubits
            hrows = results[nq]["hamming"]
            row   = findfirst(r -> abs(r["H"] - H_t) < 1e-9, hrows)
            isnothing(row) && continue
            push!(xs, nq)
            push!(ys, median(hrows[row]["goal_sa"]))
        end
        lines!(ax2,  xs, ys; color=entropy_colors[ei], linewidth=2,
               label=L"H=%$(H_t)")
        scatter!(ax2, xs, ys; color=entropy_colors[ei],
                 marker=entropy_markers[ei], markersize=12)
    end

    ax2.xticks = nqubits
    axislegend(ax2; position=:lb)
    save(joinpath(outdir, "goal_sa_scaling.pdf"), fig2)
    println("Saved goal_sa_scaling.pdf")

    # -----------------------------------------------------------------------
    # Plot 3: efficiency η vs entropy, one panel per nqubit
    # -----------------------------------------------------------------------
    fig3 = Figure(size=(400 * length(nqubits), 380))

    for (col, nq) in enumerate(nqubits)
        d      = results[nq]
        nstate = d["nstate"]
        hrows  = d["hamming"]
        H_vals = [r["H"] for r in hrows]

        ax = Axis(fig3[1, col],
            title  = L"n = %$(nq),\; N_s = %$(nstate)",
            xlabel = "Goal entropy H",
            ylabel = col == 1 ? L"\eta = (1-H(\mathrm{acc})) \cdot N_c / N_q" : "",
        )

        eta_perfect = (3^nq - 1) / (2 * nq * nstate)
        hlines!(ax, [eta_perfect]; linestyle=:dash, color=:black, linewidth=1,
                label="acc = 1")

        for (mi, (meth, label)) in enumerate(zip(methods, method_labels))
            ys = [efficiency(median(r[meth]), nq, nstate) for r in hrows]
            lines!(ax,  H_vals, ys; color=colors[mi], linewidth=2, label=label)
            scatter!(ax, H_vals, ys; color=colors[mi], marker=markers[mi], markersize=10)
        end

        col == 1 && axislegend(ax; position=:lb, labelsize=11)
    end

    save(joinpath(outdir, "efficiency_vs_entropy.pdf"), fig3)
    println("Saved efficiency_vs_entropy.pdf")

    # -----------------------------------------------------------------------
    # Plot 4: Hamming vs Bernoulli direct comparison for goal_sa
    # -----------------------------------------------------------------------
    fig4 = Figure(size=(400 * length(nqubits), 380))

    for (col, nq) in enumerate(nqubits)
        d      = results[nq]
        hrows  = d["hamming"]
        brows  = d["bernoulli"]

        ax = Axis(fig4[1, col],
            title  = L"n = %$(nq)",
            xlabel = "Goal entropy H",
            ylabel = col == 1 ? "Goal SA accuracy (median)" : "",
            limits = (nothing, nothing, 0.4, 1.05),
        )

        H_h = [r["H"]     for r in hrows]
        y_h = [median(r["goal_sa"]) for r in hrows]
        H_b = [r["H_emp"] for r in brows]
        y_b = [median(r["goal_sa"]) for r in brows]

        lines!(ax,  H_h, y_h; color=colors[1], linewidth=2, label="Hamming (fixed k)")
        scatter!(ax, H_h, y_h; color=colors[1], markersize=10)
        scatter!(ax, H_b, y_b; color=colors[2], marker=:rect, markersize=12,
                 label="Bernoulli (pzero)")

        col == 1 && axislegend(ax; position=:lb, labelsize=11)
    end

    save(joinpath(outdir, "hamming_vs_bernoulli.pdf"), fig4)
    println("Saved hamming_vs_bernoulli.pdf")
end

length(ARGS) < 1 && error("Usage: julia plot_full_comparison.jl <datafile.h5>")
main(ARGS[1])
