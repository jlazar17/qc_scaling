using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using Statistics
using LaTeXStrings

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

struct RunResult
    best_acc::Float64
    final_acc::Float64
    convergence_iter::Int   # first iteration reaching best accuracy
    accuracies::Vector{Float64}
end

function load_file(path)
    results = Dict{NamedTuple, RunResult}()
    h5open(path, "r") do h5f
        for gname in keys(h5f)
            m = match(r"nstate_(\d+)_pzero_([\d.]+)_seed_(\d+)", gname)
            m === nothing && continue
            nstate = parse(Int,    m.captures[1])
            pzero  = parse(Float64, m.captures[2])
            seed   = parse(Int,    m.captures[3])

            accs = read(h5f[gname]["accuracies"])
            best = maximum(accs)
            final = accs[end]
            conv = findfirst(==(best), accs)
            key = (nstate=nstate, pzero=pzero, seed=seed)
            results[key] = RunResult(best, final, conv, accs)
        end
    end
    return results
end

function load_both(datadir)
    base     = load_file(joinpath(datadir, "base_nqubit_8.h5"))
    improved = load_file(joinpath(datadir, "improved_nqubit_8.h5"))
    # Only keep keys present in both
    common = intersect(keys(base), keys(improved))
    return Dict(k => base[k] for k in common),
           Dict(k => improved[k] for k in common)
end

function group_by(results, field)
    out = Dict()
    for (k, v) in results
        gk = getfield(k, field)
        push!(get!(out, gk, []), (k, v))
    end
    return out
end

# ---------------------------------------------------------------------------
# Plot 1: accuracy curves — mean ± IQR over seeds, base vs improved
# One panel per (pzero, multiplier) combo, arranged as a grid.
# ---------------------------------------------------------------------------

function plot_accuracy_curves(base, improved, outdir)
    nstates = sort(unique(k.nstate for k in keys(base)))
    pzeros  = sort(unique(k.pzero  for k in keys(base)))

    nrow, ncol = length(pzeros), length(nstates)
    fig = Figure(size=(400 * ncol, 320 * nrow))

    for (ci, ns) in enumerate(nstates)
        for (ri, pz) in enumerate(pzeros)
            ax = Axis(fig[ri, ci],
                xlabel = ci == ncol && ri == nrow ? "Iteration" : "",
                ylabel = ri == 1 && ci == 1 ? "Accuracy" : "",
                title  = L"p_0=%$(pz),\; N_s=%$(ns)",
            )

            for (label, color, results) in [
                ("base",     Makie.wong_colors()[1], base),
                ("improved", Makie.wong_colors()[2], improved),
            ]
                curves = [v.accuracies for (k, v) in results
                          if k.nstate == ns && k.pzero == pz]
                isempty(curves) && continue
                mat   = reduce(hcat, curves)       # iters × seeds
                niter = size(mat, 1)
                xs    = 1:niter
                meds  = [median(mat[i, :]) for i in xs]
                q1s   = [quantile(mat[i, :], 0.25) for i in xs]
                q3s   = [quantile(mat[i, :], 0.75) for i in xs]
                lines!(ax, xs, meds;  color, label)
                band!(ax,  xs, q1s, q3s; color=(color, 0.2))
            end

            if ri == 1 && ci == ncol
                axislegend(ax; position=:rb)
            end
        end
    end

    save(joinpath(outdir, "accuracy_curves.pdf"), fig)
    println("Saved accuracy_curves.pdf")
end

# ---------------------------------------------------------------------------
# Plot 2: best accuracy — paired bar chart, base vs improved
# X-axis: pzero; one group of bars per multiplier panel.
# ---------------------------------------------------------------------------

function plot_best_accuracy(base, improved, outdir)
    nstates = sort(unique(k.nstate for k in keys(base)))
    pzeros  = sort(unique(k.pzero  for k in keys(base)))

    fig = Figure(size=(600, 400 * length(pzeros)))

    for (ri, pz) in enumerate(pzeros)
        ax = Axis(fig[ri, 1],
            xlabel = ri == length(pzeros) ? L"N_{\mathrm{state}}" : "",
            ylabel = "Best accuracy (median)",
            title  = L"p_{\mathrm{zero}} = %$(pz)",
        )
        ylims!(ax, 0, 1.05)

        off = 0.0
        for (offset, label, color, results) in [
            (-1.0, "base",     Makie.wong_colors()[1], base),
            ( 1.0, "improved", Makie.wong_colors()[2], improved),
        ]
            meds = Float64[]
            q1s  = Float64[]
            q3s  = Float64[]
            xs   = Float64[]
            for (xi, ns) in enumerate(nstates)
                vals = [v.best_acc for (k, v) in results
                        if k.nstate == ns && k.pzero == pz]
                isempty(vals) && continue
                push!(xs,   xi + offset * 0.15)
                push!(meds, median(vals))
                push!(q1s,  quantile(vals, 0.25))
                push!(q3s,  quantile(vals, 0.75))
            end
            scatter!(ax, xs, meds; color, label, markersize=10)
            errorbars!(ax, xs, meds, meds .- q1s, q3s .- meds;
                       color, whiskerwidth=6)
            lines!(ax, xs, meds; color, linestyle=:dash, linewidth=1)
        end

        ax.xticks = (1:length(nstates), string.(nstates))
        ax.xticklabelrotation = π/4

        if ri == 1
            axislegend(ax; position=:rb)
        end
    end

    save(joinpath(outdir, "best_accuracy_comparison.pdf"), fig)
    println("Saved best_accuracy_comparison.pdf")
end

# ---------------------------------------------------------------------------
# Plot 3: delta heatmap — (improved best acc) - (base best acc)
# Rows: pzero, columns: nstate multiplier. Cell = median delta across seeds.
# ---------------------------------------------------------------------------

function plot_delta_heatmap(base, improved, outdir)
    nstates = sort(unique(k.nstate for k in keys(base)))
    pzeros  = sort(unique(k.pzero  for k in keys(base)))

    mat = fill(NaN, length(nstates), length(pzeros))
    for (ci, ns) in enumerate(nstates)
        for (ri, pz) in enumerate(pzeros)
            deltas = [improved[k].best_acc - base[k].best_acc
                      for k in keys(base) if k.nstate == ns && k.pzero == pz
                          && haskey(improved, k)]
            isempty(deltas) && continue
            mat[ci, ri] = median(deltas)
        end
    end

    clim = maximum(abs.(filter(!isnan, mat)))

    fig = Figure(size=(550, 380))
    ax  = Axis(fig[1, 1],
        xlabel  = L"p_{\mathrm{zero}}",
        ylabel  = L"N_{\mathrm{state}}",
        xticks  = (1:length(pzeros),  string.(pzeros)),
        yticks  = (1:length(nstates), string.(nstates)),
        title   = "Median accuracy gain: improved − base",
    )
    hm = heatmap!(ax, 1:length(pzeros), 1:length(nstates), mat';
                  colormap=:RdBu, colorrange=(-clim, clim))
    Colorbar(fig[1, 2], hm, label="Δ accuracy")

    save(joinpath(outdir, "delta_heatmap.pdf"), fig)
    println("Saved delta_heatmap.pdf")
end

# ---------------------------------------------------------------------------
# Plot 4: convergence speed — first iteration reaching best accuracy
# ---------------------------------------------------------------------------

function plot_convergence(base, improved, outdir)
    nstates = sort(unique(k.nstate for k in keys(base)))
    pzeros  = sort(unique(k.pzero  for k in keys(base)))

    fig = Figure(size=(600, 400 * length(pzeros)))

    for (ri, pz) in enumerate(pzeros)
        ax = Axis(fig[ri, 1],
            xlabel = ri == length(pzeros) ? L"N_{\mathrm{state}}" : "",
            ylabel = "Convergence iteration (median)",
            title  = L"p_{\mathrm{zero}} = %$(pz)",
        )

        for (offset, label, color, results) in [
            (-1.0, "base",     Makie.wong_colors()[1], base),
            ( 1.0, "improved", Makie.wong_colors()[2], improved),
        ]
            meds = Float64[]
            q1s  = Float64[]
            q3s  = Float64[]
            xs   = Float64[]
            for (xi, ns) in enumerate(nstates)
                vals = Float64[v.convergence_iter for (k, v) in results
                               if k.nstate == ns && k.pzero == pz]
                isempty(vals) && continue
                push!(xs,   xi + offset * 0.15)
                push!(meds, median(vals))
                push!(q1s,  quantile(vals, 0.25))
                push!(q3s,  quantile(vals, 0.75))
            end
            scatter!(ax, xs, meds; color, label, markersize=10)
            errorbars!(ax, xs, meds, meds .- q1s, q3s .- meds;
                       color, whiskerwidth=6)
            lines!(ax, xs, meds; color, linestyle=:dash, linewidth=1)
        end

        ax.xticks = (1:length(nstates), string.(nstates))
        ax.xticklabelrotation = π/4

        if ri == 1
            axislegend(ax; position=:rt)
        end
    end

    save(joinpath(outdir, "convergence_speed.pdf"), fig)
    println("Saved convergence_speed.pdf")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(datadir)
    outdir = datadir
    println("Loading data from $datadir ...")
    base, improved = load_both(datadir)
    println("  $(length(base)) matched runs loaded")

    plot_accuracy_curves(base, improved, outdir)
    plot_best_accuracy(base, improved, outdir)
    plot_delta_heatmap(base, improved, outdir)
    plot_convergence(base, improved, outdir)

    println("\nAll plots saved to $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) < 1 && error("Usage: julia scripts/1_improved_optimizer/analyze_comparison.jl <datadir>")
    main(ARGS[1])
end
