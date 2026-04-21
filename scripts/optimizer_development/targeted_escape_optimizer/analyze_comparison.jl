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
    convergence_iter::Int
    accuracies::Vector{Float64}
end

function load_file(path)
    results = Dict{NamedTuple, RunResult}()
    h5open(path, "r") do h5f
        for gname in keys(h5f)
            m = match(r"nstate_(\d+)_pzero_([\d.]+)_seed_(\d+)", gname)
            m === nothing && continue
            nstate = parse(Int,     m.captures[1])
            pzero  = parse(Float64, m.captures[2])
            seed   = parse(Int,     m.captures[3])
            accs   = read(h5f[gname]["accuracies"])
            best   = maximum(accs)
            conv   = findfirst(==(best), accs)
            results[(nstate=nstate, pzero=pzero, seed=seed)] =
                RunResult(best, accs[end], conv, accs)
        end
    end
    return results
end

function load_all(datadir, nqubit)
    optimizers = ["improved", "targeted_escape"]
    all_results = Dict(
        name => load_file(joinpath(datadir, "$(name)_nqubit_$(nqubit).h5"))
        for name in optimizers
    )
    common = reduce(intersect, [Set(keys(r)) for r in values(all_results)])
    return Dict(name => Dict(k => v for (k,v) in res if k in common)
                for (name, res) in all_results)
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

const OPTIMIZERS = [
    ("improved",        Makie.wong_colors()[1]),
    ("targeted_escape", Makie.wong_colors()[2]),
]

function summarize(results, name, ns, pz, field)
    vals = [getfield(v, field) for (k, v) in results[name]
            if k.nstate == ns && k.pzero == pz]
    isempty(vals) && return NaN, NaN, NaN
    return median(vals), quantile(vals, 0.25), quantile(vals, 0.75)
end

# ---------------------------------------------------------------------------
# Plot 1: accuracy curves
# ---------------------------------------------------------------------------

function plot_accuracy_curves(results, outdir)
    ref    = first(values(results))
    nstates = sort(unique(k.nstate for k in keys(ref)))
    pzeros  = sort(unique(k.pzero  for k in keys(ref)))

    fig = Figure(size=(400 * length(nstates), 320 * length(pzeros)))

    for (ci, ns) in enumerate(nstates)
        for (ri, pz) in enumerate(pzeros)
            ax = Axis(fig[ri, ci],
                xlabel = ci == length(nstates) && ri == length(pzeros) ? "Iteration" : "",
                ylabel = ri == 1 && ci == 1 ? "Accuracy" : "",
                title  = L"p_0=%$(pz),\; N_s=%$(ns)",
            )
            for (name, color) in OPTIMIZERS
                haskey(results, name) || continue
                curves = [v.accuracies for (k, v) in results[name]
                          if k.nstate == ns && k.pzero == pz]
                isempty(curves) && continue
                mat  = reduce(hcat, curves)
                xs   = 1:size(mat, 1)
                meds = [median(mat[i, :]) for i in xs]
                q1s  = [quantile(mat[i, :], 0.25) for i in xs]
                q3s  = [quantile(mat[i, :], 0.75) for i in xs]
                lines!(ax, xs, meds; color, label=name)
                band!(ax,  xs, q1s, q3s; color=(color, 0.2))
            end
            ri == 1 && ci == length(nstates) && axislegend(ax; position=:rb)
        end
    end

    save(joinpath(outdir, "accuracy_curves.pdf"), fig)
    println("Saved accuracy_curves.pdf")
end

# ---------------------------------------------------------------------------
# Plot 2: best accuracy scatter
# ---------------------------------------------------------------------------

function plot_best_accuracy(results, outdir)
    ref     = first(values(results))
    nstates = sort(unique(k.nstate for k in keys(ref)))
    pzeros  = sort(unique(k.pzero  for k in keys(ref)))
    n_opt   = length(OPTIMIZERS)
    offsets = range(-0.2, 0.2, length=n_opt)

    fig = Figure(size=(600, 400 * length(pzeros)))

    for (ri, pz) in enumerate(pzeros)
        ax = Axis(fig[ri, 1],
            xlabel = ri == length(pzeros) ? L"N_{\mathrm{state}}" : "",
            ylabel = "Best accuracy (median)",
            title  = L"p_{\mathrm{zero}} = %$(pz)",
        )
        ylims!(ax, 0, 1.05)

        for ((name, color), off) in zip(OPTIMIZERS, offsets)
            haskey(results, name) || continue
            xs, meds, q1s, q3s = Float64[], Float64[], Float64[], Float64[]
            for (xi, ns) in enumerate(nstates)
                med, q1, q3 = summarize(results, name, ns, pz, :best_acc)
                isnan(med) && continue
                push!(xs, xi + off); push!(meds, med)
                push!(q1s, q1);      push!(q3s,  q3)
            end
            scatter!(ax, xs, meds; color, label=name, markersize=10)
            errorbars!(ax, xs, meds, meds .- q1s, q3s .- meds; color, whiskerwidth=6)
            lines!(ax, xs, meds; color, linestyle=:dash, linewidth=1)
        end

        ax.xticks = (1:length(nstates), string.(nstates))
        ax.xticklabelrotation = π/4
        ri == 1 && axislegend(ax; position=:rb)
    end

    save(joinpath(outdir, "best_accuracy_comparison.pdf"), fig)
    println("Saved best_accuracy_comparison.pdf")
end

# ---------------------------------------------------------------------------
# Plot 3: delta heatmaps vs base (one per non-base optimizer)
# ---------------------------------------------------------------------------

function plot_delta_heatmaps(results, outdir)
    ref     = results["improved"]
    nstates = sort(unique(k.nstate for k in keys(ref)))
    pzeros  = sort(unique(k.pzero  for k in keys(ref)))

    comparisons = [("targeted_escape", Makie.wong_colors()[2])]

    fig = Figure(size=(550 * length(comparisons), 380))

    for (ci, (name, _)) in enumerate(comparisons)
        haskey(results, name) || continue
        mat = fill(NaN, length(nstates), length(pzeros))
        for (nsi, ns) in enumerate(nstates), (pi, pz) in enumerate(pzeros)
            deltas = [results[name][k].best_acc - ref[k].best_acc
                      for k in keys(ref) if k.nstate == ns && k.pzero == pz
                                         && haskey(results[name], k)]
            isempty(deltas) || (mat[nsi, pi] = median(deltas))
        end
        clim = maximum(abs.(filter(!isnan, mat)))
        ax = Axis(fig[1, ci],
            xlabel = L"p_{\mathrm{zero}}",
            ylabel = ci == 1 ? L"N_{\mathrm{state}}" : "",
            xticks = (1:length(pzeros),  string.(pzeros)),
            yticks = (1:length(nstates), string.(nstates)),
            title  = "Δ accuracy: $name − improved",
        )
        hm = heatmap!(ax, 1:length(pzeros), 1:length(nstates), mat';
                      colormap=:RdBu, colorrange=(-clim, clim))
        Colorbar(fig[1, ci+length(comparisons)], hm, label="Δ accuracy")
    end

    save(joinpath(outdir, "delta_heatmap.pdf"), fig)
    println("Saved delta_heatmap.pdf")
end

# ---------------------------------------------------------------------------
# Plot 4: convergence speed
# ---------------------------------------------------------------------------

function plot_convergence(results, outdir)
    ref     = first(values(results))
    nstates = sort(unique(k.nstate for k in keys(ref)))
    pzeros  = sort(unique(k.pzero  for k in keys(ref)))
    n_opt   = length(OPTIMIZERS)
    offsets = range(-0.2, 0.2, length=n_opt)

    fig = Figure(size=(600, 400 * length(pzeros)))

    for (ri, pz) in enumerate(pzeros)
        ax = Axis(fig[ri, 1],
            xlabel = ri == length(pzeros) ? L"N_{\mathrm{state}}" : "",
            ylabel = "Convergence iteration (median)",
            title  = L"p_{\mathrm{zero}} = %$(pz)",
        )
        for ((name, color), off) in zip(OPTIMIZERS, offsets)
            haskey(results, name) || continue
            xs, meds, q1s, q3s = Float64[], Float64[], Float64[], Float64[]
            for (xi, ns) in enumerate(nstates)
                med, q1, q3 = summarize(results, name, ns, pz, :convergence_iter)
                isnan(med) && continue
                push!(xs, xi + off); push!(meds, med)
                push!(q1s, q1);      push!(q3s,  q3)
            end
            scatter!(ax, xs, meds; color, label=name, markersize=10)
            errorbars!(ax, xs, meds, meds .- q1s, q3s .- meds; color, whiskerwidth=6)
            lines!(ax, xs, meds; color, linestyle=:dash, linewidth=1)
        end
        ax.xticks = (1:length(nstates), string.(nstates))
        ax.xticklabelrotation = π/4
        ri == 1 && axislegend(ax; position=:rt)
    end

    save(joinpath(outdir, "convergence_speed.pdf"), fig)
    println("Saved convergence_speed.pdf")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(datadir, nqubit=8)
    println("Loading data from $datadir ...")
    results = load_all(datadir, nqubit)
    n = length(first(values(results)))
    println("  $n matched runs per optimizer")

    plot_accuracy_curves(results, datadir)
    plot_best_accuracy(results, datadir)
    plot_delta_heatmaps(results, datadir)
    plot_convergence(results, datadir)

    println("\nAll plots saved to $datadir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) < 1 && error("Usage: julia analyze_comparison.jl <datadir> [nqubit]")
    nqubit = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    main(ARGS[1], nqubit)
end
