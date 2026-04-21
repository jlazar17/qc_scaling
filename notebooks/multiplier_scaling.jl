using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using HDF5
using Statistics
using LaTeXStrings

include(joinpath(@__DIR__, "plotting_boilerplate.jl"))

# --- Data loading ---

function load_sweep_data(datadir)
    data = Dict{Int, Dict{Tuple{Int,Float64}, Vector{Float64}}}()

    for fname in readdir(datadir; join=true)
        m = match(r"nqubit_(\d+)\.h5$", fname)
        m === nothing && continue
        nqubit = parse(Int, m.captures[1])

        nq_data = Dict{Tuple{Int,Float64}, Vector{Float64}}()
        h5open(fname, "r") do h5f
            for gname in keys(h5f)
                gm = match(r"nstate_(\d+)_pzero_([\d.]+)_seed_(\d+)", gname)
                gm === nothing && continue
                nstate = parse(Int, gm.captures[1])
                pzero = parse(Float64, gm.captures[2])
                maxacc = maximum(read(h5f[gname]["accuracies"]))
                key = (nstate, pzero)
                if !haskey(nq_data, key)
                    nq_data[key] = Float64[]
                end
                push!(nq_data[key], maxacc)
            end
        end
        data[nqubit] = nq_data
    end
    return data
end

function base_nstate_exact(nqubit)
    return 3^nqubit / 2^(nqubit - 1)
end

function get_multipliers_and_pzeros(nq_data, nqubit)
    nstates = sort(unique([k[1] for k in keys(nq_data)]))
    pzeros = sort(unique([k[2] for k in keys(nq_data)]))
    bne = base_nstate_exact(nqubit)
    multipliers = [ns / bne for ns in nstates]
    return nstates, pzeros, multipliers
end

# --- Plotting ---

function plot_accuracy_vs_multiplier_per_pzero(data, outdir)
    for nqubit in sort(collect(keys(data)))
        nq_data = data[nqubit]
        nstates, pzeros, multipliers = get_multipliers_and_pzeros(nq_data, nqubit)
        colors = cgrad(:roma, length(pzeros), categorical=true)

        fig = Figure(size=(700, 450))
        ax = Axis(fig[1, 1],
            xlabel=L"N_{\mathrm{state}} / N_{\mathrm{state},0}",
            ylabel="Best accuracy",
            title=L"n_{\mathrm{qubit}} = %$(nqubit)",
        )

        for (pidx, pzero) in enumerate(pzeros)
            meds = Float64[]
            q1s = Float64[]
            q3s = Float64[]
            mults_valid = Float64[]
            for (ns, mult) in zip(nstates, multipliers)
                key = (ns, pzero)
                haskey(nq_data, key) || continue
                vals = nq_data[key]
                push!(meds, median(vals))
                push!(q1s, quantile(vals, 0.25))
                push!(q3s, quantile(vals, 0.75))
                push!(mults_valid, mult)
            end
            lines!(ax, mults_valid, meds, color=colors[pidx],
                   label=L"p_{\mathrm{zero}} = %$(pzero)")
            band!(ax, mults_valid, q1s, q3s, color=(colors[pidx], 0.2))
        end

        fig[1, 2] = Legend(fig, ax)
        save(joinpath(outdir, "accuracy_vs_multiplier_nqubit_$(nqubit).pdf"), fig)
    end
end

function plot_accuracy_vs_multiplier_per_nqubit(data, outdir)
    all_pzeros = sort(unique(vcat([collect(unique([k[2] for k in keys(nq)])) for nq in values(data)]...)))
    nqubits = sort(collect(keys(data)))
    colors = cgrad(:roma, length(nqubits), categorical=true)

    for pzero in all_pzeros
        fig = Figure(size=(700, 450))
        ax = Axis(fig[1, 1],
            xlabel=L"N_{\mathrm{state}} / N_{\mathrm{state},0}",
            ylabel="Best accuracy",
            title=L"p_{\mathrm{zero}} = %$(pzero)",
        )

        for (nidx, nqubit) in enumerate(nqubits)
            nq_data = data[nqubit]
            nstates, _, multipliers = get_multipliers_and_pzeros(nq_data, nqubit)

            meds = Float64[]
            q1s = Float64[]
            q3s = Float64[]
            mults_valid = Float64[]
            for (ns, mult) in zip(nstates, multipliers)
                key = (ns, pzero)
                haskey(nq_data, key) || continue
                vals = nq_data[key]
                push!(meds, median(vals))
                push!(q1s, quantile(vals, 0.25))
                push!(q3s, quantile(vals, 0.75))
                push!(mults_valid, mult)
            end
            length(meds) == 0 && continue
            lines!(ax, mults_valid, meds, color=colors[nidx],
                   label=L"n_{\mathrm{qubit}} = %$(nqubit)")
            band!(ax, mults_valid, q1s, q3s, color=(colors[nidx], 0.2))
        end

        fig[1, 2] = Legend(fig, ax)
        save(joinpath(outdir, "accuracy_vs_multiplier_pzero_$(pzero).pdf"), fig)
    end
end

function plot_heatmap(data, outdir)
    for nqubit in sort(collect(keys(data)))
        nq_data = data[nqubit]
        nstates, pzeros, multipliers = get_multipliers_and_pzeros(nq_data, nqubit)

        med_matrix = fill(NaN, length(multipliers), length(pzeros))
        for (midx, ns) in enumerate(nstates)
            for (pidx, pzero) in enumerate(pzeros)
                key = (ns, pzero)
                haskey(nq_data, key) || continue
                med_matrix[midx, pidx] = median(nq_data[key])
            end
        end

        fig = Figure(size=(600, 450))
        ax = Axis(fig[1, 1],
            xlabel=L"N_{\mathrm{state}} / N_{\mathrm{state},0}",
            ylabel=L"p_{\mathrm{zero}}",
            title=L"n_{\mathrm{qubit}} = %$(nqubit)",
        )

        hm = heatmap!(ax, multipliers, pzeros, med_matrix,
                       colormap=:roma)
        Colorbar(fig[1, 2], hm, label="Median best accuracy")
        save(joinpath(outdir, "heatmap_nqubit_$(nqubit).pdf"), fig)
    end
end

# --- Main ---

function main_analysis(datadir)
    data = load_sweep_data(datadir)
    if isempty(data)
        error("No nqubit_*.h5 files found in $datadir")
    end
    outdir = datadir
    println("Loaded data for nqubits: $(sort(collect(keys(data))))")
    plot_accuracy_vs_multiplier_per_pzero(data, outdir)
    plot_accuracy_vs_multiplier_per_nqubit(data, outdir)
    plot_heatmap(data, outdir)
    println("Plots saved to $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) < 1 && error("Usage: julia --project=notebooks notebooks/multiplier_scaling.jl <datadir>")
    main_analysis(ARGS[1])
end
