# Efficiency vs nstate — one subplot per nqubit, colors by entropy.
# Dashed vertical line from y=0 to the peak efficiency for each H.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))
using CairoMakie, HDF5, Printf, Statistics
include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function main()
    datafile = joinpath(@__DIR__, "data/scaling_study_adaptive.h5")
    outfile  = joinpath(@__DIR__, "data/efficiency_vs_H_by_nqubit.pdf")

    data    = Dict{Int, Dict}()
    nqubits = Int[]

    h5open(datafile, "r") do h5f
        for nq_key in sort(collect(keys(h5f)))
            g      = h5f[nq_key]
            nqubit = read(HDF5.attributes(g)["nqubit"])
            push!(nqubits, nqubit)
            data[nqubit] = Dict()
            for hkey in keys(g)
                gp = g[hkey]
                haskey(gp, "nstates") || continue
                H_t = read(HDF5.attributes(gp)["H_target"])
                data[nqubit][H_t] = (
                    nstates = read(gp["nstates"]),
                    eta_med = read(gp["eta_med"]),
                )
            end
        end
    end
    sort!(nqubits)

    all_H = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]
    n_H   = length(all_H)

    # Colors shared across subplots: plasma from dark-blue (H=0) to yellow (H=1)
    raw_colors = Makie.resample_cmap(:plasma, n_H)
    H_to_color = Dict(H => raw_colors[i] for (i, H) in enumerate(all_H))

    fig = Figure(size=(1400, 420))

    for (ci, nqubit) in enumerate(nqubits)
        ax = Axis(fig[1, ci],
            title  = "nqubit = $nqubit",
            xlabel = L"N_\mathrm{state}",
            ylabel = ci == 1 ? L"\eta_\mathrm{med}" : "",
        )
        ylims!(ax, 0, nothing)
        ci > 1 && hideydecorations!(ax, grid=false)

        for H_t in all_H
            haskey(data[nqubit], H_t) || continue
            d  = data[nqubit][H_t]
            ns = Float64.(d.nstates)
            ys = d.eta_med
            c  = H_to_color[H_t]

            peak_idx = argmax(ys)
            ns_peak  = ns[peak_idx]
            eta_peak = ys[peak_idx]

            # Dashed vertical line from y=0 up to the peak value
            lines!(ax, [ns_peak, ns_peak], [0.0, eta_peak];
                   color=c, linewidth=1.5, linestyle=:dash)
            # Main line + markers
            lines!(ax, ns, ys; color=c, linewidth=2)
            scatter!(ax, ns, ys; color=c, markersize=5)
        end
    end

    # Shared legend to the right of all subplots
    Legend(fig[1, length(nqubits) + 1],
           [LineElement(color=H_to_color[H], linewidth=2) for H in all_H],
           [@sprintf("H = %.3f", H) for H in all_H];
           framevisible=false, labelsize=14)

    save(outfile, fig)
    println("Saved $outfile")

    # -----------------------------------------------------------------------
    # Figure 2: peak efficiency vs entropy, one line per nqubit
    # Figure 3: nstate at peak efficiency vs entropy, one line per nqubit
    # -----------------------------------------------------------------------
    outdir     = dirname(datafile)
    nq_colors  = Makie.wong_colors()[1:length(nqubits)]
    nq_markers = [:circle, :rect, :utriangle][1:length(nqubits)]

    # Collect peak statistics for each (nqubit, H)
    peak_eta   = Dict{Int, Vector{Float64}}()
    peak_nstate = Dict{Int, Vector{Float64}}()
    peak_H     = Dict{Int, Vector{Float64}}()

    for nqubit in nqubits
        etas   = Float64[]
        nsts   = Float64[]
        hs     = Float64[]
        for H_t in all_H
            haskey(data[nqubit], H_t) || continue
            d = data[nqubit][H_t]
            idx = argmax(d.eta_med)
            push!(etas, d.eta_med[idx])
            push!(nsts, Float64(d.nstates[idx]))
            push!(hs,   H_t)
        end
        peak_eta[nqubit]    = etas
        peak_nstate[nqubit] = nsts
        peak_H[nqubit]      = hs
    end

    # Figure 2: peak eta vs H
    fig2 = Figure(size=(600, 420))
    ax2  = Axis(fig2[1, 1],
        xlabel = "Entropy H",
        ylabel = L"\eta_\mathrm{med}^\mathrm{peak}",
        title  = "Peak efficiency vs entropy",
    )
    for (ci, nqubit) in enumerate(nqubits)
        lines!(ax2, peak_H[nqubit], peak_eta[nqubit];
               color=nq_colors[ci], linewidth=2, label="n = $nqubit")
        scatter!(ax2, peak_H[nqubit], peak_eta[nqubit];
                 color=nq_colors[ci], marker=nq_markers[ci], markersize=10)
    end
    axislegend(ax2, position=:rt)
    ylims!(ax2, 0, nothing)
    out2 = joinpath(outdir, "peak_efficiency_vs_H.pdf")
    save(out2, fig2)
    println("Saved $out2")

    # Figure 3: nstate at peak vs H
    fig3 = Figure(size=(600, 420))
    ax3  = Axis(fig3[1, 1],
        xlabel = "Entropy H",
        ylabel = L"N_\mathrm{state}^\mathrm{peak}",
        title  = "nstate at peak efficiency vs entropy",
    )
    for (ci, nqubit) in enumerate(nqubits)
        lines!(ax3, peak_H[nqubit], peak_nstate[nqubit];
               color=nq_colors[ci], linewidth=2, label="n = $nqubit")
        scatter!(ax3, peak_H[nqubit], peak_nstate[nqubit];
                 color=nq_colors[ci], marker=nq_markers[ci], markersize=10)
    end
    axislegend(ax3, position=:lt)
    ylims!(ax3, 0, nothing)
    out3 = joinpath(outdir, "nstate_at_peak_vs_H.pdf")
    save(out3, fig3)
    println("Saved $out3")
end

main()
