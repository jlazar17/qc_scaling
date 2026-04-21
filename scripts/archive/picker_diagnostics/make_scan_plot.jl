using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using CairoMakie
using HDF5
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function eta_perfect(nqubit::Int, nstate::Real)
    return (3^nqubit - 1) / (2 * nqubit * nstate)
end

function main(datafile)
    outdir = @__DIR__

    # -------------------------------------------------------------------------
    # Load from HDF5
    # -------------------------------------------------------------------------
    nqubits      = Int[]
    base_nstates = Dict{Int,Int}()
    data         = Dict{Int, Dict}()

    h5open(datafile, "r") do h5f
        for nq_key in keys(h5f)
            gp_nq       = h5f[nq_key]
            nqubit      = read(HDF5.attributes(gp_nq)["nqubit"])
            base_nstate = read(HDF5.attributes(gp_nq)["base_nstate"])
            push!(nqubits, nqubit)
            base_nstates[nqubit] = base_nstate
            data[nqubit] = Dict()

            for hkey in keys(gp_nq)
                gp    = gp_nq[hkey]
                H_t   = read(HDF5.attributes(gp)["H_target"])
                H_act = read(HDF5.attributes(gp)["H_actual"])
                data[nqubit][H_t] = (
                    nstates  = read(gp["nstates"]),
                    eta_max  = read(gp["eta_max"]),
                    eta_med  = read(gp["eta_med"]),
                    acc_max  = read(gp["acc_max"]),
                    acc_med  = read(gp["acc_med"]),
                    H_actual = H_act,
                )
            end
        end
    end

    sort!(nqubits)
    H_targets = sort(collect(keys(data[nqubits[1]])))
    rainbow   = cgrad(:rainbow)
    h_color   = H -> rainbow[H]   # H ∈ [0,1] → color

    # Colors/markers per nqubit — sampled from rainbow at 0, 0.5, 1
    n_sample  = length(nqubits) == 1 ? [0.0] :
                [i / (length(nqubits) - 1) for i in 0:length(nqubits)-1]
    n_colors  = Dict(nq => rainbow[n_sample[i]] for (i, nq) in enumerate(nqubits))
    n_markers = Dict(nq => m for (nq, m) in zip(nqubits, [:circle, :rect, :utriangle]))

    # -------------------------------------------------------------------------
    # Helper: add acc=1 reference hyperbola to an axis
    # x_min, x_max in units of nstate/nstate₀
    # -------------------------------------------------------------------------
    function add_acc1_ref!(ax, nqubit, n0, x_min, x_max; color=:black, linewidth=1)
        xs = range(x_min, x_max, length=300)
        ys = [eta_perfect(nqubit, x * n0) for x in xs]
        lines!(ax, collect(xs), ys;
               color=color, linewidth=linewidth, linestyle=:dash)
    end

    # =========================================================================
    # Figure 1: panels by nqubit, lines colored by H
    # =========================================================================
    fig1 = Figure(size=(1100, 420))

    for (ci, nq) in enumerate(nqubits)
        ax = Axis(fig1[1, ci],
            xlabel = "nstate / nstate₀",
            ylabel = ci == 1 ? L"\eta_\mathrm{max}" : "",
            title  = "n = $nq  (nstate₀ = $(base_nstates[nq]))",
        )
        n0 = base_nstates[nq]

        x_max = 0.0
        for H_t in H_targets
            haskey(data[nq], H_t) || continue
            d    = data[nq][H_t]
            xs   = d.nstates ./ n0
            ys   = d.eta_max
            col  = h_color(H_t)
            x_max = max(x_max, maximum(xs))

            lines!(ax,   xs, ys; color=col, linewidth=2)
            scatter!(ax, xs, ys; color=col, markersize=6)

            # Vertical dashed line at η_max peak, clipped to curve height
            pk = argmax(ys)
            linesegments!(ax, [xs[pk], xs[pk]], [0.0, ys[pk]];
                          color=col, linewidth=1, linestyle=:dash, alpha=0.6)
        end

        # acc=1 reference per nqubit
        x_min_ref = base_nstates[nq] / n0   # =1 always, but explicit
        add_acc1_ref!(ax, nq, n0, x_min_ref, x_max; color=:black, linewidth=1)

        ylims!(ax, 0, nothing)
        xlims!(ax, 0, nothing)
    end

    Colorbar(fig1[1, length(nqubits)+1];
        colormap = cgrad(:rainbow, length(H_targets), categorical=true),
        limits   = (0, length(H_targets)),
        ticks    = (0.5:length(H_targets)-0.5, [@sprintf("%.3f", H) for H in H_targets]),
        label    = "Goal entropy H",
    )

    save(joinpath(outdir, "fig_scan_by_n.pdf"), fig1)
    println("Saved fig_scan_by_n.pdf")

    # =========================================================================
    # Figure 2: panels by H (H=0.0, 0.5, 1.0), lines colored by nqubit
    # =========================================================================
    H_panels = [0.0, 0.5, 1.0]

    fig2 = Figure(size=(1100, 420))

    for (ci, H_target) in enumerate(H_panels)
        ax = Axis(fig2[1, ci],
            xlabel = "nstate / nstate₀",
            ylabel = ci == 1 ? L"\eta_\mathrm{max}" : "",
            title  = @sprintf("H ≈ %.1f", H_target),
        )

        x_max_panel = 0.0
        for nq in nqubits
            isempty(data[nq]) && continue
            n0     = base_nstates[nq]
            H_keys = Float64.(collect(keys(data[nq])))
            H_near = H_keys[argmin(abs.(H_keys .- H_target))]
            abs(H_near - H_target) > 0.05 && continue

            d   = data[nq][H_near]
            xs  = d.nstates ./ n0
            ys  = d.eta_max
            col = n_colors[nq]
            x_max_panel = max(x_max_panel, maximum(xs))

            lines!(ax,   xs, ys; color=col, linewidth=2, label="n = $nq")
            scatter!(ax, xs, ys; color=col, marker=n_markers[nq], markersize=8)

            pk = argmax(ys)
            linesegments!(ax, [xs[pk], xs[pk]], [0.0, ys[pk]];
                          color=col, linewidth=1, linestyle=:dash, alpha=0.6)

            # acc=1 reference in matching color
            add_acc1_ref!(ax, nq, n0, 1.0, x_max_panel; color=col, linewidth=0.8)
        end

        ci == 1 && axislegend(ax; position=:rt)
        ylims!(ax, 0, nothing)
        xlims!(ax, 0, nothing)
    end

    save(joinpath(outdir, "fig_scan_by_H.pdf"), fig2)
    println("Saved fig_scan_by_H.pdf")

    # =========================================================================
    # Figure 3: peak nstate*/nstate₀ vs H  AND  peak η_max vs H
    # =========================================================================
    fig3 = Figure(size=(950, 420))
    ax_a = Axis(fig3[1, 1],
        xlabel = "Goal entropy H",
        ylabel = L"n^*_\mathrm{state} / n^0_\mathrm{state}",
        title  = "A   Optimal ensemble size vs H",
    )
    ax_b = Axis(fig3[1, 2],
        xlabel = "Goal entropy H",
        ylabel = L"\eta_\mathrm{max}",
        title  = "B   Peak efficiency vs H",
    )

    for nq in nqubits
        isempty(data[nq]) && continue
        n0       = base_nstates[nq]
        H_sorted = sort(Float64.(collect(keys(data[nq]))))
        col      = n_colors[nq]
        mrk      = n_markers[nq]

        peak_ns_norm = [data[nq][H].nstates[argmax(data[nq][H].eta_max)] / n0
                        for H in H_sorted]
        peak_eta     = [maximum(data[nq][H].eta_max) for H in H_sorted]

        lines!(ax_a,   H_sorted, peak_ns_norm; color=col, linewidth=2, label="n = $nq")
        scatter!(ax_a, H_sorted, peak_ns_norm; color=col, marker=mrk, markersize=8)

        lines!(ax_b,   H_sorted, peak_eta; color=col, linewidth=2, label="n = $nq")
        scatter!(ax_b, H_sorted, peak_eta; color=col, marker=mrk, markersize=8)
    end

    for ax in (ax_a, ax_b)
        xlims!(ax, -0.02, 1.02)
        ylims!(ax, 0, nothing)
        axislegend(ax; position=:rt)
    end

    save(joinpath(outdir, "fig_peak_vs_H.pdf"), fig3)
    println("Saved fig_peak_vs_H.pdf")
end

length(ARGS) < 1 && error("Usage: julia make_scan_plot.jl <datafile.h5>")
main(ARGS[1])
