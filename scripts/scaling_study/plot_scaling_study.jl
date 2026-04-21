using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function eta_perfect(nqubit::Int, nstate::Real)
    return (3^nqubit - 1) / (2 * nqubit * nstate)
end

# ---------------------------------------------------------------------------
# Poisson-Binomial accuracy prediction.
# p_anti: fraction of votes going against companion_goal (measured for nqubit=6).
# lambda: mean votes per position = nstate * npos / (n - 1).
# ---------------------------------------------------------------------------
function predict_accuracy(p_anti::Float64, lambda::Float64; V_max::Int=60)
    p_goal = 1.0 - p_anti
    lfact = zeros(V_max + 2)
    for i in 1:V_max+1; lfact[i+1] = lfact[i] + log(i); end
    lf(k) = lfact[k+1]

    pois = [exp(-lambda + v*log(lambda) - lf(v)) for v in 0:V_max]
    pois ./= sum(pois)

    p_correct = 0.0; p_nontied = 0.0
    for v in 0:V_max
        v == 0 && continue
        pv = pois[v+1]
        binom = [exp(lf(v) - lf(k) - lf(v-k) +
                     k*log(p_goal) + (v-k)*log(max(1e-300, 1.0-p_goal)))
                 for k in 0:v]
        binom ./= sum(binom)
        p_win  = sum(binom[k+1] for k in 0:v if k > v/2)
        p_tied = isodd(v) ? 0.0 : binom[v÷2+1]
        p_correct += pv * p_win
        p_nontied += pv * (1.0 - p_tied)
    end
    p2c = p_correct^2; p2n = p_nontied^2
    return p2n > 0 ? p2c / p2n : 0.0
end

# p_anti measured from script 17 (nqubit=6 only).
const PANTI_NQUBIT6 = Dict(
    0.000 => 0.167566,
    0.250 => 0.190015,
    0.500 => 0.231051,
    0.750 => 0.267702,
    1.000 => 0.273452,
)

function interpolate_panti(H::Float64)
    hs  = sort(collect(keys(PANTI_NQUBIT6)))
    pas = [PANTI_NQUBIT6[h] for h in hs]
    H <= hs[1]   && return pas[1]
    H >= hs[end] && return pas[end]
    i = findlast(h -> h <= H, hs)
    t = (H - hs[i]) / (hs[i+1] - hs[i])
    return pas[i] + t * (pas[i+1] - pas[i])
end

function main(datafile)
    outdir = dirname(datafile)

    nqubits      = Int[]
    data         = Dict{Int, Dict}()

    h5open(datafile, "r") do h5f
        for nq_key in keys(h5f)
            gp_nq  = h5f[nq_key]
            nqubit = read(HDF5.attributes(gp_nq)["nqubit"])
            push!(nqubits, nqubit)
            data[nqubit] = Dict()

            for hkey in keys(gp_nq)
                gp    = gp_nq[hkey]
                haskey(gp, "nstates") || continue   # skip in-progress groups
                H_t   = read(HDF5.attributes(gp)["H_target"])
                H_act = read(HDF5.attributes(gp)["H_actual"])

                # eta_min / eta_std / acc_min / acc_std present only in newer runs;
                # fall back to nothing so plots degrade gracefully for old data.
                _read_opt(g, k) = haskey(g, k) ? read(g[k]) : nothing

                data[nqubit][H_t] = (
                    nstates  = read(gp["nstates"]),
                    acc_med  = read(gp["acc_med"]),
                    acc_max  = read(gp["acc_max"]),
                    acc_min  = _read_opt(gp, "acc_min"),
                    acc_std  = _read_opt(gp, "acc_std"),
                    eta_med  = read(gp["eta_med"]),
                    eta_max  = read(gp["eta_max"]),
                    eta_min  = _read_opt(gp, "eta_min"),
                    eta_std  = _read_opt(gp, "eta_std"),
                    H_actual = H_act,
                )
            end
        end
    end

    sort!(nqubits)
    H_targets  = sort(collect(keys(data[nqubits[1]])))
    nq_colors  = Makie.wong_colors()[1:length(nqubits)]
    nq_markers = [:circle, :rect, :utriangle][1:length(nqubits)]

    # -----------------------------------------------------------------------
    # Helper: build one combined figure (all H as rows) for a given y-series.
    # pred_fn: optional (nqubit, H_target, nstates_vector) → predicted_ys
    # -----------------------------------------------------------------------
    function make_combined(ylabel_str, title_str, series_fn, refline_fn, outname;
                           pred_fn=nothing)
        nH  = length(H_targets)
        fig = Figure(size=(600, 300 * nH))

        for (row, H_t) in enumerate(H_targets)
            ax = Axis(fig[row, 1],
                xlabel = row == nH ? L"N_\mathrm{state}" : "",
                ylabel = ylabel_str,
                title  = @sprintf("H = %.3f", H_t),
            )

            for (ci, nq) in enumerate(nqubits)
                haskey(data[nq], H_t) || continue
                d       = data[nq][H_t]
                nstates = d.nstates
                ys      = series_fn(d)

                lines!(ax, nstates, ys; color=nq_colors[ci], linewidth=2,
                       label=L"n = %$(nq)")
                scatter!(ax, nstates, ys; color=nq_colors[ci],
                         marker=nq_markers[ci], markersize=10)

                if refline_fn !== nothing
                    ns_dense = range(nstates[1], nstates[end], length=200)
                    ys_ref   = [refline_fn(nq, ns) for ns in ns_dense]
                    lines!(ax, collect(ns_dense), ys_ref;
                           color=nq_colors[ci], linewidth=1, linestyle=:dash)
                end

                if pred_fn !== nothing
                    ys_pred = pred_fn(nq, H_t, nstates)
                    ys_pred !== nothing &&
                        lines!(ax, Float64.(nstates), ys_pred;
                               color=nq_colors[ci], linewidth=2, linestyle=:dot)
                end
            end

            row == 1 && axislegend(ax; position=:rt)
        end

        outfile = joinpath(outdir, outname)
        save(outfile, fig)
        println("Saved $outfile")
    end

    # Efficiency prediction: only available for nqubit=6.
    function binary_entropy_fn(x)
        (x <= 0.0 || x >= 1.0) && return 0.0
        return -x*log2(x) - (1-x)*log2(1-x)
    end

    function eta_pred(nq, H_t, nstates)
        nq != 6 && return nothing
        n    = 3^nq
        npos = 2^(nq-1) + 1
        pa   = interpolate_panti(H_t)
        return [begin
            acc = predict_accuracy(pa, ns * npos / (n - 1))
            (1 - binary_entropy_fn(acc)) * (n - 1) / (2 * nq * ns)
        end for ns in nstates]
    end

    # 1. η_med
    make_combined(
        L"\eta_\mathrm{med}",
        "Median efficiency vs ensemble size",
        d -> d.eta_med,
        (nq, ns) -> eta_perfect(nq, ns),
        "efficiency_med_combined.pdf";
        pred_fn=eta_pred,
    )

    # 2. η_max
    make_combined(
        L"\eta_\mathrm{max}",
        "Max efficiency vs ensemble size",
        d -> d.eta_max,
        (nq, ns) -> eta_perfect(nq, ns),
        "efficiency_max_combined.pdf";
        pred_fn=eta_pred,
    )

    # 3. acc_med
    make_combined(
        L"\mathrm{acc}_\mathrm{med}",
        "Median accuracy vs ensemble size",
        d -> d.acc_med,
        nothing,
        "accuracy_med_combined.pdf",
    )

    # 4. acc_max
    make_combined(
        L"\mathrm{acc}_\mathrm{max}",
        "Max accuracy vs ensemble size",
        d -> d.acc_max,
        nothing,
        "accuracy_max_combined.pdf",
    )

    # -----------------------------------------------------------------------
    # 5. η_med with min–max envelope + asymmetric error bars
    #    Band and error bars are omitted for runs that lack eta_min/eta_max.
    # -----------------------------------------------------------------------
    function make_band_plot(outname)
        nH  = length(H_targets)
        fig = Figure(size=(700, 320 * nH))

        for (row, H_t) in enumerate(H_targets)
            ax = Axis(fig[row, 1],
                xlabel = row == nH ? L"N_\mathrm{state}" : "",
                ylabel = L"\eta_\mathrm{med}",
                title  = @sprintf("H = %.3f", H_t),
            )

            for (ci, nq) in enumerate(nqubits)
                haskey(data[nq], H_t) || continue
                d  = data[nq][H_t]
                ns = Float64.(d.nstates)
                c  = nq_colors[ci]
                mk = nq_markers[ci]

                if d.eta_min !== nothing
                    # Shaded envelope
                    band!(ax, ns, d.eta_min, d.eta_max;
                          color=(c, 0.15))
                    # Asymmetric error bars
                    errorbars!(ax, ns, d.eta_med,
                               d.eta_med .- d.eta_min,
                               d.eta_max .- d.eta_med;
                               color=c, whiskerwidth=5, linewidth=1)
                end

                # Main η_med line + markers
                lines!(ax, ns, d.eta_med;
                       color=c, linewidth=2,
                       label=L"n = %$(nq)")
                scatter!(ax, ns, d.eta_med;
                         color=c, marker=mk, markersize=9)
            end

            row == 1 && axislegend(ax; position=:rt)
        end

        outfile = joinpath(outdir, outname)
        save(outfile, fig)
        println("Saved $outfile")
    end

    make_band_plot("efficiency_med_band.pdf")
end

length(ARGS) < 1 && error("Usage: julia plot_scaling_study.jl <datafile.h5>")
main(ARGS[1])
