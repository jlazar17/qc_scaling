using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using Statistics
using LaTeXStrings

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function main(datafile)
    outdir = dirname(datafile)

    h5open(datafile, "r") do h5f

        # -------------------------------------------------------------------
        # Plot 1: SA accuracy curves for different alpha values
        # -------------------------------------------------------------------
        cgp        = h5f["sa_curves"]
        ckpt_every = read(HDF5.attributes(cgp)["checkpoint_every"])
        nsteps     = read(HDF5.attributes(cgp)["nsteps"])
        n_restarts = read(HDF5.attributes(cgp)["n_restarts"])
        total      = nsteps * n_restarts
        alphas     = sort([parse(Float64, split(k, "_")[2]) for k in keys(cgp)
                           if startswith(k, "alpha")])

        fig1 = Figure(size=(600, 400))
        ax1  = Axis(fig1[1, 1],
            xlabel = "SA steps (×10⁶)",
            ylabel = "Best accuracy (best-so-far)",
            title  = L"SA convergence, $n=8$, $p_0=0.5$",
        )

        colors = Makie.wong_colors()
        for (i, alpha) in enumerate(alphas)
            curve = read(cgp["alpha_$(alpha)"])
            xs    = (1:length(curve)) .* ckpt_every ./ 1e6
            lines!(ax1, xs, curve; color=colors[i], label="α=$alpha", linewidth=2)
            # Mark restart boundaries
            for r in 1:(n_restarts - 1)
                vlines!(ax1, nsteps * r / 1e6; color=colors[i], linestyle=:dot, linewidth=1)
            end
        end
        axislegend(ax1; position=:rb)

        save(joinpath(outdir, "sa_curves.pdf"), fig1)
        println("Saved sa_curves.pdf")

        # -------------------------------------------------------------------
        # Plot 2: Final accuracy comparison across three methods
        # -------------------------------------------------------------------
        pzero_vals = sort([parse(Float64, split(k, "_")[2]) for k in keys(h5f)
                           if startswith(k, "pzero")])
        methods    = ["base", "improved", "sa"]
        labels     = ["Base", "Improved", "SA"]
        colors2    = Makie.wong_colors()[1:3]
        n_methods  = length(methods)
        offsets    = range(-0.25, 0.25; length=n_methods)

        fig2 = Figure(size=(500, 400))
        ax2  = Axis(fig2[1, 1],
            xlabel = L"p_{\mathrm{zero}}",
            ylabel = "Best accuracy",
            title  = L"Three-way comparison, $n=8$",
            xticks = (1:length(pzero_vals), string.(pzero_vals)),
        )
        ylims!(ax2, 0, 1)

        for (mi, (meth, label, color)) in enumerate(zip(methods, labels, colors2))
            xs_all, meds, q1s, q3s = Float64[], Float64[], Float64[], Float64[]
            for (pi, pz) in enumerate(pzero_vals)
                vals = read(h5f["pzero_$(pz)"]["$(meth)_best"])
                push!(xs_all, pi + offsets[mi])
                push!(meds, median(vals))
                push!(q1s,  quantile(vals, 0.25))
                push!(q3s,  quantile(vals, 0.75))
                # Individual seed points
                scatter!(ax2, fill(pi + offsets[mi], length(vals)), vals;
                         color=(color, 0.4), markersize=6)
            end
            scatter!(ax2,  xs_all, meds; color=color, markersize=10, label=label)
            errorbars!(ax2, xs_all, meds, meds .- q1s, q3s .- meds;
                       color=color, whiskerwidth=8)
            lines!(ax2, xs_all, meds; color=color, linestyle=:dash, linewidth=1)
        end
        axislegend(ax2; position=:rb)

        save(joinpath(outdir, "three_way_comparison.pdf"), fig2)
        println("Saved three_way_comparison.pdf")

        # -------------------------------------------------------------------
        # Plot 3: Accuracy curves (base vs improved) over iterations
        # -------------------------------------------------------------------
        fig3 = Figure(size=(700, 350 * length(pzero_vals)))
        method_pairs = [("base", "Base", colors2[1]), ("improved", "Improved", colors2[2])]

        for (ri, pz) in enumerate(pzero_vals)
            gp = h5f["pzero_$(pz)"]
            niter = read(HDF5.attributes(gp)["niter"])
            ax = Axis(fig3[ri, 1],
                xlabel = ri == length(pzero_vals) ? "Iteration" : "",
                ylabel = "Accuracy",
                title  = L"p_0 = %$(pz)",
            )

            for (key, label, color) in method_pairs
                mat  = read(gp[key == "improved" ? "impr_curves" : "$(key)_curves"])
                xs   = 1:niter
                meds = [median(mat[i, :]) for i in xs]
                q1s  = [quantile(mat[i, :], 0.25) for i in xs]
                q3s  = [quantile(mat[i, :], 0.75) for i in xs]
                lines!(ax, xs, meds; color=color, label=label, linewidth=2)
                band!(ax,  xs, q1s, q3s; color=(color, 0.2))
            end
            ri == 1 && axislegend(ax; position=:rb)
        end

        save(joinpath(outdir, "optimizer_curves.pdf"), fig3)
        println("Saved optimizer_curves.pdf")
    end
end

length(ARGS) < 1 && error("Usage: julia plot_comparison.jl <datafile.h5>")
main(ARGS[1])
