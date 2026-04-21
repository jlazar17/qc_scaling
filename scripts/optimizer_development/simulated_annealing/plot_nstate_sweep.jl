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
        nstates    = read(HDF5.attributes(h5f)["nstates"])
        pzero_vals = read(HDF5.attributes(h5f)["pzero_vals"])
        sa_nsteps  = read(HDF5.attributes(h5f)["sa_nsteps"])
        sa_alpha   = read(HDF5.attributes(h5f)["sa_alpha"])
        niter      = read(HDF5.attributes(h5f)["niter"])

        methods = [("base", "Base"), ("improved", "Improved"), ("sa", "SA")]
        colors  = Makie.wong_colors()[1:3]
        offsets = range(-0.3, 0.3; length=3)

        fig = Figure(size=(600, 380 * length(pzero_vals)))

        for (ri, pz) in enumerate(pzero_vals)
            ax = Axis(fig[ri, 1],
                xlabel = ri == length(pzero_vals) ? L"N_{\mathrm{state}}" : "",
                ylabel = "Best accuracy (median)",
                title  = L"p_0 = %$(pz),\; n=8,\; \mathrm{niter}=%$(niter),\; \mathrm{SA\ steps}=%$(sa_nsteps)",
            )
            ylims!(ax, 0, 1)

            pgp = h5f["pzero_$(pz)"]
            for ((meth, label), color, off) in zip(methods, colors, offsets)
                mat  = read(pgp[meth])   # n_nstates × nseeds
                xs   = Float64[]
                meds = Float64[]
                q1s  = Float64[]
                q3s  = Float64[]
                for (ni, ns) in enumerate(nstates)
                    vals = mat[ni, :]
                    push!(xs,   ni + off)
                    push!(meds, median(vals))
                    push!(q1s,  quantile(vals, 0.25))
                    push!(q3s,  quantile(vals, 0.75))
                    scatter!(ax, fill(ni + off, length(vals)), vals;
                             color=(color, 0.35), markersize=6)
                end
                scatter!(ax,  xs, meds; color=color, markersize=10, label=label)
                errorbars!(ax, xs, meds, meds .- q1s, q3s .- meds;
                           color=color, whiskerwidth=8)
                lines!(ax, xs, meds; color=color, linestyle=:dash, linewidth=1.5)
            end

            ax.xticks = (1:length(nstates), string.(nstates))
            ri == 1 && axislegend(ax; position=:rb)
        end

        outpath = joinpath(outdir, "nstate_sweep.pdf")
        save(outpath, fig)
        println("Saved $outpath")
    end
end

length(ARGS) < 1 && error("Usage: julia plot_nstate_sweep.jl <datafile.h5>")
main(ARGS[1])
