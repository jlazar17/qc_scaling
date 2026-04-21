using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using HDF5
using Statistics
using LaTeXStrings

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function main(datafile)
    outdir = dirname(datafile)

    keys_ordered = String[]
    h5open(datafile, "r") do h5f; append!(keys_ordered, sort(keys(h5f))); end

    h5open(datafile, "r") do h5f
        groups = [(k, h5f[k]) for k in keys_ordered]
        n      = length(groups)

        # -------------------------------------------------------------------
        # Plot 1: accuracy distributions (one panel per config)
        # -------------------------------------------------------------------
        ncols = 4
        nrows = ceil(Int, n / ncols)
        fig1  = Figure(size=(300 * ncols, 250 * nrows))

        for (i, (key, gp)) in enumerate(groups)
            nqubit = read(HDF5.attributes(gp)["nqubit"])
            nstate = read(HDF5.attributes(gp)["nstate"])
            nstate_mult = read(HDF5.attributes(gp)["nstate_mult"])
            pzero  = read(HDF5.attributes(gp)["pzero"])
            accs   = read(gp["accs"])
            max_acc = read(HDF5.attributes(gp)["max_acc"])
            n_top  = read(HDF5.attributes(gp)["n_top"])
            nseeds = read(HDF5.attributes(gp)["nseeds"])

            row = ceil(Int, i / ncols); col = mod1(i, ncols)
            ax  = Axis(fig1[row, col],
                title  = L"n=%$(nqubit),\; N_s=%$(nstate)\;(%$(nstate_mult)\times),\; p_0=%$(pzero)",
                xlabel = "Best accuracy",
                ylabel = "Count",
            )
            hist!(ax, accs; bins=15, color=(:steelblue, 0.7))
            vlines!(ax, max_acc; color=:red, linestyle=:dash, linewidth=2)
            text!(ax, max_acc, 0;
                  text="top: $(n_top)/$(nseeds)",
                  align=(:left, :bottom), fontsize=11, color=:red, offset=(4, 4))
        end

        save(joinpath(outdir, "convergence_accuracy_dist.pdf"), fig1)
        println("Saved convergence_accuracy_dist.pdf")

        # -------------------------------------------------------------------
        # Plot 2: Jaccard similarity distributions
        # -------------------------------------------------------------------
        fig2  = Figure(size=(300 * ncols, 250 * nrows))

        for (i, (key, gp)) in enumerate(groups)
            nqubit = read(HDF5.attributes(gp)["nqubit"])
            nstate = read(HDF5.attributes(gp)["nstate"])
            nstate_mult = read(HDF5.attributes(gp)["nstate_mult"])
            pzero  = read(HDF5.attributes(gp)["pzero"])
            jac_all = read(gp["jac_all"])
            jac_top = read(gp["jac_top"])
            jac_top = filter(x -> x >= 0, jac_top)

            row = ceil(Int, i / ncols); col = mod1(i, ncols)
            ax  = Axis(fig2[row, col],
                title  = L"n=%$(nqubit),\; N_s=%$(nstate)\;(%$(nstate_mult)\times),\; p_0=%$(pzero)",
                xlabel = "Jaccard similarity",
                ylabel = "Density",
            )
            xlims!(ax, 0, 1)
            isempty(jac_all) || density!(ax, jac_all; color=(:steelblue, 0.4),
                                          strokecolor=:steelblue, strokewidth=1.5,
                                          label="all pairs")
            isempty(jac_top) || density!(ax, jac_top; color=(:crimson, 0.4),
                                          strokecolor=:crimson, strokewidth=1.5,
                                          label="top pairs")
            i == 1 && axislegend(ax; position=:rt)
        end

        save(joinpath(outdir, "convergence_jaccard.pdf"), fig2)
        println("Saved convergence_jaccard.pdf")

        # -------------------------------------------------------------------
        # Plot 3: Rep vector similarity for top runs
        # -------------------------------------------------------------------
        fig3  = Figure(size=(300 * ncols, 250 * nrows))

        for (i, (key, gp)) in enumerate(groups)
            nqubit = read(HDF5.attributes(gp)["nqubit"])
            nstate = read(HDF5.attributes(gp)["nstate"])
            nstate_mult = read(HDF5.attributes(gp)["nstate_mult"])
            pzero  = read(HDF5.attributes(gp)["pzero"])
            rs_all = read(gp["rep_sim_all"])
            rs_top = read(gp["rep_sim_top"])
            rs_top = filter(x -> x >= 0, rs_top)

            row = ceil(Int, i / ncols); col = mod1(i, ncols)
            ax  = Axis(fig3[row, col],
                title  = L"n=%$(nqubit),\; N_s=%$(nstate)\;(%$(nstate_mult)\times),\; p_0=%$(pzero)",
                xlabel = "Rep vector agreement",
                ylabel = "Density",
            )
            xlims!(ax, 0, 1)
            isempty(rs_all) || density!(ax, rs_all; color=(:steelblue, 0.4),
                                         strokecolor=:steelblue, strokewidth=1.5,
                                         label="all pairs")
            isempty(rs_top) || density!(ax, rs_top; color=(:crimson, 0.4),
                                         strokecolor=:crimson, strokewidth=1.5,
                                         label="top pairs")
            i == 1 && axislegend(ax; position=:lt)
        end

        save(joinpath(outdir, "convergence_rep_sim.pdf"), fig3)
        println("Saved convergence_rep_sim.pdf")
    end
end

length(ARGS) < 1 && error("Usage: julia plot_convergence.jl <datafile.h5>")
main(ARGS[1])
