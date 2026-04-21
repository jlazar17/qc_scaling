# Plot validation: combined publication-quality figures from all validation runs.
#
# Requires (any subset; missing files are gracefully skipped):
#   data/01_solutions.h5     (from 01_collect_solutions.jl)
#   data/02_jaccard.h5       (from 02_jaccard_analysis.jl)
#   data/03_perturbation.csv (from 03_perturbation_recovery.jl)
#   data/04_convergence.h5   (from 04_convergence_nsteps.jl)
#
# Produces a single multi-panel figure suitable for a paper methods section,
# plus individual PDFs for each panel group.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie, HDF5, DelimitedFiles, Printf, Statistics

include(joinpath(@__DIR__, "../../notebooks/plotting_boilerplate.jl"))

function main()
    datadir  = joinpath(@__DIR__, "data")
    jac_h5   = joinpath(datadir, "02_jaccard.h5")
    pert_csv = joinpath(datadir, "03_perturbation.csv")
    conv_h5  = joinpath(datadir, "04_convergence.h5")
    outfile  = joinpath(datadir, "validation_combined.pdf")

    all_H    = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]
    H_colors = Makie.resample_cmap(:plasma, length(all_H))
    H_to_col = Dict(H => H_colors[i] for (i, H) in enumerate(all_H))

    conv_H_vals = [0.0, 0.5, 1.0]

    fig = Figure(size=(1400, 900))

    # -----------------------------------------------------------------------
    # Row 1: pairwise generator Jaccard and rep agreement vs H (from 02)
    # -----------------------------------------------------------------------
    if isfile(jac_h5)
        nqubits_02 = Int[]
        jac_data   = Dict{Int, Dict{Float64, Vector{Float64}}}()
        rep_data   = Dict{Int, Dict{Float64, Vector{Float64}}}()

        h5open(jac_h5, "r") do h5f
            for nq_key in sort(collect(keys(h5f)))
                g      = h5f[nq_key]
                nqubit = read(HDF5.attributes(g)["nqubit"])
                push!(nqubits_02, nqubit)
                jac_data[nqubit] = Dict{Float64, Vector{Float64}}()
                rep_data[nqubit] = Dict{Float64, Vector{Float64}}()
                for H_key in keys(g)
                    gp = g[H_key]
                    H  = read(HDF5.attributes(gp)["H"])
                    jac_data[nqubit][H] = read(gp["jac_gen"])
                    rep_data[nqubit][H] = read(gp["rep_agr"])
                end
            end
        end
        sort!(nqubits_02)
        nq_colors = Makie.wong_colors()[1:length(nqubits_02)]

        ax_jac = Axis(fig[1, 1:2],
            xlabel = "H",
            ylabel = "Pairwise generator Jaccard",
            title  = "Cross-seed solution overlap",
        )
        ylims!(ax_jac, 0, 1)

        ax_rep = Axis(fig[1, 3:4],
            xlabel = "H",
            ylabel = "Rep parity agreement",
            title  = "Cross-seed functional agreement",
        )
        ylims!(ax_rep, 0, 1)

        for (ci, nqubit) in enumerate(nqubits_02)
            hs   = sort([H for H in all_H if haskey(jac_data[nqubit], H)])
            jacs = [mean(jac_data[nqubit][H]) for H in hs]
            reps = [mean(rep_data[nqubit][H])  for H in hs]
            c    = nq_colors[ci]
            lines!(ax_jac, hs, jacs; color=c, linewidth=2, label="n=$nqubit")
            scatter!(ax_jac, hs, jacs; color=c, markersize=8)
            lines!(ax_rep, hs, reps; color=c, linewidth=2)
            scatter!(ax_rep, hs, reps; color=c, markersize=8)
        end
        axislegend(ax_jac; position=:lt)
        println("Row 1: Jaccard data loaded from $jac_h5")
    else
        println("Skipping row 1: $jac_h5 not found (run 02_jaccard_analysis.jl)")
    end

    # -----------------------------------------------------------------------
    # Row 2, left: perturbation recovery (from 03)
    # CSV columns: nqubit,H,frac,trial,n_perturb,gen_jaccard,rep_agreement,final_acc,base_acc
    # One row per trial; aggregate here for plotting.
    # -----------------------------------------------------------------------
    if isfile(pert_csv)
        raw, hdr = readdlm(pert_csv, ',', header=true)
        hdr = vec(hdr)
        col(name) = findfirst(==(name), hdr)

        H_vals_pert   = sort(unique(Float64.(raw[:, col("H")])))
        frac_vals_pert = sort(unique(Float64.(raw[:, col("frac")])))

        ax_pert_jac = Axis(fig[2, 1:2],
            xlabel = "Perturbation fraction",
            ylabel = "Generator Jaccard (recovery vs original)",
            title  = "Basin stability — generator (nqubit=6)",
        )
        ylims!(ax_pert_jac, 0, 1)

        for H in H_vals_pert
            haskey(H_to_col, H) || continue
            xs_j = Float64[]; ys_j = Float64[]; es_j = Float64[]
            for frac in frac_vals_pert
                mask = (Float64.(raw[:, col("H")]) .== H) .&
                       (Float64.(raw[:, col("frac")]) .== frac)
                any(mask) || continue
                vals = Float64.(raw[mask, col("gen_jaccard")])
                push!(xs_j, frac); push!(ys_j, mean(vals)); push!(es_j, std(vals))
            end
            c = H_to_col[H]
            lines!(ax_pert_jac, xs_j, ys_j; color=c, linewidth=2, label=@sprintf("H=%.2f", H))
            errorbars!(ax_pert_jac, xs_j, ys_j, es_j; color=c, whiskerwidth=5)
            scatter!(ax_pert_jac, xs_j, ys_j; color=c, markersize=8)
        end
        axislegend(ax_pert_jac; position=:lb)
        println("Row 2 left: perturbation data loaded from $pert_csv")
    else
        println("Skipping row 2 left: $pert_csv not found (run 03_perturbation_recovery.jl)")
    end

    # -----------------------------------------------------------------------
    # Row 2, right: convergence vs nsteps (from 04)
    # -----------------------------------------------------------------------
    if isfile(conv_h5)
        ax_conv_acc = Axis(fig[2, 3],
            xlabel = "nsteps",
            ylabel = "Median accuracy",
            title  = "Accuracy vs step budget (nqubit=6)",
            xscale = log10,
        )
        ax_conv_jac = Axis(fig[2, 4],
            xlabel = "nsteps",
            ylabel = "Median pairwise generator Jaccard",
            title  = "Reproducibility vs step budget (nqubit=6)",
            xscale = log10,
        )
        ylims!(ax_conv_acc, 0, 1); ylims!(ax_conv_jac, 0, 1)

        h5open(conv_h5, "r") do h5f
            for H in conv_H_vals
                H_key = @sprintf("H%.3f", H)
                haskey(h5f, H_key) || continue
                g  = h5f[H_key]
                ns = Float64.(read(g["nsteps_vals"]))
                am = read(g["acc_med_vec"])
                jm = read(g["jac_med_vec"])
                c  = H_to_col[H]
                lbl = @sprintf("H=%.2f", H)
                lines!(ax_conv_acc, ns, am; color=c, linewidth=2, label=lbl)
                scatter!(ax_conv_acc, ns, am; color=c, markersize=6)
                lines!(ax_conv_jac, ns, jm; color=c, linewidth=2)
                scatter!(ax_conv_jac, ns, jm; color=c, markersize=6)
            end
        end
        axislegend(ax_conv_acc; position=:rb)
        println("Row 2 right: convergence data loaded from $conv_h5")
    else
        println("Skipping row 2 right: $conv_h5 not found (run 04_convergence_nsteps.jl)")
    end

    save(outfile, fig)
    println("Saved $outfile")

    # -----------------------------------------------------------------------
    # Standalone figures for each panel group (easier to include in papers).
    # -----------------------------------------------------------------------
    if isfile(pert_csv)
        raw, hdr = readdlm(pert_csv, ',', header=true)
        hdr = vec(hdr)
        col(name) = findfirst(==(name), hdr)
        H_vals_pert    = sort(unique(Float64.(raw[:, col("H")])))
        frac_vals_pert = sort(unique(Float64.(raw[:, col("frac")])))

        for (metric_col, metric_name, out_tag) in [
                ("gen_jaccard",    "Generator Jaccard (recovery)",    "jac"),
                ("rep_agreement",  "Rep parity agreement (recovery)", "rep"),
            ]
            fig2 = Figure(size=(600, 420))
            ax2  = Axis(fig2[1, 1],
                xlabel = "Perturbation fraction",
                ylabel = metric_name,
                title  = "Basin stability (nqubit=6)",
            )
            ylims!(ax2, 0, 1)
            for H in H_vals_pert
                haskey(H_to_col, H) || continue
                xs = Float64[]; ys = Float64[]; es = Float64[]
                for frac in frac_vals_pert
                    mask = (Float64.(raw[:, col("H")]) .== H) .&
                           (Float64.(raw[:, col("frac")]) .== frac)
                    any(mask) || continue
                    vals = Float64.(raw[mask, col(metric_col)])
                    vals = filter(v -> v >= 0, vals)   # drop -1 sentinel
                    isempty(vals) && continue
                    push!(xs, frac); push!(ys, mean(vals)); push!(es, std(vals))
                end
                c = H_to_col[H]
                lines!(ax2, xs, ys; color=c, linewidth=2, label=@sprintf("H=%.2f", H))
                errorbars!(ax2, xs, ys, es; color=c, whiskerwidth=5)
                scatter!(ax2, xs, ys; color=c, markersize=8)
            end
            axislegend(ax2; position=:lb)
            out2 = joinpath(datadir, "03_recovery_$(out_tag).pdf")
            save(out2, fig2)
            println("Saved $out2")
        end
    end

    if isfile(conv_h5)
        fig3 = Figure(size=(900, 420))
        ax3a = Axis(fig3[1, 1],
            xlabel = "nsteps", ylabel = "Median accuracy",
            title  = "Accuracy vs step budget (nqubit=6)", xscale=log10)
        ax3b = Axis(fig3[1, 2],
            xlabel = "nsteps", ylabel = "Median generator Jaccard",
            title  = "Reproducibility vs step budget (nqubit=6)", xscale=log10)
        ax3c = Axis(fig3[1, 3],
            xlabel = "nsteps", ylabel = "Median rep agreement",
            title  = "Functional reproducibility (nqubit=6)", xscale=log10)
        ylims!(ax3a, 0, 1); ylims!(ax3b, 0, 1); ylims!(ax3c, 0, 1)
        h5open(conv_h5, "r") do h5f
            for H in conv_H_vals
                H_key = @sprintf("H%.3f", H)
                haskey(h5f, H_key) || continue
                g  = h5f[H_key]
                ns = Float64.(read(g["nsteps_vals"]))
                am = read(g["acc_med_vec"])
                jm = read(g["jac_med_vec"])
                rm = read(g["rep_med_vec"])
                c  = H_to_col[H]; lbl = @sprintf("H=%.2f", H)
                lines!(ax3a, ns, am; color=c, linewidth=2, label=lbl)
                scatter!(ax3a, ns, am; color=c, markersize=6)
                lines!(ax3b, ns, jm; color=c, linewidth=2)
                scatter!(ax3b, ns, jm; color=c, markersize=6)
                lines!(ax3c, ns, rm; color=c, linewidth=2)
                scatter!(ax3c, ns, rm; color=c, markersize=6)
            end
        end
        axislegend(ax3a; position=:rb)
        out3 = joinpath(datadir, "04_convergence_vs_nsteps.pdf")
        save(out3, fig3)
        println("Saved $out3")
    end
end

main()
