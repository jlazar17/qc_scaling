using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Plots
using Printf

# Enumerate ALL contexts (generator, theta_s) for each nqubit and count
# the number of both-covered pairs per context.
#
# A "both-covered pair" is a pair index j where both goal-string positions
# k1=2j-1 and k2=2j fall within the same state's context.
#
# Outputs a histogram (count per both-covered-pair count) and reports
# the context(s) with the most both-covered pairs.

function count_both_covered_pairs(generator, theta_s, cxt_master, nqubit)
    n        = 3^nqubit
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    # Count how many positions of each pair j fall in this context
    pair_hits = Dict{Int, Int}()
    for base_po in base_cxt.pos
        derived_po = generator + base_po
        k = derived_po.index
        k == n && continue          # skip identity
        j = (k + 1) ÷ 2            # 1-indexed pair
        pair_hits[j] = get(pair_hits, j, 0) + 1
    end

    # Both-covered: pair j has hits == 2
    return count(v -> v == 2, values(pair_hits))
end

function analyze_nqubit(nqubit)
    n          = 3^nqubit
    cxt_master = QCScaling.ContextMaster(nqubit)

    hist       = Dict{Int, Int}()   # n_both_covered → count of contexts
    best_count = 0
    best_ctx   = Tuple{Int,Int}[]   # (gen_idx, theta_s) with the most pairs

    for gen_idx in 0:n-1
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        for theta_s in 0:1
            nb = count_both_covered_pairs(generator, theta_s, cxt_master, nqubit)
            hist[nb] = get(hist, nb, 0) + 1

            if nb > best_count
                best_count = nb
                best_ctx   = [(gen_idx, theta_s)]
            elseif nb == best_count && best_count > 0
                push!(best_ctx, (gen_idx, theta_s))
            end
        end
    end

    n_contexts = 2 * n
    @printf("\n%s\n", "="^60)
    @printf("nqubit=%d   total contexts=%d\n", nqubit, n_contexts)
    @printf("%s\n\n", "="^60)

    @printf("  %-20s  %-12s  %-10s\n", "both-covered pairs", "# contexts", "fraction")
    println("  ", "-"^46)
    for k in sort(collect(keys(hist)))
        @printf("  %-20d  %-12d  %.4f\n", k, hist[k], hist[k] / n_contexts)
    end

    @printf("\n  Max both-covered pairs: %d\n", best_count)
    @printf("  Contexts achieving max (%d total):\n", length(best_ctx))
    for (gen_idx, theta_s) in best_ctx
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        βs_str = join(generator.βs, " ")
        @printf("    βs=[%s]  θ_s=%d  (gen_idx=%d)\n", βs_str, theta_s, gen_idx)
    end

    return hist, n_contexts
end

function main()
    nqubits = [4, 6, 8, 10]
    all_hists = Dict{Int, Dict{Int,Int}}()
    all_totals = Dict{Int, Int}()

    for nqubit in nqubits
        hist, n_ctx = analyze_nqubit(nqubit)
        all_hists[nqubit]  = hist
        all_totals[nqubit] = n_ctx
        flush(stdout)
    end

    # Collect all observed pair counts across all n, sorted
    all_counts = sort(unique(vcat([collect(keys(h)) for h in values(all_hists)]...)))
    xlabels    = string.(all_counts)
    colors     = [:steelblue, :darkorange, :seagreen, :crimson]

    plots = []
    for (i, nq) in enumerate(nqubits)
        fracs = [get(all_hists[nq], c, 0) / all_totals[nq] for c in all_counts]
        push!(plots, bar(
            xlabels, fracs,
            title      = "n = $nq",
            xlabel     = "Both-covered pairs",
            ylabel     = "Fraction of contexts",
            color      = colors[i],
            legend     = false,
            yscale     = :log10,
            ylims      = (1e-5, 1.5),
            xticks     = (1:length(xlabels), xlabels),
            bar_width  = 0.7,
            titlefont  = font(12),
        ))
    end

    p = plot(plots..., layout=(2, 2), size=(900, 650), dpi=150,
             plot_title="Both-covered pair distribution by nqubit")

    outfile = joinpath(@__DIR__, "both_covered_histogram.png")
    savefig(p, outfile)
    println("\nPlot saved to $outfile")
end

main()
