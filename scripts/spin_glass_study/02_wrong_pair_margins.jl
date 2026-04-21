# Script 02: Wrong-pair margin analysis at the accuracy ceiling.
#
# A wrong pair (k1,k2) can only be fixed by a single-state swap if the
# majority vote at k1 (or k2) has margin exactly 1 — i.e., the majority is
# held by a single vote.  If margin > 1, flipping one state's vote cannot
# change the majority.
#
# Hypothesis:
#   H=0: wrong pairs at convergence have margin = 1 (fixable).
#   H=1: wrong pairs at convergence have margin > 1 (stuck).
#        This would directly explain why oracle_pos = 0 at acc=0.77.
#
# Also measure: for CORRECT pairs, what is the margin distribution?
# High margins on correct pairs means H=0 builds a "reinforced" rep.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function analyse_margins(rep_sum, rep_ctr, goal, companion, goal_idx, nqubit)
    n = 3^nqubit
    wrong_margins  = Int[]
    correct_margins = Int[]
    uncovered = 0

    for k1 in 1:n-1
        companion[k1] == 0 && continue
        k1 > companion[k1] && continue   # canonical representative
        k2 = companion[k1]
        c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
        (c1 == 0 || c2 == 0) && (uncovered += 1; continue)
        s1 = rep_sum[k1]; s2 = rep_sum[k2]
        (2*s1 == c1 || 2*s2 == c2) && continue   # tied — skip
        r1 = 2*s1 > c1; r2 = 2*s2 > c2
        j  = goal_idx[k1]
        correct = (r1 ⊻ r2) == !iszero(goal[j])
        # margin = min of the two position margins (the tighter constraint)
        m = min(abs(2*s1 - c1), abs(2*s2 - c2))
        correct ? push!(correct_margins, m) : push!(wrong_margins, m)
    end
    return wrong_margins, correct_margins, uncovered
end

function print_margin_report(margins, label)
    isempty(margins) && (@printf("  %s: (none)\n\n", label); return)
    total = length(margins)
    println("  $label (n=$total)")
    max_m = min(maximum(margins), 15)
    for m in 1:max_m
        cnt = count(==(m), margins)
        bar = repeat("█", round(Int, 40 * cnt / total))
        @printf("    margin=%2d: %5d (%5.2f%%)  %s\n", m, cnt, 100*cnt/total, bar)
    end
    if maximum(margins) > 15
        cnt = count(>(15), margins)
        @printf("    margin>15: %5d (%5.2f%%)\n", cnt, 100*cnt/total)
    end
    @printf("    mean=%.2f  median=%.1f  frac_margin1=%.3f\n\n",
            mean(margins), median(float.(margins)),
            count(==(1), margins) / total)
end

function main()
    nqubit     = 6
    ngbits     = (3^nqubit - 1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 6

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("=" ^ 60)
        println("$H_label")
        println("=" ^ 60)

        all_wrong   = Int[]
        all_correct = Int[]
        accs        = Float64[]

        for seed in 1:n_seeds
            acc, _, rep_sum, rep_ctr, _ =
                run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master; seed=seed)
            push!(accs, acc)
            wm, cm, _ = analyse_margins(rep_sum, rep_ctr, goal, companion, goal_idx, nqubit)
            append!(all_wrong, wm)
            append!(all_correct, cm)
        end

        @printf("  Accuracy: mean=%.4f  std=%.4f\n\n",
                mean(accs), std(accs))
        print_margin_report(all_wrong,   "Wrong-pair margins  (fixable if margin=1)")
        print_margin_report(all_correct, "Correct-pair margins (higher → more robust)")
    end
end

main()
