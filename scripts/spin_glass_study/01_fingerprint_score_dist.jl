# Script 01: Fingerprint minimum score distribution at convergence.
#
# The fingerprint score for pattern j at context (gen, theta_s) is the
# Hamming distance between the pattern's parity vector and the desired-parity
# vector (companion_goal_s at each covered position).
#
# min_score = distance to the nearest PseudoGHZ codeword in the 64-pattern codebook.
#
# Hypothesis:
#   H=0: desired-parity vector converges to a near-uniform vector as accuracy
#        grows → lies close to a codeword → min_score → 0.
#   H=1: desired-parity vector is a random mix of 0s and 1s → lies far from
#        any codeword → min_score bounded away from 0.
#
# If confirmed, this is the proximal structural cause of the accuracy gap.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function probe_scores(goal, nqubit, n_probe, rep_sum, rep_ctr, rep,
                      fingerprint, cxt_master, companion, goal_idx; seed=1)
    n   = 3^nqubit
    rng = Random.MersenneTwister(seed)
    min_scores  = Int[]
    frac_valids = Float64[]
    for _ in 1:n_probe
        gen      = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        theta_s  = rand(rng, 0:1)
        ms, _, nv = all_scores(gen, theta_s, goal, rep, fingerprint,
                               cxt_master, companion, goal_idx, n)
        push!(min_scores, ms)
        push!(frac_valids, nv / length(cxt_master.base_even.pos))
    end
    return min_scores, frac_valids
end

function print_score_report(min_scores, frac_valids, label, npos)
    total = length(min_scores)
    println(label)
    @printf("  n_probe=%d  npos=%d\n", total, npos)
    @printf("  mean_min_score:  %.3f\n", mean(min_scores))
    @printf("  mean_frac_valid: %.3f\n", mean(frac_valids))
    @printf("  normalised mean: %.3f  (min_score / n_valid)\n",
            mean(min_scores[i] / max(1, round(Int, frac_valids[i]*npos))
                 for i in eachindex(min_scores)))
    println("  Distribution of min_score:")
    max_s = maximum(min_scores)
    for s in 0:min(max_s, 20)
        cnt = count(==(s), min_scores)
        bar = repeat("█", round(Int, 40 * cnt / total))
        @printf("    score=%2d: %5d (%5.2f%%)  %s\n", s, cnt, 100*cnt/total, bar)
    end
    println()
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_probe    = 5_000
    npos       = length(QCScaling.ContextMaster(nqubit).base_even.pos)

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    println("npos (context size) = $npos")
    println()

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("=" ^ 55)
        println("$H_label — running SA (nstate=$nstate, nsteps=$nsteps)")
        println("=" ^ 55)

        # Run 3 seeds and pool the probes for robustness
        all_min_scores  = Int[]
        all_frac_valids = Float64[]
        for seed in [42, 17, 99]
            acc, _, rep_sum, rep_ctr, rep =
                run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master; seed=seed)
            @printf("  seed=%d: acc=%.4f\n", seed, acc)
            ms, fv = probe_scores(goal, nqubit, n_probe ÷ 3, rep_sum, rep_ctr, rep,
                                  fingerprint, cxt_master, companion, goal_idx; seed=seed)
            append!(all_min_scores, ms)
            append!(all_frac_valids, fv)
        end
        println()
        print_score_report(all_min_scores, all_frac_valids,
                           "Min fingerprint score — $H_label", npos)
    end

    # Also probe at the START (random ensemble, before SA) to show baseline
    println("=" ^ 55)
    println("Baseline — random ensemble (no SA)")
    println("=" ^ 55)
    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))
        n    = 3^nqubit
        rng2 = Random.MersenneTwister(1)
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)
        ms, fv = probe_scores(goal, nqubit, n_probe, rep_sum, rep_ctr, rep,
                              fingerprint, cxt_master, companion, goal_idx; seed=9)
        print_score_report(ms, fv, "Baseline min score — $H_label", npos)
    end
end

main()
