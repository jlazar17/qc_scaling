# Script 04: Landscape properties vs H (continuous sweep).
#
# Sweep H ∈ {0.0, 0.1, 0.2, ..., 1.0} and measure at convergence:
#   - Final accuracy
#   - Mean min fingerprint score
#   - Fraction of wrong pairs with margin = 1 (fixable)
#   - Mean correct-pair margin
#
# Goal: find whether there is a sharp transition in landscape character
# at some critical H, or a smooth degradation.
# A sharp transition would suggest a spin-glass phase boundary.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function run_and_measure(goal, nqubit, nstate, nsteps, alpha_cool,
                         companion, goal_idx, fingerprint, cxt_master;
                         n_probe=1000, seed=42)
    n = 3^nqubit

    acc, _, rep_sum, rep_ctr, rep =
        run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                    companion, goal_idx, fingerprint, cxt_master; seed=seed)

    # Fingerprint score probe
    rng = Random.MersenneTwister(seed + 1000)
    min_scores = Int[]
    for _ in 1:n_probe
        gen     = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        theta_s = rand(rng, 0:1)
        ms, _, _ = all_scores(gen, theta_s, goal, rep, fingerprint,
                              cxt_master, companion, goal_idx, n)
        push!(min_scores, ms)
    end

    # Wrong pair margins
    wrong_margins = Int[]
    correct_margins = Int[]
    for k1 in 1:n-1
        companion[k1] == 0 && continue; k1 > companion[k1] && continue
        k2 = companion[k1]
        c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
        (c1 == 0 || c2 == 0) && continue
        s1 = rep_sum[k1]; s2 = rep_sum[k2]
        (2*s1 == c1 || 2*s2 == c2) && continue
        r1 = 2*s1 > c1; r2 = 2*s2 > c2
        j  = goal_idx[k1]
        correct = (r1 ⊻ r2) == !iszero(goal[j])
        m = min(abs(2*s1 - c1), abs(2*s2 - c2))
        correct ? push!(correct_margins, m) : push!(wrong_margins, m)
    end

    frac_fixable = isempty(wrong_margins) ? NaN :
                   count(==(1), wrong_margins) / length(wrong_margins)
    mean_wrong_margin   = isempty(wrong_margins)   ? NaN : mean(wrong_margins)
    mean_correct_margin = isempty(correct_margins) ? NaN : mean(correct_margins)

    return (
        acc               = acc,
        mean_min_score    = mean(min_scores),
        frac_score_zero   = count(==(0), min_scores) / length(min_scores),
        frac_fixable      = frac_fixable,
        mean_wrong_margin = mean_wrong_margin,
        mean_correct_margin = mean_correct_margin,
        n_wrong           = length(wrong_margins),
    )
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 12     # average over seeds for each H
    n_probe    = 1_000

    H_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    @printf("%-6s  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
            "H", "acc", "mean_ms", "frac_ms0", "frac_fix", "wrong_mg", "corr_mg")
    println(repeat("-", 82))

    for H in H_vals
        k_ones = h_to_kones(H, ngbits)
        accs = Float64[]; mean_mss = Float64[]
        frac_ms0s = Float64[]; frac_fixs = Float64[]
        wrong_mgs = Float64[]; corr_mgs  = Float64[]

        for gseed in 1:n_seeds
            rng  = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal = Random.shuffle!(rng, vcat(ones(Int, k_ones),
                                             zeros(Int, ngbits - k_ones)))
            r = run_and_measure(goal, nqubit, nstate, nsteps, alpha_cool,
                                companion, goal_idx, fingerprint, cxt_master;
                                n_probe=n_probe, seed=gseed * 31 + 7)
            push!(accs, r.acc)
            push!(mean_mss, r.mean_min_score)
            push!(frac_ms0s, r.frac_score_zero)
            !isnan(r.frac_fixable) && push!(frac_fixs, r.frac_fixable)
            !isnan(r.mean_wrong_margin) && push!(wrong_mgs, r.mean_wrong_margin)
            !isnan(r.mean_correct_margin) && push!(corr_mgs, r.mean_correct_margin)
        end

        @printf("%-6.2f  %-8.4f  %-12.4f  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n",
                H,
                mean(accs),
                mean(mean_mss),
                mean(frac_ms0s),
                isempty(frac_fixs) ? NaN : mean(frac_fixs),
                isempty(wrong_mgs) ? NaN : mean(wrong_mgs),
                isempty(corr_mgs)  ? NaN : mean(corr_mgs))
        flush(stdout)
    end
end

main()
