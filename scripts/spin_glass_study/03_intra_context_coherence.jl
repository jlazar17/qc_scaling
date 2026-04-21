# Script 03: Intra-context parity coherence.
#
# For a random PseudoGHZ state, what fraction of its covered positions vote 1?
# If states are internally coherent (tend toward all-0 or all-1 within a
# context), then:
#   - H=0 desired patterns (all want the same companion parity) are close
#     to a coherent codeword → low min_score
#   - H=1 desired patterns (mixed) can never match a coherent codeword → high min_score
#
# Also: companion_goal_s coherence.
# For a given (gen, theta_s, converged rep):
#   H=0: all companion_goal values point the same direction (high coherence)
#   H=1: companion_goal values are mixed (low coherence)
# Measured as: fraction of valid companion_goals that are 1 (vs 0).

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function main()
    nqubit  = 6
    n       = 3^nqubit
    ngbits  = (n-1) ÷ 2
    nstate  = 45
    n_probe = 5_000
    nsteps  = 300_000
    alpha_cool = 0.9999

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)
    npos        = length(cxt_master.base_even.pos)

    # ------------------------------------------------------------------
    # Part A: PseudoGHZ internal parity distribution (goal-independent)
    # For each of N random states, what fraction of covered positions
    # vote 1?  A bimodal distribution (near 0 or near 1) means states
    # are coherent; a uniform distribution means mixed.
    # ------------------------------------------------------------------
    println("=" ^ 60)
    println("Part A: Internal parity distribution of random PseudoGHZ states")
    println("=" ^ 60)

    rng = Random.MersenneTwister(42)
    frac_ones = Float64[]
    for _ in 1:10_000
        ts  = rand(rng, 0:1); tz = rand(rng, 0:1)
        alp = SVector{nqubit-1, Int}(rand(rng, 0:1, nqubit-1))
        gen = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        s   = QCScaling.PseudoGHZState(ts, tz, alp, gen)
        rep_sum_tmp = zeros(Int, n); rep_ctr_tmp = zeros(Int, n)
        apply_state!(rep_sum_tmp, rep_ctr_tmp, s, cxt_master, 1)
        covered = findall(>(0), rep_ctr_tmp)
        isempty(covered) && continue
        push!(frac_ones, sum(rep_sum_tmp[k] for k in covered) / length(covered))
    end

    println("  Fraction of covered positions voting 1 per state:")
    @printf("  mean=%.4f  std=%.4f\n", mean(frac_ones), std(frac_ones))
    # Histogram in 10 bins from 0 to 1
    edges = 0.0:0.1:1.0
    for i in 1:10
        lo, hi = edges[i], edges[i+1]
        cnt = count(x -> lo <= x < hi, frac_ones)
        bar = repeat("█", round(Int, 50 * cnt / length(frac_ones)))
        @printf("  [%.1f, %.1f): %5d (%5.2f%%)  %s\n",
                lo, hi, cnt, 100*cnt/length(frac_ones), bar)
    end
    println()

    # ------------------------------------------------------------------
    # Part B: Companion-goal coherence for H=0 vs H=1 at convergence.
    # For each random (gen, theta_s), compute companion_goal_s values
    # and measure their "coherence" = fraction that are 1 (vs 0).
    # Compare H=0 (should be very coherent) vs H=1 (should be mixed).
    # ------------------------------------------------------------------
    println("=" ^ 60)
    println("Part B: Companion-goal coherence at convergence")
    println("=" ^ 60)

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng2 = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng2, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        print("  Running SA for $H_label ... ")
        acc, _, rep_sum, rep_ctr, rep =
            run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                        companion, goal_idx, fingerprint, cxt_master; seed=42)
        @printf("acc=%.4f\n", acc)

        rng3 = Random.MersenneTwister(7)
        cg_fracs   = Float64[]   # fraction of valid companion_goals = 1
        cg_n_valid = Int[]       # number of valid companion_goals per context

        for _ in 1:n_probe
            gen     = QCScaling.ParityOperator(rand(rng3, 0:n-1), nqubit)
            theta_s = rand(rng3, 0:1)
            bc      = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt     = QCScaling.Context(gen, bc)
            cg      = [companion_goal_s(po, goal, rep, companion, goal_idx, n-1)
                       for po in cxt.pos]
            valid_cg = filter(!isnan, cg)
            isempty(valid_cg) && continue
            push!(cg_n_valid, length(valid_cg))
            push!(cg_fracs, count(!iszero, valid_cg) / length(valid_cg))
        end

        println("\n  $H_label — companion_goal_s coherence:")
        @printf("  mean_n_valid=%.1f  (out of npos=%d)\n",
                mean(cg_n_valid), npos)
        @printf("  mean_frac_cg1=%.4f  std=%.4f\n",
                mean(cg_fracs), std(cg_fracs))
        println("  Distribution of (fraction of companion_goals = 1):")
        for i in 1:10
            lo, hi = edges[i], edges[i+1]
            cnt = count(x -> lo <= x < hi, cg_fracs)
            bar = repeat("█", round(Int, 50 * cnt / length(cg_fracs)))
            @printf("    [%.1f, %.1f): %5d (%5.2f%%)  %s\n",
                    lo, hi, cnt, 100*cnt/length(cg_fracs), bar)
        end
        println()
    end
end

main()
