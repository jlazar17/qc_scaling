# Script 05: State-level selectivity in converged ensembles.
#
# HYPOTHESIS: In a converged H=0 ensemble, each state preferentially
# reinforces CORRECT pairs (votes agree with correct parity) rather than
# wrong pairs. In H=1, states have near-zero selectivity: they reinforce
# correct and wrong pairs equally.
#
# Selectivity of a state = P(vote correct | correct pair) - P(vote correct | wrong pair)
# where "vote correct" means the state's parity at position k agrees with
# the majority parity of the converged rep (which we take as the "correct" label).
#
# If selectivity is systematically positive → optimizer is genuinely finding
# states that preferentially reinforce already-correct positions.
# If selectivity ≈ 0 → frustration: states cannot discriminate.
#
# We also measure: for each state, the net change in (correct_pair_margin -
# wrong_pair_margin) it contributes to. This directly quantifies whether
# adding a state widens or narrows the gap.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# For a given state and a converged rep, compute:
#   correct_agree: fraction of covered positions in CORRECT pairs where
#                  state's parity agrees with rep majority
#   wrong_agree:   same for WRONG pairs
# Returns (correct_agree, wrong_agree, n_correct_covered, n_wrong_covered)
# ---------------------------------------------------------------------------
function state_selectivity(s, rep_sum, rep_ctr, goal, companion, goal_idx,
                           cxt_master, nqubit)
    n = 3^nqubit
    tmp_sum = zeros(Int, n); tmp_ctr = zeros(Int, n)
    apply_state!(tmp_sum, tmp_ctr, s, cxt_master, 1)
    covered = findall(>(0), tmp_ctr)

    correct_agree = 0; correct_total = 0
    wrong_agree   = 0; wrong_total   = 0

    for k in covered
        companion[k] == 0 && continue
        k2 = companion[k]
        c1 = rep_ctr[k]; c2 = rep_ctr[k2]
        (c1 == 0 || c2 == 0) && continue
        s1 = rep_sum[k]; s2 = rep_sum[k2]
        (2*s1 == c1 || 2*s2 == c2) && continue

        r1 = 2*s1 > c1; r2 = 2*s2 > c2
        j = goal_idx[k]
        is_correct = (r1 ⊻ r2) == !iszero(goal[j])

        # State's parity at position k
        state_parity = tmp_sum[k] > 0  # tmp_ctr[k]=1 so tmp_sum[k] in {0,1}
        # Majority parity at k in converged rep
        maj_parity = 2*rep_sum[k] > rep_ctr[k]
        agrees = (state_parity == maj_parity)

        if is_correct
            correct_agree += agrees
            correct_total += 1
        else
            wrong_agree += agrees
            wrong_total += 1
        end
    end

    ca = correct_total > 0 ? correct_agree / correct_total : NaN
    wa = wrong_total   > 0 ? wrong_agree   / wrong_total   : NaN
    return ca, wa, correct_total, wrong_total
end

# ---------------------------------------------------------------------------
# Margin contribution: for each covered position, the state's contribution
# to widening (or narrowing) the margin of the pair it belongs to.
# A state that votes WITH the majority widens the margin by +2;
# against the majority narrows it by -2 (or equivalently, adds -1 to |2s - c|
# if it agrees, +1 if it disagrees, both shifted since c increases by 1).
#
# Actually: current margin = |2s - c|.
# After adding the state with parity p:
#   new margin = |2(s + p) - (c + 1)|
# For the pair (k1, k2), we track the min margin change.
# ---------------------------------------------------------------------------
function pair_margin_delta(k, parity_k, rep_sum, rep_ctr)
    c = rep_ctr[k]; s = rep_sum[k]
    old_m = abs(2*s - c)
    new_m = abs(2*(s + parity_k) - (c + 1))
    return new_m - old_m
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 6

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    println("=" ^ 65)
    println("State selectivity analysis (correct vs wrong pair reinforcement)")
    println("=" ^ 65)
    println()

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("─" ^ 65)
        println("$H_label")
        println("─" ^ 65)

        all_correct_agree = Float64[]
        all_wrong_agree   = Float64[]
        all_selectivity   = Float64[]
        accs = Float64[]

        for seed in 1:n_seeds
            acc, ensemble, rep_sum, rep_ctr, rep =
                run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master; seed=seed)
            push!(accs, acc)

            for s in ensemble
                ca, wa, nc, nw = state_selectivity(s, rep_sum, rep_ctr, goal,
                                                   companion, goal_idx, cxt_master, nqubit)
                (isnan(ca) || isnan(wa)) && continue
                push!(all_correct_agree, ca)
                push!(all_wrong_agree, wa)
                push!(all_selectivity, ca - wa)
            end
        end

        @printf("  Accuracy: mean=%.4f  std=%.4f\n\n", mean(accs), std(accs))
        @printf("  States analysed: %d\n", length(all_selectivity))
        @printf("  Mean P(agree | correct pair): %.4f\n", mean(all_correct_agree))
        @printf("  Mean P(agree | wrong pair):   %.4f\n", mean(all_wrong_agree))
        @printf("  Mean selectivity (correct - wrong): %.4f  std=%.4f\n",
                mean(all_selectivity), std(all_selectivity))
        @printf("  Frac states with selectivity > 0: %.4f\n",
                count(>(0), all_selectivity) / length(all_selectivity))
        @printf("  Frac states with selectivity > 0.05: %.4f\n",
                count(>(0.05), all_selectivity) / length(all_selectivity))
        println()

        # Distribution of selectivity
        println("  Selectivity distribution:")
        edges = -1.0:0.1:1.0
        for i in 1:length(edges)-1
            lo, hi = edges[i], edges[i+1]
            cnt = count(x -> lo <= x < hi, all_selectivity)
            cnt == 0 && continue
            bar = repeat("█", round(Int, 40 * cnt / length(all_selectivity)))
            @printf("    [%+.1f, %+.1f): %5d (%5.2f%%)  %s\n",
                    lo, hi, cnt, 100*cnt/length(all_selectivity), bar)
        end
        println()
    end

    # ------------------------------------------------------------------
    # Part B: Selectivity vs accuracy across H sweep
    # Run 1 seed per H, measure mean selectivity of converged ensemble
    # ------------------------------------------------------------------
    println("=" ^ 65)
    println("Part B: Mean selectivity vs H")
    println("=" ^ 65)
    @printf("%-6s  %-8s  %-12s  %-12s  %-12s\n",
            "H", "acc", "corr_agree", "wrong_agree", "selectivity")
    println(repeat("-", 56))

    H_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rng0   = Random.MersenneTwister(99)
    companion0, goal_idx0, _ = build_shuffled_pairing(nqubit)

    for H in H_vals
        k_ones = h_to_kones(H, ngbits)
        rng2   = Random.MersenneTwister(round(Int, H * 1000) + 77)
        goal   = Random.shuffle!(rng2, vcat(ones(Int, k_ones),
                                            zeros(Int, ngbits - k_ones)))

        acc, ensemble, rep_sum, rep_ctr, _ =
            run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                        companion0, goal_idx0, fingerprint, cxt_master; seed=42)

        sels = Float64[]; cas = Float64[]; was = Float64[]
        for s in ensemble
            ca, wa, nc, nw = state_selectivity(s, rep_sum, rep_ctr, goal,
                                               companion0, goal_idx0, cxt_master, nqubit)
            (isnan(ca) || isnan(wa)) && continue
            push!(cas, ca); push!(was, wa); push!(sels, ca - wa)
        end

        @printf("%-6.2f  %-8.4f  %-12.4f  %-12.4f  %-12.4f\n",
                H, acc, mean(cas), mean(was), isempty(sels) ? NaN : mean(sels))
        flush(stdout)
    end
end

main()
