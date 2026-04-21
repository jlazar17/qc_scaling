# Script 16: Measure PseudoGHZ vote correlation within contexts.
#
# Hypothesis: PseudoGHZ states produce correlated (uniform) votes across
# positions in the same context.  If true, the 64 fingerprint columns should
# have low within-column variance (votes cluster near all-0 or all-1).
#
# This would explain why H=0 (uniform companion_goal) gets lower fp scores
# than H=1 (mixed companion_goal), and hence why H=1 correct-pair margins
# erode over time.
#
# Measurements:
#   1. For every fingerprint column, compute mean vote fraction across positions.
#      A bimodal distribution (near 0 or near 1) = correlated votes.
#
#   2. Pairwise vote correlation between positions within the same context.
#      High positive correlation = correlated structure.
#
#   3. Under a converged H=0 and H=1 ensemble, compute the companion_goal
#      vector for many random proposals, and measure its mean entropy.
#      Low entropy (uniform) = easier for fp picker.
#      High entropy (mixed)  = harder for fp picker.
#      Compare H=0 vs H=1.
#
#   4. Directly verify: for the same proposal, compare the minimum fp score
#      achievable against a uniform companion_goal vs a mixed one.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Compute mean vote fraction for each of the 64 fingerprint columns
# (per parity = base context).  Also compute inter-position correlation.
# ---------------------------------------------------------------------------
function fingerprint_correlation_stats(fp::FingerprintPacked, fingerprint::QCScaling.Fingerprint)
    npos   = fp.npos
    nalpha = size(fp.words, 4)
    ntz    = size(fp.words, 3)
    np     = 2   # parity index

    # For each parity, extract the 64 columns as binary matrices [npos × 64]
    results = Dict()
    for pi in 1:np
        mat = zeros(Int, npos, ntz * nalpha)
        col = 1
        for ai in 1:nalpha, tzi in 1:ntz
            for pos in 1:npos
                w = (pos-1) ÷ 64 + 1
                b = (pos-1) % 64
                mat[pos, col] = (fp.words[w, pi, tzi, ai] >> b) & 1
            end
            col += 1
        end

        # Column means: fraction of positions voting 1 per pattern
        col_means = vec(mean(mat, dims=1))

        # Pairwise position correlation averaged over all pairs
        # (measuring how much knowing one position's vote predicts another's)
        n_pairs_pos = npos * (npos - 1) ÷ 2
        corr_sum = 0.0
        n_valid  = 0
        for i in 1:npos, j in i+1:npos
            vi = mat[i, :]; vj = mat[j, :]
            si = std(vi); sj = std(vj)
            (si < 1e-8 || sj < 1e-8) && continue
            corr_sum += cor(vi, vj)
            n_valid  += 1
        end
        mean_corr = n_valid > 0 ? corr_sum / n_valid : 0.0

        results[pi] = (col_means=col_means, mean_corr=mean_corr,
                       frac_extreme=count(x -> x < 0.2 || x > 0.8, col_means) / length(col_means))
    end
    return results
end

# ---------------------------------------------------------------------------
# Companion-goal entropy for random proposals under a given ensemble state.
# Low = uniform target (easy for fp picker).  High = mixed (hard).
# ---------------------------------------------------------------------------
function companion_goal_entropy(goal, rep, fingerprint, cxt_master, companion, goal_idx,
                                 nqubit; n_probe=500, seed=42)
    n   = 3^nqubit
    rng = Random.MersenneTwister(seed)
    entropies = Float64[]

    for _ in 1:n_probe
        gen = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts  = rand(rng, 0:1)
        bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(gen, bc)
        cg  = [companion_goal_s(po, goal, rep, companion, goal_idx, n-1) for po in cxt.pos]
        valid = filter(!isnan, cg)
        isempty(valid) && continue
        p = mean(valid)           # fraction wanting vote=1
        (p <= 0 || p >= 1) && (push!(entropies, 0.0); continue)
        push!(entropies, -p*log2(p) - (1-p)*log2(1-p))  # binary entropy
    end
    return mean(entropies), std(entropies) / sqrt(length(entropies))
end

# ---------------------------------------------------------------------------
# Min achievable fp score for random proposals, for uniform vs mixed cg.
# ---------------------------------------------------------------------------
function min_fp_score_by_cg_type(fingerprint::FingerprintPacked, cxt_master,
                                  nqubit; n_probe=300, seed=99)
    n   = 3^nqubit
    rng = Random.MersenneTwister(seed)

    scores_uniform = Float64[]   # cg ≈ all 0 or all 1
    scores_mixed   = Float64[]   # cg ≈ half 0, half 1

    for _ in 1:n_probe
        gen = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts  = rand(rng, 0:1)
        bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(gen, bc)
        npos = length(cxt.pos)

        # Build synthetic uniform cg (all 0) and mixed cg (alternating 0,1)
        cg_uniform  = zeros(Float64, npos)         # all vote 0
        cg_mixed    = Float64[isodd(i) ? 1.0 : 0.0 for i in 1:npos]

        valid_u, vals_u = QCScaling._pack_companion_goal(cg_uniform, fingerprint.nwords)
        valid_m, vals_m = QCScaling._pack_companion_goal(cg_mixed,   fingerprint.nwords)

        parity_idx = bc.parity + 1
        nwords = fingerprint.nwords
        nalpha = size(fingerprint.words, 4)
        ntz    = size(fingerprint.words, 3)

        min_u = typemax(Int); min_m = typemax(Int)
        @inbounds for ai in 1:nalpha, tzi in 1:ntz
            su = 0; sm = 0
            for w in 1:nwords
                fw = fingerprint.words[w, parity_idx, tzi, ai]
                su += count_ones(xor(fw, vals_u[w]) & valid_u[w])
                sm += count_ones(xor(fw, vals_m[w]) & valid_m[w])
            end
            su < min_u && (min_u = su)
            sm < min_m && (min_m = sm)
        end
        push!(scores_uniform, min_u)
        push!(scores_mixed,   min_m)
    end
    return scores_uniform, scores_mixed
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit = 6
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fp_raw     = QCScaling.Fingerprint(nqubit)
    fingerprint = FingerprintPacked(fp_raw)
    cxt_master  = QCScaling.ContextMaster(nqubit)

    println("=" ^ 60)
    println("1. FINGERPRINT COLUMN STATISTICS")
    println("=" ^ 60)
    println("Each column = one (theta_z, alpha) pattern, npos=$(fingerprint.npos) positions")
    println()

    stats = fingerprint_correlation_stats(fingerprint, fp_raw)
    for pi in 1:2
        s = stats[pi]
        println("Parity $(pi-1):")
        @printf("  Mean pairwise position correlation:  %.4f\n", s.mean_corr)
        @printf("  Fraction of columns with mean ∉ [0.2,0.8]:  %.4f\n", s.frac_extreme)
        println("  Column mean distribution (vote fraction per pattern):")
        @printf("  min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f\n",
                minimum(s.col_means),
                quantile(s.col_means, 0.25),
                quantile(s.col_means, 0.5),
                quantile(s.col_means, 0.75),
                maximum(s.col_means))
        println()
    end

    println("=" ^ 60)
    println("2. MIN ACHIEVABLE FP SCORE: UNIFORM vs MIXED COMPANION-GOAL")
    println("=" ^ 60)
    su, sm = min_fp_score_by_cg_type(fingerprint, cxt_master, nqubit;
                                       n_probe=500, seed=17)
    @printf("Uniform cg (all-0): min_fp = %.2f ± %.2f\n", mean(su), std(su)/sqrt(length(su)))
    @printf("Mixed    cg (alt):  min_fp = %.2f ± %.2f\n", mean(sm), std(sm)/sqrt(length(sm)))
    @printf("Difference (mixed - uniform): %.2f\n", mean(sm) - mean(su))
    println()

    println("=" ^ 60)
    println("3. COMPANION-GOAL ENTROPY UNDER CONVERGED ENSEMBLES")
    println("=" ^ 60)
    println("(Lower entropy = more uniform target = easier for fp picker)")
    println()

    H_vals = [0.0, 1.0]
    nstate = 45; alpha_cool = 0.9999; nsteps = 300_000; n_seeds = 8

    for H in H_vals
        k_ones = h_to_kones(H, ngbits)
        ents = Float64[]
        for gseed in 1:n_seeds
            rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal     = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                       zeros(Int, ngbits - k_ones)))
            _, _, _, _, rep =
                run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master;
                            seed=gseed * 31 + 7)
            ent, _ = companion_goal_entropy(goal, rep, fingerprint, cxt_master,
                                             companion, goal_idx, nqubit;
                                             n_probe=300, seed=gseed*997)
            push!(ents, ent)
        end
        @printf("H=%.1f:  mean entropy = %.4f ± %.4f\n",
                H, mean(ents), std(ents)/sqrt(n_seeds))
    end

    println()
    println("=" ^ 60)
    println("4. N_VALID: mean non-NaN companion_goal positions per context")
    println("   and actual minimum fp score achieved")
    println("=" ^ 60)
    println("(If n_valid differs, pick_fp will differ even with identical structure)")
    println()

    for H in H_vals
        k_ones = h_to_kones(H, ngbits)
        n_valids    = Float64[]
        min_fp_acts = Float64[]
        for gseed in 1:n_seeds
            rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal     = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                       zeros(Int, ngbits - k_ones)))
            _, _, _, _, rep =
                run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master;
                            seed=gseed * 31 + 7)

            rng = Random.MersenneTwister(gseed * 997 + 1)
            for _ in 1:200
                gen = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
                ts  = rand(rng, 0:1)
                bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
                cxt = QCScaling.Context(gen, bc)
                cg  = [companion_goal_s(po, goal, rep, companion, goal_idx, n-1)
                       for po in cxt.pos]
                n_v = count(!isnan, cg)
                push!(n_valids, n_v)

                # Also measure actual minimum fp score
                valid, vals = QCScaling._pack_companion_goal(cg, fingerprint.nwords)
                pi = bc.parity + 1
                min_s = typemax(Int)
                @inbounds for ai in 1:size(fingerprint.words,4),
                              tzi in 1:size(fingerprint.words,3)
                    s = 0
                    for w in 1:fingerprint.nwords
                        s += count_ones(xor(fingerprint.words[w,pi,tzi,ai], vals[w]) & valid[w])
                    end
                    s < min_s && (min_s = s)
                end
                push!(min_fp_acts, min_s)
            end
        end
        @printf("H=%.1f:  n_valid = %.2f ± %.2f   actual_min_fp = %.2f ± %.2f\n",
                H,
                mean(n_valids), std(n_valids)/sqrt(length(n_valids)),
                mean(min_fp_acts), std(min_fp_acts)/sqrt(length(min_fp_acts)))
    end

    println()
    println("Interpretation:")
    println("  Entropy=0: all positions in context want same vote (uniform, easy)")
    println("  Entropy=1: exactly half want 0, half want 1  (maximally mixed, hard)")
    println("  If n_valid(H=0) < n_valid(H=1), H=0 has lower minimum fp score by construction.")
end

main()
