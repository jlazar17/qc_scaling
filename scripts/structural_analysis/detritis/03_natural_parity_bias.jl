# Investigate whether PseudoGHZ states have a "natural bias" toward correlated
# pair parities that advantages H=0 over H=1 independently of the both-covered
# pair mechanism.
#
# Hypothesis: for a random ensemble (no optimization), the rep will tend to
# have rep[k1] = rep[k2] for most pairs, simply because states tend to give
# parity 0 or parity 1 uniformly at all covered positions (via theta_z).
# H=0 exploits this natural correlation; H=1 must fight it.
#
# This would explain why eliminating both-covered pairs (shuffled pairing)
# still leaves a large H=0 vs H=1 gap.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Accuracy for a random ensemble (no optimization)
# ---------------------------------------------------------------------------

function random_ensemble_accuracy(nqubit, goal, nstate; seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit
    npos = length(cxt_master.base_even.pos)

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for s in ensemble
        apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
    end
    return rep_accuracy_fast(rep_sum, rep_ctr, goal)
end

# ---------------------------------------------------------------------------
# Measure the "natural correlation" of the rep:
# For a random ensemble, what fraction of pairs have rep[k1] = rep[k2]?
# If >0.5, states are naturally correlated (favoring H=0).
# ---------------------------------------------------------------------------

function natural_correlation(nqubit, nstate, n_trials; seed=42)
    rng = Random.MersenneTwister(seed)
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    cxt_master = QCScaling.ContextMaster(nqubit)

    corr_fracs = Float64[]
    for trial in 1:n_trials
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
        for s in ensemble
            apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
        end
        rep = rep_from_cache(rep_sum, rep_ctr)

        n_defined = 0; n_corr = 0
        for j in 1:ngbits
            r1 = rep[2j-1]; r2 = rep[2j]
            (isnan(r1) || isnan(r2)) && continue
            n_defined += 1
            r1 == r2 && (n_corr += 1)
        end
        n_defined > 0 && push!(corr_fracs, n_corr / n_defined)
    end
    return corr_fracs
end

# ---------------------------------------------------------------------------
# For a single state: measure the fraction of covered pairs that are
# "naturally" same-parity (XOR=0) vs anti-parity (XOR=1)
# This tests whether individual states bias toward correlation
# ---------------------------------------------------------------------------

function single_state_parity_correlation(nqubit, n_sample; seed=42)
    rng = Random.MersenneTwister(seed)
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    cxt_master = QCScaling.ContextMaster(nqubit)

    xor0_counts = Int[]  # both-covered pairs with XOR=0
    xor1_counts = Int[]  # both-covered pairs with XOR=1

    for _ in 1:n_sample
        gen_idx = rand(rng, 0:n-1)
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s = rand(rng, 0:1)
        theta_z = rand(rng, 0:1)
        nalpha = 2^(nqubit - 1)
        alpha_idx = rand(rng, 0:nalpha-1)
        alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
        state = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, gen)

        base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        ctx_parities = Dict{Int, Int}()
        for base_po in base_cxt.pos
            derived_po = gen + base_po
            p = QCScaling.parity(state, derived_po)
            p == 0.5 && continue
            ctx_parities[derived_po.index] = round(Int, p)
        end

        n0 = 0; n1 = 0
        for j in 1:ngbits
            k1 = 2j-1; k2 = 2j
            p1 = get(ctx_parities, k1, -1)
            p2 = get(ctx_parities, k2, -1)
            (p1 == -1 || p2 == -1) && continue
            if p1 == p2
                n0 += 1
            else
                n1 += 1
            end
        end
        push!(xor0_counts, n0)
        push!(xor1_counts, n1)
    end
    return xor0_counts, xor1_counts
end

# ---------------------------------------------------------------------------
# Test: does optimizing for H=0 vs H=1 change the natural correlation?
# Compare rep correlation after SA for H=0 vs H=1 and no optimization
# ---------------------------------------------------------------------------

function rep_correlation_after_sa(nqubit, goal, nstate, nsteps; seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    npos = length(cxt_master.base_even.pos)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)
    T = 0.1; alpha_cool = 0.9999
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)

    for step in 1:nsteps
        which = rand(rng, 1:nstate)
        gen_idx = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s = rand(rng, 0:1)
        base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(generator, base_cxt)
        alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
        ns = QCScaling.PseudoGHZState(alphas..., generator)

        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta = new_acc - current_acc

        if delta >= 0 || rand(rng) < exp(delta / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns
            current_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end

    # Compute correlation of the final rep
    n_defined = 0; n_corr = 0
    for j in 1:ngbits
        r1 = rep[2j-1]; r2 = rep[2j]
        (isnan(r1) || isnan(r2)) && continue
        n_defined += 1
        r1 == r2 && (n_corr += 1)
    end
    corr = n_defined > 0 ? n_corr / n_defined : NaN

    return current_acc, corr
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit = 6
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2

    @printf("nqubit=%d, n=%d, ngbits=%d\n\n", nqubit, n, ngbits)

    # ---------------------------------------------------------------------------
    # Part 1: Natural correlation in a random ensemble
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 1: Rep correlation in a RANDOM (unoptimized) ensemble")
    println("Expect: if states naturally give same parity at paired positions,")
    println("        corr > 0.5 even for random ensemble")
    println("="^65)
    println()

    for nstate in [10, 30, 100]
        corr_fracs = natural_correlation(nqubit, nstate, 100; seed=1)
        @printf("nstate=%3d: mean_corr=%.4f  std=%.4f  (corr=0.5 means no bias)\n",
                nstate, mean(corr_fracs), std(corr_fracs))
    end
    println()

    # ---------------------------------------------------------------------------
    # Part 2: Single-state parity correlation
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 2: Single-state parity correlation for both-covered pairs")
    println("Over random states, fraction of both-covered pairs with XOR=0 vs XOR=1")
    println("="^65)
    println()

    xor0, xor1 = single_state_parity_correlation(nqubit, 10000; seed=2)
    total = xor0 .+ xor1
    nonempty = total .> 0
    frac_xor0 = xor0[nonempty] ./ total[nonempty]
    @printf("Among states with ≥1 both-covered pair:\n")
    @printf("  Fraction with XOR=0: mean=%.4f  std=%.4f\n",
            mean(frac_xor0), std(frac_xor0))
    @printf("  (>0.5 means states are biased toward correlated pairs)\n")
    println()

    # ---------------------------------------------------------------------------
    # Part 3: Random ensemble accuracy for H=0 vs H=1
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 3: Accuracy of RANDOM ensemble for H=0 vs H=1")
    println("If H=0 >> H=1 for random ensemble: structural bias proven")
    println("="^65)
    println()

    rng_goals = Random.MersenneTwister(99)
    n_trials = 20

    for H_target in [0.0, 0.5, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        accs = Float64[]
        for trial in 1:n_trials
            goal = Random.shuffle!(rng_goals, vcat(ones(Int, goal_ones), zeros(Int, ngbits-goal_ones)))
            for nstate in [20, 50]
                acc = random_ensemble_accuracy(nqubit, goal, nstate; seed=trial*100)
                push!(accs, acc)
            end
        end
        @printf("H=%.2f: random ensemble accuracy: mean=%.4f  max=%.4f  min=%.4f\n",
                H_target, mean(accs), maximum(accs), minimum(accs))
    end
    println()

    # ---------------------------------------------------------------------------
    # Part 4: Rep correlation after SA for H=0 vs H=1
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 4: Rep correlation after SA optimization")
    println("H=0 rep should have high correlation (paired positions agree)")
    println("H=1 rep should have low correlation (paired positions anti-correlate)")
    println("This shows HOW DIFFERENT the SA solution structure is for H=0 vs H=1")
    println("="^65)
    println()

    rng_goals2 = Random.MersenneTwister(55)
    nstate = 50; nsteps = 30_000

    for H_target in [0.0, 0.5, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        accs = Float64[]; corrs = Float64[]
        for trial in 1:5
            goal = Random.shuffle!(rng_goals2, vcat(ones(Int, goal_ones), zeros(Int, ngbits-goal_ones)))
            acc, corr = rep_correlation_after_sa(nqubit, goal, nstate, nsteps; seed=trial)
            push!(accs, acc); push!(corrs, corr)
        end
        @printf("H=%.2f: acc=%.4f  rep_corr=%.4f\n",
                H_target, mean(accs), mean(corrs))
    end
    println()

    # ---------------------------------------------------------------------------
    # Part 5: What fraction of CONTEXTS have more both-covered pairs of one type?
    # For H=1 goal with both-covered pairs: measure the XOR distribution
    # required to satisfy them. Does the goal structure create "split" contexts?
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 5: Context-level XOR requirement distribution for H=1")
    println("For each context, count how many both-covered pairs need XOR=0 vs XOR=1")
    println("A 50/50 split means the state can't help either way (net zero)")
    println("="^65)
    println()

    goal_ones = ngbits ÷ 2
    rng3 = Random.MersenneTwister(7)
    goal = Random.shuffle!(rng3, vcat(ones(Int, goal_ones), zeros(Int, ngbits-goal_ones)))

    # Sample a batch of contexts and measure XOR distribution
    n_contexts = 500
    xor_fracs = Float64[]  # fraction of both-covered pairs needing XOR=1 per context

    cxt_master = QCScaling.ContextMaster(nqubit)
    for _ in 1:n_contexts
        gen_idx = rand(rng3, 0:n-1)
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s = rand(rng3, 0:1)
        base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

        ctx_idxs = Set{Int}()
        for base_po in base_cxt.pos
            derived_po = gen + base_po
            push!(ctx_idxs, derived_po.index)
        end

        n_need_0 = 0; n_need_1 = 0
        for j in 1:ngbits
            k1 = 2j-1; k2 = 2j
            (k1 in ctx_idxs && k2 in ctx_idxs) || continue
            if goal[j] == 0
                n_need_0 += 1
            else
                n_need_1 += 1
            end
        end
        n_total = n_need_0 + n_need_1
        n_total == 0 && continue
        push!(xor_fracs, n_need_1 / n_total)
    end

    println("Contexts with ≥1 both-covered pair:")
    @printf("  Count: %d / %d sampled\n", length(xor_fracs), n_contexts)
    if !isempty(xor_fracs)
        @printf("  Fraction of both-covered pairs needing XOR=1:\n")
        @printf("    mean=%.4f  std=%.4f  (0.5 = perfectly split)\n",
                mean(xor_fracs), std(xor_fracs))
        pct_balanced = count(0.3 .< xor_fracs .< 0.7) / length(xor_fracs)
        @printf("    Fraction of contexts with split [0.3, 0.7]: %.3f\n", pct_balanced)

        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        println("  Histogram of XOR=1 fraction per context:")
        for i in 1:length(bins)-1
            cnt = count(bins[i] .<= xor_fracs .< bins[i+1])
            @printf("    [%.1f, %.1f): %3d  (%.1f%%)\n",
                    bins[i], bins[i+1], cnt, 100*cnt/length(xor_fracs))
        end
    end
    println()

    # For H=0 (all goal=0), all both-covered pairs need XOR=0 → no split
    goal_H0 = zeros(Int, ngbits)
    xor_fracs_H0 = Float64[]
    rng4 = Random.MersenneTwister(7)
    for _ in 1:n_contexts
        gen_idx = rand(rng4, 0:n-1)
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s = rand(rng4, 0:1)
        base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        ctx_idxs = Set{Int}()
        for base_po in base_cxt.pos
            derived_po = gen + base_po
            push!(ctx_idxs, derived_po.index)
        end
        n_need_1 = 0; n_total = 0
        for j in 1:ngbits
            k1 = 2j-1; k2 = 2j
            (k1 in ctx_idxs && k2 in ctx_idxs) || continue
            n_total += 1
            goal_H0[j] == 1 && (n_need_1 += 1)
        end
        n_total == 0 && continue
        push!(xor_fracs_H0, n_need_1 / n_total)
    end
    @printf("H=0 (all zeros): fraction needing XOR=1: mean=%.4f (expected=0.0)\n",
            isempty(xor_fracs_H0) ? NaN : mean(xor_fracs_H0))
end

main()
