# Goal structure experiment: does algebraic alignment with PseudoGHZ contexts
# determine achievable accuracy, independent of entropy?
#
# Three H≈1 goal constructions:
#
#   (A) Random:      k_ones = ngbits/2 shuffled randomly. No relation to
#                    context algebra.
#
#   (B) Structured:  Derive goal from the majority parity of a large pool of
#                    random PseudoGHZ states. The parity assignment is
#                    algebraically grounded — it IS achievable at near-perfect
#                    accuracy by that pool.
#
#   (C) Anti-structured: Start from the structured parity assignment and flip
#                    the absolute parity at the position in each pair that
#                    appears most often across contexts. This maximises
#                    context-constraint violations while keeping H≈1.
#
# If (B) >> (A) >> (C) in achieved accuracy: frustration hypothesis confirmed.
# If all three are similar: entropy alone drives difficulty.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using StaticArrays
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Binary entropy
# ---------------------------------------------------------------------------
function goal_entropy(goal)
    p = mean(goal)
    (p == 0.0 || p == 1.0) && return 0.0
    return -p * log2(p) - (1-p) * log2(1-p)
end

# ---------------------------------------------------------------------------
# Construct a structured H≈1 goal from a large PseudoGHZ pool.
# Returns (goal, pool_parity) where pool_parity[k] ∈ {0,1} is the majority
# parity of the pool at position k.
# ---------------------------------------------------------------------------
function make_structured_goal(nqubit, npool, companion, goal_idx, cxt_master; seed=1)
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2
    rng    = Random.MersenneTwister(seed)

    pool    = [QCScaling.random_state(nqubit) for _ in 1:npool]
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for s in pool
        apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
    end

    # Majority parity: 1 if more votes for 1, else 0.  Ties → 0.
    par = zeros(Int, n)
    for k in 1:n-1
        rep_ctr[k] > 0 && (par[k] = 2*rep_sum[k] >= rep_ctr[k] ? 1 : 0)
    end

    goal = zeros(Int, ngbits)
    for k1 in 1:n-1
        companion[k1] == 0 && continue
        k2 = companion[k1]
        j  = goal_idx[k1];  j == 0 && continue
        goal[j] = par[k1] ⊻ par[k2]
    end
    return goal, par
end

# ---------------------------------------------------------------------------
# Construct an anti-structured goal: take the structured parity assignment
# and flip the parity of whichever position in each pair has higher context
# coverage (appears in more states' contexts in a reference pool).
# Flipping the high-coverage position disrupts the most context constraints.
# ---------------------------------------------------------------------------
function make_antistructured_goal(nqubit, par, companion, goal_idx, cxt_master; seed=2)
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2

    # Estimate per-position coverage frequency with a reference pool.
    # Use a fixed seed so results are reproducible.
    nref        = 500
    ref_sum     = zeros(Int, n)
    ref_ctr     = zeros(Int, n)
    rng_ref     = Random.MersenneTwister(seed * 1000 + 7)
    for _ in 1:nref
        ts  = rand(rng_ref, 0:1)
        tz  = rand(rng_ref, 0:1)
        alp = rand(rng_ref, 0:1, nqubit-1)
        gen = QCScaling.ParityOperator(rand(rng_ref, 0:n-1), nqubit)
        s   = QCScaling.PseudoGHZState(ts, tz,
                  StaticArrays.SVector{nqubit-1, Int}(alp), gen)
        apply_state!(ref_sum, ref_ctr, s, cxt_master, 1)
    end

    # For each pair, flip the parity of whichever position has higher coverage.
    # This disrupts as many context constraints as possible.
    new_par = copy(par)
    for k1 in 1:n-1
        companion[k1] == 0 && continue
        k2 = companion[k1]
        goal_idx[k1] == 0 && continue
        flip_k = ref_ctr[k1] >= ref_ctr[k2] ? k1 : k2
        new_par[flip_k] = 1 - new_par[flip_k]
    end

    goal = zeros(Int, ngbits)
    for k1 in 1:n-1
        companion[k1] == 0 && continue
        k2 = companion[k1]
        j  = goal_idx[k1];  j == 0 && continue
        goal[j] = new_par[k1] ⊻ new_par[k2]
    end
    return goal, new_par
end

# ---------------------------------------------------------------------------
# SA returning final accuracy (standard run, no special logging)
# ---------------------------------------------------------------------------
function run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                companion, goal_idx, fingerprint, cxt_master; seed=42)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    rng  = Random.MersenneTwister(seed)

    ensemble   = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars  = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    acc_fn(rs, rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    scratch_idxs = Vector{Int}(undef, npos); scratch_pars = Vector{Int}(undef, npos)

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        d = acc_fn(rep_sum, rep_ctr) - cur_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        d < 0 && push!(bad_deltas, abs(d))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    for _ in 1:nsteps
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc
        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs); copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 8
    sa_seeds   = [42, 17, 99, 256, 1337, 7, 314, 888]

    # Pool size for structured goal: large enough for good coverage
    npool = 10 * nstate

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    # --- Build goals ---
    println("Building goals...")

    # (A) Random H=1 (3 different random goals, averaged)
    random_goals = Vector{Vector{Int}}()
    for gseed in [99, 42, 17]
        rng  = Random.MersenneTwister(gseed)
        k_ones = ngbits ÷ 2
        g = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))
        push!(random_goals, g)
    end

    # (B) Structured H≈1 (3 different pool seeds)
    structured_goals = Vector{Vector{Int}}()
    structured_pars  = Vector{Vector{Int}}()
    for pseed in [1, 2, 3]
        g, par = make_structured_goal(nqubit, npool, companion, goal_idx, cxt_master; seed=pseed)
        push!(structured_goals, g)
        push!(structured_pars, par)
    end

    # (C) Anti-structured (derived from structured parity, flipped at high-coverage positions)
    anti_goals = Vector{Vector{Int}}()
    for (pseed, par) in zip([1, 2, 3], structured_pars)
        g, _ = make_antistructured_goal(nqubit, par, companion, goal_idx, cxt_master; seed=pseed+10)
        push!(anti_goals, g)
    end

    # Report entropies
    println("Goal entropies:")
    for (i, g) in enumerate(random_goals)
        @printf("  random[%d]:       H=%.4f  (k_ones=%d / %d)\n",
                i, goal_entropy(g), sum(g), ngbits)
    end
    for (i, g) in enumerate(structured_goals)
        @printf("  structured[%d]:   H=%.4f  (k_ones=%d / %d)\n",
                i, goal_entropy(g), sum(g), ngbits)
    end
    for (i, g) in enumerate(anti_goals)
        @printf("  anti[%d]:         H=%.4f  (k_ones=%d / %d)\n",
                i, goal_entropy(g), sum(g), ngbits)
    end
    println()

    # --- Run SA ---
    function run_goals(goals, label)
        all_accs = Float64[]
        for (gi, goal) in enumerate(goals)
            accs = Float64[]
            for seed in sa_seeds
                acc = run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                             companion, goal_idx, fingerprint, cxt_master; seed=seed)
                push!(accs, acc)
            end
            push!(all_accs, accs...)
            @printf("  %s[%d]: mean=%.4f  median=%.4f  min=%.4f  max=%.4f\n",
                    label, gi, mean(accs), median(accs), minimum(accs), maximum(accs))
        end
        @printf("  %s OVERALL: mean=%.4f  std=%.4f\n\n",
                label, mean(all_accs), std(all_accs))
        return all_accs
    end

    println("=" ^ 55)
    println("SA results (nstate=$nstate, nsteps=$nsteps, $(n_seeds) seeds each)")
    println("=" ^ 55)

    println("\n--- H=0 baseline ---")
    rng0 = Random.MersenneTwister(99)
    goal_h0 = zeros(Int, ngbits)
    accs_h0 = Float64[]
    for seed in sa_seeds
        acc = run_sa(goal_h0, nqubit, nstate, nsteps, alpha_cool,
                     companion, goal_idx, fingerprint, cxt_master; seed=seed)
        push!(accs_h0, acc)
    end
    @printf("  H=0: mean=%.4f  median=%.4f  min=%.4f  max=%.4f\n\n",
            mean(accs_h0), median(accs_h0), minimum(accs_h0), maximum(accs_h0))

    println("--- (A) Random H=1 ---")
    accs_rand = run_goals(random_goals, "random")

    println("--- (B) Structured H≈1 (derived from PseudoGHZ pool) ---")
    accs_struct = run_goals(structured_goals, "structured")

    println("--- (C) Anti-structured H≈1 (flipped at high-coverage positions) ---")
    accs_anti = run_goals(anti_goals, "anti")

    println("=" ^ 55)
    println("SUMMARY")
    println("=" ^ 55)
    @printf("  H=0:          mean=%.4f\n", mean(accs_h0))
    @printf("  random H≈1:   mean=%.4f  (delta from H=0: %.4f)\n",
            mean(accs_rand), mean(accs_rand) - mean(accs_h0))
    @printf("  structured:   mean=%.4f  (delta from random: %.4f)\n",
            mean(accs_struct), mean(accs_struct) - mean(accs_rand))
    @printf("  anti-struct:  mean=%.4f  (delta from random: %.4f)\n",
            mean(accs_anti), mean(accs_anti) - mean(accs_rand))
end

main()
