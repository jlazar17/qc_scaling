# Analyze the "both-covered pair efficiency" for H=0 vs H=1 at nqubit=6.
#
# For each state, some goal pairs are "both-covered" — both members (k1, k2)
# appear in the state's context. For those pairs, the XOR of parities is fixed
# by (alphas, theta_s): it cannot be changed by theta_z.
#
# Claim: all both-covered pairs within the same context share the SAME XOR.
# If true:
#   H=0 (all XOR=0): one alpha choice satisfies ALL both-covered pairs
#   H=1 (half XOR=0, half XOR=1): impossible to satisfy all -> max ~50%
# This would explain the ~2x efficiency ratio in scaling studies.
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

# ---------------------------------------------------------------------------
# For a given state, find all both-covered goal pairs and their XOR values
# ---------------------------------------------------------------------------

function analyze_both_covered(state, nqubit, goal)
    cxt_master = QCScaling.ContextMaster(nqubit)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    ngbits = length(goal)

    # Compute parity at all context positions
    context_parities = Dict{Int, Float64}()
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        context_parities[derived_po.index] = p
    end

    # Find both-covered pairs
    n_both = 0
    n_helped = 0   # both-covered pairs where XOR matches goal
    n_hurt  = 0   # both-covered pairs where XOR mismatches goal
    xors = Int[]

    for j in 1:ngbits
        k1 = 2j - 1; k2 = 2j
        p1 = get(context_parities, k1, 0.5)
        p2 = get(context_parities, k2, 0.5)
        (p1 == 0.5 || p2 == 0.5) && continue  # not both covered with definite parity
        n_both += 1
        xor_val = round(Int, abs(p1 - p2))
        push!(xors, xor_val)
        if xor_val == goal[j]
            n_helped += 1
        else
            n_hurt += 1
        end
    end

    return n_both, n_helped, n_hurt, xors
end

"""
Check whether all both-covered pairs in a context share the same XOR value.
Returns (all_same, xor_value) where xor_value is -1 if n_both==0.
"""
function check_xor_uniformity(state, nqubit, ngbits)
    cxt_master = QCScaling.ContextMaster(nqubit)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    context_parities = Dict{Int, Float64}()
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        context_parities[derived_po.index] = p
    end

    xors = Int[]
    for j in 1:ngbits
        k1 = 2j-1; k2 = 2j
        p1 = get(context_parities, k1, 0.5)
        p2 = get(context_parities, k2, 0.5)
        (p1 == 0.5 || p2 == 0.5) && continue
        push!(xors, round(Int, abs(p1 - p2)))
    end

    isempty(xors) && return true, -1
    return all(==(xors[1]), xors), xors[1]
end

"""
For a given context (generator + theta_s), over all alpha settings,
what XOR values are achievable for the both-covered pairs?
Returns: (set of achievable XOR patterns) — is it always uniform?
"""
function xor_achievable_patterns(generator, theta_s, nqubit, ngbits)
    nalpha = 2^(nqubit - 1)
    cxt_master = QCScaling.ContextMaster(nqubit)
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    # Find both-covered pairs for this context (independent of alpha)
    zero_state = QCScaling.PseudoGHZState(theta_s, 0, QCScaling.idx_to_alphas(0, nqubit), generator)
    context_idxs = Set{Int}()
    for base_po in base_cxt.pos
        derived_po = generator + base_po
        push!(context_idxs, derived_po.index)
    end
    both_pairs = [j for j in 1:ngbits
                  if (2j-1) in context_idxs && (2j) in context_idxs]

    isempty(both_pairs) && return Dict{Vector{Int}, Int}(), both_pairs

    # Over all alpha settings, what XOR patterns appear?
    xor_pattern_counts = Dict{Vector{Int}, Int}()
    for alpha_idx in 0:nalpha-1
        for theta_z in 0:1
            alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
            state = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator)

            pattern = Int[]
            for j in both_pairs
                k1 = 2j-1; k2 = 2j
                p1 = QCScaling.parity(state, QCScaling.ParityOperator(k1-1, nqubit))
                p2 = QCScaling.parity(state, QCScaling.ParityOperator(k2-1, nqubit))
                xor_val = (p1 == 0.5 || p2 == 0.5) ? -1 : round(Int, abs(p1 - p2))
                push!(pattern, xor_val)
            end
            xor_pattern_counts[pattern] = get(xor_pattern_counts, pattern, 0) + 1
        end
    end
    return xor_pattern_counts, both_pairs
end

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

function main()
    nqubit = 6
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    n_sample = 5000  # number of random states to sample for efficiency analysis

    @printf("nqubit=%d, n=%d, ngbits=%d\n\n", nqubit, n, ngbits)

    rng = Random.MersenneTwister(42)

    # ---------------------------------------------------------------------------
    # Part 1: Verify XOR uniformity claim
    # For a random sample of states: are all both-covered pairs always same XOR?
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 1: XOR uniformity — do all both-covered pairs in a context share XOR?")
    println("="^65)

    n_uniform = 0; n_nonuniform = 0; n_zero_both = 0
    for _ in 1:n_sample
        state = random_state(nqubit, rng)
        all_same, _ = check_xor_uniformity(state, nqubit, ngbits)
        if all_same
            n_uniform += 1
        else
            n_nonuniform += 1
        end
    end
    @printf("Uniform XOR across all both-covered pairs: %d / %d (%.1f%%)\n",
            n_uniform, n_sample, 100*n_uniform/n_sample)
    @printf("Non-uniform (would BREAK the structural argument): %d / %d\n",
            n_nonuniform, n_sample)

    # Also check: for contexts WITH both-covered pairs, uniformity
    n_uniform2 = 0; n_with_both = 0
    for _ in 1:n_sample
        state = random_state(nqubit, rng)
        nb, _, _, xors = analyze_both_covered(state, nqubit, ones(Int, ngbits))
        nb == 0 && continue
        n_with_both += 1
        all(==(xors[1]), xors) && (n_uniform2 += 1)
    end
    @printf("Among states with ≥1 both-covered pair: uniform=%d/%d (%.1f%%)\n\n",
            n_uniform2, n_with_both, 100*n_uniform2/n_with_both)

    # ---------------------------------------------------------------------------
    # Part 2: XOR achievability per context
    # For a small set of contexts, check whether BOTH XOR=0 and XOR=1 are
    # achievable, and whether the pattern is always (0,0,...,0) or (1,1,...,1)
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 2: XOR pattern space over all alpha settings")
    println("="^65)

    n_ctx_checked = 0
    always_uniform = true
    both_xors_achievable = 0
    only_xor0 = 0; only_xor1 = 0
    total_with_both = 0

    # Sample contexts
    for _ in 1:200
        gen_idx  = rand(rng, 0:n-1)
        theta_s  = rand(rng, 0:1)
        gen      = QCScaling.ParityOperator(gen_idx, nqubit)
        patterns, both_pairs = xor_achievable_patterns(gen, theta_s, nqubit, ngbits)
        isempty(both_pairs) && continue
        total_with_both += 1
        n_ctx_checked += 1

        # Check: is every pattern uniform (all same XOR)?
        for (pat, _) in patterns
            if !all(==(pat[1]), pat) || pat[1] == -1
                always_uniform = false
            end
        end

        # Extract unique XOR values achievable across all alpha/theta_z
        achievable_xors = Set{Int}()
        for (pat, _) in patterns
            pat[1] == -1 && continue
            push!(achievable_xors, pat[1])
        end
        if 0 in achievable_xors && 1 in achievable_xors
            both_xors_achievable += 1
        elseif 0 in achievable_xors
            only_xor0 += 1
        elseif 1 in achievable_xors
            only_xor1 += 1
        end
    end

    @printf("Contexts with ≥1 both-covered pair checked: %d\n", n_ctx_checked)
    @printf("All XOR patterns uniform (all-0 or all-1): %s\n", always_uniform ? "YES ✓" : "NO ✗")
    @printf("Contexts where BOTH XOR=0 and XOR=1 achievable: %d / %d\n",
            both_xors_achievable, n_ctx_checked)
    @printf("Contexts where ONLY XOR=0: %d  |  ONLY XOR=1: %d\n\n",
            only_xor0, only_xor1)

    # ---------------------------------------------------------------------------
    # Part 3: Per-state both-covered efficiency for H=0 vs H=1 goals
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 3: Both-covered efficiency for H=0 vs H=1")
    println("="^65)
    println("Efficiency = fraction of both-covered pairs that a state HELPS")
    println()

    goal_rng = Random.MersenneTwister(13)
    n_trials = 10

    for H_target in [0.0, 0.5, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        efficiencies = Float64[]
        n_both_counts = Int[]

        for trial in 1:n_trials
            goal = shuffle!(goal_rng, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

            # Over a large sample of states, compute both-covered efficiency
            trial_effs = Float64[]
            for _ in 1:n_sample
                state = random_state(nqubit, rng)
                nb, nh, nhurt, _ = analyze_both_covered(state, nqubit, goal)
                nb == 0 && continue
                push!(trial_effs, nh / nb)
                push!(n_both_counts, nb)
            end
            !isempty(trial_effs) && push!(efficiencies, mean(trial_effs))
        end

        @printf("H=%.2f (ones=%d): mean_efficiency=%.4f  std=%.4f  (mean n_both=%.1f)\n",
                H_target, goal_ones,
                mean(efficiencies), std(efficiencies),
                mean(n_both_counts))
    end

    println()

    # ---------------------------------------------------------------------------
    # Part 4: Net contribution — states help - hurt
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 4: Net contribution per state (helped_pairs - hurt_pairs)")
    println("  Positive = state pushes ensemble toward goal")
    println("  Zero = state is useless for goal")
    println("  Negative = state actively hurts")
    println("="^65)
    println()

    goal_rng2 = Random.MersenneTwister(77)
    for H_target in [0.0, 0.5, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        goal = shuffle!(goal_rng2, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

        # Sample states, but CHOOSE the best XOR for each
        # (simulate the picker choosing the best alpha)
        net_best_dist = Float64[]
        net_rand_dist = Float64[]

        for _ in 1:n_sample
            gen_idx = rand(rng, 0:n-1)
            gen = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s = rand(rng, 0:1)
            nalpha = 2^(nqubit - 1)

            best_net = -Inf
            rand_nb = 0; rand_nh = 0

            for alpha_idx in 0:nalpha-1
                for theta_z in 0:1
                    alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                    state = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, gen)
                    nb, nh, nhurt, _ = analyze_both_covered(state, nqubit, goal)
                    nb == 0 && continue
                    net = (nh - nhurt) / nb
                    best_net = max(best_net, net)
                    # Pick one random alpha to compare
                    if alpha_idx == 0 && theta_z == 0
                        rand_nb = nb; rand_nh = nh
                    end
                end
            end
            best_net > -Inf && push!(net_best_dist, best_net)
            rand_nb > 0 && push!(net_rand_dist, (rand_nh - (rand_nb - rand_nh)) / rand_nb)
        end

        @printf("H=%.2f: best_net: mean=%.4f, median=%.4f\n",
                H_target, mean(net_best_dist), median(net_best_dist))
        @printf("         rand_net: mean=%.4f, median=%.4f\n",
                mean(net_rand_dist), median(net_rand_dist))
        println()
    end

    # ---------------------------------------------------------------------------
    # Part 5: Verify independence — for the shuffled pairing, is there still an
    # asymmetry between H=0 and H=1? Check by computing "single-covered" pair
    # efficiency.
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Part 5: Can independent coverage resolve the gap?")
    println("  For each pair (k1,k2), what fraction of states give:")
    println("  - definite parity 0 at k1 only (singly-covered)")
    println("  - definite parity 1 at k1 only (singly-covered)")
    println("="^65)
    println()

    # For each position k, what fraction of random states cover k with parity 0 vs 1?
    n_pos_sample = 20
    sample_positions = rand(rng, 1:2*ngbits, n_pos_sample)
    println("Sampled positions: coverage and parity distribution:")
    @printf("  %-6s  %-10s  %-10s  %-10s\n", "pos", "frac_cov", "frac_p0|cov", "frac_p1|cov")
    for k in sample_positions[1:10]
        n_cov = 0; n_p0 = 0; n_p1 = 0
        for _ in 1:n_sample
            state = random_state(nqubit, rng)
            p = QCScaling.parity(state, QCScaling.ParityOperator(k-1, nqubit))
            if p != 0.5
                n_cov += 1
                p == 0.0 ? (n_p0 += 1) : (n_p1 += 1)
            end
        end
        @printf("  %-6d  %-10.4f  %-10.4f  %-10.4f\n",
                k, n_cov/n_sample,
                n_cov > 0 ? n_p0/n_cov : NaN,
                n_cov > 0 ? n_p1/n_cov : NaN)
    end
    println()
    println("Note: if frac_p0 ≈ frac_p1 ≈ 0.5, parity is equiprobable → independent")
    println("control over each position. H=0 and H=1 should be equally achievable.")
end

function random_state(nqubit, rng)
    n = 3^nqubit
    gen_idx = rand(rng, 0:n-1)
    gen = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s = rand(rng, 0:1)
    theta_z = rand(rng, 0:1)
    nalpha = 2^(nqubit - 1)
    alpha_idx = rand(rng, 0:nalpha-1)
    alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
    return QCScaling.PseudoGHZState(theta_s, theta_z, alphas, gen)
end

main()
