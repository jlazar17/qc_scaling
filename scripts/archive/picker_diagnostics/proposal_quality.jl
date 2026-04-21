using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal / rep helpers (same as other scripts)
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N; return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0; H >= 1.0 && return N ÷ 2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), 0:N÷2)
    return (0:N÷2)[idx]
end

function apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]
        rep[i] = c == 0 ? NaN : (v = rep_sum[i]/c; v == 0.5 ? NaN : round(v))
    end
end

# ---------------------------------------------------------------------------
# Measure proposal quality:
# For each proposed state, compute:
#   pair_acc_prop: fraction of covered pairs where proposed state's XOR matches goal
#   pair_acc_cg:   fraction of covered pairs where companion_goal target XOR matches goal
#   delta_acc:     change in global accuracy if proposal were accepted
# ---------------------------------------------------------------------------

function measure_proposal_quality(goal, nqubit, nstate, nsteps; seed=42)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n)
    rep_ctr  = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    # Stores per-step diagnostics
    prop_pair_acc  = Float64[]   # fraction of covered pairs where state XOR = goal
    cg_pair_acc    = Float64[]   # fraction of covered pairs where cg XOR = goal
    global_deltas  = Float64[]   # Δ global accuracy from proposal

    for step in 1:nsteps
        which = rand(rng, 1:nstate)

        # Generate smart proposal
        gen_idx   = rand(rng, 0:3^nqubit-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)

        # Compute companion goals BEFORE picking alphas (to audit what picker sees)
        cg = QCScaling.companion_goal(cxt, goal, rep)

        alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns = QCScaling.PseudoGHZState(alphas..., generator)

        # Evaluate proposed state's pair coverage and XOR alignment
        covered_pairs = Dict{Int, Vector{Float64}}()
        for base_po in base_cxt.pos
            derived_po = ns.generator + base_po
            k = derived_po.index
            k == n && continue  # unpaired last position
            j = (k + 1) ÷ 2    # 1-indexed pair
            p_val = QCScaling.parity(ns, derived_po)
            p_val == 0.5 && continue
            if !haskey(covered_pairs, j)
                covered_pairs[j] = Float64[]
            end
            push!(covered_pairs[j], p_val)
        end

        n_both = 0; n_correct = 0
        n_cg_both = 0; n_cg_correct = 0

        # Evaluate actual XOR of proposed state at each covered pair
        for (j, parities) in covered_pairs
            length(parities) == 2 || continue  # need both positions
            n_both += 1
            xor_state = parities[1] != parities[2] ? 1 : 0
            n_correct += (xor_state == goal[j])
        end

        # Evaluate XOR implied by companion_goal targets
        # cg is indexed over base_cxt.pos; we need to track pair positions
        cg_pairs = Dict{Int, Vector{Float64}}()
        for (pi, base_po) in enumerate(base_cxt.pos)
            derived_po = ns.generator + base_po
            k = derived_po.index
            k == n && continue
            j = (k + 1) ÷ 2
            isnan(cg[pi]) && continue
            if !haskey(cg_pairs, j)
                cg_pairs[j] = Float64[]
            end
            push!(cg_pairs[j], cg[pi])
        end

        for (j, cg_vals) in cg_pairs
            length(cg_vals) == 2 || continue
            n_cg_both += 1
            xor_cg = cg_vals[1] != cg_vals[2] ? 1 : 0
            n_cg_correct += (xor_cg == goal[j])
        end

        n_both > 0 && push!(prop_pair_acc, n_correct / n_both)
        n_cg_both > 0 && push!(cg_pair_acc, n_cg_correct / n_cg_both)

        # Measure Δ global accuracy
        current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        push!(global_deltas, new_acc - current_acc)

        # Run SA step (always accept for diagnostic purposes — we want to see the distribution
        # across SA states, not just static ensemble)
        if new_acc >= current_acc || rand(rng) < 0.1  # accept improving or 10% random
            update_rep_at!(rep, rep_sum, rep_ctr, ensemble[which], cxt_master)
            update_rep_at!(rep, rep_sum, rep_ctr, ns,              cxt_master)
            ensemble[which] = ns
        else
            apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        end
    end

    return prop_pair_acc, cg_pair_acc, global_deltas
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit    = 6
    nstate    = 40
    nsteps    = 5_000
    base_seed = 42

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    @printf("nqubit=%d  nstate=%d  nsteps=%d\n\n", nqubit, nstate, nsteps)
    @printf("%-8s  %-10s  %-10s  %-10s  %-12s  %-12s\n",
            "H_target", "prop_acc", "cg_acc", "frac_impr", "mean_delta+", "mean_delta-")
    println("-"^72)
    flush(stdout)

    for H_target in [0.0, 0.25, 0.5, 0.75, 1.0]
        k    = k_from_entropy(H_target, ngbits)
        rng  = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal = shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))

        prop_acc, cg_acc, deltas = measure_proposal_quality(
            goal, nqubit, nstate, nsteps; seed=base_seed)

        frac_impr = mean(deltas .> 0)
        pos_deltas = filter(>(0), deltas)
        neg_deltas = filter(<(0), deltas)

        @printf("%-8.4f  %-10.4f  %-10.4f  %-10.4f  %-12.6f  %-12.6f\n",
                hamming_entropy(k, ngbits),
                isempty(prop_acc) ? NaN : mean(prop_acc),
                isempty(cg_acc)   ? NaN : mean(cg_acc),
                frac_impr,
                isempty(pos_deltas) ? NaN : mean(pos_deltas),
                isempty(neg_deltas) ? NaN : mean(neg_deltas))
        flush(stdout)
    end
end

main()
