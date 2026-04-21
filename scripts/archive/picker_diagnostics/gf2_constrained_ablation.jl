using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using HDF5
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# GF(2)-constrained picker ablation: n=6 and n=8
#
# For proposals with both-covered pairs, the standard picker may choose a
# (theta_z, alpha) config that violates the goal XOR constraints at those
# pairs. The GF(2)-constrained picker:
#   1. Enumerates all 2*nalpha configs
#   2. Filters to those satisfying the XOR constraint at all both-covered pairs
#   3. If none satisfy: resamples the generator (up to max_tries)
#   4. If some satisfy: among valid configs, picks the one with best
#      companion_goal score at all positions
#
# For generators with no both-covered pairs, falls back to standard
# pick_new_alphas (FingerprintPacked) — no overhead.
#
# Comparison: :smart (standard) vs :gf2 (GF(2)-constrained)
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
    pref = rep_sum ./ rep_ctr; pref[pref .== 0.5] .= NaN; return round.(pref)
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
# GF(2)-constrained proposal
#
# Uses the plain Fingerprint (.a array) to enumerate and filter configs.
# Falls back to FingerprintPacked for generators with no both-covered pairs.
# ---------------------------------------------------------------------------

function gf2_constrained_proposal(nqubit, rep, goal, fp_plain, fp_packed,
                                  cxt_master, rng; max_tries=20)
    n = 3^nqubit

    for _ in 1:max_tries
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        parity_idx = base_cxt.parity + 1  # = theta_s + 1

        # Find both-covered pairs and their base-context position indices
        # pi: 1-indexed position in base_cxt.pos (= fingerprint index)
        # k: derived position index (1-indexed, k==n is the special position)
        pair_first_pi = Dict{Int, Int}()   # j => pi of first encounter
        pair_first_k  = Dict{Int, Int}()   # j => k of first encounter
        both_covered  = Vector{Tuple{Int,Int,Int}}()  # (j, k1_pi, k2_pi), k1 odd

        pi = 0
        for base_po in base_cxt.pos
            pi += 1
            derived_po = generator + base_po
            k = derived_po.index
            k == n && continue
            j = (k + 1) ÷ 2
            if haskey(pair_first_pi, j)
                other_pi = pair_first_pi[j]
                other_k  = pair_first_k[j]
                if isodd(k)
                    push!(both_covered, (j, pi, other_pi))  # k (odd) = k1
                else
                    push!(both_covered, (j, other_pi, pi))  # other = k1 (must be odd)
                end
            else
                pair_first_pi[j] = pi
                pair_first_k[j]  = k
            end
        end

        # No both-covered pairs: standard fast picker
        if isempty(both_covered)
            alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
            return QCScaling.PseudoGHZState(alphas..., generator)
        end

        # Enumerate configs, filter by GF(2) constraint
        fa       = fp_plain.a
        nalpha   = size(fa, 4)
        ntz      = size(fa, 3)
        npos_fp  = size(fa, 1)

        # companion_goals for scoring
        cg = QCScaling.companion_goal(cxt, goal, rep)

        best_score = typemin(Int)
        best_tzi   = -1
        best_ai    = -1

        @inbounds for tzi in 1:ntz
            for ai in 1:nalpha
                # Check GF(2) constraint at all both-covered pairs
                feasible = true
                for (j, k1_pi, k2_pi) in both_covered
                    p1 = fa[k1_pi, parity_idx, tzi, ai]
                    p2 = fa[k2_pi, parity_idx, tzi, ai]
                    xor_pred = p1 != p2 ? 1 : 0
                    if xor_pred != goal[j]
                        feasible = false; break
                    end
                end
                !feasible && continue

                # Score by companion_goal agreement at all positions
                score = 0
                pi2 = 0
                for base_po in base_cxt.pos
                    pi2 += 1
                    isnan(cg[pi2]) && continue
                    pred = fa[pi2, parity_idx, tzi, ai]
                    score += pred == round(Int, cg[pi2]) ? 1 : -1
                end

                if score > best_score
                    best_score = score
                    best_tzi   = tzi
                    best_ai    = ai
                end
            end
        end

        # No feasible config: try a new generator
        best_tzi == -1 && continue

        alphas = QCScaling.idx_to_alphas(best_ai - 1, nqubit)
        return QCScaling.PseudoGHZState(theta_s, best_tzi - 1, alphas, generator)
    end

    # Fallback: standard proposal (should be rare)
    gen_idx   = rand(rng, 0:n-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

# ---------------------------------------------------------------------------
# T0 calibration and SA runner
# ---------------------------------------------------------------------------

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fp_packed, fp_plain, picker;
                      target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns = picker == :smart ?
            smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng) :
            gf2_constrained_proposal(nqubit, rep, goal, fp_plain, fp_packed, cxt_master, rng)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
        delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

function run_sa(goal, nqubit, nstate, nsteps, alpha, picker; n_restarts=3, seed=42)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    fp_plain   = QCScaling.Fingerprint(nqubit)
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2
    min_delta  = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))
    best_acc   = -Inf

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fp_packed, fp_plain, picker; rng=rng)
        current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best     = current_acc
        last_improvement = 0

        for step in 1:nsteps
            which = rand(rng, 1:nstate)
            ns = picker == :smart ?
                smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng) :
                gf2_constrained_proposal(nqubit, rep, goal, fp_plain, fp_packed, cxt_master, rng)
            old_state = ensemble[which]

            apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
                update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
                ensemble[which] = ns
                current_acc     = new_acc
                if new_acc > restart_best + min_delta
                    restart_best     = new_acc
                    last_improvement = step
                end
                new_acc > best_acc && (best_acc = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha

            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

# ---------------------------------------------------------------------------
# Main: n=6 then n=8
# ---------------------------------------------------------------------------

function run_ablation(nqubit, nstate, nsteps, n_restarts, nseeds, base_seed,
                      entropy_targets, alpha)
    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2
    pickers = [:smart, :gf2]

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    @printf("\n%s\n", "="^60)
    @printf("nqubit=%d  nstate=%d  nsteps=%d  nseeds=%d  n_restarts=%d\n",
            nqubit, nstate, nsteps, nseeds, n_restarts)
    @printf("%-8s  %-6s  %-12s  %-12s  %-12s\n",
            "H_act", "k", "acc_smart", "acc_gf2", "delta")
    println("-"^56)
    flush(stdout)

    nH = length(entropy_targets)
    acc_mat = zeros(nH, 2, nseeds)  # [H × picker × seed]

    for (hi, H_target) in enumerate(entropy_targets)
        k     = k_from_entropy(Float64(H_target), ngbits)
        H_act = hamming_entropy(k, ngbits)

        rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

        for (pi, picker) in enumerate(pickers)
            for (si, seed) in enumerate(seeds)
                acc_mat[hi, pi, si] = run_sa(goals[si], nqubit, nstate, nsteps,
                                             alpha, picker;
                                             n_restarts=n_restarts, seed=seed)
            end
        end

        @printf("%-8.4f  %-6d  %-12.4f  %-12.4f  %-+12.4f\n",
                H_act, k,
                median(acc_mat[hi, 1, :]),
                median(acc_mat[hi, 2, :]),
                median(acc_mat[hi, 2, :]) - median(acc_mat[hi, 1, :]))
        flush(stdout)
    end

    return acc_mat
end

function main()
    alpha     = 0.99999
    base_seed = 42

    # n=6: full H sweep at the H=0-optimal nstate
    acc6 = run_ablation(
        6, 40, 500_000, 3, 10, base_seed,
        0.0:0.1:1.0, alpha
    )

    # n=8: three H levels at the H=0-optimal nstate
    acc8 = run_ablation(
        8, 147, 1_000_000, 2, 5, base_seed,
        [0.0, 0.25, 0.5, 0.75, 1.0], alpha
    )

    println("\nDone.")
end

main()
