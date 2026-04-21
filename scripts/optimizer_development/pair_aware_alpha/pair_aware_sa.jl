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
# Goal / entropy helpers
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

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function efficiency(acc::Float64, nqubit::Int, nstate::Int)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    return (1 - binary_entropy(acc)) * n_classical / n_quantum
end

# ---------------------------------------------------------------------------
# Rep helpers (identical to scaling_study_adaptive.jl)
# ---------------------------------------------------------------------------

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
# Pair-aware alpha picker
#
# The current pick_new_alphas minimizes L1 distance between the state's
# predicted parities and companion_goal at each position. companion_goal(k)
# is "what parity does k need to satisfy the XOR, given rep[companion(k)]?"
#
# Problem for high entropy: about half the pairs want XOR=1 (opposite parities)
# and half want XOR=0 (same). The per-position L1 objective doesn't account for
# the fact that satisfying a XOR=1 pair requires a state that votes *against*
# its companion's current direction. The score is per-position, not per-pair.
#
# Pair-aware scoring:
# 1. For positions where BOTH k1 and k2 of pair j are covered by the proposed
#    state, score by predicted XOR directly (correct = +1, wrong = -1).
# 2. For positions where only one member of the pair is covered, fall back to
#    companion_goal logic using rep as a proxy for the missing partner.
# 3. For positions with no companion information (rep[companion]=NaN), add a
#    small coverage bonus rather than skipping entirely.
#
# This change directly optimizes the accuracy metric (XOR pair correctness)
# rather than a proxy (per-position companion agreement).
# ---------------------------------------------------------------------------

function pick_alphas_pair_aware(
    cxt,
    goal::Vector,
    rep::Vector,
    fp::QCScaling.Fingerprint,
    base_cxt
)
    parity_idx = base_cxt.parity + 1
    nalpha     = size(fp.a, 4)
    ntz        = size(fp.a, 3)
    npos       = length(cxt.pos)

    # Actual position indices covered by this state (1-indexed in n-dim space)
    covered = [p.index for p in cxt.pos]
    covered_set = Set(covered)

    best_score = -Inf
    best_tz    = 1
    best_alpha = 1

    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0.0
            counted = falses(npos)  # avoid double-counting pairs

            for pi in 1:npos
                idx = covered[pi]
                idx == length(rep) && continue  # last position: special case
                companion_idx = idx % 2 == 1 ? idx + 1 : idx - 1
                companion_idx == length(rep) && continue
                j = (min(idx, companion_idx) + 1) ÷ 2  # 1-indexed pair

                pred_parity = fp.a[pi, parity_idx, tzi, ai]

                if companion_idx in covered_set
                    # Both positions of pair j in this state's context:
                    # score by the predicted XOR directly.
                    counted[pi] && continue
                    companion_pi = findfirst(==(companion_idx), covered)
                    companion_pi === nothing && continue
                    counted[companion_pi] = true

                    pred_companion = fp.a[companion_pi, parity_idx, tzi, ai]
                    pred_xor = pred_parity != pred_companion ? 1 : 0
                    correct   = pred_xor == goal[j]

                    # Downweight pairs that are already correctly covered by ensemble
                    rep_valid = !isnan(rep[idx]) && !isnan(rep[companion_idx])
                    if rep_valid
                        rep_xor = Int(rep[idx]) != Int(rep[companion_idx]) ? 1 : 0
                        already_correct = rep_xor == goal[j]
                        score += correct ? (already_correct ? 0.2 : 1.0) :
                                          (already_correct ? -1.0 : -0.2)
                    else
                        # Uncovered pair: reward completing it correctly
                        score += correct ? 1.0 : 0.0
                    end
                else
                    # Companion not covered by this state: fall back to companion_goal logic
                    if !isnan(rep[companion_idx])
                        cg = goal[j] == 1 ? 1.0 - rep[companion_idx] : rep[companion_idx]
                        score += abs(pred_parity - cg) < 0.5 ? 0.5 : -0.5
                    else
                        # Neither position covered by ensemble: small coverage bonus
                        score += 0.1
                    end
                end
            end

            if score > best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end

    return (base_cxt.parity, best_tz - 1,
            QCScaling.idx_to_alphas(best_alpha - 1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Proposal functions
# ---------------------------------------------------------------------------

function smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function pair_aware_proposal(nqubit, rep, goal, fp_plain, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = pick_alphas_pair_aware(cxt, goal, rep, fp_plain, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

# ---------------------------------------------------------------------------
# SA runner with picker selector (:smart | :pair_aware | :random)
# ---------------------------------------------------------------------------

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fp_packed, fp_plain, picker;
                      target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns = if picker == :smart
            smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng)
        elseif picker == :pair_aware
            pair_aware_proposal(nqubit, rep, goal, fp_plain, cxt_master, rng)
        else
            QCScaling.random_state(nqubit)
        end
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
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fp_packed   = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    fp_plain    = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2
    best_acc    = -Inf

    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

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
            ns = if picker == :smart
                smart_proposal(nqubit, rep, goal, fp_packed, cxt_master, rng)
            elseif picker == :pair_aware
                pair_aware_proposal(nqubit, rep, goal, fp_plain, cxt_master, rng)
            else
                QCScaling.random_state(nqubit)
            end
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
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit          = 4
    nstate          = 20
    nsteps          = 200_000
    alpha           = 0.99999
    n_restarts      = 3
    nseeds          = 20
    base_seed       = 42
    entropy_targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pickers         = [:random, :smart, :pair_aware]

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "pair_aware_ablation.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    @printf("nqubit=%d  nstate=%d  nseeds=%d\n\n", nqubit, nstate, nseeds)
    @printf("%-8s  %-6s  %-10s  %-10s  %-12s\n",
            "H_act", "k", "acc_rand", "acc_smart", "acc_pair_aware")
    println("-"^52)
    flush(stdout)

    nH     = length(entropy_targets)
    npick  = length(pickers)
    H_acts = zeros(nH)
    ks_out = zeros(Int, nH)
    acc_mat = zeros(nH, npick, nseeds)  # [H × picker × seed]

    for (hi, H_target) in enumerate(entropy_targets)
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        H_acts[hi] = H_act
        ks_out[hi] = k

        rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

        for (pi, picker) in enumerate(pickers)
            for (si, seed) in enumerate(seeds)
                acc_mat[hi, pi, si] = run_sa(goals[si], nqubit, nstate, nsteps, alpha, picker;
                                             n_restarts=n_restarts, seed=seed)
            end
        end

        @printf("%-8.4f  %-6d  %-10.4f  %-10.4f  %-12.4f\n",
                H_act, k,
                median(acc_mat[hi, 1, :]),
                median(acc_mat[hi, 2, :]),
                median(acc_mat[hi, 3, :]))
        flush(stdout)
    end

    h5open(outfile, "w") do h5f
        HDF5.attributes(h5f)["nqubit"]    = nqubit
        HDF5.attributes(h5f)["nstate"]    = nstate
        HDF5.attributes(h5f)["nseeds"]    = nseeds
        HDF5.attributes(h5f)["base_seed"] = base_seed
        h5f["H_acts"]  = H_acts
        h5f["ks"]      = ks_out
        h5f["acc_mat"] = acc_mat   # [nH × npicker × nseeds]
        h5f["pickers"] = string.(pickers)
    end

    println("\nSaved to $outfile")
end

main()
