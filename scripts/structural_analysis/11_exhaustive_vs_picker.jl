# Exhaustive oracle vs picker: does a better pattern exist?
#
# For each proposed (gen, theta_s, which), try all 64 (tz, alpha) patterns
# and record whether the best possible pattern gives delta >= 0.
# Compare to the picker's rate.
#
# If exhaustive rate >> picker rate: picker is finding the wrong pattern.
# If exhaustive rate ≈ picker rate:  no pattern works — the landscape is stuck.
#
# Run SA to checkpoints, probe at each one, compare H=0 vs H=1.

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
# Build the ensemble + rep via SA to a target accuracy.
# Returns (ensemble, cache_idxs, cache_pars, rep_sum, rep_ctr, rep).
# ---------------------------------------------------------------------------
function run_sa_to_acc(goal, nqubit, nstate, target_acc, max_steps, alpha_cool,
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

    # Temperature calibration
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

    for step in 1:max_steps
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
        cur_acc >= target_acc && break
    end
    return ensemble, cache_idxs, cache_pars, rep_sum, rep_ctr, rep, cur_acc
end

# ---------------------------------------------------------------------------
# Probe: for n_probe random (gen, theta_s, which), compute
#   picker_delta  = accuracy delta from pick_alphas_s recommendation
#   oracle_delta  = best accuracy delta across all 64 (tz, alpha) patterns
# Returns (picker_pos_rate, oracle_pos_rate, mean_picker_delta, mean_oracle_delta)
# ---------------------------------------------------------------------------
function probe_exhaustive(goal, nqubit, nstate, n_probe,
                          ensemble, cache_idxs, cache_pars,
                          rep_sum, rep_ctr, rep,
                          companion, goal_idx, fingerprint, cxt_master; seed=1)
    n      = 3^nqubit
    npos   = length(cxt_master.base_even.pos)
    nalpha = 1 << (nqubit - 1)
    rng    = Random.MersenneTwister(seed)

    acc_fn(rs, rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    scratch_idxs  = Vector{Int}(undef, npos)
    scratch_pars  = Vector{Int}(undef, npos)
    alphas_buf    = zeros(Int, nqubit - 1)

    n_picker_pos  = 0
    n_oracle_pos  = 0
    picker_deltas = Float64[]
    oracle_deltas = Float64[]

    # Temporary copies so we don't mutate the ensemble
    tmp_sum = copy(rep_sum)
    tmp_ctr = copy(rep_ctr)

    for _ in 1:n_probe
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd

        # --- Picker recommendation ---
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc,
                               companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)

        copy!(tmp_sum, rep_sum); copy!(tmp_ctr, rep_ctr)
        apply_state_cached!(tmp_sum, tmp_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(tmp_sum, tmp_ctr, scratch_idxs, scratch_pars, 1)
        picker_d = acc_fn(tmp_sum, tmp_ctr) - cur_acc
        push!(picker_deltas, picker_d)
        picker_d >= 0 && (n_picker_pos += 1)

        # --- Exhaustive oracle: try all (tz, alpha) ---
        best_d = typemin(Float64)
        for tz in 0:1
            for ai in 0:nalpha-1
                for b in 0:nqubit-2; alphas_buf[b+1] = (ai >> b) & 1; end
                sv  = SVector{nqubit-1, Int}(alphas_buf)
                ns2 = QCScaling.PseudoGHZState(ts, tz, sv, gen)
                fill_state_cache!(scratch_idxs, scratch_pars, ns2, cxt_master)
                copy!(tmp_sum, rep_sum); copy!(tmp_ctr, rep_ctr)
                apply_state_cached!(tmp_sum, tmp_ctr, cache_idxs[which], cache_pars[which], -1)
                apply_state_cached!(tmp_sum, tmp_ctr, scratch_idxs, scratch_pars, 1)
                d = acc_fn(tmp_sum, tmp_ctr) - cur_acc
                d > best_d && (best_d = d)
            end
        end
        push!(oracle_deltas, best_d)
        best_d >= 0 && (n_oracle_pos += 1)
    end

    return (
        picker_pos  = n_picker_pos / n_probe,
        oracle_pos  = n_oracle_pos / n_probe,
        picker_mean = mean(picker_deltas),
        oracle_mean = mean(oracle_deltas),
    )
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
    max_steps  = 500_000
    n_probe    = 2000

    checkpoints = [0.65, 0.70, 0.75, 0.78]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    @printf("%-8s  %-8s  %-12s  %-12s  %-12s  %-12s\n",
            "H", "acc", "picker_pos", "oracle_pos", "picker_mean", "oracle_mean")
    println(repeat("-", 70))

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("\n$H_label")
        prev_acc = 0.0
        for target_acc in checkpoints
            print("  Running SA to acc=$(target_acc) ... ")
            ens, ci, cp, rs, rc, rep, reached =
                run_sa_to_acc(goal, nqubit, nstate, target_acc, max_steps, alpha_cool,
                              companion, goal_idx, fingerprint, cxt_master; seed=42)
            @printf("reached %.4f\n", reached)

            # Only probe if we actually reached near the target
            reached < target_acc - 0.02 && (prev_acc = reached; continue)

            print("  Probing ($n_probe proposals) ... ")
            r = probe_exhaustive(goal, nqubit, nstate, n_probe, ens, ci, cp, rs, rc, rep,
                                 companion, goal_idx, fingerprint, cxt_master; seed=7)
            println("done.")
            @printf("  %-8s  acc=%-6.3f  picker_pos=%-8.4f  oracle_pos=%-8.4f  picker_mean=%-10.6f  oracle_mean=%.6f\n",
                    H_label, reached,
                    r.picker_pos, r.oracle_pos,
                    r.picker_mean, r.oracle_mean)
            prev_acc = reached
        end
    end
end

main()
