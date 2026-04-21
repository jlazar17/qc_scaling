# Basin of attraction analysis: H=0 vs H=1
#
# Protocol:
#   1. Run N_INIT SA runs to convergence → N_INIT converged ensembles
#   2. For each ensemble, replace k states with fresh random states
#      (k = 1, 2, 3, 5, 10)
#   3. Re-run SA from the perturbed starting point (same nsteps)
#   4. Compare recovery accuracy to original
#
# Large basins (H=0 hypothesis): recovery_acc ≈ original_acc for all k
# Small basins (H=1 hypothesis): recovery_acc degrades as k grows

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# SA from a given initial ensemble. Returns (final_acc, ensemble, cache_idxs, cache_pars)
# ---------------------------------------------------------------------------
function run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                companion, goal_idx, fingerprint, cxt_master;
                seed=42, init_ensemble=nothing)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    rng  = Random.MersenneTwister(seed)

    if init_ensemble === nothing
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    else
        ensemble = deepcopy(init_ensemble)
    end

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
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    # Temperature calibration
    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                               bc, companion, goal_idx, n-1)
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
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                               bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc
        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns
            cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc, ensemble
end

# ---------------------------------------------------------------------------
# Perturb an ensemble by replacing k randomly chosen states with fresh random
# states. Returns a new ensemble (does not modify original).
# ---------------------------------------------------------------------------
function perturb_ensemble(ensemble, k, nqubit; seed=1)
    rng = Random.MersenneTwister(seed)
    new_ens = deepcopy(ensemble)
    idxs = randperm(rng, length(ensemble))[1:k]
    for i in idxs
        new_ens[i] = QCScaling.random_state(nqubit)
    end
    return new_ens
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

    n_init     = 10
    k_vals     = [1, 2, 3, 5, 10]
    init_seeds = collect(1:n_init)

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("=" ^ 60)
        println("$H_label")
        println("=" ^ 60)

        # Step 1: run N_INIT SA runs to convergence
        println("  Running $n_init initial SA runs...")
        orig_accs      = Float64[]
        orig_ensembles = Vector{Vector{QCScaling.PseudoGHZState}}()
        for (i, seed) in enumerate(init_seeds)
            acc, ens = run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                              companion, goal_idx, fingerprint, cxt_master; seed=seed)
            push!(orig_accs, acc)
            push!(orig_ensembles, ens)
            @printf("    init %2d (seed=%d): acc=%.4f\n", i, seed, acc)
        end
        @printf("  Original: mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n\n",
                mean(orig_accs), std(orig_accs), minimum(orig_accs), maximum(orig_accs))

        # Step 2 & 3: perturb each ensemble by k states, re-run SA, record recovery
        println("  Perturbation recovery:")
        @printf("  %-6s  %-10s  %-10s  %-10s  %-10s\n",
                "k", "mean_rec", "std_rec", "mean_delta", "frac_recovered")
        println("  " * repeat("-", 52))

        for k in k_vals
            deltas    = Float64[]
            recovered = Int[]   # 1 if recovery_acc >= orig_acc - 0.005, else 0
            for (i, (orig_acc, orig_ens)) in enumerate(zip(orig_accs, orig_ensembles))
                perturbed = perturb_ensemble(orig_ens, k, nqubit; seed=i * 100 + k)
                rec_acc, _ = run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=i * 100 + k + 999,
                                    init_ensemble=perturbed)
                push!(deltas, rec_acc - orig_acc)
                push!(recovered, rec_acc >= orig_acc - 0.005 ? 1 : 0)
            end
            @printf("  k=%-4d  mean=%.4f  std=%.4f  delta=%-+.4f  frac=%.2f\n",
                    k, mean(orig_accs) + mean(deltas), std(deltas),
                    mean(deltas), mean(recovered))
        end
        println()
    end
end

main()
