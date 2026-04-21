# Script 06: Accuracy ceiling vs nstate for H=0 and H=1.
#
# QUESTION: Is the ~77% ceiling for H=1 fundamental (does not improve with
# more states), or does it improve with nstate → ∞?
#
# Prediction from frustration hypothesis:
#   H=0: accuracy improves with nstate (more states → stronger correct margins)
#   H=1: accuracy saturates quickly (frustration prevents improvement)
#
# If H=1 does improve, it means the ceiling is finite-nstate, not fundamental.
# If H=1 flatlines, it confirms the spin-glass interpretation.
#
# Method: run SA for nstate ∈ {10, 20, 30, 45, 60, 90, 120, 180}
# with 4 seeds each, at nqubit=6.  nsteps scales with nstate to keep
# nsteps/nstate ≈ 6667 (same per-state budget as 300k/45).

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# Run SA with adjustable nstate; keep per-state step budget fixed.
function run_sa_nstate(goal, nqubit, nstate, nsteps, alpha_cool,
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
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:min(300, nsteps)
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
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    n_seeds    = 5
    alpha_cool = 0.9999

    # Per-state step budget: 300_000 / 45 ≈ 6667 steps per state
    steps_per_state = 300_000 ÷ 45

    nstate_vals = [10, 20, 30, 45, 60, 90, 120, 180]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("=" ^ 65)
        println("$H_label — accuracy vs nstate (fixed per-state budget = $steps_per_state steps)")
        println("=" ^ 65)
        @printf("%-8s  %-10s  %-8s  %-8s  %-8s\n",
                "nstate", "nsteps", "mean_acc", "std_acc", "max_acc")
        println(repeat("-", 50))

        for nstate in nstate_vals
            nsteps = nstate * steps_per_state
            # Use a longer cooling for larger ensembles so T still decays to ~0
            alpha  = exp(log(1e-4) / nsteps)  # T → T * 1e-4 over nsteps
            accs   = Float64[]
            for seed in 1:n_seeds
                acc = run_sa_nstate(goal, nqubit, nstate, nsteps, alpha,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=seed * 31 + 7)
                push!(accs, acc)
            end
            @printf("%-8d  %-10d  %-8.4f  %-8.4f  %-8.4f\n",
                    nstate, nsteps, mean(accs), std(accs), maximum(accs))
            flush(stdout)
        end
        println()
    end

    # Also: fixed nsteps=300k to isolate nstate effect from step budget
    println("=" ^ 65)
    println("Bonus: fixed nsteps=300k (increasing nstate is pure additional capacity)")
    println("=" ^ 65)
    alpha_fixed = 0.9999  # same cooling always

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))
        println("$H_label (fixed nsteps=300k)")
        @printf("%-8s  %-8s  %-8s\n", "nstate", "mean_acc", "std_acc")
        println(repeat("-", 30))

        for nstate in nstate_vals
            accs = Float64[]
            for seed in 1:n_seeds
                acc = run_sa_nstate(goal, nqubit, nstate, 300_000, alpha_fixed,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=seed * 31 + 7)
                push!(accs, acc)
            end
            @printf("%-8d  %-8.4f  %-8.4f\n", nstate, mean(accs), std(accs))
            flush(stdout)
        end
        println()
    end
end

main()
