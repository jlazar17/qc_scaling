# Script 09: nstate scaling for nqubit=8.
#
# nqubit=4: both H=0 and H=1 reach 1.000 at nstate=20 (no gap)
# nqubit=6: H=0 reaches 1.000 at nstate=90, H=1 at nstate=180 (2x gap)
# nqubit=8: ? — does the gap grow further?
#
# n=3^8=6561, ngbits=3280, npos = 2^7+1 = 129.
# Run at nstate ∈ {45, 90, 180, 270, 360} with 3 seeds each.
# Use the same per-state step budget (6666 steps/state).

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function run_sa_nq(goal, nqubit, nstate, nsteps, alpha_cool,
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

    for step in 1:nsteps
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
        # Progress checkpoint every 10% of steps
        if step % (nsteps ÷ 10) == 0
            @printf("    step=%d/%d  T=%.6f  acc=%.4f\n", step, nsteps, T, cur_acc)
            flush(stdout)
        end
    end
    return cur_acc
end

function main()
    nqubit = 8
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2
    steps_per_state = 300_000 ÷ 45  # 6666

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fp = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cm = QCScaling.ContextMaster(nqubit)
    npos = length(cm.base_even.pos)

    @printf("nqubit=%d  n=%d  ngbits=%d  npos=%d\n", nqubit, n, ngbits, npos)

    rng0 = Random.MersenneTwister(99)
    goal0 = zeros(Int, ngbits)  # H=0: all-zero

    rng1 = Random.MersenneTwister(99)
    goal1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2), zeros(Int, ngbits - ngbits ÷ 2)))  # H=1

    n_seeds = 3
    nstate_vals = [45, 90, 180, 270, 360]

    println("=" ^ 60)
    println("nqubit=8: H=0 vs H=1 accuracy vs nstate")
    println("=" ^ 60)
    @printf("%-8s  %-10s  %-10s\n", "nstate", "H=0 acc", "H=1 acc")
    println(repeat("-", 32))

    for nstate in nstate_vals
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs0 = Float64[]; accs1 = Float64[]

        @printf("\nnstate=%d  nsteps=%d\n", nstate, nsteps)
        for seed in 1:n_seeds
            @printf("  H=0 seed=%d:\n", seed)
            a0 = run_sa_nq(goal0, nqubit, nstate, nsteps, alpha,
                           companion, goal_idx, fp, cm; seed=seed*31+7)
            push!(accs0, a0)
            @printf("  H=1 seed=%d:\n", seed)
            a1 = run_sa_nq(goal1, nqubit, nstate, nsteps, alpha,
                           companion, goal_idx, fp, cm; seed=seed*31+7)
            push!(accs1, a1)
        end
        @printf("%-8d  %-10.4f  %-10.4f\n", nstate, mean(accs0), mean(accs1))
        flush(stdout)
    end
end

main()
