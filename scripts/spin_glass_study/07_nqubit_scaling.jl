# Script 07: Required nstate vs nqubit for H=0 and H=1.
#
# KEY QUESTION: How does the accuracy gap between H=0 and H=1 scale with nqubit?
#
# Method: for each nqubit ∈ {4, 5, 6}, sweep nstate over a range and record
# the accuracy achieved. Use a fixed per-state step budget.
# Report the nstate required to reach 90% and 99% accuracy.
#
# If the RATIO of required nstates (H=1 vs H=0) grows with nqubit,
# H=1 is fundamentally harder at scale.
# If the ratio stays constant, it's a fixed overhead.
#
# nqubit=7 may be too slow for a full sweep; we run it at fixed nstates only.

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

function sweep_nstate(nqubit, goal, companion, goal_idx, fingerprint, cxt_master,
                      nstate_vals, steps_per_state, n_seeds)
    accs_by_nstate = Dict{Int, Vector{Float64}}()
    for nstate in nstate_vals
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs   = Float64[]
        for seed in 1:n_seeds
            acc = run_sa_nq(goal, nqubit, nstate, nsteps, alpha,
                            companion, goal_idx, fingerprint, cxt_master;
                            seed=seed * 31 + 7)
            push!(accs, acc)
        end
        accs_by_nstate[nstate] = accs
    end
    return accs_by_nstate
end

function main()
    n_seeds = 5

    # Steps per state chosen so nqubit=6 matches our baseline (300k/45 ≈ 6667)
    steps_per_state = 300_000 ÷ 45

    println("=" ^ 70)
    println("Accuracy vs nstate for different nqubit values (H=0 and H=1)")
    println("=" ^ 70)
    println()

    for nqubit in [4, 5, 6]
        n      = 3^nqubit
        ngbits = (n-1) ÷ 2

        companion, goal_idx, _ = build_shuffled_pairing(nqubit)
        fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
        cxt_master  = QCScaling.ContextMaster(nqubit)
        npos = length(cxt_master.base_even.pos)

        # nstate range: scale with n so per-pair coverage is comparable
        # For nqubit=6, baseline is nstate=45. Scale as nstate ~ ngbits^0.5 roughly.
        # Just use a fixed range for all nqubit and see what fits.
        nstate_vals = [5, 10, 20, 30, 45, 60, 90, 120, 180]

        println("─" ^ 70)
        @printf("nqubit=%d  n=%d  ngbits=%d  npos=%d\n", nqubit, n, ngbits, npos)
        println("─" ^ 70)
        @printf("%-8s", "nstate")
        for H_label in ["H=0 acc", "H=1 acc", "ratio"]
            @printf("  %-10s", H_label)
        end
        println()
        println(repeat("-", 45))

        rng0 = Random.MersenneTwister(99)
        goal0 = Random.shuffle!(rng0, vcat(zeros(Int, ngbits)))

        rng1 = Random.MersenneTwister(99)
        goal1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2),
                                           zeros(Int, ngbits - ngbits ÷ 2)))

        for nstate in nstate_vals
            nsteps = nstate * steps_per_state
            alpha  = exp(log(1e-4) / nsteps)
            accs0 = Float64[]; accs1 = Float64[]
            for seed in 1:n_seeds
                a0 = run_sa_nq(goal0, nqubit, nstate, nsteps, alpha,
                               companion, goal_idx, fingerprint, cxt_master;
                               seed=seed * 31 + 7)
                a1 = run_sa_nq(goal1, nqubit, nstate, nsteps, alpha,
                               companion, goal_idx, fingerprint, cxt_master;
                               seed=seed * 31 + 7)
                push!(accs0, a0); push!(accs1, a1)
            end
            m0 = mean(accs0); m1 = mean(accs1)
            ratio = m1 > 0 ? m0 / m1 : NaN
            @printf("%-8d  %-10.4f  %-10.4f  %-10.4f\n", nstate, m0, m1, ratio)
            flush(stdout)
        end
        println()
    end

    # nqubit=7: too slow to sweep; just run a few nstate values
    println("─" ^ 70)
    println("nqubit=7 (fixed nstate spot-check)")
    println("─" ^ 70)
    nqubit = 7
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2
    companion7, goal_idx7, _ = build_shuffled_pairing(nqubit)
    fingerprint7 = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master7  = QCScaling.ContextMaster(nqubit)
    npos7 = length(cxt_master7.base_even.pos)
    @printf("n=%d  ngbits=%d  npos=%d\n", n, ngbits, npos7)

    rng7_0 = Random.MersenneTwister(99)
    goal7_0 = Random.shuffle!(rng7_0, zeros(Int, ngbits))
    rng7_1 = Random.MersenneTwister(99)
    goal7_1 = Random.shuffle!(rng7_1, vcat(ones(Int, ngbits ÷ 2),
                                           zeros(Int, ngbits - ngbits ÷ 2)))

    @printf("%-8s  %-10s  %-10s\n", "nstate", "H=0 acc", "H=1 acc")
    println(repeat("-", 32))
    for nstate in [45, 90, 180]
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs0 = Float64[]; accs1 = Float64[]
        for seed in 1:3
            a0 = run_sa_nq(goal7_0, nqubit, nstate, nsteps, alpha,
                           companion7, goal_idx7, fingerprint7, cxt_master7;
                           seed=seed * 31 + 7)
            a1 = run_sa_nq(goal7_1, nqubit, nstate, nsteps, alpha,
                           companion7, goal_idx7, fingerprint7, cxt_master7;
                           seed=seed * 31 + 7)
            push!(accs0, a0); push!(accs1, a1)
        end
        @printf("%-8d  %-10.4f  %-10.4f\n", nstate, mean(accs0), mean(accs1))
        flush(stdout)
    end
end

main()
