# Script 08: Deep nstate sweep for nqubit=7, plus nqubit=5 anomaly check.
#
# nqubit=7 shows H=1 at only 63% with 180 states while H=0 is perfect at 90.
# This script:
#   1. Extends the nstate sweep for nqubit=7 to see if H=1 converges.
#   2. Checks the nqubit=5 non-monotonic behavior with more seeds (was it noise?).
#   3. Compares nqubit=5 H=1 with a random alpha picker to isolate fingerprint effect.
#
# Also: report what fraction of fingerprint positions have parity=0.5 at nqubit=5,7
# (positions where our floor() fix changes the value).

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

function run_sa_full_nq(goal, nqubit, nstate, nsteps, alpha_cool,
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

function count_undefined_parity(nqubit)
    fp = QCScaling.Fingerprint(nqubit)
    total = length(fp.a)
    # The original parity() returns 0.5 for undefined; we stored floor(Int, ...)
    # Check by recomputing: for each state, recompute raw parity and count 0.5s
    n05 = 0
    for theta_s in 0:1
        base_cxt = QCScaling.generate_base_context(nqubit, theta_s)
        nalpha = 2^(nqubit-1)
        for idx in 0:nalpha-1
            alphas = QCScaling.idx_to_alphas(idx, nqubit)
            for theta_z in 0:1
                state = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, base_cxt.pos[1])
                ps = QCScaling.parity(state, base_cxt)
                n05 += count(p -> p == 0.5, ps)
            end
        end
    end
    return n05, total
end

function main()
    steps_per_state = 300_000 ÷ 45  # 6666
    n_seeds = 5

    # ------------------------------------------------------------------
    # Part A: Fraction of undefined-parity positions in fingerprint
    # ------------------------------------------------------------------
    println("=" ^ 60)
    println("Part A: Undefined parity (=0.5) fraction in Fingerprint")
    println("=" ^ 60)
    for nq in [4, 5, 6, 7]
        n05, total = count_undefined_parity(nq)
        @printf("  nqubit=%d: n_undefined=%d / %d = %.4f\n",
                nq, n05, total, n05/total)
    end
    println()

    # ------------------------------------------------------------------
    # Part B: nqubit=5 anomaly — more seeds
    # ------------------------------------------------------------------
    println("=" ^ 60)
    println("Part B: nqubit=5, H=1 non-monotonic behavior (n_seeds=10)")
    println("=" ^ 60)
    nqubit = 5
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2

    companion5, goal_idx5, _ = build_shuffled_pairing(nqubit)
    fp5 = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cm5 = QCScaling.ContextMaster(nqubit)

    rng5 = Random.MersenneTwister(99)
    goal5_1 = Random.shuffle!(rng5, vcat(ones(Int, ngbits ÷ 2), zeros(Int, ngbits - ngbits ÷ 2)))

    n_seeds_b = 10
    @printf("%-8s  %-8s  %-8s\n", "nstate", "mean_acc", "std_acc")
    println(repeat("-", 28))
    for nstate in [30, 45, 60, 90, 120, 180, 240]
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs = Float64[]
        for seed in 1:n_seeds_b
            acc = run_sa_full_nq(goal5_1, nqubit, nstate, nsteps, alpha,
                                 companion5, goal_idx5, fp5, cm5; seed=seed*31+7)
            push!(accs, acc)
        end
        @printf("%-8d  %-8.4f  %-8.4f\n", nstate, mean(accs), std(accs))
        flush(stdout)
    end
    println()

    # ------------------------------------------------------------------
    # Part C: nqubit=7, extended nstate sweep
    # ------------------------------------------------------------------
    println("=" ^ 60)
    println("Part C: nqubit=7, H=0 and H=1 extended nstate sweep")
    println("=" ^ 60)
    nqubit = 7
    n      = 3^nqubit
    ngbits = (n-1) ÷ 2

    companion7, goal_idx7, _ = build_shuffled_pairing(nqubit)
    fp7 = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cm7 = QCScaling.ContextMaster(nqubit)
    npos7 = length(cm7.base_even.pos)
    @printf("n=%d  ngbits=%d  npos=%d\n", n, ngbits, npos7)

    rng7_0 = Random.MersenneTwister(99)
    goal7_0 = zeros(Int, ngbits)  # H=0

    rng7_1 = Random.MersenneTwister(99)
    goal7_1 = Random.shuffle!(rng7_1, vcat(ones(Int, ngbits ÷ 2), zeros(Int, ngbits - ngbits ÷ 2)))

    n_seeds_c = 3
    @printf("%-8s  %-10s  %-10s\n", "nstate", "H=0 acc", "H=1 acc")
    println(repeat("-", 32))

    for nstate in [45, 90, 180, 270, 360]
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs0 = Float64[]; accs1 = Float64[]
        for seed in 1:n_seeds_c
            a0 = run_sa_full_nq(goal7_0, nqubit, nstate, nsteps, alpha,
                                companion7, goal_idx7, fp7, cm7; seed=seed*31+7)
            a1 = run_sa_full_nq(goal7_1, nqubit, nstate, nsteps, alpha,
                                companion7, goal_idx7, fp7, cm7; seed=seed*31+7)
            push!(accs0, a0); push!(accs1, a1)
        end
        @printf("%-8d  %-10.4f  %-10.4f\n", nstate, mean(accs0), mean(accs1))
        flush(stdout)
    end
end

main()
