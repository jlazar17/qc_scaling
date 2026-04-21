# profile_n10_cluster.jl
#
# Run this on the cluster before submitting the full n=10 sweep to get
# accurate per-step timing and projected runtimes.
#
# Usage:
#   julia --threads=20 profile_n10_cluster.jl
#
# Output: timing breakdown, projected wall time per H job, and a suggested
# TIME_LIMIT for submit_n10.sh.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling, Statistics, Random, Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Minimal SA loop matching run_goal_aware_sa exactly (no restarts, no
# stagnation check) so the timing is pure per-step cost.
# ---------------------------------------------------------------------------
function time_sa_steps(nqubit, nstate, nsteps; seed=1)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2

    goal     = zeros(Int, ngbits)   # H=0; timing doesn't depend on goal content
    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    npos     = length(cxt_master.base_even.pos)

    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = 0.01
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)

    function gen_proposal()
        gen_idx  = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
        return QCScaling.PseudoGHZState(alphas..., generator)
    end

    t0 = time_ns()
    for _ in 1:nsteps
        which = rand(rng, 1:nstate)
        ns    = gen_proposal()
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs,      scratch_pars,       1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta   = new_acc - current_acc
        if delta >= 0 || rand(rng) < exp(delta / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns
            current_acc     = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs,      scratch_pars,       -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which],   1)
        end
        T *= 0.99999
    end
    return (time_ns() - t0) / 1e9 / nsteps   # seconds per step
end

# ---------------------------------------------------------------------------
# Thread scaling: time a batch of seeds in parallel the same way
# evaluate_nstate does, at nstate0.
# ---------------------------------------------------------------------------
function time_threaded_batch(nqubit, nstate, nsteps_per_seed, n_seeds)
    times = Vector{Float64}(undef, n_seeds)
    t0 = time()
    Threads.@threads for si in 1:n_seeds
        t1 = time()
        time_sa_steps(nqubit, nstate, nsteps_per_seed; seed=si)
        times[si] = time() - t1
    end
    wall = time() - t0
    return wall, times
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit      = 10
    nstate0     = Int(ceil(3^nqubit / 2^(nqubit - 1)))   # 116
    nthreads    = Threads.nthreads()
    n_warmup    = 200
    n_time      = 1_000
    nseeds      = 20
    n_restarts  = 3
    sa_nsteps   = 5_000_000
    nstate_max  = 4000
    n_coarse    = 12
    n_refine    = 2
    n_nstates   = n_coarse + 2 * n_refine   # approximate total nstate evaluations

    println("="^70)
    @printf("Host        : %s\n", gethostname())
    @printf("nqubit      : %d  (n = %d,  nstate0 = %d)\n", nqubit, 3^nqubit, nstate0)
    @printf("Julia       : %s\n", VERSION)
    @printf("Threads     : %d\n", nthreads)
    println("="^70)

    # Warmup to trigger JIT compilation
    print("\nWarming up JIT ... ")
    flush(stdout)
    time_sa_steps(nqubit, nstate0, n_warmup; seed=0)
    println("done.")

    # Per-step timing at nstate0
    print("Timing $n_time steps at nstate0=$nstate0 ... ")
    flush(stdout)
    sec_per_step = time_sa_steps(nqubit, nstate0, n_time; seed=1)
    us_per_step  = sec_per_step * 1e6
    println("done.")
    @printf("  %.2f µs/step\n", us_per_step)

    # Thread scaling: simulate one nstate evaluation (nseeds seeds in parallel)
    short_steps = min(5_000, sa_nsteps)
    @printf("\nThread scaling: %d seeds × %d steps on %d threads ... ",
            nseeds, short_steps, nthreads)
    flush(stdout)
    wall_t, seed_times = time_threaded_batch(nqubit, nstate0, short_steps, nseeds)
    serial_t = sum(seed_times)
    speedup  = serial_t / wall_t
    parallel_efficiency = speedup / nthreads * 100
    println("done.")
    @printf("  Wall time  : %.2f s  (serial sum: %.2f s)\n", wall_t, serial_t)
    @printf("  Speedup    : %.2f×  (%.0f%% parallel efficiency on %d threads)\n",
            speedup, parallel_efficiency, nthreads)

    # ---------------------------------------------------------------------------
    # Projections
    # ---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PROJECTED RUNTIMES")
    println("-"^70)

    # Per-seed, per-restart: worst case (no stagnation termination)
    t_restart_wc   = sa_nsteps * sec_per_step / 60   # minutes
    t_seed_wc      = n_restarts * t_restart_wc        # minutes per seed
    t_nstate_wc    = t_seed_wc                         # with perfect thread parallelism

    # More realistic: stagnation terminates at ~500k steps on average
    stag_window    = round(Int, -5.0 / log(0.99999))  # ≈ 500_000
    t_restart_est  = stag_window * sec_per_step / 60  # minutes (stagnation estimate)
    t_seed_est     = n_restarts * t_restart_est
    t_nstate_est   = t_seed_est

    @printf("\nPer SA restart:\n")
    @printf("  Worst case  (all %d steps)     : %5.1f min\n", sa_nsteps, t_restart_wc)
    @printf("  Typical     (stagnation ~%dk)  : %5.1f min\n",
            stag_window÷1000, t_restart_est)

    # t_nstate_wc is already the wall time for one nstate evaluation: seeds run
    # in parallel on their own threads, so wall = one seed's time.  Do NOT divide
    # by speedup again — that would double-count the parallelism.  The speedup
    # measurement above is reported only to characterise thread efficiency.
    @printf("\nPer nstate evaluation (%d seeds, %d threads):\n", nseeds, nthreads)
    @printf("  Worst case  : %5.1f min wall\n", t_nstate_wc)
    @printf("  Typical est : %5.1f min wall\n", t_nstate_est)

    @printf("\nPer H value (%d nstate points):\n", n_nstates)
    @printf("  Worst case  : %5.1f hours\n", n_nstates * t_nstate_wc / 60)
    @printf("  Typical est : %5.1f hours\n", n_nstates * t_nstate_est / 60)

    # Suggested SLURM time limit (worst case + 20% buffer, rounded up to hour)
    wc_hrs    = n_nstates * t_nstate_wc / 60
    suggested = ceil(Int, wc_hrs * 1.2)
    @printf("\nSuggested TIME_LIMIT for submit_n10.sh : %02d:00:00\n", suggested)

    println()
    println("="^70)
    println("If parallel efficiency is low (<50%), consider reducing --threads")
    println("or check whether the cluster node has enough cores allocated.")
    println("="^70)
end

main()
