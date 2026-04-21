using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

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

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fp_packed; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        gen_idx   = rand(rng, 0:3^nqubit-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns = QCScaling.PseudoGHZState(alphas..., generator)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

# Run SA and record accuracy at log-spaced intervals.
function run_sa_trajectory(goal, nqubit, nstate, nsteps, alpha; seed=42, log_interval=1000)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

    steps_log = Int[]
    acc_log   = Float64[]

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fp_packed; rng=rng)
    current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc         = current_acc
    last_improvement = 0

    push!(steps_log, 0); push!(acc_log, current_acc)

    for step in 1:nsteps
        which     = rand(rng, 1:nstate)
        gen_idx   = rand(rng, 0:3^nqubit-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns = QCScaling.PseudoGHZState(alphas..., generator)
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
            if new_acc > best_acc + min_delta
                best_acc         = new_acc
                last_improvement = step
            end
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end
        T *= alpha

        if step % log_interval == 0
            push!(steps_log, step); push!(acc_log, current_acc)
        end

        step > stag_window && (step - last_improvement) >= stag_window && break
    end

    return steps_log, acc_log, best_acc
end

function main()
    nqubit   = 6
    nstate   = 40
    nsteps   = 500_000
    alpha    = 0.99999
    base_seed = 42
    nseeds   = 5
    log_interval = 10_000

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    entropy_targets = [0.0, 0.25, 0.5, 0.75, 1.0]

    @printf("Accuracy trajectories: nqubit=%d  nstate=%d  nsteps=%d\n\n", nqubit, nstate, nsteps)
    @printf("Logging every %d steps, %d seeds per entropy level\n\n", log_interval, nseeds)

    for H_target in entropy_targets
        k    = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng  = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal = goal_from_hamming(k, ngbits, rng)

        @printf("H=%.4f (k=%d)\n", H_act, k)
        @printf("  seed   step_plateau   acc_at_10K   acc_at_100K   acc_final\n")

        for seed in base_seed:(base_seed+nseeds-1)
            steps, accs, best = run_sa_trajectory(
                goal, nqubit, nstate, nsteps, alpha; seed=seed, log_interval=log_interval)

            # Find approximate plateau step (first step where acc ≥ 99% of best)
            plateau_idx = findfirst(a -> a >= 0.99 * best, accs)
            plateau_step = plateau_idx === nothing ? steps[end] : steps[plateau_idx]

            # Accuracy at specific steps
            idx_10k  = findfirst(s -> s >= 10_000, steps)
            idx_100k = findfirst(s -> s >= 100_000, steps)
            acc_10k  = idx_10k  === nothing ? NaN : accs[idx_10k]
            acc_100k = idx_100k === nothing ? NaN : accs[idx_100k]

            @printf("  %-6d %-14d %-12.4f %-13.4f %.4f\n",
                    seed, plateau_step, acc_10k, acc_100k, best)
        end
        println()
        flush(stdout)
    end
end

main()
