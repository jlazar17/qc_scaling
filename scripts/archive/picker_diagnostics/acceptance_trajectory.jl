using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Acceptance trajectory
#
# At each logging interval, record proposal acceptance statistics:
#   - frac_improve:  fraction accepted as improvements (delta > 0)
#   - frac_uphill:   fraction accepted as uphill moves  (delta < 0, accepted)
#   - frac_reject:   fraction rejected
#   - mean_delta_improve: mean delta of accepted improvements
#   - mean_delta_uphill:  mean |delta| of accepted uphill moves
#
# The accuracy_trajectory.jl results showed the gap between H=0 and H=1 opens
# in the warm SA regime (steps 10K-100K). This script asks WHY:
#   - Are there fewer improving proposals for H=1 (flatter landscape)?
#   - Or does temperature drop too fast, killing uphill exploration?
#   - Or are improvements happening but smaller in magnitude?
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

function run_acceptance_trajectory(goal, nqubit, nstate, nsteps, alpha;
                                   seed=42, log_interval=10_000)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2
    min_delta  = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

    steps_log         = Int[]
    acc_log           = Float64[]
    frac_improve_log  = Float64[]
    frac_uphill_log   = Float64[]
    frac_reject_log   = Float64[]
    mean_dimprove_log = Float64[]
    mean_duphill_log  = Float64[]
    T_log             = Float64[]

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fp_packed; rng=rng)
    current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc         = current_acc
    last_improvement = 0

    # Counters for the current window
    n_improve = 0; n_uphill = 0; n_reject = 0
    delta_improve = Float64[]; delta_uphill = Float64[]

    function flush_log(step)
        total = n_improve + n_uphill + n_reject
        push!(steps_log, step)
        push!(acc_log, current_acc)
        push!(T_log, T)
        push!(frac_improve_log,  total > 0 ? n_improve / total : NaN)
        push!(frac_uphill_log,   total > 0 ? n_uphill  / total : NaN)
        push!(frac_reject_log,   total > 0 ? n_reject  / total : NaN)
        push!(mean_dimprove_log, isempty(delta_improve) ? NaN : mean(delta_improve))
        push!(mean_duphill_log,  isempty(delta_uphill)  ? NaN : mean(delta_uphill))
        # Reset window
        n_improve = 0; n_uphill = 0; n_reject = 0
        empty!(delta_improve); empty!(delta_uphill)
    end

    # Log initial state (step 0, window not yet meaningful — skip counters)
    push!(steps_log, 0); push!(acc_log, current_acc); push!(T_log, T)
    push!(frac_improve_log, NaN); push!(frac_uphill_log, NaN); push!(frac_reject_log, NaN)
    push!(mean_dimprove_log, NaN); push!(mean_duphill_log, NaN)

    for step in 1:nsteps
        which     = rand(rng, 1:nstate)
        gen_idx   = rand(rng, 0:3^nqubit-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns        = QCScaling.PseudoGHZState(alphas..., generator)
        old_state = ensemble[which]

        apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta   = new_acc - current_acc

        if delta > 0
            update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
            update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
            ensemble[which] = ns; current_acc = new_acc
            n_improve += 1; push!(delta_improve, delta)
            new_acc > best_acc + min_delta && (best_acc = new_acc; last_improvement = step)
        elseif rand(rng) < exp(delta / T)
            update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
            update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
            ensemble[which] = ns; current_acc = new_acc
            n_uphill += 1; push!(delta_uphill, abs(delta))
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            n_reject += 1
        end
        T *= alpha

        step % log_interval == 0 && flush_log(step)
        step > stag_window && (step - last_improvement) >= stag_window && break
    end

    return steps_log, acc_log, T_log,
           frac_improve_log, frac_uphill_log, frac_reject_log,
           mean_dimprove_log, mean_duphill_log
end

function main()
    nqubit       = 6
    nstate       = 40
    nsteps       = 500_000
    alpha        = 0.99999
    base_seed    = 42
    nseeds       = 5
    log_interval = 10_000
    entropy_targets = [0.0, 0.5, 1.0]

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    @printf("Acceptance trajectory: nqubit=%d  nstate=%d  nsteps=%d\n\n", nqubit, nstate, nsteps)
    @printf("Tracks fraction of proposals that are: improvements / uphill-accepted / rejected\n")
    @printf("and mean delta magnitude in each category, over 10K-step windows.\n\n")

    for H_target in entropy_targets
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng   = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal  = goal_from_hamming(k, ngbits, rng)

        @printf("=== H=%.4f (k=%d) ===\n", H_act, k)
        @printf("%-6s  %-7s  %-8s  %-9s  %-9s  %-9s  %-12s  %-12s\n",
                "seed", "step", "acc", "f_impr", "f_uphill", "f_reject",
                "mean_d_impr", "mean_d_uphl")
        println("-"^82)

        for seed in base_seed:(base_seed + nseeds - 1)
            steps, accs, Ts, f_imp, f_uph, f_rej, d_imp, d_uph =
                run_acceptance_trajectory(goal, nqubit, nstate, nsteps, alpha;
                                          seed=seed, log_interval=log_interval)

            for i in eachindex(steps)
                @printf("%-6d  %-7d  %-8.4f  %-9.4f  %-9.4f  %-9.4f  %-12.6f  %-12.6f\n",
                        seed, steps[i], accs[i],
                        isnan(f_imp[i]) ? -1.0 : f_imp[i],
                        isnan(f_uph[i]) ? -1.0 : f_uph[i],
                        isnan(f_rej[i]) ? -1.0 : f_rej[i],
                        isnan(d_imp[i]) ? -1.0 : d_imp[i],
                        isnan(d_uph[i]) ? -1.0 : d_uph[i])
            end
            println()
            flush(stdout)
        end
    end
end

main()
