using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Rep convergence trajectory
#
# At each logging interval, record:
#   - Overall pair accuracy
#   - Pair completeness: fraction of pairs where BOTH k1 and k2 are non-NaN in rep
#   - Conditional accuracy: accuracy restricted to complete pairs
#   - Rep bias: mean value across non-NaN rep entries (0.5 = unbiased)
#
# The hypothesis: for H=0, the rep fills in with values that trivially satisfy
# the XOR=0 constraint (any equal pair works). For H=1, the rep must converge
# to a specific mixed pattern — and may stall in a biased state that gives
# misleading companion_goal guidance.
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

# Compute rep statistics at a logging step.
function rep_stats(rep, goal, n, ngbits)
    n_zero = 0; n_one = 0; n_nan = 0
    for k in 1:n
        v = rep[k]
        if isnan(v); n_nan += 1
        elseif v == 0.0; n_zero += 1
        else; n_one += 1
        end
    end
    rep_bias = (n_zero + n_one) > 0 ? n_one / (n_zero + n_one) : NaN  # fraction of defined entries that are 1

    n_complete = 0; n_correct = 0
    for j in 1:ngbits
        k1 = 2j - 1; k2 = 2j
        (isnan(rep[k1]) || isnan(rep[k2])) && continue
        n_complete += 1
        xor_rep = Int(rep[k1]) != Int(rep[k2]) ? 1 : 0
        n_correct += (xor_rep == goal[j])
    end
    pair_completeness = n_complete / ngbits
    cond_accuracy     = n_complete > 0 ? n_correct / n_complete : NaN

    return rep_bias, pair_completeness, cond_accuracy
end

function run_trajectory(goal, nqubit, nstate, nsteps, alpha; seed=42, log_interval=10_000)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2
    min_delta  = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

    steps_log       = Int[]
    acc_log         = Float64[]
    rep_bias_log    = Float64[]
    pair_comp_log   = Float64[]
    cond_acc_log    = Float64[]

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fp_packed; rng=rng)
    current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc         = current_acc
    last_improvement = 0

    function log_step(step)
        push!(steps_log, step)
        push!(acc_log, current_acc)
        rb, pc, ca = rep_stats(rep, goal, n, ngbits)
        push!(rep_bias_log, rb)
        push!(pair_comp_log, pc)
        push!(cond_acc_log, ca)
    end

    log_step(0)

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

        step % log_interval == 0 && log_step(step)
        step > stag_window && (step - last_improvement) >= stag_window && break
    end

    return steps_log, acc_log, rep_bias_log, pair_comp_log, cond_acc_log
end

function main()
    nqubit        = 6
    nstate        = 40
    nsteps        = 500_000
    alpha         = 0.99999
    base_seed     = 42
    nseeds        = 5
    log_interval  = 10_000
    entropy_targets = [0.0, 0.5, 1.0]

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    @printf("Rep convergence trajectory: nqubit=%d  nstate=%d  nsteps=%d\n\n", nqubit, nstate, nsteps)
    @printf("Columns: step | acc | rep_bias (frac 1s in defined rep) | pair_completeness | cond_accuracy\n\n")

    for H_target in entropy_targets
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng   = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal  = goal_from_hamming(k, ngbits, rng)

        @printf("=== H=%.4f (k=%d) ===\n", H_act, k)
        @printf("%-8s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
                "seed", "step", "acc", "rep_bias", "pair_comp", "cond_acc")
        println("-"^62)

        for seed in base_seed:(base_seed + nseeds - 1)
            steps, accs, biases, comps, cond_accs = run_trajectory(
                goal, nqubit, nstate, nsteps, alpha; seed=seed, log_interval=log_interval)

            for i in eachindex(steps)
                @printf("%-8d  %-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
                        seed, steps[i], accs[i],
                        isnan(biases[i]) ? -1.0 : biases[i],
                        comps[i],
                        isnan(cond_accs[i]) ? -1.0 : cond_accs[i])
            end
            println()
            flush(stdout)
        end
    end
end

main()
