using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Warm-start experiment: does an H=0 solution help H=1 SA?
#
# The parity-flip symmetry argument implies equal numbers of high-accuracy
# ensembles exist for H=0 and H=1. If so, starting H=1 SA from a converged
# H=0 ensemble (rather than random init) should not systematically help or
# hurt — the optima are in different parts of ensemble space, but equally
# reachable.
#
# If warm-start from H=0 HURTS H=1 accuracy: the H=0 optima and H=1 optima
# are in different basins with a barrier between them, and SA from H=0 starts
# on the wrong side.
#
# If warm-start from H=0 HELPS: the H=0 optima and H=1 optima are nearby,
# suggesting the difficulty is in *finding* the basin, not the basin structure.
#
# We run three conditions per seed:
#   :cold_h1   — standard H=1 SA from random init
#   :warm_h0   — H=1 SA initialized from the final H=0 ensemble (same seed)
#   :cold_h0   — standard H=0 SA (reference accuracy ceiling)
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

# Run SA and return (best_acc, final_ensemble).
function run_sa(goal, nqubit, nstate, nsteps, alpha;
                init_ensemble=nothing, n_restarts=1, seed=42)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2
    min_delta  = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))
    best_acc   = -Inf
    best_ensemble = nothing

    for restart in 1:n_restarts
        if init_ensemble !== nothing && restart == 1
            ensemble = copy(init_ensemble)
        else
            ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        end

        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fp_packed; rng=rng)
        current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best     = current_acc
        last_improvement = 0

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
                if new_acc > restart_best + min_delta
                    restart_best     = new_acc
                    last_improvement = step
                end
                if new_acc > best_acc
                    best_acc      = new_acc
                    best_ensemble = copy(ensemble)
                end
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha

            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end

    return best_acc, best_ensemble
end

function main()
    nqubit   = 6
    nstate   = 40
    nsteps   = 500_000
    alpha    = 0.99999
    base_seed = 42
    nseeds   = 10

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    rng_seeds = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng_seeds, UInt32, nseeds))

    k_h0 = k_from_entropy(0.0, ngbits)
    k_h1 = k_from_entropy(1.0, ngbits)

    rng_g0 = Random.MersenneTwister(base_seed)
    rng_g1 = Random.MersenneTwister(base_seed + 1000)
    goals_h0 = [goal_from_hamming(k_h0, ngbits, rng_g0) for _ in 1:nseeds]
    goals_h1 = [goal_from_hamming(k_h1, ngbits, rng_g1) for _ in 1:nseeds]

    @printf("Warm-start experiment: nqubit=%d  nstate=%d  nseeds=%d\n\n", nqubit, nstate, nseeds)
    @printf("For each seed: run H=0 SA to convergence, then run H=1 SA from:\n")
    @printf("  cold: random init\n")
    @printf("  warm: final H=0 ensemble as starting point\n\n")
    @printf("%-6s  %-10s  %-12s  %-12s  %-10s\n",
            "seed", "acc_h0", "acc_h1_cold", "acc_h1_warm", "warm_delta")
    println("-"^56)
    flush(stdout)

    acc_h0_all   = Float64[]
    acc_cold_all = Float64[]
    acc_warm_all = Float64[]

    for (si, seed) in enumerate(seeds)
        # Run H=0 SA, save final ensemble
        acc_h0, ens_h0 = run_sa(goals_h0[si], nqubit, nstate, nsteps, alpha;
                                 n_restarts=3, seed=seed)

        # Cold-start H=1
        acc_h1_cold, _ = run_sa(goals_h1[si], nqubit, nstate, nsteps, alpha;
                                 n_restarts=3, seed=seed)

        # Warm-start H=1 from H=0 ensemble (1 restart only, from H=0 init)
        acc_h1_warm, _ = run_sa(goals_h1[si], nqubit, nstate, nsteps, alpha;
                                 init_ensemble=ens_h0, n_restarts=1, seed=seed)

        push!(acc_h0_all,   acc_h0)
        push!(acc_cold_all, acc_h1_cold)
        push!(acc_warm_all, acc_h1_warm)

        @printf("%-6d  %-10.4f  %-12.4f  %-12.4f  %-+10.4f\n",
                seed, acc_h0, acc_h1_cold, acc_h1_warm, acc_h1_warm - acc_h1_cold)
        flush(stdout)
    end

    println()
    @printf("Summary (median across seeds):\n")
    @printf("  acc_h0:       %.4f\n", median(acc_h0_all))
    @printf("  acc_h1_cold:  %.4f\n", median(acc_cold_all))
    @printf("  acc_h1_warm:  %.4f\n", median(acc_warm_all))
    @printf("  warm - cold:  %+.4f\n", median(acc_warm_all) - median(acc_cold_all))
    println()
    @printf("Interpretation:\n")
    @printf("  warm >> cold → H=1 optima near H=0 optima; difficulty is finding the basin\n")
    @printf("  warm ≈ cold  → H=0 and H=1 optima unrelated; no warm-start advantage\n")
    @printf("  warm << cold → H=0 ensemble actively misleads H=1 SA (wrong basin)\n")
end

main()
