using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Run SA at the H=1-optimal nstate and compare η to H=0 optimal nstate.
#
# From scaling study (scaling_study_adaptive.log):
#   n=6: H=0 peak at nstate=40 η=1.068, H=1 peak at nstate=97 η=0.495
#   n=8: H=0 peak at nstate=147 η=1.733, H=1 peak at nstate=599 η=0.364
#
# Question: does running H=1 at its own optimal nstate recover η comparable
# to H=0? If yes, the gap is purely a "needs more states" story.
# If no, something more fundamental is limiting H=1 efficiency.
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N; return -p * log2(p) - (1 - p) * log2(1 - p)
end

function binary_entropy(acc)
    (acc <= 0.0 || acc >= 1.0) && return 0.0
    return -acc * log2(acc) - (1 - acc) * log2(1 - acc)
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
    nstate = length(ensemble)
    n = 3^nqubit
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which     = rand(rng, 1:nstate)
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns        = QCScaling.PseudoGHZState(alphas..., generator)

        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
        delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

function run_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=3, seed=42)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2
    min_delta  = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))
    best_acc   = -Inf

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
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
            gen_idx   = rand(rng, 0:n-1)
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
                new_acc > best_acc && (best_acc = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha

            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

function run_experiment(nqubit, configs, nsteps, alpha, nseeds, base_seed)
    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    @printf("\n%s\n", "="^60)
    @printf("nqubit=%d  nsteps=%d  nseeds=%d\n", nqubit, nsteps, nseeds)
    @printf("%-8s  %-8s  %-8s  %-10s  %-10s  %-10s\n",
            "H", "nstate", "nqubit_res", "med_acc", "η", "note")
    println("-"^60)
    flush(stdout)

    for (H_target, nstate, note) in configs
        k     = H_target >= 1.0 ? ngbits ÷ 2 : round(Int, H_target * ngbits / 2) * 2 ÷ 2
        # Use exact k from entropy targets
        k = H_target <= 0.0 ? 0 : (H_target >= 1.0 ? ngbits ÷ 2 :
            begin
                _, idx = findmin(kk -> abs((-kk/ngbits*log2(kk/ngbits+1e-12) - (1-kk/ngbits)*log2(1-kk/ngbits+1e-12)) - H_target), 0:ngbits÷2)
                (0:ngbits÷2)[idx]
            end)
        # Simpler: just pass H directly
        H_target == 0.0 && (k = 0)
        H_target == 1.0 && (k = ngbits ÷ 2)

        H_act = hamming_entropy(k, ngbits)
        N_classical = ngbits
        N_quantum   = nqubit * nstate

        g_rng = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goals = [shuffle!(Random.MersenneTwister(base_seed + round(Int, H_target*1000) + si),
                          vcat(ones(Int,k), zeros(Int,ngbits-k))) for si in 1:nseeds]

        accs = [run_sa(goals[si], nqubit, nstate, nsteps, alpha;
                       n_restarts=3, seed=seeds[si]) for si in 1:nseeds]

        med_acc = median(accs)
        η = (1 - binary_entropy(med_acc)) * N_classical / N_quantum

        @printf("%-8.3f  %-8d  %-8d  %-10.4f  %-10.4f  %s\n",
                H_act, nstate, N_quantum, med_acc, η, note)
        flush(stdout)
    end
end

function main()
    alpha     = 0.99999
    base_seed = 42
    nseeds    = 5

    # n=6: compare H=0 at nstate=40 vs H=1 at nstate=40 (current) vs H=1 at nstate=97 (optimal)
    run_experiment(6, [
        (0.0, 40,  "H=0 @ optimal nstate"),
        (1.0, 40,  "H=1 @ H=0-optimal nstate"),
        (1.0, 97,  "H=1 @ H=1-optimal nstate"),
    ], 500_000, alpha, nseeds, base_seed)

    # n=8: compare H=0 at nstate=147 vs H=1 at nstate=147 (current) vs H=1 at nstate=599 (optimal)
    run_experiment(8, [
        (0.0, 147, "H=0 @ optimal nstate"),
        (1.0, 147, "H=1 @ H=0-optimal nstate"),
        (1.0, 599, "H=1 @ H=1-optimal nstate"),
    ], 2_000_000, alpha, nseeds, base_seed)
end

main()
