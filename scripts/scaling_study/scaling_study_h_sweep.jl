using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Additional H values for nqubit=6 and nqubit=8.
# Existing scan has H=0.0, 0.5, 1.0.  This adds H=0.25, 0.625, 0.75, 0.875.
# Same adaptive grid logic and SA parameters as scaling_study_adaptive.jl.
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

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function efficiency(acc::Float64, nqubit::Int, nstate::Int)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    return (1 - binary_entropy(acc)) * n_classical / n_quantum
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

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns    = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
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

function run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=3, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2
    best_acc    = -Inf
    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint; rng=rng)
        current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best     = current_acc
        last_improvement = 0

        for step in 1:nsteps
            which     = rand(rng, 1:nstate)
            ns        = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
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

function evaluate_nstate(nstate, goals, nqubit, nsteps, alpha, n_restarts, seeds)
    accs = [run_goal_aware_sa(goals[si], nqubit, nstate, nsteps, alpha;
                              n_restarts=n_restarts, seed=seeds[si])
            for si in eachindex(seeds)]
    etas = [efficiency(a, nqubit, nstate) for a in accs]
    return accs, etas
end

function format_seeds(vals)
    join([@sprintf("%.4f", v) for v in vals], " ")
end

function log_spaced_nstates(nstate_min, nstate_max, n_coarse)
    lo   = log(nstate_min); hi = log(nstate_max)
    vals = unique(round.(Int, exp.(range(lo, hi, length=n_coarse))))
    return filter(v -> v >= nstate_min && v <= nstate_max, vals)
end

function adaptive_grid(goals, nqubit, base_nstate, nsteps, alpha,
                       n_restarts, seeds, nstate_max; n_coarse=12, n_refine=2)
    coarse  = log_spaced_nstates(base_nstate, nstate_max, n_coarse)
    results = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    @printf("  Phase 1: coarse grid %s\n", string(coarse))
    for ns in coarse
        accs, etas = evaluate_nstate(ns, goals, nqubit, nsteps, alpha, n_restarts, seeds)
        results[ns] = (accs, etas)
        @printf("    nstate=%-5d  acc_med=%.4f  η_med=%.4f  η_max=%.4f  η_seeds=[%s]\n",
                ns, median(accs), median(etas), maximum(etas), format_seeds(etas))
        flush(stdout)
    end

    for phase in 1:n_refine
        sorted_ns = sort(collect(keys(results)))
        max_etas  = [maximum(results[ns][2]) for ns in sorted_ns]
        peak_idx  = argmax(max_etas)
        lo = peak_idx > 1                ? sorted_ns[peak_idx-1] : sorted_ns[peak_idx]
        hi = peak_idx < length(max_etas) ? sorted_ns[peak_idx+1] : sorted_ns[peak_idx]
        new_ns = Int[]
        for mid in [(lo + sorted_ns[peak_idx]) ÷ 2, (sorted_ns[peak_idx] + hi) ÷ 2]
            if mid >= base_nstate && mid <= nstate_max && !haskey(results, mid)
                push!(new_ns, mid)
            end
        end
        isempty(new_ns) && break
        @printf("  Phase %d refinement: bisecting around peak (nstate=%d, η_max=%.4f) → add %s\n",
                phase+1, sorted_ns[peak_idx], max_etas[peak_idx], string(new_ns))
        for ns in new_ns
            accs, etas = evaluate_nstate(ns, goals, nqubit, nsteps, alpha, n_restarts, seeds)
            results[ns] = (accs, etas)
            @printf("    nstate=%-5d  acc_med=%.4f  η_med=%.4f  η_max=%.4f  η_seeds=[%s]\n",
                    ns, median(accs), median(etas), maximum(etas), format_seeds(etas))
            flush(stdout)
        end
    end

    sorted_ns = sort(collect(keys(results)))
    return [(ns, median(results[ns][1]), maximum(results[ns][2])) for ns in sorted_ns]
end

function main()
    base_seed  = 42
    nseeds     = 20
    n_restarts = 3
    n_coarse   = 12
    n_refine   = 2

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))
    println("Seeds: $seeds\n")

    configs = [
        # (nqubit, base_nstate, nstate_max, nsteps, alpha)
        (6, 23,  460,  500_000,   0.99999),
        (8, 52, 1340, 2_000_000,  0.99999),
    ]
    new_H_targets = [0.25, 0.625, 0.75, 0.875]

    for (nqubit, base_nstate, nstate_max, nsteps, alpha) in configs
        ngbits = (3^nqubit - 1) ÷ 2
        println("="^70)
        @printf("nqubit=%d  base_nstate=%d  nstate_max=%d  nsteps=%d\n",
                nqubit, base_nstate, nstate_max, nsteps)

        for H_target in new_H_targets
            k     = k_from_entropy(H_target, ngbits)
            H_act = hamming_entropy(k, ngbits)
            rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

            println()
            @printf("H=%.3f (k=%d, H_act=%.4f)\n", H_target, k, H_act)

            pts = adaptive_grid(goals, nqubit, base_nstate, nsteps, alpha,
                                n_restarts, seeds, nstate_max;
                                n_coarse=n_coarse, n_refine=n_refine)

            best = argmax(p -> p[3], pts)
            @printf("  → η peak at nstate=%d: acc_med=%.4f  η_max=%.4f\n\n",
                    best[1], best[2], best[3])
            flush(stdout)
        end
    end
end

main()
