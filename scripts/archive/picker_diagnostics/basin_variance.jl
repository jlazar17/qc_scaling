using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Basin accessibility test
#
# Run many independent seeds with the SAME fixed goal at each entropy level.
# Vary only the random initialization and SA trajectory.
#
# If H=1 has higher variance in final accuracy than H=0, it suggests a
# narrower, harder-to-find basin: some seeds land near it and converge well,
# others don't find it at all. A bimodal accuracy distribution is the
# clearest possible signature.
#
# H=0: naturally fixed (all-zeros goal)
# H=0.5, H=1.0: one specific random goal held fixed across all seeds
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

function print_distribution(accs, ngbits)
    sorted = sort(accs)
    @printf("  mean=%.4f  std=%.4f  median=%.4f  min=%.4f  max=%.4f\n",
            mean(accs), std(accs), median(accs), minimum(accs), maximum(accs))

    # Histogram over accuracy bins
    edges = 0.5:0.05:1.01
    @printf("  Distribution (bin width 0.05):\n")
    for i in 1:(length(edges)-1)
        lo = edges[i]; hi = edges[i+1]
        count = sum(lo .<= accs .< hi)
        bar = repeat("#", count)
        @printf("    [%.2f, %.2f)  %3d  %s\n", lo, hi, count, bar)
    end

    # Fraction above key thresholds
    for thresh in [0.80, 0.85, 0.90, 0.95]
        @printf("  frac >= %.2f: %.3f\n", thresh, mean(accs .>= thresh))
    end
end

function main()
    nqubit     = 6
    nstate     = 40
    nsteps     = 500_000
    alpha      = 0.99999
    n_restarts = 3
    nseeds     = 100
    base_seed  = 42

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    entropy_targets = [0.0, 0.5, 1.0]

    # Generate fixed goals (one per H level, same across all seeds)
    goals = Dict{Float64, Vector{Int}}()
    for H in entropy_targets
        k = k_from_entropy(H, ngbits)
        g_rng = Random.MersenneTwister(base_seed + round(Int, H * 1000))
        goals[H] = goal_from_hamming(k, ngbits, g_rng)
    end

    # Generate 100 independent seeds
    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    @printf("Basin variance test: nqubit=%d  nstate=%d  nseeds=%d\n", nqubit, nstate, nseeds)
    @printf("Fixed goal per H level; only the SA initialization and trajectory vary.\n")
    @printf("Testing whether H=1 shows higher variance (bimodal distribution) than H=0.\n\n")
    flush(stdout)

    for H in entropy_targets
        k     = k_from_entropy(H, ngbits)
        H_act = hamming_entropy(k, ngbits)
        goal  = goals[H]

        @printf("=== H=%.4f (k=%d) ===\n", H_act, k)
        flush(stdout)

        accs = Float64[]
        for (i, seed) in enumerate(seeds)
            acc = run_sa(goal, nqubit, nstate, nsteps, alpha;
                         n_restarts=n_restarts, seed=seed)
            push!(accs, acc)
            # Print each result as it arrives
            @printf("  seed %3d/%d  acc=%.4f\n", i, nseeds, acc)
            flush(stdout)
        end

        println()
        print_distribution(accs, ngbits)
        println()
        flush(stdout)
    end
end

main()
