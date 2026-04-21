using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

# ---------------------------------------------------------------------------
# Per-state frustration vs entropy H
#
# For each (nqubit, H) pair, samples ngens random (generator, theta_s) pairs
# and computes the MAXIMUM achievable companion_goal agreement fraction over
# all (theta_z, alpha) configurations.
#
# This isolates inherent frustration: even with the best possible alpha
# config, what fraction of covered positions can a single state satisfy?
#
# For H=0, companion_goals are consistent (all want XOR=0), so the max
# fraction should be high. For H=1, companion_goals conflict within a
# context, imposing a ceiling on per-state contribution.
#
# Uses a warm rep (from a short SA run) to make companion_goal meaningful.
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
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

# Build a warm rep by running a short SA without caring about H
function warm_rep(nqubit, nstate, nsteps, goal, cxt_master, fp_packed, rng)
    n = 3^nqubit
    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n)
    rep_ctr  = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = 0.05
    for _ in 1:nsteps
        which     = rand(rng, 1:nstate)
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        ns        = QCScaling.PseudoGHZState(alphas..., generator)

        old = ensemble[which]
        apply_state!(rep_sum, rep_ctr, old, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,  cxt_master,  1)
        new_acc = sum(!isnan, rep_from_cache(rep_sum, rep_ctr)) / n  # proxy
        apply_state!(rep_sum, rep_ctr, ns,  cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, old, cxt_master,  1)

        # Accept unconditionally for warm-up
        apply_state!(rep_sum, rep_ctr, old, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,  cxt_master,  1)
        ensemble[which] = ns
        rep = rep_from_cache(rep_sum, rep_ctr)
    end
    return rep
end

# Compute max achievable companion_goal agreement fraction for a given generator
function max_cg_fraction(generator, theta_s, goal, rep, fp_plain, cxt_master, nqubit)
    base_cxt   = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt        = QCScaling.Context(generator, base_cxt)
    cg         = QCScaling.companion_goal(cxt, goal, rep)
    parity_idx = theta_s + 1
    fa         = fp_plain.a
    nalpha     = size(fa, 4)
    ntz        = size(fa, 3)

    n_defined = sum(!isnan, cg)
    n_defined == 0 && return NaN

    best_score = typemin(Int)
    @inbounds for tzi in 1:ntz, ai in 1:nalpha
        score = 0
        for pi in eachindex(cg)
            isnan(cg[pi]) && continue
            score += fa[pi, parity_idx, tzi, ai] == round(Int, cg[pi]) ? 1 : -1
        end
        score > best_score && (best_score = score)
    end

    return (best_score + n_defined) / (2 * n_defined)
end

function run_frustration(nqubit, nstate, ngens, entropy_targets, base_seed)
    rng        = Random.MersenneTwister(base_seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_plain   = QCScaling.Fingerprint(nqubit)
    fp_packed  = FingerprintPacked(fp_plain)
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    @printf("\n%s\n", "="^60)
    @printf("nqubit=%d  nstate=%d  ngens=%d\n", nqubit, nstate, ngens)
    @printf("%-8s  %-6s  %-12s  %-12s  %-12s\n",
            "H_act", "k", "mean_max", "med_max", "frac_lt50")
    println("-"^56)
    flush(stdout)

    for H_target in entropy_targets
        k     = k_from_entropy(Float64(H_target), ngbits)
        H_act = hamming_entropy(k, ngbits)

        g_rng = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal  = goal_from_hamming(k, ngbits, g_rng)

        # Build a warm rep via short SA
        rep = warm_rep(nqubit, nstate, 20_000, goal, cxt_master, fp_packed,
                       Random.MersenneTwister(base_seed + 1))

        fracs = Float64[]
        for _ in 1:ngens
            gen_idx   = rand(rng, 0:n-1)
            generator = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s   = rand(rng, 0:1)
            f = max_cg_fraction(generator, theta_s, goal, rep, fp_plain, cxt_master, nqubit)
            isnan(f) && continue
            push!(fracs, f)
        end

        @printf("%-8.4f  %-6d  %-12.4f  %-12.4f  %-12.4f\n",
                H_act, k,
                mean(fracs),
                median(fracs),
                mean(fracs .< 0.5))
        flush(stdout)
    end
end

function main()
    base_seed = 42
    ngens     = 5_000

    entropy_targets = [0.0, 0.25, 0.5, 0.75, 1.0]

    run_frustration(6, 40,  ngens, entropy_targets, base_seed)
    run_frustration(8, 147, ngens, entropy_targets, base_seed)
end

main()
