using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Shared rep helpers
# ---------------------------------------------------------------------------

function apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

function rep_accuracy(rep_sum, rep_ctr, goal)
    s = 0
    @inbounds for i in eachindex(goal)
        i1 = 2i - 1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1] / c1
        p2 = rep_sum[i2] / c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function _update_rep_cache!(rep_sum, rep_ctr, state, cxt_master, sign)
    apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
end

function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    _update_rep_cache!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    _update_rep_cache!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

# ---------------------------------------------------------------------------
# Improved optimizer (stripped of HDF5 I/O for timing purposes)
# ---------------------------------------------------------------------------

function run_improved(goal, nqubit, nstate, niter; seed=42,
                      nreplace=1, p_mutate=0.3, n_same_tol=10)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]

    rep_sum, rep_ctr = zeros(Float64, 3^nqubit), zeros(Int, 3^nqubit)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    best_acc = accuracy(rep, goal)
    n_same   = 0
    last_acc = best_acc

    t0 = time()
    for _ in 1:niter
        scores  = QCScaling.score(states, rep, goal, cxt_master)
        sorter  = sortperm(scores)
        scores  = scores[sorter]
        states  = states[sorter]

        acc = accuracy(rep, goal)
        acc > best_acc && (best_acc = acc)

        n_same = (acc == last_acc) ? n_same + 1 : 0
        last_acc = acc

        if n_same <= n_same_tol
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace; replace=false)
            for which in whiches
                rplc = states[which]
                if rand() < p_mutate
                    new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(new_cxt.pos)
                    base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = new_cxt
                else
                    new_generator = rplc.generator
                    base_cxt      = rplc.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = QCScaling.Context(new_generator, base_cxt)
                end
                alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                new_state = QCScaling.PseudoGHZState(alphas..., new_generator)
                replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
            end
        else
            n_same = 0
            for _ in 1:nreplace
                replace_idx   = rand(1:length(states))
                new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                new_generator = first(new_cxt.pos)
                base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                alphas        = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
                new_state     = QCScaling.PseudoGHZState(alphas..., new_generator)
                replace_state!(states, replace_idx, new_state, rep_sum, rep_ctr, cxt_master)
            end
        end

        rep = rep_from_cache(rep_sum, rep_ctr)
    end

    return best_acc, time() - t0
end

# ---------------------------------------------------------------------------
# SA (random state generation, no pool)
# ---------------------------------------------------------------------------

function random_state(nqubit, rng)
    theta_s = rand(rng, 0:1)
    theta_z = rand(rng, 0:1)
    alphas  = rand(rng, Bool, nqubit - 1)
    gen_idx = rand(rng, 0:3^nqubit - 1)
    return QCScaling.PseudoGHZState(theta_s, theta_z, alphas, QCScaling.ParityOperator(gen_idx, nqubit))
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, goal, nqubit, cxt_master;
                      target_rate=0.8, n_samples=500, rng)
    nstate      = length(ensemble)
    current_acc = rep_accuracy(rep_sum, rep_ctr, goal)
    bad_deltas  = Float64[]
    for _ in 1:n_samples
        which     = rand(rng, 1:nstate)
        new_state = random_state(nqubit, rng)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, new_state,       cxt_master,  1)
        new_acc = rep_accuracy(rep_sum, rep_ctr, goal)
        apply_state!(rep_sum, rep_ctr, new_state,       cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        delta = new_acc - current_acc
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

function run_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=5, seed=42)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    best_acc   = -Inf

    t0 = time()
    for _ in 1:n_restarts
        ensemble = [random_state(nqubit, rng) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n)
        rep_ctr  = zeros(Int,     n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end

        T            = calibrate_T0(ensemble, rep_sum, rep_ctr, goal, nqubit, cxt_master; rng=rng)
        current_acc  = rep_accuracy(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for _ in 1:nsteps
            which     = rand(rng, 1:nstate)
            new_state = random_state(nqubit, rng)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, new_state,       cxt_master,  1)
            new_acc = rep_accuracy(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc
            if delta >= 0 || rand(rng) < exp(delta / T)
                ensemble[which] = new_state
                current_acc     = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, new_state,       cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
            end
            T *= alpha
        end
        restart_best > best_acc && (best_acc = restart_best)
    end

    return best_acc, time() - t0
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit     = 8
    pzero      = 0.5
    ngbits     = (3^nqubit - 1) ÷ 2
    nstate     = Int(ceil(3^nqubit / 2^(nqubit - 1)))
    niter      = 5_000
    nsteps     = 200_000
    alpha      = 0.99995
    n_restarts = 5
    nseeds     = 5
    base_seed  = 42

    rng   = Random.MersenneTwister(base_seed)
    seeds = rand(rng, UInt32, nseeds)

    println("nqubit=$nqubit  nstate=$nstate  pzero=$pzero")
    println("improved: niter=$niter")
    println("SA:       nsteps=$nsteps  alpha=$alpha  n_restarts=$n_restarts")
    println()
    @printf("%-6s  %-10s  %-8s    %-10s  %-8s\n",
            "seed", "impr_acc", "impr_t(s)", "sa_acc", "sa_t(s)")
    println("-"^55)

    impr_accs, impr_times = Float64[], Float64[]
    sa_accs,   sa_times   = Float64[], Float64[]

    for seed in seeds
        rng_g = Random.MersenneTwister(seed)
        goal  = sample(rng_g, 0:1, Weights([pzero, 1 - pzero]), ngbits)

        impr_acc, impr_t = run_improved(goal, nqubit, nstate, niter; seed=Int(seed))
        sa_acc,   sa_t   = run_sa(goal, nqubit, nstate, nsteps, alpha;
                                   n_restarts=n_restarts, seed=Int(seed))

        push!(impr_accs, impr_acc); push!(impr_times, impr_t)
        push!(sa_accs,   sa_acc);   push!(sa_times,   sa_t)

        @printf("%-6d  %-10.4f  %-8.2f    %-10.4f  %-8.2f\n",
                seed, impr_acc, impr_t, sa_acc, sa_t)
        flush(stdout)
    end

    println("-"^55)
    @printf("%-6s  %-10.4f  %-8.2f    %-10.4f  %-8.2f\n",
            "median",
            median(impr_accs), median(impr_times),
            median(sa_accs),   median(sa_times))
end

main()
