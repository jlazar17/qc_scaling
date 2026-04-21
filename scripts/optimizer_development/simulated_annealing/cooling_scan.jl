using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using Printf

# ---------------------------------------------------------------------------
# Incremental rep helpers (allocation-free)
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

# ---------------------------------------------------------------------------
# Random state generation (avoids pre-building a pool for large nqubit)
# ---------------------------------------------------------------------------

function random_state(nqubit, rng)
    theta_s = rand(rng, 0:1)
    theta_z = rand(rng, 0:1)
    alphas  = rand(rng, Bool, nqubit - 1)
    gen_idx = rand(rng, 0:3^nqubit - 1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    return QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator)
end

# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Simulated annealing (samples from full state space on the fly)
# ---------------------------------------------------------------------------

function simulated_annealing(goal, nqubit, nstate, nsteps, alpha;
                              n_restarts=5, cxt_master=nothing,
                              rng=Random.default_rng())
    isnothing(cxt_master) && (cxt_master = QCScaling.ContextMaster(nqubit))
    n = 3^nqubit

    best_acc = -Inf

    for _ in 1:n_restarts
        ensemble = [random_state(nqubit, rng) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n)
        rep_ctr  = zeros(Int,     n)
        for s in ensemble
            apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
        end

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

    return best_acc
end

# ---------------------------------------------------------------------------
# Cooling scan
#
# For each (alpha, nsteps) pair, runs SA with n_restarts restarts on
# n_goals random goals at pzero=0.5 and reports median accuracy.
# ---------------------------------------------------------------------------

function cooling_scan(nqubit;
                      pzero       = 0.5,
                      n_goals     = 3,
                      n_restarts  = 3,
                      alphas      = [0.999, 0.9995, 0.9999, 0.99995, 0.99999],
                      nsteps_vals = [200_000, 500_000, 1_000_000],
                      nstate_mult = 3,
                      seed        = 42)

    ngbits     = (3^nqubit - 1) ÷ 2
    nstate     = nstate_mult * Int(ceil(3^nqubit / 2^(nqubit - 1)))
    cxt_master = QCScaling.ContextMaster(nqubit)
    rng        = Random.MersenneTwister(seed)

    goals = [sample(rng, 0:1, Weights([pzero, 1 - pzero]), ngbits) for _ in 1:n_goals]

    println("nqubit=$nqubit  nstate=$nstate  pzero=$pzero  n_goals=$n_goals  n_restarts=$n_restarts")
    println()

    # Print header: columns = nsteps, rows = alpha
    @printf("%-10s", "alpha \\ ns")
    for ns in nsteps_vals
        @printf("  %8d", ns)
    end
    println()
    println("-" ^ (10 + 10 * length(nsteps_vals)))

    for alpha in alphas
        @printf("%-10.5f", alpha)
        for nsteps in nsteps_vals
            accs = Float64[]
            for goal in goals
                acc = simulated_annealing(
                    goal, nqubit, nstate, nsteps, alpha;
                    n_restarts=n_restarts, cxt_master=cxt_master,
                    rng=Random.MersenneTwister(rand(rng, UInt32)),
                )
                push!(accs, acc)
            end
            @printf("  %8.4f", median(accs))
            flush(stdout)
        end
        println()
    end
end

# ---------------------------------------------------------------------------
# Main: scan both nqubit=6 and nqubit=8
# ---------------------------------------------------------------------------

println("\n" * "="^60)
println("nqubit = 8 (extended nsteps)")
println("="^60)
cooling_scan(8;
    alphas      = [0.999, 0.9999, 0.99999],
    nsteps_vals = [1_000_000, 2_000_000, 5_000_000],
    n_goals     = 3,
    n_restarts  = 3,
    nstate_mult = 3,
)
