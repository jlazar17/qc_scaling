using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using Printf

# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

function all_states(nqubit)
    states = QCScaling.PseudoGHZState[]
    for idx in 0:3^nqubit - 1
        generator = QCScaling.ParityOperator(idx, nqubit)
        for theta_s in 0:1, theta_z in 0:1
            for alpha_idx in 0:2^(nqubit-1) - 1
                alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                push!(states, QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator))
            end
        end
    end
    return states
end

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
        i1 = 2i - 1
        i2 = 2i
        c1 = rep_ctr[i1];  c1 == 0 && continue
        c2 = rep_ctr[i2];  c2 == 0 && continue
        p1 = rep_sum[i1] / c1
        p2 = rep_sum[i2] / c2
        (p1 == 0.5 || p2 == 0.5) && continue
        x = abs(round(p1) - round(p2))
        s += (x == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

# ---------------------------------------------------------------------------
# Temperature calibration
#
# Samples random single-state swaps to estimate the typical magnitude of
# accuracy loss for bad moves. Sets T0 so that bad moves of that magnitude
# are accepted with `target_rate` probability initially.
# ---------------------------------------------------------------------------

function calibrate_T0(pool, ensemble, rep_sum, rep_ctr, goal, cxt_master;
                      target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]

    for _ in 1:n_samples
        which     = rand(rng, 1:nstate)
        new_state = rand(rng, pool)

        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, new_state,       cxt_master,  1)
        new_acc = rep_accuracy(rep_sum, rep_ctr, goal)
        apply_state!(rep_sum, rep_ctr, new_state,       cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)

        delta = new_acc - current_acc
        delta < 0 && push!(bad_deltas, abs(delta))
    end

    isempty(bad_deltas) && return 0.1   # already at a local maximum; low T is fine
    mean_delta = mean(bad_deltas)
    # exp(-mean_delta / T0) = target_rate  =>  T0 = -mean_delta / log(target_rate)
    return -mean_delta / log(target_rate)
end

# ---------------------------------------------------------------------------
# Simulated annealing
#
# Fixes ensemble size to `nstate` and optimizes the set of states via
# single-state swaps with a geometric cooling schedule. Returns the best
# accuracy found across all restarts.
# ---------------------------------------------------------------------------

function simulated_annealing(pool, goal, nqubit, nstate;
                              nsteps=200_000, alpha=nothing,
                              n_restarts=5, target_T_ratio=1e-3,
                              checkpoint_every=nothing,
                              cxt_master=nothing, rng=Random.default_rng())
    isnothing(cxt_master) && (cxt_master = QCScaling.ContextMaster(nqubit))
    n = 3^nqubit

    # Default alpha: cool by target_T_ratio over nsteps
    isnothing(alpha) && (alpha = exp(log(target_T_ratio) / nsteps))

    tracking  = !isnothing(checkpoint_every)
    total_steps = nsteps * n_restarts
    n_checkpoints = tracking ? div(total_steps, checkpoint_every) : 0
    # best-so-far curve across all restarts (step index = global step count)
    acc_curve = tracking ? fill(-Inf, n_checkpoints) : Float64[]
    best_acc  = -Inf
    global_step = 0

    for _ in 1:n_restarts
        # Random initialization
        ensemble = Vector{QCScaling.PseudoGHZState}(sample(rng, pool, nstate; replace=false))
        rep_sum  = zeros(Float64, n)
        rep_ctr  = zeros(Int,     n)
        for s in ensemble
            apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
        end

        T = calibrate_T0(pool, ensemble, rep_sum, rep_ctr, goal, cxt_master; rng=rng)
        current_acc  = rep_accuracy(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for step in 1:nsteps
            which     = rand(rng, 1:nstate)
            new_state = rand(rng, pool)

            # Tentative swap
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, new_state,       cxt_master,  1)
            new_acc = rep_accuracy(rep_sum, rep_ctr, goal)

            delta = new_acc - current_acc
            if delta >= 0 || rand(rng) < exp(delta / T)
                ensemble[which] = new_state
                current_acc     = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                # Undo
                apply_state!(rep_sum, rep_ctr, new_state,       cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
            end

            T *= alpha

            if tracking
                global_step += 1
                ci = div(global_step - 1, checkpoint_every) + 1
                if step % checkpoint_every == 0 && ci <= n_checkpoints
                    best_acc = max(best_acc, restart_best)
                    acc_curve[ci] = best_acc
                end
            end
        end

        restart_best > best_acc && (best_acc = restart_best)
    end

    return tracking ? (best_acc, acc_curve) : best_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function run_analysis(nqubit;
                      pzero_values = [0.0, 0.3, 0.5],
                      nstate_values = nothing,
                      ngoals        = 20,
                      nsteps        = 200_000,
                      n_restarts    = 5,
                      pool_size     = nothing,
                      seed          = 42)

    ngbits        = (3^nqubit - 1) ÷ 2
    base_nstate   = 3^nqubit / 2^(nqubit - 1)
    isnothing(nstate_values) && (nstate_values = unique(Int.(ceil.(base_nstate .* [1, 2, 4, 6, 8]))))
    rng           = Random.MersenneTwister(seed)

    println("Building state pool for nqubit=$nqubit...")
    full_pool = all_states(nqubit)
    pool = if isnothing(pool_size) || pool_size >= length(full_pool)
        println("  Using full pool: $(length(full_pool)) states")
        full_pool
    else
        sampled = sample(rng, full_pool, pool_size; replace=false)
        println("  Sampled $(length(sampled)) / $(length(full_pool)) states")
        sampled
    end

    cxt_master = QCScaling.ContextMaster(nqubit)

    println("  nsteps=$nsteps, n_restarts=$n_restarts, ngoals=$ngoals")
    println("  nstate_values=$nstate_values")
    println("  pzero_values=$pzero_values\n")

    @printf("%-6s  %s\n", "pzero", join([@sprintf("ns=%-6d", ns) for ns in nstate_values], "  "))
    println("-" ^ (8 + 9 * length(nstate_values)))

    for pzero in pzero_values
        ng    = pzero == 0.0 || pzero == 1.0 ? 1 : ngoals
        rng_g = Random.MersenneTwister(seed + round(Int, pzero * 1000))
        goals = [sample(rng_g, 0:1, Weights([pzero, 1 - pzero]), ngbits) for _ in 1:ng]

        medians = Float64[]
        for nstate in nstate_values
            accs = Float64[]
            for goal in goals
                acc = simulated_annealing(
                    pool, goal, nqubit, nstate;
                    nsteps=nsteps, n_restarts=n_restarts,
                    cxt_master=cxt_master,
                    rng=Random.MersenneTwister(rand(rng_g, UInt32)),
                )
                push!(accs, acc)
            end
            push!(medians, median(accs))
            @printf("  pzero=%.1f  ns=%-4d  acc=%.4f\n", pzero, nstate, medians[end])
            flush(stdout)
        end
        @printf("%-6.1f  %s\n", pzero, join([@sprintf("%.4f  ", v) for v in medians], "  "))
    end
end

nqubit    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
pool_size = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing
run_analysis(nqubit; pool_size=pool_size)
