using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using Printf

# ---------------------------------------------------------------------------
# Enumerate every possible PseudoGHZ state for a given nqubit
# ---------------------------------------------------------------------------

function all_states(nqubit)
    states = QCScaling.PseudoGHZState[]
    for idx in 0:3^nqubit - 1
        generator = QCScaling.ParityOperator(idx, nqubit)
        for theta_s in 0:1
            for theta_z in 0:1
                for alpha_idx in 0:2^(nqubit-1) - 1
                    alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                    push!(states, QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator))
                end
            end
        end
    end
    return states
end

# ---------------------------------------------------------------------------
# Greedy accuracy maximization
#
# Given a pool of all states and a goal, greedily add states one at a time,
# each time picking the state that most improves accuracy. Returns the
# accuracy curve as a function of nstate.
# ---------------------------------------------------------------------------

function greedy_max_accuracy(all_states_pool, goal, nqubit, max_nstate)
    cxt_master = QCScaling.ContextMaster(nqubit)

    # Start with the single state that gives the highest accuracy
    best_acc   = -Inf
    best_state = first(all_states_pool)
    for s in all_states_pool
        rep = QCScaling.calculate_representation([s])
        acc = accuracy(rep, goal)
        if acc > best_acc
            best_acc   = acc
            best_state = s
        end
    end

    selected = [best_state]
    acc_curve = [best_acc]

    for _ in 2:max_nstate
        best_acc_new = -Inf
        best_s_new   = nothing
        rep_current  = QCScaling.calculate_representation(selected)
        for s in all_states_pool
            rep = QCScaling.calculate_representation(vcat(selected, [s]))
            acc = accuracy(rep, goal)
            if acc > best_acc_new
                best_acc_new = acc
                best_s_new   = s
            end
        end
        push!(selected, best_s_new)
        push!(acc_curve, best_acc_new)
    end

    return acc_curve
end

function accuracy(rep, goal)
    pred = abs.(rep[1:2:end-2] .- rep[2:2:end-1])
    s = 0
    for (x, y) in zip(pred, goal)
        isnan(x) && continue
        s += (x == y) ? 1 : 0
    end
    return s / length(goal)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit     = 4
    ngoals     = 50      # random goals per pzero value to average over
    max_nstate = 20
    pzero_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ngbits = (3^nqubit - 1) ÷ 2

    println("Enumerating all states for nqubit=$nqubit...")
    pool = all_states(nqubit)
    println("  $(length(pool)) states in pool")

    println("\nGreedy accuracy ceiling vs pzero (averaged over $ngoals goals each):\n")
    println("pzero  | max_acc (at nstate=$max_nstate) | mean_acc_curve")
    println("-------|-------------------------------|---------------")

    rng = Random.MersenneTwister(42)
    results = Dict{Float64, Vector{Float64}}()

    for pzero in pzero_values
        final_accs = Float64[]
        for _ in 1:ngoals
            goal = sample(rng, 0:1, Weights([pzero, 1-pzero]), ngbits)
            curve = greedy_max_accuracy(pool, goal, nqubit, max_nstate)
            push!(final_accs, curve[end])
        end
        results[pzero] = final_accs
        @printf("%.1f    | %.4f ± %.4f\n",
                pzero, mean(final_accs), std(final_accs))
    end

    # Also print the curve at nstate = 1, 5, 10, 20 for pzero = 0.0, 0.5, 1.0
    println("\nDetailed curves for pzero ∈ {0.0, 0.5, 1.0}:")
    for pzero in [0.0, 0.5, 1.0]
        curves = Float64[]
        rng2 = Random.MersenneTwister(42)
        for _ in 1:ngoals
            goal  = sample(rng2, 0:1, Weights([pzero, 1-pzero]), ngbits)
            curve = greedy_max_accuracy(pool, goal, nqubit, max_nstate)
            push!(curves, curve[end])
        end
        # redo to get per-nstate medians
        all_curves = Matrix{Float64}(undef, max_nstate, ngoals)
        rng3 = Random.MersenneTwister(42)
        for gi in 1:ngoals
            goal = sample(rng3, 0:1, Weights([pzero, 1-pzero]), ngbits)
            all_curves[:, gi] = greedy_max_accuracy(pool, goal, nqubit, max_nstate)
        end
        med_curve = [median(all_curves[i, :]) for i in 1:max_nstate]
        println("pzero=$pzero: " * join([@sprintf("%.3f", v) for v in med_curve[[1,2,3,5,10,15,20]]], "  ") *
                "  (at nstate = 1,2,3,5,10,15,20)")
    end
end

main()
