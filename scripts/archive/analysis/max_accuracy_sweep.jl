using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using Printf

# ---------------------------------------------------------------------------
# Accuracy helper (NaN predictions count as incorrect)
# ---------------------------------------------------------------------------

function accuracy(rep, goal)
    s = 0
    for i in eachindex(goal)
        x = abs(rep[2i-1] - rep[2i])
        isnan(x) && continue
        s += (x == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

# ---------------------------------------------------------------------------
# Incremental greedy selection
#
# Maintains rep_sum/rep_ctr, adds one state at a time picking whichever
# candidate from the pool most improves accuracy. O(pool × context_size)
# per greedy step rather than recomputing from scratch.
# ---------------------------------------------------------------------------

function greedy_max_accuracy(pool, goal, nqubit, max_nstate; rng=Random.default_rng())
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    rep_sum    = zeros(Float64, n)
    rep_ctr    = zeros(Int,     n)

    function add!(state, sign)
        base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        for base_po in base_cxt.pos
            derived_po = state.generator + base_po
            p = QCScaling.parity(state, derived_po)
            rep_sum[derived_po.index] += sign * p
            rep_ctr[derived_po.index] += sign
        end
    end

    function current_acc()
        pref = rep_sum ./ rep_ctr
        pref[pref .== 0.5] .= NaN
        rep = round.(pref)
        return accuracy(rep, goal)
    end

    acc_curve = Float64[]

    for step in 1:max_nstate
        best_acc   = -Inf
        best_state = first(pool)

        for s in pool
            add!(s,  1)
            a = current_acc()
            add!(s, -1)
            if a > best_acc
                best_acc   = a
                best_state = s
            end
        end

        add!(best_state, 1)
        push!(acc_curve, best_acc)
    end

    return acc_curve
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(nqubit; pool_size=nothing, ngoals=20, max_nstate=nothing, seed=42)
    ngbits       = (3^nqubit - 1) ÷ 2
    pzero_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    max_nstate   = isnothing(max_nstate) ? min(40, 3 * Int(ceil(3^nqubit / 2^(nqubit-1)))) : max_nstate
    rng          = Random.MersenneTwister(seed)

    # Build or sample pool
    println("Building state pool for nqubit=$nqubit...")
    full_pool = QCScaling.PseudoGHZState[]
    for idx in 0:3^nqubit - 1
        generator = QCScaling.ParityOperator(idx, nqubit)
        for theta_s in 0:1, theta_z in 0:1
            for alpha_idx in 0:2^(nqubit-1) - 1
                alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                push!(full_pool, QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator))
            end
        end
    end

    pool = if isnothing(pool_size) || pool_size >= length(full_pool)
        println("  Using full pool: $(length(full_pool)) states")
        full_pool
    else
        sampled = sample(rng, full_pool, pool_size; replace=false)
        println("  Sampled $(length(sampled)) / $(length(full_pool)) states")
        sampled
    end

    println("  ngoals=$ngoals per pzero, max_nstate=$max_nstate, ngbits=$ngbits\n")
    println("pzero  | ceiling (nstate=$max_nstate)")
    println("-------|----------------------")

    all_curves = Dict{Float64, Matrix{Float64}}()

    for pzero in pzero_values
        curves = Matrix{Float64}(undef, max_nstate, ngoals)
        rng_g  = Random.MersenneTwister(seed + round(Int, pzero * 1000))
        for gi in 1:ngoals
            goal = sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits)
            curves[:, gi] = greedy_max_accuracy(pool, goal, nqubit, max_nstate; rng=rng_g)
        end
        all_curves[pzero] = curves
        med = median(curves[end, :])
        std_ = std(curves[end, :])
        @printf("%.1f    | %.4f ± %.4f\n", pzero, med, std_)
    end

    println("\nMedian accuracy curves (nstate = 1, 3, 5, 10, $(max_nstate÷2), $max_nstate):")
    checkpoints = unique([1, 3, 5, 10, max_nstate÷2, max_nstate])
    @printf("%-6s  %s\n", "pzero", join([@sprintf("ns=%-4d", c) for c in checkpoints], "  "))
    for pzero in pzero_values
        vals = [median(all_curves[pzero][c, :]) for c in checkpoints]
        @printf("%-6.1f  %s\n", pzero, join([@sprintf("%.3f ", v) for v in vals], "  "))
    end
end

nqubit = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
pool_size = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (nqubit <= 4 ? nothing : 4000)
main(nqubit; pool_size=pool_size, ngoals=20, seed=42)
