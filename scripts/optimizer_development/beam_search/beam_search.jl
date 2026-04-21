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
# Accuracy helper
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
# Incremental rep helpers
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

# Allocation-free accuracy computed directly from rep_sum/rep_ctr.
# Avoids materializing pref and rep arrays on every candidate evaluation.
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
# Beam search
#
# Maintains beam_width partial solutions simultaneously. At each step, every
# beam tries every pool state and the best beam_width (beam, state) pairs
# are kept as the new generation. Returns accuracy curve of the best beam
# at each step.
# ---------------------------------------------------------------------------

mutable struct Beam
    rep_sum  :: Vector{Float64}
    rep_ctr  :: Vector{Int}
    selected :: BitSet           # indices into pool already used
    acc      :: Float64
end

function beam_search(pool, goal, nqubit, max_nstate, beam_width; cxt_master=nothing)
    isnothing(cxt_master) && (cxt_master = QCScaling.ContextMaster(nqubit))
    n = 3^nqubit

    beams = [Beam(zeros(Float64, n), zeros(Int, n), BitSet(), -Inf)]

    acc_curve = Float64[]

    for step in 1:max_nstate
        # Collect all (beam_index, pool_index, new_acc) candidates
        candidates = Tuple{Int,Int,Float64}[]
        sizehint!(candidates, length(beams) * length(pool))

        for (bi, beam) in enumerate(beams)
            rs = beam.rep_sum
            rc = beam.rep_ctr
            for (pi, s) in enumerate(pool)
                pi in beam.selected && continue
                apply_state!(rs, rc, s, cxt_master, 1)
                a = rep_accuracy(rs, rc, goal)
                apply_state!(rs, rc, s, cxt_master, -1)
                push!(candidates, (bi, pi, a))
            end
        end

        # Partial sort: only order the top beam_width entries, O(N) vs O(N log N)
        partialsort!(candidates, 1:min(beam_width, length(candidates)), by=x -> x[3], rev=true)

        new_beams = Beam[]
        for (bi, pi, new_acc) in candidates
            length(new_beams) >= beam_width && break
            old = beams[bi]
            new_rs = copy(old.rep_sum)
            new_rc = copy(old.rep_ctr)
            apply_state!(new_rs, new_rc, pool[pi], cxt_master, 1)
            new_sel = copy(old.selected)
            push!(new_sel, pi)
            push!(new_beams, Beam(new_rs, new_rc, new_sel, new_acc))
        end

        beams = new_beams
        push!(acc_curve, maximum(b.acc for b in beams))
    end

    return acc_curve
end

# Greedy baseline (beam_width=1) using the same incremental rep logic
function greedy_search(pool, goal, nqubit, max_nstate; cxt_master=nothing)
    return beam_search(pool, goal, nqubit, max_nstate, 1; cxt_master=cxt_master)
end

# ---------------------------------------------------------------------------
# Main: validate on n=4 (full pool), then run n=6 (full pool)
# ---------------------------------------------------------------------------

function run_analysis(nqubit; beam_width=20, ngoals=20, max_nstate=nothing, seed=42,
                      pool_size=nothing, pzero_values=nothing, ngoals_per_pzero=nothing)
    ngbits     = (3^nqubit - 1) ÷ 2
    max_nstate = isnothing(max_nstate) ? min(40, 3 * Int(ceil(3^nqubit / 2^(nqubit-1)))) : max_nstate
    rng        = Random.MersenneTwister(seed)

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

    cxt_master   = QCScaling.ContextMaster(nqubit)
    isnothing(pzero_values) && (pzero_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    println("  beam_width=$beam_width, ngoals=$ngoals, max_nstate=$max_nstate, ngbits=$ngbits")
    println("  pzero_values=$pzero_values\n")

    checkpoints = unique([1, 3, 5, 10, max_nstate÷2, max_nstate])
    @printf("%-6s  %-8s  %s\n", "pzero", "method", join([@sprintf("ns=%-4d", c) for c in checkpoints], "  "))
    println("-" ^ (8 + 10 + 8 * length(checkpoints)))

    for pzero in pzero_values
        ng = isnothing(ngoals_per_pzero) ? ngoals : get(ngoals_per_pzero, pzero, ngoals)
        beam_mat   = Matrix{Float64}(undef, max_nstate, ng)
        greedy_mat = Matrix{Float64}(undef, max_nstate, ng)
        rng_g = Random.MersenneTwister(seed + round(Int, pzero * 1000))

        for gi in 1:ng
            goal = sample(rng_g, 0:1, Weights([pzero, 1 - pzero]), ngbits)
            beam_mat[:,   gi] = beam_search(pool, goal, nqubit, max_nstate, beam_width; cxt_master=cxt_master)
            greedy_mat[:, gi] = greedy_search(pool, goal, nqubit, max_nstate; cxt_master=cxt_master)
        end

        bmed  = median(beam_mat[end, :])
        gmed  = median(greedy_mat[end, :])
        delta = bmed - gmed

        beam_vals   = [median(beam_mat[c, :])   for c in checkpoints]
        greedy_vals = [median(greedy_mat[c, :]) for c in checkpoints]

        @printf("%-6.1f  %-8s  %s\n", pzero, "beam",   join([@sprintf("%.3f ", v) for v in beam_vals],   "  "))
        @printf("%-6s  %-8s  %s    Δ=%.4f\n", "", "greedy", join([@sprintf("%.3f ", v) for v in greedy_vals], "  "), delta)
        println()
    end
end

nqubit     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
beam_width = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
pool_size  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
run_analysis(
    nqubit;
    beam_width      = beam_width,
    pool_size       = pool_size,
    pzero_values    = [0.0, 0.3, 0.5],
    ngoals_per_pzero = Dict(0.0 => 1),   # deterministic goal, one run suffices
    ngoals          = 20,
    seed            = 42,
)
