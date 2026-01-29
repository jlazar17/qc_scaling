module QCScaling

export ParityOperator

using ProgressBars
using Statistics
using OhMyThreads
using StaticArrays

include("./parity_observable.jl")
include("./contexts.jl")
include("./states.jl")
include("./fingerprint.jl")
include("./parities.jl")

function get_goal_index(po::ParityOperator)
    goal_index = (po.index+1) ÷ 2
    return goal_index
end

function get_companion_index(po::ParityOperator)
    return isone(po.index % 2) ? po.index + 1 : po.index - 1
end

function calculate_preference(
    states::Vector{<:PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    nqubit = length(first(states))
    representation = zeros(3 ^ nqubit)
    ctr = zeros(Int, 3 ^ nqubit)
    for state in states
        base_cxt = state.theta_s==0 ? base_even : base_odd
        for base_po in base_cxt.pos
            derived_po = state.generator + base_po
            p = parity(state, derived_po)
            representation[derived_po.index] += p
            ctr[derived_po.index] += 1
        end
    end
    pref = representation ./ ctr
    return pref
end

function calculate_representation(
    states::Vector{<:PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    pref = calculate_preference(states, base_even, base_odd)
    pref[pref.==0.5] .= NaN
    rep = round.(pref)
    return rep
end

function calculate_representation(
    states::Vector{<:PseudoGHZState},
)
    nqubit = length(first(states))
    base_even = generate_base_context(nqubit, 0)
    base_odd = generate_base_context(nqubit, 1)
    return calculate_representation(states, base_even, base_odd)
end

function score(
    states::Vector{<:PseudoGHZState},
    rep::Vector,
    goal::Vector{Int},
    cxt_master::ContextMaster;
    multi_threading=true
)

    itr = states
    undefined_idxs = @views Set(findall(isnan, (rep[1:2:end-1] .- rep[2:2:end])))
    scores = OhMyThreads.@tasks for state in itr
        @set begin
            collect = true
            ntasks = multi_threading ? Threads.nthreads() : 1
        end
        score(state, rep, undefined_idxs, goal, cxt_master)
    end
    return scores
end

function score(
    states::Vector{<:PseudoGHZState},
    goal::Vector{Int},
    cxt_master::ContextMaster
)
    rep = calculate_representation(
        states,
        cxt_master.base_even,
        cxt_master.base_odd
    )
    scores = score(states, rep, goal, cxt_master)
    return scores
end

function score(
    state::PseudoGHZState,
    rep::Vector{Float64},
    undefined_idxs::Set,
    goal::Vector{Int},
    cxt_master::ContextMaster;
)
    nqubit = cxt_master.nqubit

    base_cxt = ifelse(
        state.theta_s==0,
        cxt_master.base_even,
        cxt_master.base_odd
    )

    score = 0
    tnq = 3^nqubit
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        if derived_po.index==tnq
            continue
        end
        reduced_idx = get_goal_index(derived_po)

        if reduced_idx in undefined_idxs
            continue
        end

        companion_idx = get_companion_index(derived_po)
        p = parity(state, derived_po)
        @assert p ∈ (0,1)
        predicted_bit = abs(p - rep[companion_idx])
        diff = predicted_bit==goal[reduced_idx] ? 1 : -1
        score += diff
    end
    return score
end

function get_new_contexts(
    states::Vector,
    rep::Vector,
    cxt_master::ContextMaster,
    nnew::Int
)
    counter = zeros(Int, 3^cxt_master.nqubit)
    counter[isnan.(rep)] .= 1
    scores = zeros(Int, 3^cxt_master.nqubit)
    base_cxt = ifelse(
        rand() > 0.5,
        cxt_master.base_even,
        cxt_master.base_odd
    )
    for idx in 1:3^cxt_master.nqubit
        generator = ParityOperator(idx-1, cxt_master.nqubit)
        cxt = Context(generator, base_cxt)
        idxs =  map(x->x.index, Context(generator, base_cxt))
        scores[idx] += sum(counter[idxs])
    end
    chosen_idxs = sortperm(-scores)[1:nnew] .- 1
    pos = ParityOperator.(chosen_idxs, cxt_master.nqubit)
    cxts = Context.(pos, Ref(base_cxt))
    return cxts
end

function get_new_contexts(states::Vector, cxt_master::ContextMaster, nnew::Int)
    rep = calculate_representation(states, cxt_master.base_even, cxt_master.base_odd)
    return get_new_contexts(states, rep, cxt_master, nnew)
end

"""
    Find the desired direction for a PO to move based on current representation
    [0.7, 0.3, ...] -> [1, 0, 1, 0, 1, 0, 1, NaN, 1]
"""
function companion_goal(po::ParityOperator, goal::Vector, rep::Vector)
    if po.index==length(rep)
        return NaN
    end
    companion_idx = po.index % 2==1 ? po.index + 1 : po.index - 1
    if isnan(rep[companion_idx])
        return NaN
    end
    goal_index = get_goal_index(po)
    cg = goal[goal_index]==1 ? 1 - rep[companion_idx] : rep[companion_idx]
    return cg
end

function companion_goal(cxt::Context, goal::Vector, rep::Vector)
    return companion_goal.(cxt.pos, Ref(goal), Ref(rep))
end

function pick_new_alphas(
    cxt::Context,
    goal::Vector,
    rep::Vector,
    fingerprint::Fingerprint,
    base_cxt::Context
)
    cg = companion_goal(cxt, goal, rep)
    fa = fingerprint.a
    parity_idx = base_cxt.parity + 1
    npos = size(fa, 1)
    ntz = size(fa, 3)
    nalpha = size(fa, 4)
    best_sum = typemax(Float64)
    best_tz = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            s = 0.0
            for pi in 1:npos
                c = cg[pi]
                isnan(c) && continue
                s += abs(fa[pi, parity_idx, tzi, ai] - c)
            end
            if s < best_sum
                best_sum = s
                best_tz = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

end # module QCScaling
