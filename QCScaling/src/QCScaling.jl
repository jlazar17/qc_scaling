module QCScaling

export ParityOperator

using ProgressBars
using StatsBase

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
    return po.index % 2==1 ? po.index + 1 : po.index - 1
end

function calculate_preference(
    states::Vector{PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    nqubit = length(first(states))
    representation = zeros(3 ^ nqubit)
    ctr = zeros(Int, 3 ^ nqubit)
    for state in states
        base_cxt = state.theta_s==0 ? base_even : base_odd
        cxt = Context(state.generator, base_cxt)
        idxs = map(po->po.index, cxt.pos)
        representation[idxs] .+= parity(state, cxt)
        ctr[idxs] .+= 1
    end
    pref = representation ./ ctr
    return pref
end

function calculate_representation(
    states::Vector{PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    pref = calculate_preference(states, base_even, base_odd)
    pref[pref.==0.5] .= NaN
    rep = round.(pref)
    return rep
end

function calculate_representation(
    states::Vector{PseudoGHZState},
)
    nqubit = length(first(states))
    base_even = generate_base_context(nqubit, 0)
    base_odd = generate_base_context(nqubit, 1)
    return calculate_representation(states, base_even, base_odd)
end

function score(
    states::Vector{PseudoGHZState},
    rep::Vector,
    goal::Vector{Int},
    base_even::Context,
    base_odd::Context;
)
    scores = Int[]
    itr = states
    for state in itr
        push!(scores, score(state, rep, goal, base_even, base_odd))
    end
    return scores
end

function score(
    states::Vector{PseudoGHZState},
    goal::Vector{Int},
    base_even::Context,
    base_odd::Context
)
    rep = calculate_representation(states, base_even, base_odd)
    scores = score(states, rep, goal, base_even, base_odd)
    return scores
end

function score(
    state::PseudoGHZState,
    rep::Vector{Float64},
    goal::Vector{Int},
    base_even::Context,
    base_odd::Context
)
    nqubit = length(state.generator)
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))

    base_cxt = state.theta_s==0 ? base_even : base_odd
    cxt = Context(state.generator, base_cxt)
    p = sortperm(QCScaling.to_index(cxt))
    pos_sorted = cxt.pos[p]
    
    score = 0
    for po in cxt.pos[p]
        # We actually do not care about this since it is "extra"
        if po.index==3^nqubit
            continue
        end
        # Find where on goal this PO maps
        reduced_idx = get_goal_index(po)
        
        if reduced_idx in undefined_idxs
            continue
        end
        companion_idx = get_companion_index(po) 
        p = parity(state, po)
        @assert p in [0,1]
        predicted_bit = abs(p - rep[companion_idx])
        diff = predicted_bit==goal[reduced_idx] ? 1 : -1
        score += diff
    end
    return score
end

function get_new_generators(
    states::Vector,
    rep::Vector,
    base_even::Context,
    base_odd::Context,
    nnew::Int
)::Vector{ParityOperator}
    counter = zeros(Int, length(rep))
    nqubit = length(first(base_even))
    for state in states
        base_cxt = state.theta_s==0 ? base_even : base_odd
        idxs =  map(x->x.index, Context(state.generator, base_cxt))
        counter[idxs] .+= 1
    end
    weights = Weights(maximum(counter) .- counter)
    chosen_idxs = sample(0:length(rep)-1, weights, nnew, replace=false)
    pos = ParityOperator.(chosen_idxs, nqubit)
    return pos
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
    hamming_distance = abs.(fingerprint.a[:, base_cxt.parity+1, :, :] .- cg)
    nan_mask = isnan.(hamming_distance)
    hamming_distance[nan_mask] .= 0
    where = argmin(sum(hamming_distance, dims=1))
    return (base_cxt.parity, where[2]-1, idx_to_alphas(where[3]-1, length(cxt.pos[1])))
end

end # module QCScaling
