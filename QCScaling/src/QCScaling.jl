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
    cxt_master::ContextMaster
)
    scores = Int[]
    itr = states
    for state in itr
        push!(scores, score(state, rep, goal, cxt_master))
    end
    return scores
end

function score(
    states::Vector{PseudoGHZState},
    goal::Vector{Int},
    cxt_master::ContextMaster
)
    rep = calculate_representation(states,cxt_master.base_even, cxt_master.base_odd)
    scores = score(states, rep, goal, cxt_master)
    return scores
end

function score(
    state::PseudoGHZState,
    rep::Vector{Float64},
    goal::Vector{Int},
    cxt_master::ContextMaster
)
    nqubit = cxt_master.nqubit
    # Something is undefined if either / both of the paired positions is NaN
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))

    base_cxt = ifelse(
        state.theta_s==0,
        cxt_master.base_even,
        cxt_master.base_odd
    )
    cxt = Context(state.generator, base_cxt)
    p = sortperm(QCScaling.to_index(cxt))
    pos_sorted = cxt.pos[p]
    
    score = 0
    for po in cxt.pos[p]
        # We actually do not care about this since it is "extra"
        # TODO check if this is an off by one error
        if po.index==3^nqubit
            continue
        end
        # Find where on goal this PO maps
        # This is the position in the 3^n/2 bitstring
        reduced_idx = get_goal_index(po)
        
        if reduced_idx in undefined_idxs
            continue
        end

        # This is the companion in the 3^n string
        companion_idx = get_companion_index(po) 
        p = parity(state, po)
        @assert p ∈ [0,1]
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
)::Vector{Context}
    counter = zeros(Int, 3^cxt_master.nqubit)
    counter[isnan.(rep)] .= 1
    ## Count how many times a po is covered
    #for state in states
    #    base_cxt = ifelse(
    #        state.theta_s==0,
    #        cxt_master.base_even,
    #        cxt_master.base_odd
    #    )
    #    idxs =  map(x->x.index, Context(state.generator, base_cxt))
    #    counter[idxs] .+= 1
    #end
    #@show counter
    #@show findall(counter.==0)
    # Look through every state and score them based on how much coverage there is
    scores = zeros(Int, 3^cxt_master.nqubit)
    # TODO look through even and odd
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
    #weights = Weights(maximum(scores) .- scores)
    #chosen_idxs = sample(0:3^cxt_master.nqubit-1, weights, nnew, replace=false)
    chosen_idxs = sortperm(-scores)[1:nnew] .- 1
    pos = ParityOperator.(chosen_idxs, cxt_master.nqubit)
    cxts = Context.(pos, Ref(base_cxt))
    return cxts
end

function get_new_contexts(states::Vector, cxt_master::ContextMaster, nnew::Int)::Vector{Context}
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
    # TODO Use the nice helper function you wrote for this
    # hamming_distance = abs.(fingerprint.a[cxt] .- cg)
    hamming_distance = abs.(fingerprint.a[:, base_cxt.parity+1, :, :] .- cg)
    nan_mask = isnan.(hamming_distance)
    hamming_distance[nan_mask] .= 0
    w = argmin(sum(hamming_distance, dims=1))
    return (base_cxt.parity, w[2]-1, idx_to_alphas(w[3]-1, length(cxt.pos[1])))
end

end # module QCScaling
