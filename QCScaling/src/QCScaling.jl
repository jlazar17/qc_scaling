module QCScaling

export ParityOperator

using ProgressBars
using StatsBase

include("./parity_observable.jl")
include("./contexts.jl")
include("./states.jl")
include("./fingerprint.jl")

function parity(state::PseudoGHZState, measurement_po::ParityOperator) :: Float64
    beta_diff = mod.(state.generator.βs - measurement_po.βs, 3)
    J = 1 .- mod.((beta_diff .- 1), 2)
    if all(beta_diff .== 0)
        return mod(sum(state.alphas), 2)
    elseif any(beta_diff .== 0)
        return 0.5
    elseif mod(sum(J) + state.theta_s, 2) == 1
        return 0.5
    else
        return mod(state.theta_z + (sum(J) + state.theta_s) // 2 + sum(J[1:end-1] .* state.alphas), 2)
    end
end

function parity(state::PseudoGHZState, cxt::Context) :: Vector{Float64}
    parity.(Ref(state), cxt)
end

function calculate_representation(
    states::Vector{PseudoGHZState},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    nqubit = length(first(states).generator)
    representation = zeros(3 ^ nqubit)
    ctr = zeros(Int, 3 ^ nqubit)
    for state in states
        base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt
        cxt = Context(state.generator, base_cxt)
        idxs = map(po->po.index, cxt.pos)
        representation[idxs] .+= parity(state, cxt)
        ctr[idxs] .+= 1
    end
    rep = representation ./ ctr
    rep[rep.==0.5] .= NaN
    return round.(rep)
end

function score(
    states::Vector{PseudoGHZState},
    rep::Vector,
    goal::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context;
    track=true
)
    scores = Int[]
    itr = states
    if track
        itr = ProgressBar(itr)
    end
    for state in itr
        push!(scores, score(state, rep, goal, even_base_cxt, odd_base_cxt))
    end
    return scores
end

function score(
    states::Vector{PseudoGHZState},
    goal::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    rep = calculate_representation(states, even_base_cxt, odd_base_cxt)
    scores = score(states, rep, goal, even_base_cxt, odd_base_cxt)
    return scores
end

function score(
    state::PseudoGHZState,
    rep::Vector{Float64},
    goal::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    nqubit = length(state.generator)
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))

    base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt
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
        reduced_idx = (po.index+1) ÷ 2
        
        if reduced_idx in undefined_idxs
            continue
        end
        companion = po.index % 2==1 ? 1 : -1
        p = parity(state, po)
        @assert p in [0,1]
        predicted_bit = abs(p - rep[po.index + companion])
        diff = predicted_bit==goal[reduced_idx] ? 1 : -1
        score += diff
    end
    return score
end

function get_new_generators(states::Vector, rep::Vector, base_even, base_odd, nnew)
    counter = zeros(length(rep))
    nqubit = length(first(base_even))
    pos = ParityOperator[]
    for state in states
        base_cxt = state.theta_s==0 ? base_even : base_odd
        idxs =  map(x->x.index, QCScaling.Context(state.generator, base_cxt))
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
    # Find what direction each po should move, if any
end

function companion_goal(cxt::Context, goal::Vector, rep::Vector)
    # broadcast accross all POs in context
    # [f(state) for state in context.states]
    # f(context)
end

function pick_new_alphas(cxt::Context, goal::Vector, rep::Vector)
    cg = companion_goal(cxt, goal, rep)
    # hamming_distance = cg - fingerprint
    # Pick alpha by which one is the closest.
    # Down the line, we could explore reordering the fingerprint so that
    # close parities are nearby and we don't need to scan all of them
    # idx = argmin(hamming_distance)
    # Could also do non-deterministic sampling with hamming_distance
    # return alphas[idx]
end

end # module QCScaling
