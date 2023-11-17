module QCScaling

export ParityOperator

include("./parity_observable.jl")
include("./contexts.jl")
include("./states.jl")

function parity(state::PseudoGHZState, measurement_po::ParityOperator) :: Float64
    po_diff = state.generator - measurement_po
    J = 1 .- mod.((po_diff.βs .- 1), 2)
    if all(po_diff.βs .== 0)
        return mod(sum(state.alphas), 2)
    elseif any(po_diff.βs .== 0)
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

function calculate_representation(states::Vector{PseudoGHZState}, even_base_cxt::Context, odd_base_cxt::Context)
    nqubit = length(first(states).generator)
    representation = zeros(3 ^ nqubit)
    ctr = zeros(3 ^ nqubit)
    for state in states
        base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt
        cxt = Context(state.generator, base_cxt)
        idxs = [po.index for po in cxt]
        representation[idxs] .+= parity(state, cxt)
        ctr[idxs] .+= 1
    end
    return round.(representation ./ ctr)
end

function score(
    state::PseudoGHZState,
    rep::Vector{Int},
    goal_representation::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    nqubit = length(first(states).generator)
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))
    base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt

    cxt = Context(state.generator, base_cxt)
    score = 0
    for po in cxt
        if ((po.index+1) ÷ 2) in undefined_idxs || po.index==3^nqubit
            continue
        end
        companion = 1
        if po.index // 2!=po.index / 2
            companion = -1
        end
        p = parity(state, po)
        if p==0.5
            continue
        end
        predicted_bit = abs( - rep[po.index + companion])
        diff = predicted_bit==goal_representation[(po.index+1) ÷ 2] ? 1 : -1
        score += diff
    end
end
    

function score_states(
    states::Vector{PseudoGHZState},
    goal_representation::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    nqubit = length(first(states).generator)
    rep = calculate_representation(states, even_base_cxt, odd_base_cxt)
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))
    scores = Int[]
    for state in states
        score = 0
        base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt
        cxt = Context(state.generator, base_cxt)
        for po in cxt
            if ((po.index+1) ÷ 2) in undefined_idxs || po.index==3^nqubit
                continue
            end
            companion = 1
            if po.index // 2!=po.index / 2
                companion = -1
            end
            p = parity(state, po)
            if p==0.5
                continue
            end
            predicted_bit = abs( - rep[po.index + companion])
            diff = predicted_bit==goal_representation[(po.index+1) ÷ 2] ? 1 : -1
            score += diff
        end
        push!(scores, score)
    end
    return scores
end

end # module QCScaling
