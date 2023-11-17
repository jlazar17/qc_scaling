module QCScaling

export ParityOperator

using ProgressBars

include("./parity_observable.jl")
include("./contexts.jl")
include("./states.jl")

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
    states::Vector{PseudoGHZState},
    goal::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    rep = QCScaling.calculate_representation(states, even_base_cxt, odd_base_cxt)
    scores = Int[]
    for state in ProgressBar(states)
        push!(scores, score(state, rep, goal, even_base_cxt, odd_base_cxt))
    end
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
    score = 0
    for po in cxt
        if ((po.index+1) ÷ 2) in undefined_idxs || po.index==3^nqubit
            continue
        end
        companion = po.index % 2==0 ? 1 : -1
        p = parity(state, po)
        if p==0.5
            continue
        end
        predicted_bit = abs(p - rep[po.index + companion])
        diff = predicted_bit==goal[(po.index+1) ÷ 2] ? 1 : -1
        score += diff
    end
    return score
end
    

function score_states(
    states::Vector{PseudoGHZState},
    goal::Vector{Int},
    even_base_cxt::Context,
    odd_base_cxt::Context
)
    nqubit = length(first(states).generator)
    rep = calculate_representation(states, even_base_cxt, odd_base_cxt)
    undefined_idxs = findall(isnan, abs.(rep[1:2:end-1] .- rep[2:2:end]))
    scores = Int[]
    idx=1
    for state in states
        println(idx)
        idx+=1
        score = 0
        base_cxt = state.theta_s==0 ? even_base_cxt : odd_base_cxt
        cxt = Context(state.generator, base_cxt)
        for po in cxt
            if ((po.index+1) ÷ 2) in undefined_idxs || po.index==3^nqubit
                continue
            end
            companion = po.index % 2==0 ? 1 : -1
            p = parity(state, po)
            if p==0.5
                continue
            end

            predicted_bit = xor(Int(p), Int(rep[po.index + companion]))
            diff = predicted_bit==goal[(po.index+1) ÷ 2] ? 1 : -1
            score += diff
        end
        push!(scores, score)
    end
    return scores
end

end # module QCScaling
