module QCScaling

export ParityOperator

using ProgressBars
using StatsBase

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
    odd_base_cxt::Context
)
    scores = Int[]
    for state in ProgressBar(states)
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
    
    stepper = 1
    next_undefined = undefined_idxs[stepper]
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
    
function find_missing_βs(rep::Vector)
    βs = Vector{Int}[]
    nqubit = Int(round(log(3, length(rep))))
    for index in ProgressBar(findall(isnan.(rep)))
        if index==3^nqubit
            continue
        end
        β = QCScaling.to_ternary(index)
        if length(β) < nqubit
            β = vcat(zeros(Int, nqubit - length(β)), β)
        end
        push!(βs, β)
    end
    return βs
end

function good_β(missing_βs)
    nqubit = length(first(missing_βs))
    # preallocate an array of X,Y,Z for each qubit
    a = [[0,0,0] for _ in 1:nqubit]
    for missing_β in missing_βs
        for (idx, β) in enumerate(missing_β)
            a[idx][β+1] += 1
        end
    end
    return argmin.(a) .- 1
end

function get_new_generators(rep::Vector; nnew::Int=50, ntries::Int=200, nchange::Int=7)
    missing_βs = find_missing_βs(rep)
    nqubit = length(first(missing_βs))
    pos, scores = ParityOperator[], Int[]
    good_beta = good_β(missing_βs)
    for _ in 1:ntries
        dummy = copy(good_beta)
        dummy[sample(1:nqubit, nchange; replace=false)] .= sample(0:2, nchange)
        po = ParityOperator(dummy)
        score = score_parity_operator(po, missing_βs)
        if length(pos) >= nnew && score < minimum(scores)
            continue
        end
        push!(pos, po)
        push!(scores, score)
        p = sortperm(-1 .* scores)
        scores, pos = scores[p], pos[p]
        if length(scores) > nnew
            scores, pos = scores[1:nnew], pos[1:nnew]
        end
    end
    return scores, pos
end

function score_parity_operator(po::ParityOperator, missing_βs)
    score = 0
    for missing_β in missing_βs
        if any(po.βs.==missing_β)
            continue
        end
        score += 1
    end
    return score
end

end # module QCScaling
