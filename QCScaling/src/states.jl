struct PseudoGHZState
    theta_s::Int
    theta_z::Int
    alphas::Vector{Int}
    generator::ParityOperator
    function PseudoGHZState(theta_s, theta_z, alphas, generator)
        @assert (length(alphas) + 1 == length(generator))
        @assert theta_s ∈ [0, 1]
        @assert theta_z ∈ [0, 1]
        @assert all([x in [0, 1] for x in alphas])
        return new(theta_s, theta_z, alphas, generator)
    end
end

function random_state(nqubit::Int)
    theta_s, theta_z = rand(0:1, 2)
    alphas = rand(0:1, nqubit-1)
    po = QCScaling.ParityOperator(rand(0:2, nqubit))
    state = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, po)
    return state
end

function random_state(nqubit::Int, n::Int)
    states = QCScaling.PseudoGHZState[]
    for _ in 1:n
        state = random_state(nqubit)
        push!(states, state)
    end
    return states
end
