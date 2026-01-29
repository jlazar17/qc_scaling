struct PseudoGHZState{M, N}
    theta_s::Int
    theta_z::Int
    alphas::SVector{M, Int}
    generator::ParityOperator{N}
    function PseudoGHZState(theta_s, theta_z, alphas::SVector{M, Int}, generator::ParityOperator{N}) where {M, N}
        @assert M + 1 == N
        @assert theta_s ∈ [0, 1]
        @assert theta_z ∈ [0, 1]
        @assert all(x -> x in (0, 1), alphas)
        return new{M, N}(theta_s, theta_z, alphas, generator)
    end
end

function PseudoGHZState(theta_s, theta_z, alphas::Vector{<:Integer}, generator::ParityOperator{N}) where N
    @assert length(alphas) + 1 == N
    sv = SVector{N-1, Int}(alphas)
    return PseudoGHZState(theta_s, theta_z, sv, generator)
end

function PseudoGHZState(v::Vector{T}) where T <: Integer
    nqubit = Int((length(v) - 1) / 2)
    theta_s = v[1]
    theta_z = v[2]
    alphas = SVector{nqubit-1, Int}(v[3:1+nqubit])
    po = ParityOperator(v[end-nqubit+1:end])
    return PseudoGHZState(theta_s, theta_z, alphas, po)
end

function random_state(nqubit::Integer)
    theta_s, theta_z = rand(0:1, 2)
    alphas = SVector{nqubit-1, Int}(rand(0:1, nqubit-1))
    po = ParityOperator(SVector{nqubit, Int}(rand(0:2, nqubit)))
    state = PseudoGHZState(theta_s, theta_z, alphas, po)
    return state
end

function random_state(nqubit::Integer, n::Integer)
    states = PseudoGHZState[]
    for _ in 1:n
        state = random_state(nqubit)
        push!(states, state)
    end
    return states
end

Base.length(state::PseudoGHZState{M, N}) where {M, N} = N
