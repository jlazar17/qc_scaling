struct ParityOperator{N}
    βs::SVector{N, Int}
    index::Int
    function ParityOperator(βs::SVector{N, T}, index::Integer) where {N, T <: Integer}
        @assert all(<=(2), βs) "Some betas are not in <2> $(βs) $(index)"
        @assert index <= 3 ^ N "Index is too big"
        return new{N}(SVector{N, Int}(βs), index)
    end
end

function to_index(βs::SVector{N, T}) where {N, T <: Integer}
    idx = 1
    exp = 0
    for x in Iterators.reverse(βs)
        idx += x * 3 ^ exp
        exp += 1
    end
    return idx
end

function to_index(βs::Vector{T}) where T <: Integer
    idx = 1
    exp = 0
    for x in Iterators.reverse(βs)
        idx += x * 3 ^ exp
        exp += 1
    end
    return idx
end

function ParityOperator(index::Integer, nqubit::Integer)
    βs = QCScaling.to_ternary(index, Val(nqubit))
    return ParityOperator(βs, index)
end

function ParityOperator(βs::SVector{N, T}) where {N, T <: Integer}
    index = to_index(βs)
    return ParityOperator(βs, index)
end

function ParityOperator(βs::Vector{T}) where T <: Integer
    sv = SVector{length(βs), Int}(βs)
    return ParityOperator(sv)
end

Base.:+(po0::ParityOperator{N}, po1::ParityOperator{N}) where N = ParityOperator(SVector{N,Int}((po0.βs .+ po1.βs) .% 3))
Base.:-(po0::ParityOperator{N}, po1::ParityOperator{N}) where N = ParityOperator(SVector{N,Int}((po0.βs .- po1.βs) .% 3))
Base.length(po::ParityOperator{N}) where N = N
