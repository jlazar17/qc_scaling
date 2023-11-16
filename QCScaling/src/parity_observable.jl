struct ParityOperator
    βs::Vector{Int}
    index::Int
    function ParityOperator(βs::Vector{Int}, index::Int)
        @assert length(βs) % 2==0 "Odd number of betas"
        @assert all(βs .<= 2) "Some betas are not in <2>"
        @assert index <= 3 ^ length(βs) "Index is too big"
        return new(βs, index)
    end
end

function to_index(βs::Vector{Int})
    idx = 1
    exp = 0
    for x in reverse(βs)
        idx += x * 3 ^ exp
        exp += 1
    end
    return idx
end

# Doing this basis change every time might involve some overhead.
# Be sure to check if that is killing us
function ParityOperator(βs::Vector{Int})
    index = to_index(βs)
    return ParityOperator(βs, index)
end

Base.:+(po0::ParityOperator, po1::ParityOperator) = ParityOperator((po0.βs + po1.βs) .% 3)
Base.:-(po0::ParityOperator, po1::ParityOperator) = ParityOperator((po0.βs - po1.βs) .% 3)
Base.length(po::ParityOperator) = length(po.βs)
