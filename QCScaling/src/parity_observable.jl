struct ParityOperator
    βs::Vector{Int}
    index::Int
    function ParityOperator(βs::Vector{Int}, index::Int)
        @assert length(βs) % 2==0 "Odd number of betas: $(βs)"
        @assert all(βs .<= 2) "Some betas are not in <2> $(βs) $(index)"
        @assert index <= 3 ^ length(βs) "Index is too big"
        return new(βs, index)
    end
end

function to_index(βs::Vector{Int})
    #return sum([3^exp for exp in 0:length(βs)-1] .* βs) + 1
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
function ParityOperator(index::Int, nqubit::Int)
    βs = QCScaling.to_ternary(index)
    if length(βs) < nqubit
        βs = vcat(zeros(Int, nqubit - length(βs)), βs)
    end
    return ParityOperator(βs, index)
end

function ParityOperator(βs::Vector{Int})
    index = to_index(βs)
    return ParityOperator(βs, index)
end

Base.:+(po0::ParityOperator, po1::ParityOperator) = ParityOperator((po0.βs + po1.βs) .% 3)
Base.:-(po0::ParityOperator, po1::ParityOperator) = ParityOperator((po0.βs - po1.βs) .% 3)
Base.length(po::ParityOperator) = length(po.βs)
