struct Context
    pos::Vector{ParityOperator}
    function Context(pos)
        nqubit = length(first(pos))
        @assert length(pos)==(2^(nqubit -1) + 1) "Not the right number of POs"
        return new(pos)
    end
end

function Context(generator::ParityOperator, base_cxt::Context)
    pos = [generator + po for po in base_cxt.pos]
    return Context(pos)
end

Base.iterate(cxt::Context, idx::Int) = iterate(cxt.pos, idx)
Base.iterate(cxt::Context) = iterate(cxt.pos)
Base.length(cxt::Context) = length(cxt.pos)

function to_ternary(x)
    pow = Int(floor(log(3, x)))
    digs = Int[]
    while pow >= 0
        dig, x = divrem(x, 3^pow)
        pow -= 1
        push!(digs, dig)
    end
    return digs
end

function generate_base_context(nqubit::Int, evenoddbit::Int)
    @assert evenoddbit ∈ [0, 1]
    pos = ParityOperator[ParityOperator(fill(0, nqubit))]
    for idx in 1:3^nqubit - 1
        βs = to_ternary(idx)
        if length(βs) != nqubit || any(βs.==0) || ~(sum(βs[βs.==1]) % 2==evenoddbit)
            continue
        end
        push!(pos, ParityOperator(βs))
    end
    return Context(pos)
end
