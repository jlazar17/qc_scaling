struct Context
    pos::Vector{ParityOperator}
    parity::Int
    function Context(pos, parity)
        nqubit = length(first(pos))
        @assert parity in [0, 1]
        @assert length(pos)==(2^(nqubit -1) + 1) "Not the right number of POs"
        return new(pos, parity)
    end
end

function Context(generator::ParityOperator, base_cxt::Context)
    pos = [generator + po for po in base_cxt.pos]
    return Context(pos, base_cxt.parity)
end

struct ContextMaster
    base_even::Context
    base_odd::Context
    nqubit::Int
    function ContextMaster(base_even, base_odd, nqubit)
        @assert length(base_even.pos)==(2^(nqubit -1) + 1) "Not the right number of POs"
        @assert length(base_odd.pos)==(2^(nqubit -1) + 1) "Not the right number of POs"
        return new(base_even, base_odd, nqubit)
    end
end

function ContextMaster(nqubit::Int)
    base_even = generate_base_context(nqubit, 0)
    base_odd = generate_base_context(nqubit, 1)
    return ContextMaster(base_even, base_odd, nqubit)
end

Base.iterate(cxt::Context, idx::Int) = iterate(cxt.pos, idx)
Base.iterate(cxt::Context) = iterate(cxt.pos)
Base.length(cxt::Context) = length(cxt.pos)

function to_index(cxt::Context)
    return [po.index for po in cxt]
end

function to_ternary(x)
    if x ==0
        return [0]
    end
    pow = log(3, x)
    if abs(pow - round(pow)) < 1e-8
        pow = round(pow)
    end
    pow = Int(floor(pow))
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
    return Context(pos, evenoddbit)
end
