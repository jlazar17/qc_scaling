struct Context{N}
    pos::Vector{ParityOperator{N}}
    parity::Int
    idxs::Vector{Int}
    function Context(pos::Vector{ParityOperator{N}}, parity) where N
        @assert parity in (0, 1)
        @assert length(pos)==(2^(N -1) + 1) "Not the right number of POs"
        idxs = [po.index for po in pos]
        return new{N}(pos, parity, idxs)
    end
end

function Context(generator::ParityOperator{N}, base_cxt::Context{N}) where N
    pos = Ref(generator) .+ base_cxt.pos
    return Context(pos, base_cxt.parity)
end

struct ContextMaster
    base_even::Context
    base_odd::Context
    nqubit::Int
    function ContextMaster(base_even::Context{N}, base_odd::Context{N}, nqubit) where N
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
    return cxt.idxs
end

function to_ternary(x::Integer, ::Val{N}) where N
    t = ntuple(N) do i
        (x ÷ 3^(N - i)) % 3
    end
    return SVector{N, Int}(t)
end

to_ternary(x::Integer, nqubit::Integer) = to_ternary(x, Val(nqubit))


function generate_base_context(nqubit::Int, evenoddbit::Int)
    @assert evenoddbit ∈ [0, 1]
    _generate_base_context(Val(nqubit), evenoddbit)
end

function _generate_base_context(::Val{N}, evenoddbit::Int) where N
    zero_po = ParityOperator(SVector{N, Int}(ntuple(_ -> 0, N)))
    pos = ParityOperator{N}[zero_po]
    for idx in 1:3^N - 1
        βs = to_ternary(idx, Val(N))
        if any(βs .== 0) || ~(sum(βs[βs.==1]) % 2==evenoddbit)
            continue
        end
        push!(pos, ParityOperator(βs))
    end
    return Context(pos, evenoddbit)
end
