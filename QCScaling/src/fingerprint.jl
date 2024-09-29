struct Fingerprint
    a::Array{Int, 4}
    function Fingerprint(a::Array{Int, 4})
        @assert size(a)[3]==2
        @assert size(a)[2]==2
        nqubit = Int(log(2, size(a)[4]) + 1)
        @assert size(a)[1]==2^(nqubit-1) + 1
        return new(a)
    end
end

function idx_to_alphas(idx::Int, nqubit::Int) :: Vector{Int}
    @assert idx < 2^(nqubit-1)
    r = zeros(Int, nqubit-1)
    b = digits(idx, base=2)
    r[1:length(b)] .= b
    return r
end

function Fingerprint(fname::String, groupname::String)
    error("unimplemented")
end

function Fingerprint(nqubit::Int)
    fingerprint = Array{Int}(undef, (2^(nqubit-1)+1, 2, 2, 2^(nqubit-1)))
    for theta_s in 0:1
        base_cxt = generate_base_context(nqubit, theta_s)
        for theta_z in 0:1
            for idx in 0:2^(nqubit-1) - 1
                alphas = idx_to_alphas(idx, nqubit)
                state = PseudoGHZState(
                    theta_s,
                    theta_z,
                    alphas,
                    base_cxt.pos[1]
                )
                fingerprint[:, theta_s+1, theta_z+1, idx+1,] .= parity(state, base_cxt)
           end
       end
    end
    return Fingerprint(fingerprint)
end

function Base.:-(fp::Fingerprint, v::Vector)
    @assert size(fp.a)[1]==length(v)
    return fp.a .- v
end

function Base.getindex(fp::Fingerprint, state::PseudoGHZState)
    theta_s = state.theta_s
    theta_z = state.theta_z
    nqubit = length(state)
    idx = sum([2^n for n in 0:nqubit-2] .* state.alphas)
    return fp.a[:, theta_s+1, theta_z+1, idx+1]
end
