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

function idx_to_alphas(idx::Int, nqubit::Int)
    @assert idx < 2^(nqubit-1)
    M = nqubit - 1
    return SVector{M, Int}(ntuple(i -> (idx >> (i-1)) & 1, M))
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
                fingerprint[:, theta_s+1, theta_z+1, idx+1,] .= floor.(Int, parity(state, base_cxt))
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
    idx = sum(state.alphas[i] << (i-1) for i in eachindex(state.alphas))
    return fp.a[:, theta_s+1, theta_z+1, idx+1]
end

# ---------------------------------------------------------------------------
# Bit-packed fingerprint
#
# Each column of Fingerprint.a (binary 0/1 values) is stored as a vector of
# UInt64 words.  L1 distance in pick_new_alphas reduces to XOR + popcount,
# giving ~50x speedup over float comparison for n=10.
# ---------------------------------------------------------------------------

struct FingerprintPacked
    words ::Array{UInt64, 4}   # [nwords × 2 × 2 × nalpha]
    npos  ::Int
    nwords::Int
end

function FingerprintPacked(fp::Fingerprint)
    npos   = size(fp.a, 1)
    np     = size(fp.a, 2)
    ntz    = size(fp.a, 3)
    nalpha = size(fp.a, 4)
    nwords = cld(npos, 64)
    words  = zeros(UInt64, nwords, np, ntz, nalpha)
    @inbounds for ai in 1:nalpha, tzi in 1:ntz, pi in 1:np
        for bit in 1:npos
            if fp.a[bit, pi, tzi, ai] == 1
                w = (bit - 1) ÷ 64 + 1
                b = (bit - 1) % 64
                words[w, pi, tzi, ai] |= (UInt64(1) << b)
            end
        end
    end
    return FingerprintPacked(words, npos, nwords)
end
