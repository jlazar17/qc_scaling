module QCScaling

export ParityOperator, FingerprintPacked, rep_accuracy_fast,
       apply_state!, rep_from_cache, update_rep_at!,
       fill_state_cache!, apply_state_cached!, update_rep_at_cached!,
       build_shuffled_pairing, rep_accuracy_shuffled,
       companion_goal_s, pick_alphas_s

using ProgressBars
using Statistics
using OhMyThreads
using StaticArrays

include("./parity_observable.jl")
include("./contexts.jl")
include("./states.jl")
include("./fingerprint.jl")
include("./parities.jl")
include("./rep_cache.jl")
include("./shuffled_pairing.jl")

function get_goal_index(po::ParityOperator)
    goal_index = (po.index+1) ÷ 2
    return goal_index
end

function get_companion_index(po::ParityOperator)
    return isone(po.index % 2) ? po.index + 1 : po.index - 1
end

function calculate_preference(
    states::Vector{<:PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    nqubit = length(first(states))
    representation = zeros(3 ^ nqubit)
    ctr = zeros(Int, 3 ^ nqubit)
    for state in states
        base_cxt = state.theta_s==0 ? base_even : base_odd
        for base_po in base_cxt.pos
            derived_po = state.generator + base_po
            p = parity(state, derived_po)
            representation[derived_po.index] += p
            ctr[derived_po.index] += 1
        end
    end
    pref = representation ./ ctr
    return pref
end

function calculate_representation(
    states::Vector{<:PseudoGHZState},
    base_even::Context,
    base_odd::Context
)
    pref = calculate_preference(states, base_even, base_odd)
    pref[pref.==0.5] .= NaN
    rep = round.(pref)
    return rep
end

function calculate_representation(
    states::Vector{<:PseudoGHZState},
)
    nqubit = length(first(states))
    base_even = generate_base_context(nqubit, 0)
    base_odd = generate_base_context(nqubit, 1)
    return calculate_representation(states, base_even, base_odd)
end

function score(
    states::Vector{<:PseudoGHZState},
    rep::Vector,
    goal::Vector{Int},
    cxt_master::ContextMaster;
    multi_threading=true
)

    itr = states
    undefined_idxs = @views Set(findall(isnan, (rep[1:2:end-1] .- rep[2:2:end])))
    scores = OhMyThreads.@tasks for state in itr
        @set begin
            collect = true
            ntasks = multi_threading ? Threads.nthreads() : 1
        end
        score(state, rep, undefined_idxs, goal, cxt_master)
    end
    return scores
end

function score(
    states::Vector{<:PseudoGHZState},
    goal::Vector{Int},
    cxt_master::ContextMaster
)
    rep = calculate_representation(
        states,
        cxt_master.base_even,
        cxt_master.base_odd
    )
    scores = score(states, rep, goal, cxt_master)
    return scores
end

function score(
    state::PseudoGHZState,
    rep::Vector{Float64},
    undefined_idxs::Set,
    goal::Vector{Int},
    cxt_master::ContextMaster;
)
    nqubit = cxt_master.nqubit

    base_cxt = ifelse(
        state.theta_s==0,
        cxt_master.base_even,
        cxt_master.base_odd
    )

    score = 0
    tnq = 3^nqubit
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        if derived_po.index==tnq
            continue
        end
        reduced_idx = get_goal_index(derived_po)

        if reduced_idx in undefined_idxs
            continue
        end

        companion_idx = get_companion_index(derived_po)
        p = parity(state, derived_po)
        @assert p ∈ (0,1)
        predicted_bit = abs(p - rep[companion_idx])
        diff = predicted_bit==goal[reduced_idx] ? 1 : -1
        score += diff
    end
    return score
end

function get_new_contexts(
    states::Vector,
    rep::Vector,
    cxt_master::ContextMaster,
    nnew::Int
)
    counter = zeros(Int, 3^cxt_master.nqubit)
    counter[isnan.(rep)] .= 1
    scores = zeros(Int, 3^cxt_master.nqubit)
    base_cxt = ifelse(
        rand() > 0.5,
        cxt_master.base_even,
        cxt_master.base_odd
    )
    for idx in 1:3^cxt_master.nqubit
        generator = ParityOperator(idx-1, cxt_master.nqubit)
        cxt = Context(generator, base_cxt)
        idxs =  map(x->x.index, Context(generator, base_cxt))
        scores[idx] += sum(counter[idxs])
    end
    chosen_idxs = sortperm(-scores)[1:nnew] .- 1
    pos = ParityOperator.(chosen_idxs, cxt_master.nqubit)
    cxts = Context.(pos, Ref(base_cxt))
    return cxts
end

function get_new_contexts(states::Vector, cxt_master::ContextMaster, nnew::Int)
    rep = calculate_representation(states, cxt_master.base_even, cxt_master.base_odd)
    return get_new_contexts(states, rep, cxt_master, nnew)
end

"""
    Find the desired direction for a PO to move based on current representation
    [0.7, 0.3, ...] -> [1, 0, 1, 0, 1, 0, 1, NaN, 1]
"""
function companion_goal(po::ParityOperator, goal::Vector, rep::Vector)
    if po.index==length(rep)
        return NaN
    end
    companion_idx = po.index % 2==1 ? po.index + 1 : po.index - 1
    if isnan(rep[companion_idx])
        return NaN
    end
    goal_index = get_goal_index(po)
    cg = goal[goal_index]==1 ? 1 - rep[companion_idx] : rep[companion_idx]
    return cg
end

function companion_goal(cxt::Context, goal::Vector, rep::Vector)
    return companion_goal.(cxt.pos, Ref(goal), Ref(rep))
end

function pick_new_alphas(
    cxt::Context,
    goal::Vector,
    rep::Vector,
    fingerprint::Fingerprint,
    base_cxt::Context
)
    cg = companion_goal(cxt, goal, rep)
    fa = fingerprint.a
    parity_idx = base_cxt.parity + 1
    npos = size(fa, 1)
    ntz = size(fa, 3)
    nalpha = size(fa, 4)
    best_sum = typemax(Float64)
    best_tz = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            s = 0.0
            for pi in 1:npos
                c = cg[pi]
                isnan(c) && continue
                s += abs(fa[pi, parity_idx, tzi, ai] - c)
            end
            if s < best_sum
                best_sum = s
                best_tz = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Bit-packed pick_new_alphas
#
# companion_goal values are binary (0.0 / 1.0 / NaN).  Pack them into
# (valid_mask, value_mask) UInt64 words, then compute L1 distance as
# popcount(xor(fp_words, value_words) & valid_words).
# ---------------------------------------------------------------------------

function _pack_companion_goal(cg::Vector, nwords::Int)
    valid = zeros(UInt64, nwords)
    vals  = zeros(UInt64, nwords)
    @inbounds for i in eachindex(cg)
        isnan(cg[i]) && continue
        w = (i - 1) ÷ 64 + 1
        b = (i - 1) % 64
        valid[w] |= (UInt64(1) << b)
        if !iszero(cg[i])
            vals[w] |= (UInt64(1) << b)
        end
    end
    return valid, vals
end

# Mutating version: fills pre-allocated valid/vals in-place (zero-alloc).
function _pack_companion_goal!(valid::Vector{UInt64}, vals::Vector{UInt64}, cg::Vector)
    fill!(valid, zero(UInt64))
    fill!(vals,  zero(UInt64))
    @inbounds for i in eachindex(cg)
        isnan(cg[i]) && continue
        w = (i - 1) ÷ 64 + 1
        b = (i - 1) % 64
        valid[w] |= (UInt64(1) << b)
        if !iszero(cg[i])
            vals[w] |= (UInt64(1) << b)
        end
    end
end

function pick_new_alphas(
    cxt::Context,
    goal::Vector,
    rep::Vector,
    fp::FingerprintPacked,
    base_cxt::Context
)
    cg          = companion_goal(cxt, goal, rep)
    valid, vals = _pack_companion_goal(cg, fp.nwords)
    parity_idx  = base_cxt.parity + 1
    nwords      = fp.nwords
    nalpha      = size(fp.words, 4)
    ntz         = size(fp.words, 3)

    best_score = typemax(Int)
    best_tz    = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0
            for w in 1:nwords
                score += count_ones(xor(fp.words[w, parity_idx, tzi, ai], vals[w]) & valid[w])
            end
            if score < best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Allocation-free pick_new_alphas: takes generator + base_cxt directly,
# avoiding the Context(generator, base_cxt) construction which broadcasts
# ParityOperator addition across npos elements (~2000 pool allocs per call).
# ---------------------------------------------------------------------------

@inline function _companion_goal_from_idx(idx::Int, goal, rep)
    idx == length(rep) && return NaN
    companion_idx = isone(idx & 1) ? idx + 1 : idx - 1
    isnan(rep[companion_idx]) && return NaN
    goal_index = (idx + 1) >> 1
    return goal[goal_index] == 1 ? 1.0 - rep[companion_idx] : rep[companion_idx]
end

function pick_new_alphas(
    generator ::ParityOperator{N},
    base_cxt  ::Context{N},
    goal      ::Vector,
    rep       ::Vector,
    fp        ::FingerprintPacked
) where N
    gen_βs     = generator.βs
    parity_idx = base_cxt.parity + 1
    nwords     = fp.nwords
    npos       = length(base_cxt.pos)

    # Compute companion_goal for each derived position without materialising
    # the derived Context (saves ~2000 pool allocs per call at nqubit=10).
    cg = Vector{Float64}(undef, npos)
    @inbounds for (i, base_po) in enumerate(base_cxt.pos)
        base_βs = base_po.βs
        didx = 1; pw = 1
        @inbounds for j in N:-1:1
            didx += mod(gen_βs[j] + base_βs[j], 3) * pw
            pw   *= 3
        end
        cg[i] = _companion_goal_from_idx(didx, goal, rep)
    end

    valid, vals = _pack_companion_goal(cg, nwords)
    nalpha = size(fp.words, 4)
    ntz    = size(fp.words, 3)
    best_score = typemax(Int)
    best_tz    = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0
            for w in 1:nwords
                score += count_ones(xor(fp.words[w, parity_idx, tzi, ai], vals[w]) & valid[w])
            end
            if score < best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz - 1, idx_to_alphas(best_alpha - 1, N))
end

# ---------------------------------------------------------------------------
# Zero-allocation pick_new_alphas: caller pre-allocates scratch_cg, scratch_valid,
# scratch_vals (sized npos and fp.nwords respectively) and passes them in.
# This eliminates all per-call heap allocations from the SA hot path.
# ---------------------------------------------------------------------------

function pick_new_alphas(
    generator    ::ParityOperator{N},
    base_cxt     ::Context{N},
    goal         ::Vector,
    rep          ::Vector,
    fp           ::FingerprintPacked,
    scratch_cg   ::Vector{Float64},
    scratch_valid::Vector{UInt64},
    scratch_vals ::Vector{UInt64}
) where N
    gen_βs     = generator.βs
    parity_idx = base_cxt.parity + 1
    nwords     = fp.nwords
    npos       = length(base_cxt.pos)

    @inbounds for (i, base_po) in enumerate(base_cxt.pos)
        base_βs = base_po.βs
        didx = 1; pw = 1
        @inbounds for j in N:-1:1
            didx += mod(gen_βs[j] + base_βs[j], 3) * pw
            pw   *= 3
        end
        scratch_cg[i] = _companion_goal_from_idx(didx, goal, rep)
    end

    _pack_companion_goal!(scratch_valid, scratch_vals, scratch_cg)

    nalpha = size(fp.words, 4)
    ntz    = size(fp.words, 3)
    best_score = typemax(Int)
    best_tz    = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0
            for w in 1:nwords
                score += count_ones(xor(fp.words[w, parity_idx, tzi, ai], scratch_vals[w]) & scratch_valid[w])
            end
            if score < best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz - 1, idx_to_alphas(best_alpha - 1, N))
end

# ---------------------------------------------------------------------------
# rep_accuracy_fast
#
# Computes fraction of goal bits correctly predicted by the current rep cache.
# Reformulates p = sum/count comparisons as integer ops (2*sum vs count),
# eliminating all divisions.  p == 0.5 ⟺ 2*sum == count (integer equality).
# p > 0.5 ⟺ 2*sum > count (integer comparison).
# ---------------------------------------------------------------------------

function rep_accuracy_fast(
    rep_sum ::AbstractVector{Float64},
    rep_ctr ::AbstractVector{Int},
    goal    ::AbstractVector{Int}
)
    s = 0
    n = length(goal)
    # Branchless + @simd formulation:
    #   p = sum/count comparisons rewritten as 2*sum vs count (no division).
    #   p == 0.5 ⟺ 2*sum == count; p > 0.5 ⟺ 2*sum > count.
    #   valid flag masks out zero-count and ambiguous positions.
    @inbounds @simd for i in 1:n
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c2 = rep_ctr[i2]
        fc1 = Float64(c1); fc2 = Float64(c2)
        s1_2 = 2.0 * rep_sum[i1]; s2_2 = 2.0 * rep_sum[i2]
        valid = (c1 > 0) & (c2 > 0) & (s1_2 != fc1) & (s2_2 != fc2)
        r1 = s1_2 > fc1
        r2 = s2_2 > fc2
        s += valid & ((r1 ⊻ r2) == !iszero(goal[i]))
    end
    return s / n
end

# Integer overload: parity values are always 0 or 1 for even-qubit systems,
# so rep_sum accumulates only integers.  Ambiguity (average == 0.5) arises
# when equal numbers of states give parity 0 and parity 1, detected by
# 2*rep_sum == rep_ctr (pure integer comparison, no floating point).
function rep_accuracy_fast(
    rep_sum ::AbstractVector{Int},
    rep_ctr ::AbstractVector{Int},
    goal    ::AbstractVector{Int}
)
    s = 0
    n = length(goal)
    @inbounds @simd for i in 1:n
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c2 = rep_ctr[i2]
        s1 = rep_sum[i1];  s2 = rep_sum[i2]
        valid = (c1 > 0) & (c2 > 0) & (2*s1 != c1) & (2*s2 != c2)
        r1 = 2*s1 > c1
        r2 = 2*s2 > c2
        s += valid & ((r1 ⊻ r2) == !iszero(goal[i]))
    end
    return s / n
end

end # module QCScaling
