# ---------------------------------------------------------------------------
# Shuffled pairing: a perfect matching of ternary positions where every pair
# (k1, k2) differs at an odd number of ternary digits.  This eliminates
# "both-covered" pairs, ensuring every pair contributes independently to
# accuracy.  Used throughout the structural analysis.
# ---------------------------------------------------------------------------

"""
    build_shuffled_pairing(nqubit) -> (companion, goal_idx, n_pairs)

Build a perfect matching of the 3^nqubit - 1 non-zero ternary positions such
that every matched pair (k1, k2) differs at an odd number of ternary digits
(guaranteeing zero both-covered pairs).

Returns:
- `companion[k]`: the index paired with position k (0 if k is unpaired)
- `goal_idx[k]`: the goal-pair index for position k (1-based)
- `n_pairs`: total number of pairs (= ngbits = (3^nqubit - 1) / 2)
"""
function build_shuffled_pairing(nqubit)
    n = 3^nqubit
    betas = [to_ternary(k-1, Val(nqubit)) for k in 1:n-1]
    matched = falses(n-1); pairs = Tuple{Int,Int}[]
    for k1 in 1:n-1
        matched[k1] && continue
        for k2 in k1+1:n-1
            matched[k2] && continue
            if isodd(count(i -> betas[k1][i] != betas[k2][i], 1:nqubit))
                push!(pairs, (k1, k2)); matched[k1] = matched[k2] = true; break
            end
        end
    end
    companion = zeros(Int, n); goal_idx = zeros(Int, n)
    for (gi, (k1, k2)) in enumerate(pairs)
        companion[k1]=k2; companion[k2]=k1; goal_idx[k1]=gi; goal_idx[k2]=gi
    end
    return companion, goal_idx, length(pairs)
end

"""
    rep_accuracy_shuffled(rep_sum, rep_ctr, goal, companion, goal_idx) -> Float64

Compute accuracy against `goal` using the shuffled pairing defined by
`companion` and `goal_idx`.  Iterates over canonical pair representatives
(k1 < companion[k1]) and counts correctly predicted pairs.
"""
function rep_accuracy_shuffled(rep_sum, rep_ctr, goal, companion, goal_idx)
    s = 0; n = length(goal)
    for k1 in 1:length(rep_sum)-1
        companion[k1]==0 && continue; k1>companion[k1] && continue
        k2=companion[k1]; c1=rep_ctr[k1]; c2=rep_ctr[k2]
        s1=rep_sum[k1]; s2=rep_sum[k2]
        valid=(c1>0)&(c2>0)&(2*s1!=c1)&(2*s2!=c2)
        r1=2*s1>c1; r2=2*s2>c2; j=goal_idx[k1]
        s += valid & ((r1 ⊻ r2) == !iszero(goal[j]))
    end
    return s / n
end

"""
    companion_goal_s(po, goal, rep, companion, goal_idx, n_rep) -> Float64 or NaN

For a parity observable `po`, return the desired value of `rep[companion[k]]`
that would make this position vote correctly for its pair goal.  Returns NaN
if the position is unpaired, unrepresented, or out of range.
"""
function companion_goal_s(po, goal, rep, companion, goal_idx, n_rep)
    k=po.index; (k<=0||k>n_rep) && return NaN
    comp=companion[k]; comp==0 && return NaN; isnan(rep[comp]) && return NaN
    j=goal_idx[k]; j==0 && return NaN
    return goal[j]==1 ? 1-rep[comp] : rep[comp]
end

"""
    pick_alphas_s(cxt, goal, rep, fp, base_cxt, companion, goal_idx, n_rep)

Shuffled-pairing variant of `pick_new_alphas`.  Selects the (parity, theta_z,
alphas) combination that minimises the number of positions voting away from
their companion goals, using the bit-packed fingerprint scoring.
"""
function pick_alphas_s(cxt, goal, rep, fp::FingerprintPacked, base_cxt,
                       companion, goal_idx, n_rep)
    cg = [companion_goal_s(po, goal, rep, companion, goal_idx, n_rep) for po in cxt.pos]
    valid, vals = _pack_companion_goal(cg, fp.nwords)
    parity_idx=base_cxt.parity+1; nwords=fp.nwords; nalpha=size(fp.words,4); ntz=size(fp.words,3)
    best_score=typemax(Int); best_tz=1; best_alpha=1
    @inbounds for ai in 1:nalpha, tzi in 1:ntz
        score=0
        for w in 1:nwords; score+=count_ones(xor(fp.words[w,parity_idx,tzi,ai],vals[w])&valid[w]); end
        if score<best_score; best_score=score; best_tz=tzi; best_alpha=ai; end
    end
    return (base_cxt.parity, best_tz-1, idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end
