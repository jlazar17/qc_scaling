# ---------------------------------------------------------------------------
# Incremental rep cache maintenance
#
# The "rep cache" is a pair (rep_sum::Vector{Int}, rep_ctr::Vector{Int})
# where rep_sum[i] counts the number of ensemble states that give parity 1
# at position i, and rep_ctr[i] counts total coverage.  Because parity is
# always 0 or 1 for even-qubit systems (never 0.5), both arrays are integer.
#
# The rounded representation rep[i] ∈ {0.0, 1.0, NaN} is derived on demand:
#   - NaN  if rep_ctr[i] == 0 (uncovered) or 2*rep_sum[i] == rep_ctr[i] (tied)
#   - 1.0  if 2*rep_sum[i] > rep_ctr[i]
#   - 0.0  otherwise
# ---------------------------------------------------------------------------

"""
    apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)

Add (`sign=+1`) or remove (`sign=-1`) `state`'s parity contributions to the
rep cache by recomputing derived positions and parities on the fly.
Use in non-hot-path code (initialisation, calibration).  For the SA inner
loop use `fill_state_cache!` + `apply_state_cached!` instead.
"""
function apply_state!(rep_sum::Vector{Int}, rep_ctr::Vector{Int},
                      state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = round(Int, parity(state, derived_po))
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

"""
    rep_from_cache(rep_sum, rep_ctr) -> Vector{Float64}

Convert the integer rep cache to the rounded representation vector.
Positions with zero coverage or a tied vote are set to `NaN`.
"""
function rep_from_cache(rep_sum::Vector{Int}, rep_ctr::Vector{Int})
    rep = fill(NaN, length(rep_sum))
    @inbounds for i in eachindex(rep_sum)
        c = rep_ctr[i]; c == 0 && continue
        s2 = 2 * rep_sum[i]
        s2 == c && continue
        rep[i] = s2 > c ? 1.0 : 0.0
    end
    return rep
end

"""
    update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)

Refresh the entries of `rep` at every position covered by `state`,
reading current values from the integer rep cache.  Called after
each accepted SA move to keep the rounded rep in sync.
"""
function update_rep_at!(rep, rep_sum::Vector{Int}, rep_ctr::Vector{Int},
                        state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]; s2 = 2 * rep_sum[i]
        rep[i] = (c == 0 || s2 == c) ? NaN : (s2 > c ? 1.0 : 0.0)
    end
end

# ---------------------------------------------------------------------------
# Precomputed state cache
#
# For each state in the SA ensemble we precompute the derived position indices
# and integer parity values once.  Subsequent apply/update calls become pure
# indexed array operations — no ternary arithmetic, no parity() call.
#
# base_even and base_odd have the same length (2^(N-1)+1), so a single npos-
# length pair of buffers works for both parities.
# ---------------------------------------------------------------------------

"""
    fill_state_cache!(idxs, pars, state, cxt_master)

Fill pre-allocated buffers `idxs` and `pars` with the derived position
indices and integer parities for `state`.  Costs the same as one
`apply_state!` call; the savings are realised on every subsequent
`apply_state_cached!` / `update_rep_at_cached!` call.
"""
function fill_state_cache!(idxs::Vector{Int}, pars::Vector{Int},
                           state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    @inbounds for (i, base_po) in enumerate(base_cxt.pos)
        derived_po = state.generator + base_po
        idxs[i] = derived_po.index
        pars[i] = round(Int, parity(state, derived_po))
    end
end

"""
    apply_state_cached!(rep_sum, rep_ctr, idxs, pars, sign)

Add (`sign=+1`) or remove (`sign=-1`) a state's contributions using its
precomputed index and parity buffers.  O(npos) indexed array ops, no
ternary arithmetic or parity recomputation.
"""
function apply_state_cached!(rep_sum::Vector{Int}, rep_ctr::Vector{Int},
                              idxs::Vector{Int}, pars::Vector{Int}, sign)
    @inbounds for i in eachindex(idxs)
        rep_sum[idxs[i]] += sign * pars[i]
        rep_ctr[idxs[i]] += sign
    end
end

"""
    update_rep_at_cached!(rep, rep_sum, rep_ctr, idxs)

Refresh `rep` at the positions given by `idxs` (a precomputed index
buffer).  Used in place of `update_rep_at!` in the SA hot loop.
"""
function update_rep_at_cached!(rep, rep_sum::Vector{Int}, rep_ctr::Vector{Int},
                                idxs::Vector{Int})
    @inbounds for i in eachindex(idxs)
        idx = idxs[i]; c = rep_ctr[idx]; s2 = 2 * rep_sum[idx]
        rep[idx] = (c == 0 || s2 == c) ? NaN : (s2 > c ? 1.0 : 0.0)
    end
end
