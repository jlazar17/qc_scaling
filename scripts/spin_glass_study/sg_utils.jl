# Shared utilities for spin-glass study scripts.
# Include after activating the project and loading QCScaling.

using QCScaling
using Random
using Statistics
using StaticArrays
using Printf

include(joinpath(@__DIR__, "../utils/optimization_utils.jl"))

# ---------------------------------------------------------------------------
# Alternative pickers
#
# pick_alphas_margin: same as pick_alphas_s but only counts positions with
#   margin <= margin_threshold (the only positions a single vote can flip).
#   margin=0 includes uncovered (rep_ctr==0) and tied positions.
#
# pick_alphas_local_oracle: bit-packed local oracle.  For each of 64 patterns,
#   computes the expected accuracy change at margin-flippable positions:
#     +1 for each currently-wrong margin-1/0 position voted correctly
#     -1 for each currently-right margin-1 position voted incorrectly
#   Picks the pattern that maximises this score.
# ---------------------------------------------------------------------------

function pick_alphas_margin(cxt, goal, rep, rep_sum, rep_ctr,
                            fp::FingerprintPacked, base_cxt,
                            companion, goal_idx, n_rep;
                            margin_threshold::Int=1)
    cg_full = [companion_goal_s(po, goal, rep, companion, goal_idx, n_rep)
               for po in cxt.pos]
    # Zero out (NaN) positions with margin > threshold
    cg = map(eachindex(cg_full)) do i
        isnan(cg_full[i]) && return NaN
        k = cxt.pos[i].index
        (k <= 0 || k > n_rep) && return NaN
        mg = rep_ctr[k] == 0 ? 0 : abs(2*rep_sum[k] - rep_ctr[k])
        mg <= margin_threshold ? cg_full[i] : NaN
    end
    valid, vals = QCScaling._pack_companion_goal(cg, fp.nwords)
    parity_idx = base_cxt.parity + 1
    nwords = fp.nwords
    best_score = typemax(Int); best_tz = 1; best_alpha = 1
    @inbounds for ai in 1:size(fp.words,4), tzi in 1:size(fp.words,3)
        score = 0
        for w in 1:nwords
            score += count_ones(xor(fp.words[w,parity_idx,tzi,ai], vals[w]) & valid[w])
        end
        if score < best_score; best_score = score; best_tz = tzi; best_alpha = ai; end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

function pick_alphas_local_oracle(cxt, goal, rep, rep_sum, rep_ctr,
                                   fp::FingerprintPacked, base_cxt,
                                   companion, goal_idx, n_rep)
    nwords = fp.nwords

    # For each position, determine which of 4 categories it falls in:
    #   want1_wrong: cg=1, currently wrong/untied  => gain if fp_vote=1
    #   want0_wrong: cg=0, currently wrong/untied  => gain if fp_vote=0
    #   want1_right: cg=1, currently right, margin=1 => lose if fp_vote=0
    #   want0_right: cg=0, currently right, margin=1 => lose if fp_vote=1
    # Positions with margin > 1 are skipped (cannot flip).
    want1_wrong = zeros(UInt64, nwords)
    want0_wrong = zeros(UInt64, nwords)
    want1_right = zeros(UInt64, nwords)
    want0_right = zeros(UInt64, nwords)

    for i in eachindex(cxt.pos)
        po = cxt.pos[i]
        cg = companion_goal_s(po, goal, rep, companion, goal_idx, n_rep)
        isnan(cg) && continue
        k = po.index
        (k <= 0 || k > n_rep) && continue

        rc = rep_ctr[k]
        mg = rc == 0 ? 0 : abs(2*rep_sum[k] - rep_ctr[k])

        # Only flippable positions: margin <= 1
        # (margin=0 covers uncovered and tied; margin=1 can flip with one vote)
        mg > 1 && continue

        w = (i-1) ÷ 64 + 1
        b = (i-1) % 64
        bit = UInt64(1) << b

        if !iszero(cg)   # want vote = 1
            # "currently right" means majority == cg == 1, i.e. 2*rep_sum > rep_ctr
            currently_right = rc > 0 && mg == 1 && 2*rep_sum[k] > rep_ctr[k]
            if currently_right
                want1_right[w] |= bit
            else
                want1_wrong[w] |= bit
            end
        else             # want vote = 0
            # "currently right" means majority == cg == 0, i.e. 2*rep_sum < rep_ctr
            currently_right = rc > 0 && mg == 1 && 2*rep_sum[k] < rep_ctr[k]
            if currently_right
                want0_right[w] |= bit
            else
                want0_wrong[w] |= bit
            end
        end
    end

    parity_idx = base_cxt.parity + 1
    best_score = typemin(Int); best_tz = 1; best_alpha = 1
    @inbounds for ai in 1:size(fp.words,4), tzi in 1:size(fp.words,3)
        score = 0
        for w in 1:nwords
            fw = fp.words[w, parity_idx, tzi, ai]
            score += count_ones( fw & want1_wrong[w])   # gain: vote=1 fixes wrong
            score -= count_ones( fw & want0_right[w])   # lose: vote=1 breaks right
            score += count_ones(~fw & want0_wrong[w])   # gain: vote=0 fixes wrong
            score -= count_ones(~fw & want1_right[w])   # lose: vote=0 breaks right
        end
        if score > best_score; best_score = score; best_tz = tzi; best_alpha = ai; end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# pick_alphas_hybrid: combines full fp score with a bonus for patterns that
# vote correctly on margin-flippable currently-wrong positions.
#
# score = fp_score - bonus * (count of margin<=1 currently-wrong positions
#                             where fp_vote == companion_goal)
#
# A higher bonus shifts weight toward fixing wrong pairs; bonus=0 reduces
# to the standard fp picker.
# ---------------------------------------------------------------------------
function pick_alphas_hybrid(cxt, goal, rep, rep_sum, rep_ctr,
                             fp::FingerprintPacked, base_cxt,
                             companion, goal_idx, n_rep;
                             bonus::Int=2)
    nwords = fp.nwords

    # Standard companion_goal packing (for full fp score)
    cg_full = [companion_goal_s(po, goal, rep, companion, goal_idx, n_rep)
               for po in cxt.pos]
    valid, vals = QCScaling._pack_companion_goal(cg_full, fp.nwords)

    # Bonus masks: positions where margin<=1 and currently wrong
    # Voting correctly here gives +bonus to the pattern score
    bonus1_mask = zeros(UInt64, nwords)  # want vote=1, currently wrong
    bonus0_mask = zeros(UInt64, nwords)  # want vote=0, currently wrong

    for i in eachindex(cxt.pos)
        isnan(cg_full[i]) && continue
        k = cxt.pos[i].index
        (k <= 0 || k > n_rep) && continue
        rc = rep_ctr[k]
        mg = rc == 0 ? 0 : abs(2*rep_sum[k] - rep_ctr[k])
        mg > 1 && continue

        # Check currently wrong: majority != companion_goal
        if rc == 0
            currently_wrong = true   # uncovered: treat as wrong
        elseif 2*rep_sum[k] == rep_ctr[k]
            currently_wrong = true   # tied: treat as wrong
        else
            majority = 2*rep_sum[k] > rep_ctr[k] ? 1 : 0
            currently_wrong = (majority != round(Int, cg_full[i]))
        end
        !currently_wrong && continue

        w = (i-1) ÷ 64 + 1
        b = (i-1) % 64
        bit = UInt64(1) << b
        if !iszero(cg_full[i])
            bonus1_mask[w] |= bit
        else
            bonus0_mask[w] |= bit
        end
    end

    parity_idx = base_cxt.parity + 1
    best_score = typemax(Int); best_tz = 1; best_alpha = 1
    @inbounds for ai in 1:size(fp.words,4), tzi in 1:size(fp.words,3)
        score = 0
        for w in 1:nwords
            fw = fp.words[w, parity_idx, tzi, ai]
            # Standard fp disagreement count
            score += count_ones(xor(fw, vals[w]) & valid[w])
            # Subtract bonus for correctly voting on margin-1 wrong positions
            score -= bonus * count_ones( fw & bonus1_mask[w])
            score -= bonus * count_ones(~fw & bonus0_mask[w])
        end
        if score < best_score; best_score = score; best_tz = tzi; best_alpha = ai; end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# Unified call signature for all pickers:
#   picker(cxt, goal, rep, rep_sum, rep_ctr, fp, bc, companion, goal_idx, n_rep)
const PICKER_FP = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_s(cxt, goal, rep, fp, bc, comp, gi, nr)

const PICKER_MARGIN1 = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_margin(cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr; margin_threshold=1)

const PICKER_MARGIN2 = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_margin(cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr; margin_threshold=2)

const PICKER_LOCAL_ORACLE = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_local_oracle(cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr)

const PICKER_HYBRID2 = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_hybrid(cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr; bonus=2)

const PICKER_HYBRID4 = (cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr) ->
    pick_alphas_hybrid(cxt, goal, rep, rs, rc, fp, bc, comp, gi, nr; bonus=4)

# ---------------------------------------------------------------------------
# Standard SA returning (final_acc, ensemble, rep_sum, rep_ctr, rep)
# ---------------------------------------------------------------------------
function run_sa_full(goal, nqubit, nstate, nsteps, alpha_cool,
                     companion, goal_idx, fingerprint, cxt_master;
                     seed=42, picker=PICKER_FP)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    rng  = Random.MersenneTwister(seed)

    ensemble   = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars  = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    acc_fn(rs, rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = picker(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
                        fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        d = acc_fn(rep_sum, rep_ctr) - cur_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        d < 0 && push!(bad_deltas, abs(d))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    for _ in 1:nsteps
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = picker(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
                        fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc
        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc, ensemble, rep_sum, rep_ctr, rep
end

# ---------------------------------------------------------------------------
# Compute all 64 fingerprint scores for a given (gen, theta_s, rep).
# Returns (min_score, all_scores, n_valid) where n_valid = number of
# positions with a valid companion_goal signal.
# ---------------------------------------------------------------------------
function all_scores(gen, theta_s, goal, rep, fp, cxt_master, companion, goal_idx, n)
    bc  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt = QCScaling.Context(gen, bc)
    cg  = [companion_goal_s(po, goal, rep, companion, goal_idx, n-1) for po in cxt.pos]
    valid, vals = QCScaling._pack_companion_goal(cg, fp.nwords)
    n_valid = count(!isnan, cg)

    parity_idx = bc.parity + 1
    nwords     = fp.nwords
    nalpha     = size(fp.words, 4)
    ntz        = size(fp.words, 3)

    scores = Vector{Int}(undef, ntz * nalpha)
    idx = 1
    @inbounds for ai in 1:nalpha, tzi in 1:ntz
        s = 0
        for w in 1:nwords
            s += count_ones(xor(fp.words[w, parity_idx, tzi, ai], vals[w]) & valid[w])
        end
        scores[idx] = s; idx += 1
    end
    return minimum(scores), scores, n_valid
end

# ---------------------------------------------------------------------------
# Margin at a position: |2*rep_sum[k] - rep_ctr[k]|.  0 = tied/uncovered.
# ---------------------------------------------------------------------------
margin(rep_sum, rep_ctr, k) =
    rep_ctr[k] == 0 ? 0 : abs(2 * rep_sum[k] - rep_ctr[k])

# ---------------------------------------------------------------------------
# h_to_kones: find k such that H(k/ngbits) ≈ H_target.
# ---------------------------------------------------------------------------
function h_to_kones(H_target, ngbits)
    H_target == 0.0 && return 0
    H_target >= 1.0 && return ngbits ÷ 2
    lo, hi = 0.0, 0.5
    for _ in 1:60
        p = (lo + hi) / 2
        h = -p * log2(p) - (1 - p) * log2(1 - p)
        h < H_target ? (lo = p) : (hi = p)
    end
    return round(Int, (lo + hi) / 2 * ngbits)
end
