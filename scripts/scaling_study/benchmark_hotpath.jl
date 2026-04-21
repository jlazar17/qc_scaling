using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using BenchmarkTools
using Random
using Statistics
using StaticArrays
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Current implementations (mirrors the SA loop)
# ---------------------------------------------------------------------------

function apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

function rep_accuracy_fast(rep_sum, rep_ctr, goal)
    s = 0
    @inbounds for i in eachindex(goal)
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1]/c1; p2 = rep_sum[i2]/c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

# ---------------------------------------------------------------------------
# Optimized: bit-packed fingerprint for pick_new_alphas
#
# Fingerprint values and companion_goal are both binary (0/1 or NaN).
# L1 distance = count of disagreements = popcount(xor(fp_bits, cg_bits) & valid)
# Packs each column of the fingerprint into a Vector{UInt64}.
# ---------------------------------------------------------------------------

struct FingerprintPacked
    # [nwords × 2 × 2 × nalpha]: packed bits for each (parity, theta_z, alpha_idx) column
    words::Array{UInt64, 4}
    npos::Int
    nwords::Int
end

function FingerprintPacked(fp::QCScaling.Fingerprint)
    npos   = size(fp.a, 1)
    np     = size(fp.a, 2)
    ntz    = size(fp.a, 3)
    nalpha = size(fp.a, 4)
    nwords = cld(npos, 64)
    words  = zeros(UInt64, nwords, np, ntz, nalpha)
    for ai in 1:nalpha, tzi in 1:ntz, pi in 1:np
        col = view(fp.a, :, pi, tzi, ai)
        for bit in 1:npos
            if col[bit] == 1
                w = (bit - 1) ÷ 64 + 1
                b = (bit - 1) % 64
                words[w, pi, tzi, ai] |= (UInt64(1) << b)
            end
        end
    end
    return FingerprintPacked(words, npos, nwords)
end

function pack_companion_goal(cg::Vector)
    npos   = length(cg)
    nwords = cld(npos, 64)
    valid  = zeros(UInt64, nwords)   # 1 where cg is not NaN
    vals   = zeros(UInt64, nwords)   # cg value (0 or 1)
    for i in 1:npos
        isnan(cg[i]) && continue
        w = (i - 1) ÷ 64 + 1
        b = (i - 1) % 64
        valid[w] |= (UInt64(1) << b)
        if cg[i] != 0.0
            vals[w] |= (UInt64(1) << b)
        end
    end
    return valid, vals
end

function pick_new_alphas_packed(
    cxt::QCScaling.Context,
    goal::Vector,
    rep::Vector,
    fp_packed::FingerprintPacked,
    base_cxt::QCScaling.Context
)
    cg          = QCScaling.companion_goal(cxt, goal, rep)
    valid, vals = pack_companion_goal(cg)
    parity_idx  = base_cxt.parity + 1
    nwords      = fp_packed.nwords
    nalpha      = size(fp_packed.words, 4)
    ntz         = size(fp_packed.words, 3)

    best_score = typemax(Int)
    best_tz    = 1
    best_alpha = 1

    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0
            for w in 1:nwords
                score += count_ones(xor(fp_packed.words[w, parity_idx, tzi, ai], vals[w]) & valid[w])
            end
            if score < best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Optimized: apply_state! using the fingerprint (avoids recomputing parity)
#
# parity(state, gen + base_po) is independent of gen — it equals
# parity(state_with_zero_gen, base_po). The fingerprint stores exactly this.
# ---------------------------------------------------------------------------

function apply_state_fp!(rep_sum, rep_ctr, state, cxt_master, fp::QCScaling.Fingerprint, sign)
    base_cxt  = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    fp_col    = fp[state]   # precomputed parity column, Int vector of length npos
    for (i, base_po) in enumerate(base_cxt.pos)
        derived_idx = (state.generator + base_po).index
        p = fp_col[i]
        rep_sum[derived_idx] += sign * p
        rep_ctr[derived_idx] += sign
    end
end

# ---------------------------------------------------------------------------
# Optimized: loop-reordered pick_new_alphas (pi innermost for cache locality)
# ---------------------------------------------------------------------------

function pick_new_alphas_reordered(
    cxt::QCScaling.Context,
    goal::Vector,
    rep::Vector,
    fp::QCScaling.Fingerprint,
    base_cxt::QCScaling.Context
)
    cg         = QCScaling.companion_goal(cxt, goal, rep)
    fa         = fp.a
    parity_idx = base_cxt.parity + 1
    npos       = size(fa, 1)
    ntz        = size(fa, 3)
    nalpha     = size(fa, 4)
    best_sum   = typemax(Float64)
    best_tz    = 1
    best_alpha = 1

    # scores[tzi, ai] accumulated over pi — avoids branching inside inner loop
    scores = fill(0.0, ntz, nalpha)
    @inbounds for pi in 1:npos
        c = cg[pi]
        isnan(c) && continue
        for tzi in 1:ntz
            for ai in 1:nalpha
                scores[tzi, ai] += abs(fa[pi, parity_idx, tzi, ai] - c)
            end
        end
    end
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            if scores[tzi, ai] < best_sum
                best_sum   = scores[tzi, ai]
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Setup for a given nqubit
# ---------------------------------------------------------------------------

function make_bench_state(nqubit; seed=42)
    rng    = Random.MersenneTwister(seed)
    nstate = Int(ceil(3^nqubit / 2^(nqubit-1))) * 3
    ngbits = (3^nqubit - 1) ÷ 2
    n      = 3^nqubit

    cm     = QCScaling.ContextMaster(nqubit)
    fp     = QCScaling.Fingerprint(nqubit)
    states = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    goal   = rand(rng, 0:1, ngbits)

    rep_sum = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cm, +1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    state   = first(states)
    base_cxt = state.theta_s == 0 ? cm.base_even : cm.base_odd
    gen_idx  = rand(rng, 0:3^nqubit-1)
    gen      = QCScaling.ParityOperator(gen_idx, nqubit)
    cxt      = QCScaling.Context(gen, base_cxt)

    fp_packed = FingerprintPacked(fp)

    return (; cm, fp, fp_packed, states, goal, rep, rep_sum, rep_ctr,
              state, cxt, base_cxt, nqubit, n, ngbits)
end

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

function run_benchmarks(nqubit)
    @printf("\n%s\n", "="^60)
    @printf("n = %d\n", nqubit)
    @printf("%s\n", "="^60)

    env = make_bench_state(nqubit)
    (; cm, fp, fp_packed, states, goal, rep, rep_sum, rep_ctr,
       state, cxt, base_cxt) = env

    rs_copy  = copy(rep_sum)
    rc_copy  = copy(rep_ctr)

    # ---- apply_state! ----
    b_apply  = @benchmark apply_state!($rs_copy, $rc_copy, $state, $cm, 1) setup=(rs_copy=copy($rep_sum); rc_copy=copy($rep_ctr)) evals=1 samples=200
    b_applyfp = @benchmark apply_state_fp!($rs_copy, $rc_copy, $state, $cm, $fp, 1) setup=(rs_copy=copy($rep_sum); rc_copy=copy($rep_ctr)) evals=1 samples=200

    @printf("\napply_state! (current):      %8.2f µs\n", median(b_apply).time   / 1e3)
    @printf("apply_state! (fingerprint):  %8.2f µs  (%.1fx)\n",
            median(b_applyfp).time / 1e3,
            median(b_apply).time / median(b_applyfp).time)

    # ---- rep_accuracy_fast ----
    b_acc = @benchmark rep_accuracy_fast($rep_sum, $rep_ctr, $goal) samples=500
    @printf("\nrep_accuracy_fast:           %8.2f µs\n", median(b_acc).time / 1e3)

    # ---- pick_new_alphas ----
    b_pick     = @benchmark QCScaling.pick_new_alphas($cxt, $goal, $rep, $fp, $base_cxt) samples=200
    b_pick_re  = @benchmark pick_new_alphas_reordered($cxt, $goal, $rep, $fp, $base_cxt) samples=200
    b_pick_pk  = @benchmark pick_new_alphas_packed($cxt, $goal, $rep, $fp_packed, $base_cxt) samples=200

    @printf("\npick_new_alphas (current):   %8.2f µs\n", median(b_pick).time    / 1e3)
    @printf("pick_new_alphas (reordered): %8.2f µs  (%.1fx)\n",
            median(b_pick_re).time / 1e3,
            median(b_pick).time / median(b_pick_re).time)
    @printf("pick_new_alphas (packed):    %8.2f µs  (%.1fx)\n",
            median(b_pick_pk).time / 1e3,
            median(b_pick).time / median(b_pick_pk).time)

    # ---- verify correctness of optimized versions ----
    r0  = QCScaling.pick_new_alphas(cxt, goal, rep, fp, base_cxt)
    r1  = pick_new_alphas_reordered(cxt, goal, rep, fp, base_cxt)
    r2  = pick_new_alphas_packed(cxt, goal, rep, fp_packed, base_cxt)
    match_re = r0[1] == r1[1] && r0[2] == r1[2] && r0[3] == r1[3]
    match_pk = r0[1] == r2[1] && r0[2] == r2[2] && r0[3] == r2[3]
    @printf("\nCorrectness: reordered=%s  packed=%s\n",
            match_re ? "OK" : "FAIL", match_pk ? "OK" : "FAIL")

    t_step_current  = 2*median(b_apply).time + median(b_acc).time + median(b_pick).time
    t_step_opt      = 2*median(b_applyfp).time + median(b_acc).time + median(b_pick_pk).time
    @printf("\nProjected step time (current):  %.2f µs\n", t_step_current / 1e3)
    @printf("Projected step time (opt):      %.2f µs  (%.1fx)\n",
            t_step_opt / 1e3, t_step_current / t_step_opt)
end

for nq in [8, 10]
    run_benchmarks(nq)
end
