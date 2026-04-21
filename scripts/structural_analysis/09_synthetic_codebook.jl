# Synthetic codebook comparison: PseudoGHZ patterns vs random patterns.
#
# Tests whether the H-dependent p_pos collapse is specific to the PseudoGHZ
# algebraic structure or generic to any 64-pattern constrained code.
#
# No SA required. A synthetic rep is built directly at target accuracy by
# assigning parities to pairs. Then for each random generator + theta_s:
#
#   (A) PseudoGHZ codebook: enumerate all 64 (theta_z, alphas) patterns via
#       fill_state_cache!, compute best achievable delta.
#   (B) Random codebook: draw 64 random binary patterns over the same covered
#       positions, compute best achievable delta.
#
# p_pos = fraction of generators where best delta > 0.
# Compare H=0 vs H=1 under both codebooks.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using StaticArrays
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Build a synthetic rep at target accuracy (no SA).
# Returns parity[k] ∈ {0,1} for every position k, with exactly
# round(acc * ngbits) pairs correct.
# ---------------------------------------------------------------------------

function build_synthetic_rep(goal, companion, goal_idx, nqubit, target_acc, rng)
    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    pairs = [(k1, companion[k1]) for k1 in 1:n-1
             if companion[k1] > 0 && k1 < companion[k1]]
    @assert length(pairs) == ngbits

    n_correct  = round(Int, target_acc * ngbits)
    is_correct = vcat(trues(n_correct), falses(ngbits - n_correct))
    shuffle!(rng, is_correct)

    parity = zeros(Int, n)
    for (i, (k1, k2)) in enumerate(pairs)
        g  = goal[goal_idx[k1]]
        p1 = rand(rng, 0:1)
        p2 = is_correct[i] ? (p1 ⊻ g) : (p1 ⊻ (1 - g))
        parity[k1] = p1
        parity[k2] = p2
    end
    return parity
end

# ---------------------------------------------------------------------------
# Compute delta when replacing any existing state with a new state whose
# parities at covered positions are given by `new_pars`.
# Uses margin-1 assumption: any change in a covered position's parity flips
# the majority.
# Only counts pairs where exactly one of (k, companion[k]) is covered
# (shuffled pairing guarantees this).
# ---------------------------------------------------------------------------

function delta_from_pattern(new_pars, covered_idxs, parity, companion, goal_idx, goal)
    delta = 0
    for (i, k) in enumerate(covered_idxs)
        km = companion[k]
        (km == 0 || goal_idx[k] == 0) && continue
        g             = goal[goal_idx[k]]
        old_correct   = (parity[k] ⊻ parity[km]) == g
        new_correct   = (new_pars[i] ⊻ parity[km]) == g
        delta        += Int(new_correct) - Int(old_correct)
    end
    return delta
end

# ---------------------------------------------------------------------------
# For one context (gen, theta_s): return best delta under PseudoGHZ codebook
# and best delta under a fresh random codebook of the same size.
# ---------------------------------------------------------------------------

function best_deltas(gen, theta_s, nqubit, cxt_master, parity,
                     companion, goal_idx, goal, rng)
    n       = 3^nqubit
    nalpha  = 2^(nqubit - 1)   # 32 for nqubit=6
    npos    = length(cxt_master.base_even.pos)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    # --- PseudoGHZ codebook: 2 (theta_z) × nalpha combinations ---
    best_pghz = typemin(Int)
    alphas_buf = zeros(Int, nqubit - 1)
    for tz in 0:1
        for ai in 0:nalpha-1
            for b in 0:nqubit-2; alphas_buf[b+1] = (ai >> b) & 1; end
            sv    = SVector{nqubit-1, Int}(alphas_buf)
            state = QCScaling.PseudoGHZState(theta_s, tz, sv, gen)
            fill_state_cache!(scratch_idxs, scratch_pars, state, cxt_master)
            d = delta_from_pattern(scratch_pars, scratch_idxs, parity,
                                   companion, goal_idx, goal)
            best_pghz = max(best_pghz, d)
        end
    end

    # --- Random codebook: 64 random binary patterns over same positions ---
    # (Use the theta_s=0 covered positions for consistency)
    ref_state = QCScaling.PseudoGHZState(theta_s, 0, SVector{nqubit-1,Int}(zeros(Int,nqubit-1)), gen)
    fill_state_cache!(scratch_idxs, scratch_pars, ref_state, cxt_master)
    covered = copy(scratch_idxs)  # positions covered by this (gen, theta_s)

    best_rand = typemin(Int)
    for _ in 1:64
        rand_pars = rand(rng, 0:1, npos)
        d = delta_from_pattern(rand_pars, covered, parity, companion, goal_idx, goal)
        best_rand = max(best_rand, d)
    end

    return best_pghz, best_rand
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit    = 6
    n         = 3^nqubit
    ngbits    = (n - 1) ÷ 2
    n_gens    = 2000   # generators probed per (H, acc, trial)
    n_trials  = 8

    acc_vals = [0.70, 0.80, 0.88, 0.93]
    H_vals   = [0.0, 0.25, 0.5, 0.75, 1.0]

    println("Building shuffled pairing...")
    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    cxt_master = QCScaling.ContextMaster(nqubit)
    println("  done.\n")

    @printf("%-8s  %-6s  %-12s  %-12s\n", "acc", "H", "p_pos_pghz", "p_pos_rand")
    println(repeat("-", 48))

    for H in H_vals
        k_ones = round(Int, _h_to_p(H) * ngbits)
        for acc in acc_vals
            ppos_pghz = Float64[]
            ppos_rand = Float64[]
            for trial in 1:n_trials
                rng  = Random.MersenneTwister(trial * 137 + round(Int, H * 1000) +
                                               round(Int, acc * 10000))
                goal = Random.shuffle!(rng, vcat(ones(Int, k_ones),
                                                  zeros(Int, ngbits - k_ones)))
                parity = build_synthetic_rep(goal, companion, goal_idx,
                                              nqubit, acc, rng)

                n_pos_pghz = 0; n_pos_rand = 0
                for _ in 1:n_gens
                    gen_idx  = rand(rng, 0:n-1)
                    gen      = QCScaling.ParityOperator(gen_idx, nqubit)
                    theta_s  = rand(rng, 0:1)
                    bp, br   = best_deltas(gen, theta_s, nqubit, cxt_master,
                                           parity, companion, goal_idx, goal, rng)
                    bp > 0 && (n_pos_pghz += 1)
                    br > 0 && (n_pos_rand += 1)
                end
                push!(ppos_pghz, n_pos_pghz / n_gens)
                push!(ppos_rand, n_pos_rand / n_gens)
            end
            @printf("%-8.2f  %-6.2f  %-12.4f  %-12.4f\n",
                    acc, H, mean(ppos_pghz), mean(ppos_rand))
        end
        println()
    end
end

# Binary entropy inversion: return p ∈ [0, 0.5] such that H(p) = H_target
function _h_to_p(H_target)
    H_target == 0.0 && return 0.0
    H_target == 1.0 && return 0.5
    lo, hi = 0.0, 0.5
    for _ in 1:60
        p = (lo + hi) / 2
        h = -p * log2(p) - (1 - p) * log2(1 - p)
        h < H_target ? (lo = p) : (hi = p)
    end
    return (lo + hi) / 2
end

main()
