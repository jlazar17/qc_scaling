using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

# ---------------------------------------------------------------------------
# GF(2) capacity as a function of entropy H, for n=4,6,8
#
# For each (nqubit, H) pair, samples ngens random generators and ngoals random
# goals at that entropy level. Reports:
#   frac_zero:    fraction of both-covered-pair generators where H=1 has 0
#                 valid alpha configs (averaged over goals)
#   mean_frac:    mean fraction of configs satisfying all constraints
#
# This reveals whether the GF(2) wall is a pathology of H=1 exactly, or
# whether it grows continuously as H increases.
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N; return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0; H >= 1.0 && return N ÷ 2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), 0:N÷2)
    return (0:N÷2)[idx]
end

function compute_pair_xors(generator, theta_s, cxt_master, nqubit)
    n      = 3^nqubit
    nalpha = 2^(nqubit - 1)
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    pair_to_positions = Dict{Int, Vector{Int}}()
    covered_k = Int[]
    for base_po in base_cxt.pos
        derived_po = generator + base_po
        k = derived_po.index
        k == n && continue
        j = (k + 1) ÷ 2
        if !haskey(pair_to_positions, j); pair_to_positions[j] = Int[]; end
        push!(pair_to_positions[j], length(covered_k) + 1)
        push!(covered_k, k)
    end

    both_pairs = Int[]
    both_pair_local = Vector{Tuple{Int,Int}}()
    for (j, locals) in pair_to_positions
        length(locals) == 2 || continue
        push!(both_pairs, j)
        k1_l, k2_l = locals[1], locals[2]
        if isodd(covered_k[k1_l])
            push!(both_pair_local, (k1_l, k2_l))
        else
            push!(both_pair_local, (k2_l, k1_l))
        end
    end

    isempty(both_pairs) && return Int[], Matrix{Int}(undef, 0, 0)

    npos     = length(covered_k)
    npairs   = length(both_pairs)
    nconfigs = 2 * nalpha
    xor_mat  = zeros(Int, nconfigs, npairs)

    config_idx = 0
    for theta_z in 0:1
        for alpha_idx in 0:nalpha-1
            config_idx += 1
            alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
            state  = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator)
            parities = zeros(Int, npos)
            pos_idx = 0
            for base_po in base_cxt.pos
                derived_po = generator + base_po
                k = derived_po.index
                k == n && continue
                pos_idx += 1
                p = QCScaling.parity(state, derived_po)
                parities[pos_idx] = p == 0.5 ? -1 : round(Int, p)
            end
            for (pi, (k1_l, k2_l)) in enumerate(both_pair_local)
                p1 = parities[k1_l]; p2 = parities[k2_l]
                xor_mat[config_idx, pi] = (p1 < 0 || p2 < 0) ? -1 : xor(p1, p2)
            end
        end
    end

    return both_pairs, xor_mat
end

function count_satisfying(both_pairs, xor_mat, goal)
    isempty(both_pairs) && return 0, 0
    nconfigs, npairs = size(xor_mat)
    n_sat = 0; n_valid = 0
    for ci in 1:nconfigs
        any(xor_mat[ci, pi] < 0 for pi in 1:npairs) && continue
        n_valid += 1
        all(xor_mat[ci, pi] == goal[both_pairs[pi]] for pi in 1:npairs) && (n_sat += 1)
    end
    return n_sat, n_valid
end

function run_sweep(nqubit, ngens, ngoals, base_seed)
    rng        = Random.MersenneTwister(base_seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    entropy_targets = 0.0:0.1:1.0

    @printf("nqubit=%d  ngbits=%d  nalpha=%d  ngens=%d  ngoals=%d\n",
            nqubit, ngbits, 2^(nqubit-1), ngens, ngoals)
    @printf("%-8s  %-6s  %-12s  %-14s  %-14s\n",
            "H_target", "k", "H_act", "mean_frac_sat", "frac_zero_valid")
    println("-"^60)

    # Pre-compute XOR matrices for all generators (shared across H levels)
    both_pairs_all = Vector{Vector{Int}}()
    xor_mats_all   = Vector{Matrix{Int}}()
    for _ in 1:ngens
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        bp, xm    = compute_pair_xors(generator, theta_s, cxt_master, nqubit)
        push!(both_pairs_all, bp)
        push!(xor_mats_all,   xm)
    end
    # Only keep generators that have both-covered pairs
    has_pairs = findall(i -> !isempty(both_pairs_all[i]), 1:ngens)

    for H_target in entropy_targets
        k     = k_from_entropy(Float64(H_target), ngbits)
        H_act = hamming_entropy(k, ngbits)

        # Average over ngoals random goals at this entropy level
        frac_zero_per_goal = Float64[]
        mean_frac_per_goal = Float64[]

        for gi in 1:ngoals
            g_rng = Random.MersenneTwister(base_seed + gi * 1000 + round(Int, H_target * 100))
            goal  = goal_from_hamming(k, ngbits, g_rng)

            n_zero = 0; fracs = Float64[]
            for i in has_pairs
                n_sat, n_valid = count_satisfying(both_pairs_all[i], xor_mats_all[i], goal)
                n_valid == 0 && continue
                frac = n_sat / n_valid
                push!(fracs, frac)
                frac == 0.0 && (n_zero += 1)
            end

            push!(frac_zero_per_goal, length(fracs) > 0 ? n_zero / length(fracs) : NaN)
            push!(mean_frac_per_goal, length(fracs) > 0 ? mean(fracs) : NaN)
        end

        @printf("%-8.1f  %-6d  %-12.4f  %-14.6f  %-14.6f\n",
                H_target, k, H_act,
                mean(filter(!isnan, mean_frac_per_goal)),
                mean(filter(!isnan, frac_zero_per_goal)))
    end
    println()
    flush(stdout)
end

function main()
    base_seed = 42
    ngoals    = 5   # goals averaged per (n, H) point

    # ngens scaled to keep runtime reasonable at n=8
    ngens_by_n = Dict(4 => 10_000, 6 => 10_000, 8 => 5_000)

    @printf("GF(2) capacity vs entropy H, for n=4,6,8\n")
    @printf("Averaged over %d random goals per (n,H) point.\n\n", ngoals)
    flush(stdout)

    for nqubit in [4, 6, 8]
        run_sweep(nqubit, ngens_by_n[nqubit], ngoals, base_seed)
    end
end

main()
