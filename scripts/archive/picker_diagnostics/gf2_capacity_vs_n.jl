using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

# ---------------------------------------------------------------------------
# GF(2) capacity vs nqubit
#
# Runs the both-covered pair XOR capacity test at n=4, 6, 8.
# Key question: does the fraction of generators where H=1 has ZERO valid
# alpha configs grow with n? If so, the structural constraint tightens with
# system size, directly explaining why H=1 efficiency declines at larger n.
#
# For each n and generator, we enumerate all 2*nalpha (theta_z, alpha) configs
# and count what fraction satisfy ALL both-covered pair XOR constraints for
# H=0 and H=1 goals.
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
        if !haskey(pair_to_positions, j)
            pair_to_positions[j] = Int[]
        end
        push!(pair_to_positions[j], length(covered_k) + 1)
        push!(covered_k, k)
    end

    both_pairs = Int[]
    both_pair_local = Vector{Tuple{Int,Int}}()
    for (j, locals) in pair_to_positions
        length(locals) == 2 || continue
        push!(both_pairs, j)
        k1_local, k2_local = locals[1], locals[2]
        if isodd(covered_k[k1_local])
            push!(both_pair_local, (k1_local, k2_local))
        else
            push!(both_pair_local, (k2_local, k1_local))
        end
    end

    isempty(both_pairs) && return Int[], Matrix{Int}(undef, 0, 0)

    npos    = length(covered_k)
    npairs  = length(both_pairs)
    nconfigs = 2 * nalpha

    xor_mat = zeros(Int, nconfigs, npairs)
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

function count_satisfying_configs(both_pairs, xor_mat, goal)
    isempty(both_pairs) && return 0, 0
    nconfigs, npairs = size(xor_mat)
    n_satisfy_all = 0; n_valid = 0
    for ci in 1:nconfigs
        any(xor_mat[ci, pi] < 0 for pi in 1:npairs) && continue
        n_valid += 1
        all(xor_mat[ci, pi] == goal[both_pairs[pi]] for pi in 1:npairs) && (n_satisfy_all += 1)
    end
    return n_satisfy_all, n_valid
end

function run_capacity_test(nqubit, ngens, base_seed)
    rng        = Random.MersenneTwister(base_seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    goals = Dict{Float64, Vector{Int}}()
    for H in [0.0, 1.0]
        k = k_from_entropy(H, ngbits)
        g_rng = Random.MersenneTwister(base_seed + round(Int, H * 1000))
        goals[H] = goal_from_hamming(k, ngbits, g_rng)
    end

    n_total = 0; n_with_pairs = 0
    frac_h0 = Float64[]; frac_h1 = Float64[]
    n_zero_h1 = 0  # generators where H=1 has 0 satisfying configs
    n_pairs_hist = Int[]

    for _ in 1:ngens
        n_total += 1
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)

        both_pairs, xor_mat = compute_pair_xors(generator, theta_s, cxt_master, nqubit)
        isempty(both_pairs) && continue
        n_with_pairs += 1
        push!(n_pairs_hist, length(both_pairs))

        n_h0, n_valid = count_satisfying_configs(both_pairs, xor_mat, goals[0.0])
        n_h1, _       = count_satisfying_configs(both_pairs, xor_mat, goals[1.0])

        push!(frac_h0, n_valid > 0 ? n_h0 / n_valid : NaN)
        push!(frac_h1, n_valid > 0 ? n_h1 / n_valid : NaN)
        n_h1 == 0 && (n_zero_h1 += 1)
    end

    frac_with_pairs = n_with_pairs / n_total

    @printf("  nqubit=%d  ngens=%d  frac_with_both_covered_pairs=%.3f\n",
            nqubit, n_total, frac_with_pairs)
    @printf("  avg_both_covered_pairs_per_gen (when any): %.2f\n",
            mean(n_pairs_hist))
    @printf("  H=0: mean_frac_configs_satisfying=%.4f  std=%.4f\n",
            mean(filter(!isnan, frac_h0)), std(filter(!isnan, frac_h0)))
    @printf("  H=1: mean_frac_configs_satisfying=%.4f  std=%.4f\n",
            mean(filter(!isnan, frac_h1)), std(filter(!isnan, frac_h1)))
    @printf("  H=1: frac_generators_with_ZERO_valid_configs=%.4f  (%d / %d)\n",
            n_zero_h1 / n_with_pairs, n_zero_h1, n_with_pairs)
    println()
end

function main()
    ngens     = 10_000
    base_seed = 42

    @printf("GF(2) capacity vs nqubit\n")
    @printf("Tests whether the structural XOR constraint tightens with system size.\n")
    @printf("Key metric: frac_generators_with_ZERO_valid_H=1_configs.\n\n")
    flush(stdout)

    for nqubit in [4, 6, 8]
        run_capacity_test(nqubit, ngens, base_seed)
        flush(stdout)
    end
end

main()
