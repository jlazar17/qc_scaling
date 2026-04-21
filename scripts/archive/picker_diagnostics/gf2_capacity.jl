using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

# ---------------------------------------------------------------------------
# For each random generator, enumerate ALL (theta_z, alpha) configurations,
# compute XOR at every both-covered pair, and count how many configs satisfy
# the full set of pair XOR requirements for different goals.
#
# This tests fundamental alpha-space capacity — decoupled from the picker.
#
# XOR formula: XOR(pair j) = D_j XOR (XOR_{i in S_j} alpha_i)
# where D_j = C_bp1 XOR C_bp2 (a constant, theta_z-independent).
#
# The question: for H=0 (all XOR=0) vs random goals (H=0.5, H=1),
# are there more alpha configs satisfying ALL pair constraints?
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

# For a given generator + theta_s, compute:
#   - The set of both-covered pairs (pairs where both k1 and k2 are covered)
#   - For each (theta_z, alpha_idx): the XOR at each both-covered pair
# Returns: (pair_indices, xor_matrix) where xor_matrix[config_idx, pair_idx] ∈ {0,1}
function compute_pair_xors(generator, theta_s, cxt_master, nqubit)
    n       = 3^nqubit
    nalpha  = 2^(nqubit - 1)
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

    # Map: pair_index → list of (config_idx, parity) for each covered position
    # We need both k1 and k2 to be covered for a pair to count
    pair_to_positions = Dict{Int, Vector{Int}}()  # pair_j → [local_pos_indices for k1, k2]

    covered_k = Int[]
    for base_po in base_cxt.pos
        derived_po = generator + base_po
        k = derived_po.index
        k == n && continue
        j = (k + 1) ÷ 2  # 1-indexed pair
        if !haskey(pair_to_positions, j)
            pair_to_positions[j] = Int[]
        end
        push!(pair_to_positions[j], length(covered_k) + 1)  # local index
        push!(covered_k, k)
    end

    # Keep only both-covered pairs
    both_pairs = Int[]
    both_pair_local = Vector{Tuple{Int,Int}}()  # (local_idx_k1, local_idx_k2)
    for (j, locals) in pair_to_positions
        length(locals) == 2 || continue
        push!(both_pairs, j)
        k1_local, k2_local = locals[1], locals[2]
        # Determine which local index corresponds to k1 (odd) vs k2 (even)
        if isodd(covered_k[k1_local])
            push!(both_pair_local, (k1_local, k2_local))
        else
            push!(both_pair_local, (k2_local, k1_local))
        end
    end

    isempty(both_pairs) && return Int[], Matrix{Int}(undef, 0, 0)

    npos    = length(covered_k)
    npairs  = length(both_pairs)
    nconfigs = 2 * nalpha  # theta_z ∈ {0,1}, alpha_idx ∈ {0..nalpha-1}

    # Build state for each config and compute parities at covered positions
    xor_mat = zeros(Int, nconfigs, npairs)

    config_idx = 0
    for theta_z in 0:1
        for alpha_idx in 0:nalpha-1
            config_idx += 1
            alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
            state  = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator)

            # Compute parity at each covered position
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

            # Compute XOR at each both-covered pair
            for (pi, (k1_l, k2_l)) in enumerate(both_pair_local)
                p1 = parities[k1_l]
                p2 = parities[k2_l]
                if p1 < 0 || p2 < 0
                    xor_mat[config_idx, pi] = -1  # ambiguous
                else
                    xor_mat[config_idx, pi] = xor(p1, p2)
                end
            end
        end
    end

    return both_pairs, xor_mat
end

# Count configs satisfying all pair XOR constraints for a given goal.
function count_satisfying_configs(both_pairs, xor_mat, goal)
    isempty(both_pairs) && return 0, 0
    nconfigs, npairs = size(xor_mat)
    n_satisfy_all = 0
    n_valid = 0  # configs with no ambiguous parities

    for ci in 1:nconfigs
        has_ambig = any(xor_mat[ci, pi] < 0 for pi in 1:npairs)
        has_ambig && continue
        n_valid += 1
        correct = all(xor_mat[ci, pi] == goal[both_pairs[pi]] for pi in 1:npairs)
        n_satisfy_all += correct
    end
    return n_satisfy_all, n_valid
end

function run_capacity_test(nqubit, ngens, base_seed)
    rng        = Random.MersenneTwister(base_seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    nalpha     = 2^(nqubit - 1)
    ngbits     = (n - 1) ÷ 2

    entropy_targets = [0.0, 0.25, 0.5, 0.75, 1.0]
    goals = Dict{Float64, Vector{Int}}()
    for H in entropy_targets
        k = k_from_entropy(H, ngbits)
        g_rng = Random.MersenneTwister(base_seed + round(Int, H * 1000))
        goals[H] = goal_from_hamming(k, ngbits, g_rng)
    end

    # Accumulators
    n_gens_with_pairs = 0
    n_total_gens = 0
    frac_satisfy = Dict(H => Float64[] for H in entropy_targets)
    n_pairs_hist = Int[]
    # Also track: among generators with both-covered pairs,
    # does H=0 have MORE satisfying configs than other goals?
    h0_vs_other = Dict(H => Int[] for H in entropy_targets if H > 0)

    for _ in 1:ngens
        n_total_gens += 1
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)

        both_pairs, xor_mat = compute_pair_xors(generator, theta_s, cxt_master, nqubit)
        isempty(both_pairs) && continue
        n_gens_with_pairs += 1
        push!(n_pairs_hist, length(both_pairs))

        counts = Dict{Float64, Int}()
        valid_count = 0
        for H in entropy_targets
            n_sat, n_valid = count_satisfying_configs(both_pairs, xor_mat, goals[H])
            counts[H] = n_sat
            valid_count = max(valid_count, n_valid)
        end

        for H in entropy_targets
            push!(frac_satisfy[H], valid_count > 0 ? counts[H] / valid_count : NaN)
        end
        for H in entropy_targets
            H == 0.0 && continue
            push!(h0_vs_other[H], sign(counts[0.0] - counts[H]))
        end
    end

    @printf("\nCapacity test: nqubit=%d, ngens=%d (of which %d had both-covered pairs, %.1f%%)\n\n",
            nqubit, n_total_gens, n_gens_with_pairs, 100*n_gens_with_pairs/n_total_gens)
    @printf("Average both-covered pairs per generator (when any): %.2f\n\n", mean(n_pairs_hist))

    @printf("%-10s  %-20s  %-20s\n", "H_target", "mean_frac_configs_sat", "std")
    println("-"^54)
    for H in entropy_targets
        fs = filter(!isnan, frac_satisfy[H])
        @printf("%-10.4f  %-20.6f  %-20.6f\n", H, mean(fs), std(fs))
    end

    println()
    @printf("Fraction of generators where H=0 satisfies MORE configs than other goals:\n")
    for H in sort(collect(keys(h0_vs_other)))
        signs = h0_vs_other[H]
        frac_h0_more = mean(signs .> 0)
        frac_equal   = mean(signs .== 0)
        frac_other_more = mean(signs .< 0)
        @printf("  vs H=%.2f:  H=0 more: %.3f  equal: %.3f  H=%.2f more: %.3f\n",
                H, frac_h0_more, frac_equal, H, frac_other_more)
    end

    # Also check: among generators where H=0 has 0 satisfying configs, how does H=1 do?
    println()
    n_h0_zero = sum(frac_satisfy[0.0] .== 0)
    @printf("Generators where H=0 has 0 satisfying configs: %d (%.1f%% of those with pairs)\n",
            n_h0_zero, 100*n_h0_zero/n_gens_with_pairs)

    # Distribution of satisfying config counts
    println()
    @printf("Distribution of frac_configs_satisfying for H=0 and H=1:\n")
    for H in [0.0, 0.5, 1.0]
        fs = filter(!isnan, frac_satisfy[H])
        frac_zero = mean(fs .== 0.0)
        frac_half = mean(0.4 .< fs .< 0.6)
        frac_all  = mean(fs .== 1.0)
        @printf("  H=%.2f: frac=0: %.3f  frac≈0.5: %.3f  frac=1: %.3f  mean=%.4f\n",
                H, frac_zero, frac_half, frac_all, mean(fs))
    end
end

function main()
    nqubit    = 6
    ngens     = 10_000
    base_seed = 42
    @printf("GF(2) capacity test: nqubit=%d, ngens=%d\n", nqubit, ngens)
    @printf("Counting alpha configs satisfying ALL both-covered pair XOR constraints.\n")
    @printf("Decoupled from picker — pure structural measurement.\n")
    flush(stdout)
    run_capacity_test(nqubit, ngens, base_seed)
end

main()
