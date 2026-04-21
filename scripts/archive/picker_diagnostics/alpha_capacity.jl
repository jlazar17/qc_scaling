using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# For each random generator, compute the BEST achievable pair accuracy
# using pick_new_alphas with an "oracle" rep — where rep is set so that
# companion_goal always points in the correct direction for the given goal.
#
# Specifically: for goal g, set rep[k] to the correct majority vote direction
# that would satisfy the XOR constraint with whatever the best alpha gives.
# We approximate this by: for a random goal, set rep[k1]=0, rep[k2]=goal[j]
# (so that companion_goal for k1 targets 0 XOR goal[j], and for k2 targets goal[j]).
#
# This tests the intrinsic capacity of the alpha space to satisfy XOR constraints,
# decoupled from the actual SA dynamics.
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

# Build an "oracle" rep consistent with the given goal:
# For each pair j: set rep[k1]=0, rep[k2]=goal[j].
# Then companion_goal[k1] = goal[j]==1 ? 1-rep[k2]=1-goal[j]=0 : rep[k2]=0  → wants 0 at k1
#       companion_goal[k2] = goal[j]==1 ? 1-rep[k1]=1 : rep[k1]=0             → wants goal[j] at k2
# So this rep makes the XOR target = goal[j] for each pair (since we want (0, goal[j]) which has XOR=goal[j]).
function build_oracle_rep(goal, n)
    rep = fill(NaN, n)
    ngbits = length(goal)
    for j in 1:ngbits
        k1 = 2j - 1
        k2 = 2j
        rep[k1] = 0.0
        rep[k2] = Float64(goal[j])
    end
    return rep
end

# Count pairs covered by the proposed state and how many match the goal XOR.
function count_pair_accuracy(ns, goal, cxt_master, n)
    theta_s  = ns.theta_s
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    pair_parities = Dict{Int, Vector{Float64}}()
    for base_po in base_cxt.pos
        derived_po = ns.generator + base_po
        k = derived_po.index
        k == n && continue
        j = (k + 1) ÷ 2
        p = QCScaling.parity(ns, derived_po)
        p == 0.5 && continue
        if !haskey(pair_parities, j)
            pair_parities[j] = Float64[]
        end
        push!(pair_parities[j], p)
    end
    n_both = 0; n_correct = 0
    for (j, pars) in pair_parities
        length(pars) == 2 || continue
        n_both += 1
        xor_state = pars[1] != pars[2] ? 1 : 0
        n_correct += (xor_state == goal[j])
    end
    return n_both, n_correct
end

function run_capacity_test(nqubit, ngens, base_seed)
    rng        = Random.MersenneTwister(base_seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    fp_packed  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n          = 3^nqubit
    ngbits     = (n - 1) ÷ 2

    # Test across entropy levels
    entropy_targets = [0.0, 0.25, 0.5, 0.75, 1.0]

    @printf("\n%-8s  %-12s  %-12s  %-12s\n",
            "H_target", "best_pair_frac", "rand_pair_frac", "n_pairs_both")
    println("-"^52)

    for H_target in entropy_targets
        k    = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        goal_rng = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
        goal = goal_from_hamming(k, ngbits, goal_rng)
        oracle_rep = build_oracle_rep(goal, n)

        best_fracs = Float64[]   # fraction of both-covered pairs satisfied by best alpha
        rand_fracs = Float64[]   # fraction satisfied by random alpha (baseline)

        for _ in 1:ngens
            gen_idx   = rand(rng, 0:3^nqubit-1)
            generator = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s   = rand(rng, 0:1)
            base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt       = QCScaling.Context(generator, base_cxt)

            # Smart pick with oracle rep
            alphas = QCScaling.pick_new_alphas(cxt, goal, oracle_rep, fp_packed, base_cxt)
            ns     = QCScaling.PseudoGHZState(alphas..., generator)
            n_both, n_corr = count_pair_accuracy(ns, goal, cxt_master, n)
            n_both > 0 && push!(best_fracs, n_corr / n_both)

            # Random pick (baseline)
            rand_alphas = (theta_s, rand(rng, 0:1), QCScaling.idx_to_alphas(rand(rng, 0:2^(nqubit-1)-1), nqubit))
            ns_rand = QCScaling.PseudoGHZState(rand_alphas..., generator)
            n_both_r, n_corr_r = count_pair_accuracy(ns_rand, goal, cxt_master, n)
            n_both_r > 0 && push!(rand_fracs, n_corr_r / n_both_r)
        end

        @printf("%-8.4f  %-12.4f  %-12.4f  %-12d\n",
                H_act,
                isempty(best_fracs) ? NaN : mean(best_fracs),
                isempty(rand_fracs) ? NaN : mean(rand_fracs),
                length(best_fracs))
    end
end

function main()
    nqubit   = 6
    ngens    = 5000
    base_seed = 42

    @printf("Alpha capacity test: nqubit=%d, ngens=%d\n", nqubit, ngens)
    @printf("Oracle rep: companion_goals always point in the correct direction\n")
    @printf("Shows intrinsic capacity of alpha space to satisfy goal XOR at covered pairs\n")
    flush(stdout)

    run_capacity_test(nqubit, ngens, base_seed)

    println()
    @printf("If best_pair_frac ≈ rand_pair_frac → alpha space has no extra capacity for these goals\n")
    @printf("If best_pair_frac >> rand_pair_frac → picker successfully exploits alpha structure\n")
    @printf("If best_pair_frac differs by H → asymmetric capacity between H=0 and H=1\n")
    flush(stdout)
end

main()
