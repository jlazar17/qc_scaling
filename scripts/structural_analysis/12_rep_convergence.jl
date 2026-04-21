# Rep convergence across seeds: do different runs find the same rep?
#
# For H=0 the goal is identical across seeds. If all runs converge to the
# same rep (up to a global parity flip), that suggests a unique global
# attractor. For H=1 multiple local optima may exist.
#
# Method: run SA with N seeds, record the final majority-parity assignment
# at each position (sign of 2*rep_sum - rep_ctr). Then for each pair of
# seeds compute the agreement rate (accounting for the global flip symmetry).
# Also compare margin distributions across seeds.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

function run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                companion, goal_idx, fingerprint, cxt_master; seed=42)
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
    scratch_idxs = Vector{Int}(undef, npos); scratch_pars = Vector{Int}(undef, npos)

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
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
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc
        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs); copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return rep_sum, rep_ctr, cur_acc
end

# Majority parity at each covered position: +1, -1, or 0 (tied/uncovered)
function majority_parity(rep_sum, rep_ctr)
    n = length(rep_sum)
    out = zeros(Int, n)
    for k in 1:n
        rep_ctr[k] == 0 && continue
        v = 2 * rep_sum[k] - rep_ctr[k]
        out[k] = v > 0 ? 1 : (v < 0 ? -1 : 0)
    end
    return out
end

# Agreement between two parity vectors, maximised over global flip.
# Only counts positions covered in both.
function parity_agreement(p1, p2)
    both = findall(i -> p1[i] != 0 && p2[i] != 0, eachindex(p1))
    isempty(both) && return NaN, NaN
    agree      = count(i -> p1[i] == p2[i],  both) / length(both)
    agree_flip = count(i -> p1[i] == -p2[i], both) / length(both)
    return max(agree, agree_flip), min(agree, agree_flip)
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 8
    seeds      = [42, 17, 99, 256, 1337, 7, 314, 888]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("=" ^ 60)
        println("$H_label")
        println("=" ^ 60)

        parities = Vector{Vector{Int}}(undef, n_seeds)
        margins  = Vector{Vector{Int}}(undef, n_seeds)
        accs     = Vector{Float64}(undef, n_seeds)

        for (i, seed) in enumerate(seeds)
            print("  seed=$seed ... ")
            rs, rc, acc = run_sa(goal, nqubit, nstate, nsteps, alpha_cool,
                                  companion, goal_idx, fingerprint, cxt_master; seed=seed)
            parities[i] = majority_parity(rs, rc)
            margins[i]  = [abs(2*rs[k] - rc[k]) for k in 1:n-1]
            accs[i]     = acc
            @printf("acc=%.4f  mean_margin=%.2f  med_margin=%.1f\n",
                    acc, mean(margins[i]), median(float.(margins[i])))
        end

        println()
        println("  Pairwise parity agreement (best / worst over global flip):")
        @printf("  %-6s", "")
        for j in 1:n_seeds; @printf("  s%-5d", seeds[j]); end
        println()
        for i in 1:n_seeds
            @printf("  s%-5d", seeds[i])
            for j in 1:n_seeds
                if i == j
                    @printf("  %-6s", "  ---")
                else
                    best, _ = parity_agreement(parities[i], parities[j])
                    @printf("  %-6.3f", best)
                end
            end
            println()
        end

        println()
        println("  Cross-seed margin correlation (Pearson r):")
        @printf("  %-6s", "")
        for j in 1:n_seeds; @printf("  s%-5d", seeds[j]); end
        println()
        for i in 1:n_seeds
            @printf("  s%-5d", seeds[i])
            for j in 1:n_seeds
                if i == j
                    @printf("  %-6s", "  ---")
                else
                    r = cor(float.(margins[i]), float.(margins[j]))
                    @printf("  %-6.3f", r)
                end
            end
            println()
        end
        println()
    end
end

main()
