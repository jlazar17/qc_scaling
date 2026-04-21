# Verify there is no hard ceiling for H=1 accuracy.
#
# If both H=0 and H=1 converge to the same accuracy at large nstate,
# then the gap is NOT a fundamental barrier — H=1 just needs more states
# (because per-state efficiency is lower due to both-covered pair constraints).
#
# Also tests the "2x more states" prediction: does H=1 need ~2x nstate to
# achieve the same efficiency as H=0?

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

function run_sa(goal, nqubit, nstate, nsteps, alpha_cool;
                n_restarts=3, seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit
    ngbits = length(goal)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))

    stag_window = round(Int, -5.0 / log(alpha_cool))
    npos = length(cxt_master.base_even.pos)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars  = Vector{Int}(undef, npos)

    best_acc = -Inf

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
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

        current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        bad_deltas = Float64[]
        for _ in 1:500
            which = rand(rng, 1:nstate)
            gen_idx = rand(rng, 0:n-1)
            gen = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s = rand(rng, 0:1)
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, base_cxt)
            alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
            ns = QCScaling.PseudoGHZState(alphas..., gen)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
            delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
            delta < 0 && push!(bad_deltas, abs(delta))
        end
        T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
        current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        last_improvement = 0
        min_delta = 1.0 / ngbits

        for step in 1:nsteps
            which = rand(rng, 1:nstate)
            gen_idx = rand(rng, 0:n-1)
            gen = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s = rand(rng, 0:1)
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, base_cxt)
            alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
            ns = QCScaling.PseudoGHZState(alphas..., gen)

            fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
                update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
                copy!(cache_idxs[which], scratch_idxs)
                copy!(cache_pars[which], scratch_pars)
                ensemble[which] = ns
                current_acc = new_acc
                new_acc > best_acc && (best_acc = new_acc)
                new_acc > (best_acc - min_delta) + min_delta && (last_improvement = step)
            else
                apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
                apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
            end
            T *= alpha_cool
            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

function binary_entropy(x)
    (x <= 0.0 || x >= 1.0) && return 0.0
    -x*log2(x) - (1-x)*log2(1-x)
end

function efficiency(acc, nqubit, nstate)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    (1 - binary_entropy(acc)) * n_classical / n_quantum
end

function main()
    nqubit = 6
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2

    # Extended nstate range: both small (peak H=0) and large (where H=1 should converge)
    nstate_min = Int(ceil(3^nqubit / 2^(nqubit-1)))
    # Extend to 16x the min to let H=1 fully converge
    nstates = unique(round.(Int, 10 .^ range(log10(nstate_min), log10(16*nstate_min), length=16)))

    nsteps = 80_000
    alpha  = 0.9999
    n_seeds = 10
    n_restarts = 3

    rng_goals = Random.MersenneTwister(99)
    seeds = rand(Random.MersenneTwister(7), UInt32, n_seeds)

    @printf("nstate_min=%d, grid: %s\n", nstate_min, string(nstates))
    @printf("nsteps=%d, n_seeds=%d\n\n", nsteps, n_seeds)

    println("="^75)
    println("Extended scaling: H=0.00 vs H=1.00")
    println("Testing whether H=1 accuracy converges to H=0 accuracy at large nstate")
    println("="^75)
    @printf("%-8s  %-12s  %-12s  %-12s  %-12s  %-10s\n",
            "nstate", "H0 acc_max", "H1 acc_max", "H0 eta_max", "H1 eta_max", "eta ratio")
    println(repeat("-", 75))

    for H_target in [0.0, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        goals = [Random.shuffle!(Random.MersenneTwister(si*100), vcat(ones(Int, goal_ones), zeros(Int, ngbits-goal_ones))) for si in 1:n_seeds]

        H_target == 0.0 && (H0_accs_per_nstate = Float64[])
        H_target == 1.0 && (H1_accs_per_nstate = Float64[])
        H_target == 0.0 && (H0_nstates_list = Int[])
        H_target == 1.0 && (H1_nstates_list = Int[])
    end

    H0_results = Dict{Int, Float64}()
    H1_results = Dict{Int, Float64}()

    for nstate in nstates
        accs_H0 = zeros(Float64, n_seeds)
        accs_H1 = zeros(Float64, n_seeds)

        Threads.@threads for si in 1:n_seeds
            rng = Random.MersenneTwister(seeds[si])
            seed = Int(seeds[si])

            goal_H0 = zeros(Int, ngbits)
            goal_H1 = Random.shuffle!(rng, vcat(ones(Int, ngbits÷2), zeros(Int, ngbits-ngbits÷2)))

            accs_H0[si] = run_sa(goal_H0, nqubit, nstate, nsteps, alpha;
                                  n_restarts=n_restarts, seed=seed)
            accs_H1[si] = run_sa(goal_H1, nqubit, nstate, nsteps, alpha;
                                  n_restarts=n_restarts, seed=seed+1)
        end

        eta_H0 = maximum(efficiency.(accs_H0, nqubit, nstate))
        eta_H1 = maximum(efficiency.(accs_H1, nqubit, nstate))
        acc_H0 = maximum(accs_H0)
        acc_H1 = maximum(accs_H1)
        H0_results[nstate] = acc_H0
        H1_results[nstate] = acc_H1

        ratio = eta_H0 > 0 ? eta_H1 / eta_H0 : NaN
        @printf("%-8d  %-12.4f  %-12.4f  %-12.4f  %-12.4f  %-10.3f\n",
                nstate, acc_H0, acc_H1, eta_H0, eta_H1, ratio)
        flush(stdout)
    end

    println()
    println("="^75)
    println("Summary: accuracy vs nstate for H=0 and H=1")
    println("Key test: at large nstate, do accuracies converge?")
    println("="^75)
    for nstate in nstates
        acc0 = get(H0_results, nstate, NaN)
        acc1 = get(H1_results, nstate, NaN)
        @printf("  nstate=%4d: H=0 acc=%.4f  H=1 acc=%.4f  diff=%.4f\n",
                nstate, acc0, acc1, acc0 - acc1)
    end
end

main()
