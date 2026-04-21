# Run the same saturation test as script 04 but with the shuffled (non-co-occurring)
# pairing, to see if both H=0 and H=1 converge at the same nstate.
#
# If shuffled pairing equalises the nstate-to-convergence, the both-covered pair
# mechanism is the full story. If a gap persists, something else is at play.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# SA loop
# ---------------------------------------------------------------------------

function run_sa_shuffled(goal, nqubit, nstate, nsteps, alpha_cool,
                          companion, goal_idx, fingerprint;
                          n_restarts=3, seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit
    ngbits = length(goal)
    n_rep = n - 1
    stag_window = round(Int, -5.0 / log(alpha_cool))
    npos = length(cxt_master.base_even.pos)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars  = Vector{Int}(undef, npos)
    min_delta = 1.0 / ngbits

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

        acc_fn(rs, rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
        current_acc = acc_fn(rep_sum, rep_ctr)

        bad_deltas = Float64[]
        for _ in 1:500
            which = rand(rng, 1:nstate)
            gen_idx = rand(rng, 0:n-1)
            gen = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s = rand(rng, 0:1)
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, base_cxt)
            alphas = pick_alphas_s(cxt, goal, rep, fingerprint, base_cxt,
                                               companion, goal_idx, n_rep)
            ns = QCScaling.PseudoGHZState(alphas..., gen)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
            delta = acc_fn(rep_sum, rep_ctr) - current_acc
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
            delta < 0 && push!(bad_deltas, abs(delta))
        end
        T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
        current_acc = acc_fn(rep_sum, rep_ctr)
        last_improvement = 0

        for step in 1:nsteps
            which = rand(rng, 1:nstate)
            gen_idx = rand(rng, 0:n-1)
            gen = QCScaling.ParityOperator(gen_idx, nqubit)
            theta_s = rand(rng, 0:1)
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, base_cxt)
            alphas = pick_alphas_s(cxt, goal, rep, fingerprint, base_cxt,
                                               companion, goal_idx, n_rep)
            ns = QCScaling.PseudoGHZState(alphas..., gen)

            fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
            new_acc = acc_fn(rep_sum, rep_ctr)
            delta = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
                update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
                copy!(cache_idxs[which], scratch_idxs); copy!(cache_pars[which], scratch_pars)
                ensemble[which] = ns; current_acc = new_acc
                new_acc > best_acc && (best_acc = new_acc)
                new_acc > (last_improvement > 0 ? best_acc - min_delta : -Inf) && (last_improvement = step)
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
    (1 - binary_entropy(acc)) * (3^nqubit - 1) / 2 / (nqubit * nstate)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit = 6
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2

    println("Building shuffled pairing...")
    companion, goal_idx, npairs = build_shuffled_pairing(nqubit)
    @assert npairs == ngbits
    println("  Done: $npairs pairs, zero both-covered pairs")

    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))

    nstate_min = Int(ceil(3^nqubit / 2^(nqubit-1)))
    nstates = unique(round.(Int, 10 .^ range(log10(nstate_min), log10(16*nstate_min), length=16)))
    nsteps = 80_000; alpha = 0.9999; n_seeds = 10; n_restarts = 3

    @printf("nstate grid: %s\n", string(nstates))
    @printf("nsteps=%d, n_seeds=%d\n\n", nsteps, n_seeds)

    println("="^80)
    println("Shuffled pairing saturation: H=0.00 vs H=1.00")
    println("Comparing with original pairing (from script 04) side-by-side")
    println("="^80)
    @printf("%-8s  %-12s  %-12s  %-14s  %-10s\n",
            "nstate", "H0 acc_max", "H1 acc_max", "H0/H1 eta_max", "eta ratio")
    println(repeat("-", 65))

    H0_results = Dict{Int, Float64}()
    H1_results = Dict{Int, Float64}()

    for nstate in nstates
        accs_H0 = zeros(Float64, n_seeds)
        accs_H1 = zeros(Float64, n_seeds)

        seeds = rand(Random.MersenneTwister(7), UInt32, n_seeds)

        Threads.@threads for si in 1:n_seeds
            rng = Random.MersenneTwister(seeds[si])
            seed = Int(seeds[si])
            goal_H0 = zeros(Int, ngbits)
            goal_H1 = Random.shuffle!(rng, vcat(ones(Int, ngbits÷2), zeros(Int, ngbits-ngbits÷2)))

            accs_H0[si] = run_sa_shuffled(goal_H0, nqubit, nstate, nsteps, alpha,
                                           companion, goal_idx, fingerprint;
                                           n_restarts=n_restarts, seed=seed)
            accs_H1[si] = run_sa_shuffled(goal_H1, nqubit, nstate, nsteps, alpha,
                                           companion, goal_idx, fingerprint;
                                           n_restarts=n_restarts, seed=seed+1)
        end

        eta_H0 = maximum(efficiency.(accs_H0, nqubit, nstate))
        eta_H1 = maximum(efficiency.(accs_H1, nqubit, nstate))
        acc_H0 = maximum(accs_H0)
        acc_H1 = maximum(accs_H1)
        H0_results[nstate] = acc_H0; H1_results[nstate] = acc_H1

        ratio = eta_H0 > 0 ? eta_H1 / eta_H0 : NaN
        @printf("%-8d  %-12.4f  %-12.4f  %-6.4f / %-6.4f  %-10.3f\n",
                nstate, acc_H0, acc_H1, eta_H0, eta_H1, ratio)
        flush(stdout)
    end

    println()
    println("Summary: accuracy gap (H=0 acc - H=1 acc)")
    println("Original pairing (from script 04) vs shuffled pairing:")
    orig_H0 = Dict(23=>0.6731, 33=>0.8214, 48=>0.9368, 70=>0.9945,
                   84=>1.0000, 121=>1.0000, 176=>1.0000)
    orig_H1 = Dict(23=>0.5632, 33=>0.6676, 48=>0.7775, 70=>0.8846,
                   84=>0.9066, 121=>0.9808, 176=>1.0000)
    @printf("%-8s  %-14s  %-14s\n", "nstate", "orig gap", "shuffled gap")
    println(repeat("-", 40))
    for nstate in nstates
        shuf_acc0 = get(H0_results, nstate, NaN)
        shuf_acc1 = get(H1_results, nstate, NaN)
        # Find closest nstate in original
        closest = argmin(abs.(collect(keys(orig_H0)) .- nstate))
        ns_orig = collect(keys(orig_H0))[closest]
        orig_gap = get(orig_H0, ns_orig, NaN) - get(orig_H1, ns_orig, NaN)
        shuf_gap = shuf_acc0 - shuf_acc1
        @printf("%-8d  %-14.4f  %-14.4f\n", nstate, orig_gap, shuf_gap)
    end
end

main()
