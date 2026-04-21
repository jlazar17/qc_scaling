using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using HDF5
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal generation (fixed Hamming weight)
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    @assert 0 <= k <= ngbits
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N
    return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0
    H >= 1.0 && return N ÷ 2
    ks = 0:N÷2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), ks)
    return ks[idx]
end

# ---------------------------------------------------------------------------
# Efficiency metric: (1 - H(x)) * N_classical / N_quantum
# where H(x) = -x*log2(x) - (1-x)*log2(1-x), x = accuracy
# ---------------------------------------------------------------------------

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function efficiency(acc::Float64, nqubit::Int, nstate::Int)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    return (1 - binary_entropy(acc)) * n_classical / n_quantum
end

# ---------------------------------------------------------------------------
# Rep helpers
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
        i1 = 2i - 1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1] / c1; p2 = rep_sum[i2] / c2
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

function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]
        rep[i] = c == 0 ? NaN : (v = rep_sum[i] / c; v == 0.5 ? NaN : round(v))
    end
end

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit - 1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns    = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
        delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

function run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                           n_restarts=3, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit
    best_acc    = -Inf

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint; rng=rng)
        current_acc  = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for _ in 1:nsteps
            which     = rand(rng, 1:nstate)
            ns        = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
            old_state = ensemble[which]

            apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
                update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
                ensemble[which] = ns
                current_acc     = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha
        end

        restart_best > best_acc && (best_acc = restart_best)
    end
    return best_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    # nqubit → (nstate_mults, sa_nsteps, sa_alpha)
    nqubit_configs = [
        (4,  [1, 2, 3, 4, 6],    200_000, 0.999),
        (6,  [1, 2, 3, 4, 6],    500_000, 0.999),
        (8,  [1, 2, 3, 4, 6],  2_000_000, 0.999),
    ]
    entropy_vals = [0.0, 0.5, 1.0]
    nseeds       = 5
    n_restarts   = 3
    base_seed    = 42
    outdir       = joinpath(@__DIR__, "data")
    mkpath(outdir)

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))
    println("Seeds: $seeds\n")

    outfile = joinpath(outdir, "scaling_study.h5")
    h5open(outfile, "w") do h5f
    HDF5.attributes(h5f)["nseeds"]    = nseeds
    HDF5.attributes(h5f)["base_seed"] = base_seed

    for (nqubit, nstate_mults, sa_nsteps, sa_alpha) in nqubit_configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        ngbits      = (3^nqubit - 1) ÷ 2
        nstates     = nstate_mults .* base_nstate

        println("="^70)
        @printf("nqubit=%d  base_nstate=%d  nstates=%s  SA: nsteps=%d  alpha=%.4f\n",
                nqubit, base_nstate, string(nstates), sa_nsteps, sa_alpha)

        gp_nq = create_group(h5f, "nqubit_$(nqubit)")
        HDF5.attributes(gp_nq)["nqubit"]      = nqubit
        HDF5.attributes(gp_nq)["base_nstate"] = base_nstate
        HDF5.attributes(gp_nq)["nstates"]     = nstates
        HDF5.attributes(gp_nq)["sa_nsteps"]   = sa_nsteps
        HDF5.attributes(gp_nq)["sa_alpha"]    = sa_alpha

        for H_target in entropy_vals
            k        = k_from_entropy(H_target, ngbits)
            H_actual = hamming_entropy(k, ngbits)
            rng_g    = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals    = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

            println("\n  H=$(H_target) (k=$(k), H_act=$(round(H_actual, digits=4)))")
            @printf("  %-6s  %-8s  %-10s  %-10s\n", "nstate", "mult", "acc(med)", "η(med)")
            println("  " * "-"^40)

            key  = @sprintf("H%.2f_k%d", H_target, k)
            gp_H = create_group(gp_nq, key)
            HDF5.attributes(gp_H)["H_target"] = H_target
            HDF5.attributes(gp_H)["H_actual"] = H_actual
            HDF5.attributes(gp_H)["k"]        = k
            HDF5.attributes(gp_H)["ngbits"]   = ngbits

            acc_mat = zeros(length(nstates), nseeds)
            eta_mat = zeros(length(nstates), nseeds)

            for (ni, nstate) in enumerate(nstates)
                for (si, seed) in enumerate(seeds)
                    acc = run_goal_aware_sa(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                           n_restarts=n_restarts, seed=seed)
                    acc_mat[ni, si] = acc
                    eta_mat[ni, si] = efficiency(acc, nqubit, nstate)
                end
                @printf("  %-6d  %-8d  %-10.4f  %-10.4f\n",
                        nstate, nstate_mults[ni],
                        median(acc_mat[ni, :]), median(eta_mat[ni, :]))
                flush(stdout)
            end

            gp_H["acc"] = acc_mat   # [n_nstates × nseeds]
            gp_H["eta"] = eta_mat
            gp_H["nstate_mults"] = nstate_mults
        end
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
