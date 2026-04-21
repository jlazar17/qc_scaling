using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using HDF5
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal / entropy helpers (identical to scaling_study_adaptive.jl)
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
# Rep helpers (identical to scaling_study_adaptive.jl)
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

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr; pref[pref .== 0.5] .= NaN; return round.(pref)
end

function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]
        rep[i] = c == 0 ? NaN : (v = rep_sum[i]/c; v == 0.5 ? NaN : round(v))
    end
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

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint, use_smart; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns    = use_smart ? smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng) :
                            QCScaling.random_state(nqubit)
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

# ---------------------------------------------------------------------------
# SA with use_smart flag
# ---------------------------------------------------------------------------

function run_sa(goal, nqubit, nstate, nsteps, alpha, use_smart; n_restarts=3, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2
    best_acc    = -Inf

    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint, use_smart; rng=rng)
        current_acc      = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best     = current_acc
        last_improvement = 0

        for step in 1:nsteps
            which     = rand(rng, 1:nstate)
            ns        = use_smart ? smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng) :
                                    QCScaling.random_state(nqubit)
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
                if new_acc > restart_best + min_delta
                    restart_best     = new_acc
                    last_improvement = step
                end
                new_acc > best_acc && (best_acc = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha

            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit      = 6
    nstate      = 40        # efficiency-peak nstate for n=6 H=0.0; H=1.0 acc=0.755 here
    nsteps      = 500_000
    alpha       = 0.99999
    n_restarts  = 3
    nseeds      = 10        # goals per entropy level
    base_seed   = 42
    entropy_targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "ablation_smart_vs_random.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    @printf("nqubit=%d  nstate=%d  nseeds=%d\n\n", nqubit, nstate, nseeds)
    @printf("%-8s  %-6s  %-10s  %-10s  %-10s  %-10s\n",
            "H_act", "k", "acc_smart", "acc_rand", "eta_smart", "eta_rand")
    println("-"^66)
    flush(stdout)

    h5open(outfile, "w") do h5f
        HDF5.attributes(h5f)["nqubit"]    = nqubit
        HDF5.attributes(h5f)["nstate"]    = nstate
        HDF5.attributes(h5f)["nseeds"]    = nseeds
        HDF5.attributes(h5f)["base_seed"] = base_seed

        H_acts   = Float64[]
        ks       = Int[]
        acc_smart_mat = Matrix{Float64}(undef, length(entropy_targets), nseeds)
        acc_rand_mat  = Matrix{Float64}(undef, length(entropy_targets), nseeds)

        for (hi, H_target) in enumerate(entropy_targets)
            k     = k_from_entropy(H_target, ngbits)
            H_act = hamming_entropy(k, ngbits)
            push!(H_acts, H_act); push!(ks, k)

            rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

            for (si, seed) in enumerate(seeds)
                acc_smart_mat[hi, si] = run_sa(goals[si], nqubit, nstate, nsteps, alpha, true;
                                               n_restarts=n_restarts, seed=seed)
                acc_rand_mat[hi, si]  = run_sa(goals[si], nqubit, nstate, nsteps, alpha, false;
                                               n_restarts=n_restarts, seed=seed)
            end

            ms = median(acc_smart_mat[hi, :])
            mr = median(acc_rand_mat[hi, :])
            @printf("%-8.4f  %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
                    H_act, k, ms, mr,
                    efficiency(ms, nqubit, nstate), efficiency(mr, nqubit, nstate))
            flush(stdout)
        end

        h5f["H_acts"]        = H_acts
        h5f["ks"]            = ks
        h5f["acc_smart_mat"] = acc_smart_mat
        h5f["acc_rand_mat"]  = acc_rand_mat
    end

    println("\nSaved to $outfile")
end

main()
