using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using HDF5
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal generation
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
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1]/c1; p2 = rep_sum[i2]/c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
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
                      cxt_master, fingerprint; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
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

# ---------------------------------------------------------------------------
# Phase 1: PT warm-up — returns all chain states for use as seeds
# ---------------------------------------------------------------------------

function run_pt_warmup(goal, nqubit, nstate, nsteps_warmup;
                       n_chains=3, T_ratio=20.0, swap_every=500, rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit

    ensembles = [[QCScaling.random_state(nqubit) for _ in 1:nstate] for _ in 1:n_chains]
    rep_sums  = [zeros(Float64, n) for _ in 1:n_chains]
    rep_ctrs  = [zeros(Int,     n) for _ in 1:n_chains]
    reps      = [zeros(Float64, n) for _ in 1:n_chains]

    for c in 1:n_chains
        for s in ensembles[c]; apply_state!(rep_sums[c], rep_ctrs[c], s, cxt_master, 1); end
        reps[c] = rep_from_cache(rep_sums[c], rep_ctrs[c])
    end

    T_min = calibrate_T0(ensembles[1], rep_sums[1], rep_ctrs[1], reps[1], goal, nqubit,
                          cxt_master, fingerprint; rng=rng)
    T_min = max(T_min, 1e-6)
    T_max = T_min * T_ratio
    temps = [T_min * (T_max / T_min)^((c-1) / (n_chains-1)) for c in 1:n_chains]

    current_accs = [rep_accuracy_fast(rep_sums[c], rep_ctrs[c], goal) for c in 1:n_chains]

    for step in 1:nsteps_warmup
        for c in 1:n_chains
            which     = rand(rng, 1:nstate)
            ns        = smart_proposal(nqubit, reps[c], goal, fingerprint, cxt_master, rng)
            old_state = ensembles[c][which]

            apply_state!(rep_sums[c], rep_ctrs[c], old_state, cxt_master, -1)
            apply_state!(rep_sums[c], rep_ctrs[c], ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sums[c], rep_ctrs[c], goal)
            delta   = new_acc - current_accs[c]

            if delta >= 0 || rand(rng) < exp(delta / temps[c])
                update_rep_at!(reps[c], rep_sums[c], rep_ctrs[c], old_state, cxt_master)
                update_rep_at!(reps[c], rep_sums[c], rep_ctrs[c], ns,        cxt_master)
                ensembles[c][which] = ns
                current_accs[c]     = new_acc
            else
                apply_state!(rep_sums[c], rep_ctrs[c], ns,        cxt_master, -1)
                apply_state!(rep_sums[c], rep_ctrs[c], old_state, cxt_master,  1)
            end
        end

        if step % swap_every == 0
            for c in 1:n_chains-1
                log_prob = (current_accs[c+1] - current_accs[c]) * (1/temps[c] - 1/temps[c+1])
                if log_prob >= 0 || rand(rng) < exp(log_prob)
                    ensembles[c],    ensembles[c+1]    = ensembles[c+1],    ensembles[c]
                    rep_sums[c],     rep_sums[c+1]     = rep_sums[c+1],     rep_sums[c]
                    rep_ctrs[c],     rep_ctrs[c+1]     = rep_ctrs[c+1],     rep_ctrs[c]
                    reps[c],         reps[c+1]          = reps[c+1],         reps[c]
                    current_accs[c], current_accs[c+1] = current_accs[c+1], current_accs[c]
                end
            end
        end
    end

    # Return each chain's ensemble and rep cache for use as goal_sa seeds
    return [(deepcopy(ensembles[c]), copy(rep_sums[c]), copy(rep_ctrs[c]), copy(reps[c]))
            for c in 1:n_chains]
end

# ---------------------------------------------------------------------------
# Phase 2: goal-aware SA annealing from a pre-initialized ensemble state
# ---------------------------------------------------------------------------

function run_gsa_anneal(goal, nqubit, nsteps_refine, alpha,
                        ensemble, rep_sum, rep_ctr, rep; rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    nstate      = length(ensemble)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fingerprint; rng=rng)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc    = current_acc

    for _ in 1:nsteps_refine
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
            new_acc > best_acc && (best_acc = new_acc)
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end

        T *= alpha
    end

    return best_acc
end

# ---------------------------------------------------------------------------
# Hybrid: PT warm-up → independent goal-SA annealing from each chain
#
# Budget accounting (equal to goal_sa with n_restarts=3):
#   PT phase:    n_chains × nsteps_warmup
#   anneal phase: n_chains × nsteps_refine
#   total:        n_chains × nsteps  =  3 × nsteps  ✓
#
# warmup_frac controls the split; default 1/3 warmup, 2/3 annealing.
# ---------------------------------------------------------------------------

function run_hybrid(goal, nqubit, nstate, nsteps, alpha;
                    n_chains=3, T_ratio=20.0, swap_every=500,
                    warmup_frac=1/3, seed=42)
    rng = Random.MersenneTwister(seed)

    nsteps_warmup = round(Int, nsteps * warmup_frac)
    nsteps_refine = nsteps - nsteps_warmup

    chain_states = run_pt_warmup(goal, nqubit, nstate, nsteps_warmup;
                                 n_chains=n_chains, T_ratio=T_ratio,
                                 swap_every=swap_every, rng=rng)

    best_acc = -Inf
    for (ensemble, rep_sum, rep_ctr, rep) in chain_states
        acc = run_gsa_anneal(goal, nqubit, nsteps_refine, alpha,
                             ensemble, rep_sum, rep_ctr, rep; rng=rng)
        acc > best_acc && (best_acc = acc)
    end

    return best_acc
end

# ---------------------------------------------------------------------------
# Main — same configs, seeds, and goals as full_comparison.jl
# ---------------------------------------------------------------------------

function main()
    configs = [
        # (nqubit, nstate_mult, sa_nsteps, sa_alpha)
        (4,  2,    200_000, 0.999),
        (6,  3,    500_000, 0.999),
        (8,  3,  2_000_000, 0.999),
    ]
    entropy_vals = [0.0, 0.5, 1.0]
    pzero_vals   = [0.0, 0.5]
    nseeds       = 5
    base_seed    = 42
    outdir       = joinpath(@__DIR__, "data")
    mkpath(outdir)

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))
    println("Seeds: $seeds")
    println("Note: same seeds and goals as full_comparison.jl for direct comparability.\n")
    println("Budget: 3 × nsteps total (1/3 PT warmup, 2/3 goal-SA annealing)\n")

    outfile = joinpath(outdir, "hybrid_pt_gsa.h5")
    h5open(outfile, "w") do h5f

    for (nqubit, nstate_mult, sa_nsteps, sa_alpha) in configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        nstate      = nstate_mult * base_nstate
        ngbits      = (3^nqubit - 1) ÷ 2
        nsteps_warmup = round(Int, sa_nsteps * (1/3))
        nsteps_refine = sa_nsteps - nsteps_warmup

        println("="^70)
        @printf("nqubit=%d  nstate=%d (%d×)  nsteps=%d  alpha=%.4f\n",
                nqubit, nstate, nstate_mult, sa_nsteps, sa_alpha)
        @printf("  warmup=%d steps (PT, 3 chains)  refine=%d steps (goal-SA × 3)\n",
                nsteps_warmup, nsteps_refine)
        @printf("%-22s  %-12s\n", "condition", "hybrid_pt_gsa")
        println("-"^40)

        gp_nq = create_group(h5f, "nqubit_$(nqubit)")
        HDF5.attributes(gp_nq)["nqubit"]        = nqubit
        HDF5.attributes(gp_nq)["nstate"]        = nstate
        HDF5.attributes(gp_nq)["nstate_mult"]   = nstate_mult
        HDF5.attributes(gp_nq)["nsteps"]        = sa_nsteps
        HDF5.attributes(gp_nq)["nsteps_warmup"] = nsteps_warmup
        HDF5.attributes(gp_nq)["nsteps_refine"] = nsteps_refine
        HDF5.attributes(gp_nq)["alpha"]         = sa_alpha

        # Fixed-Hamming sweep
        for H_target in entropy_vals
            k     = k_from_entropy(H_target, ngbits)
            H_act = hamming_entropy(k, ngbits)
            rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

            accs = Float64[]
            for (si, seed) in enumerate(seeds)
                acc = run_hybrid(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                 n_chains=3, warmup_frac=1/3, seed=seed)
                push!(accs, acc)
            end

            label = @sprintf("H=%.2f k=%-5d H_act=%.4f", H_target, k, H_act)
            @printf("  %-20s  %-12.4f\n", label, median(accs))
            flush(stdout)

            key = @sprintf("H%.2f_k%d", H_target, k)
            gp  = create_group(gp_nq, key)
            HDF5.attributes(gp)["mode"]     = "hamming"
            HDF5.attributes(gp)["H_target"] = H_target
            HDF5.attributes(gp)["H_actual"] = H_act
            HDF5.attributes(gp)["k"]        = k
            HDF5.attributes(gp)["ngbits"]   = ngbits
            gp["hybrid_pt_gsa"] = accs
        end

        # Bernoulli sweep
        for pzero in pzero_vals
            rng_g = Random.MersenneTwister(base_seed + round(Int, pzero * 10_000) + 99_999)
            goals = [sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits) for _ in 1:nseeds]
            H_emp = mean(hamming_entropy(sum(g), length(g)) for g in goals)

            accs = Float64[]
            for (si, seed) in enumerate(seeds)
                acc = run_hybrid(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                 n_chains=3, warmup_frac=1/3, seed=seed)
                push!(accs, acc)
            end

            label = @sprintf("pzero=%.1f H_emp=%.4f     ", pzero, H_emp)
            @printf("  %-20s  %-12.4f\n", label, median(accs))
            flush(stdout)

            key = @sprintf("pzero_%.1f_H_emp=%.4f", pzero, H_emp)
            gp  = create_group(gp_nq, replace(key, "." => "p"))
            HDF5.attributes(gp)["mode"]        = "bernoulli"
            HDF5.attributes(gp)["pzero"]       = pzero
            HDF5.attributes(gp)["H_empirical"] = H_emp
            HDF5.attributes(gp)["ngbits"]      = ngbits
            gp["hybrid_pt_gsa"] = accs
        end
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
