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
# Shared helpers (copied from run_comparison.jl)
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

function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    apply_state!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    apply_state!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

function run_base(goal, nqubit, nstate, niter; seed=42, nreplace=1)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    best_acc    = -Inf
    for _ in 1:niter
        scores = QCScaling.score(states, goal, cxt_master)
        sorter = sortperm(scores); scores = scores[sorter]; states = states[sorter]
        rep    = QCScaling.calculate_representation(states)
        acc    = accuracy(rep, goal)
        acc > best_acc && (best_acc = acc)
        if rand() > 1e-3
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace; replace=false)
            for which in whiches
                rplc = states[which]
                base_cxt = rplc.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                cxt      = QCScaling.Context(rplc.generator, base_cxt)
                alphas   = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                states[which] = QCScaling.PseudoGHZState(alphas..., rplc.generator)
            end
        else
            replace_idx   = rand(1:length(states))
            new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
            new_generator = first(new_cxt.pos)
            base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
            alphas        = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
            states[replace_idx] = QCScaling.PseudoGHZState(alphas..., new_generator)
        end
    end
    return best_acc
end

function run_improved(goal, nqubit, nstate, niter; seed=42, nreplace=1,
                      p_mutate=0.3, n_same_tol=10)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum = zeros(Float64, 3^nqubit); rep_ctr = zeros(Int, 3^nqubit)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)
    best_acc = -Inf; n_same = 0; last_acc = -1.0
    for _ in 1:niter
        scores = QCScaling.score(states, rep, goal, cxt_master)
        sorter = sortperm(scores); scores = scores[sorter]; states = states[sorter]
        acc = accuracy(rep, goal)
        acc > best_acc && (best_acc = acc)
        n_same = (acc == last_acc) ? n_same + 1 : 0; last_acc = acc
        if n_same <= n_same_tol
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace; replace=false)
            for which in whiches
                rplc = states[which]
                if rand() < p_mutate
                    new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(new_cxt.pos)
                    base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = new_cxt
                else
                    new_generator = rplc.generator
                    base_cxt      = rplc.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = QCScaling.Context(new_generator, base_cxt)
                end
                alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                replace_state!(states, which, QCScaling.PseudoGHZState(alphas..., new_generator),
                               rep_sum, rep_ctr, cxt_master)
            end
        else
            n_same = 0
            for _ in 1:nreplace
                replace_idx   = rand(1:length(states))
                new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                new_generator = first(new_cxt.pos)
                base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                alphas        = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
                replace_state!(states, replace_idx, QCScaling.PseudoGHZState(alphas..., new_generator),
                               rep_sum, rep_ctr, cxt_master)
            end
        end
        rep = rep_from_cache(rep_sum, rep_ctr)
    end
    return best_acc
end

function random_pghz(nqubit, rng)
    theta_s = rand(rng, 0:1); theta_z = rand(rng, 0:1)
    alphas  = rand(rng, Bool, nqubit - 1)
    gen_idx = rand(rng, 0:3^nqubit - 1)
    return QCScaling.PseudoGHZState(theta_s, theta_z, alphas, QCScaling.ParityOperator(gen_idx, nqubit))
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, goal, nqubit, cxt_master;
                      target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate); ns = random_pghz(nqubit, rng)
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

function run_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=3, seed=42)
    rng = Random.MersenneTwister(seed); cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit; best_acc = -Inf
    for _ in 1:n_restarts
        ensemble = [random_pghz(nqubit, rng) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        T = calibrate_T0(ensemble, rep_sum, rep_ctr, goal, nqubit, cxt_master; rng=rng)
        current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal); restart_best = current_acc
        for _ in 1:nsteps
            which = rand(rng, 1:nstate); ns = random_pghz(nqubit, rng)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal); delta = new_acc - current_acc
            if delta >= 0 || rand(rng) < exp(delta / T)
                ensemble[which] = ns; current_acc = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
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
    nqubit       = 8
    base_nstate  = Int(ceil(3^nqubit / 2^(nqubit - 1)))   # 52
    nstate_mults = [1, 2, 3, 4, 6]
    nstates      = nstate_mults .* base_nstate
    ngbits       = (3^nqubit - 1) ÷ 2
    niter        = 5_000
    sa_nsteps    = 2_000_000
    sa_alpha     = 0.999
    n_restarts   = 3
    nseeds       = 5
    pzero_vals   = [0.0, 0.5]
    base_seed    = 42
    outdir       = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile      = joinpath(outdir, "nstate_sweep_nqubit$(nqubit).h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("nqubit=$nqubit  base_nstate=$base_nstate  nstates=$nstates")
    println("niter=$niter  SA: alpha=$sa_alpha  nsteps=$sa_nsteps  n_restarts=$n_restarts")
    println("Seeds: $seeds\n")

    h5open(outfile, "w") do h5f
        attributes(h5f)["nstates"]    = nstates
        attributes(h5f)["pzero_vals"] = pzero_vals
        attributes(h5f)["nseeds"]     = nseeds
        attributes(h5f)["niter"]      = niter
        attributes(h5f)["sa_nsteps"]  = sa_nsteps
        attributes(h5f)["sa_alpha"]   = sa_alpha

        for pzero in pzero_vals
            println("="^55)
            println("pzero=$pzero")
            @printf("%-6s  %-10s  %-10s  %-10s\n", "nstate", "base", "improved", "sa")
            println("-"^42)

            pgp     = create_group(h5f, "pzero_$(pzero)")
            results = Dict("base" => zeros(length(nstates), nseeds),
                           "improved" => zeros(length(nstates), nseeds),
                           "sa"       => zeros(length(nstates), nseeds))

            rng_g = Random.MersenneTwister(base_seed + round(Int, pzero * 1000))
            goals = [sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits) for _ in 1:nseeds]

            for (ni, nstate) in enumerate(nstates)
                for (si, seed) in enumerate(seeds)
                    results["base"][ni, si]     = run_base(goals[si],     nqubit, nstate, niter; seed=seed)
                    results["improved"][ni, si] = run_improved(goals[si], nqubit, nstate, niter; seed=seed)
                    results["sa"][ni, si]       = run_sa(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                                         n_restarts=n_restarts, seed=seed)
                end
                b = median(results["base"][ni, :])
                i = median(results["improved"][ni, :])
                s = median(results["sa"][ni, :])
                @printf("%-6d  %-10.4f  %-10.4f  %-10.4f\n", nstate, b, i, s)
                flush(stdout)
            end

            for (meth, mat) in results
                pgp[meth] = mat   # [n_nstates × nseeds]
            end
        end
    end

    println("\nSaved to $outfile")
end

main()
