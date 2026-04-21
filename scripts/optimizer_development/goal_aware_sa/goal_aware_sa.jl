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
# Shared rep helpers (same as other scripts)
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

# Update rep in-place at positions touched by `state`.
# Call this after rep_sum/rep_ctr have been updated to reflect the accepted swap.
function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index
        c = rep_ctr[i]
        if c == 0
            rep[i] = NaN
        else
            v = rep_sum[i] / c
            rep[i] = v == 0.5 ? NaN : round(v)
        end
    end
end

function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    apply_state!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    apply_state!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

function random_pghz(nqubit, rng)
    theta_s = rand(rng, 0:1); theta_z = rand(rng, 0:1)
    alphas  = rand(rng, Bool, nqubit - 1)
    gen_idx = rand(rng, 0:3^nqubit - 1)
    return QCScaling.PseudoGHZState(theta_s, theta_z, alphas, QCScaling.ParityOperator(gen_idx, nqubit))
end

# ---------------------------------------------------------------------------
# Goal-aware proposal: pick a random generator, then use pick_new_alphas to
# set alphas optimally for the current rep and goal.
# ---------------------------------------------------------------------------

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx  = rand(rng, 0:3^nqubit - 1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s  = rand(rng, 0:1)
    base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt      = QCScaling.Context(generator, base_cxt)
    alphas   = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint; p_smart=1.0, target_rate=0.8,
                      n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]

    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns = rand(rng) < p_smart ? smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng) :
                                   random_pghz(nqubit, rng)
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
# Goal-aware simulated annealing
#
# p_smart: probability of using a goal-aware (pick_new_alphas) proposal;
#          1.0 = fully goal-aware, 0.0 = same as random SA.
# ---------------------------------------------------------------------------

function run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                           n_restarts=3, p_smart=1.0, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit
    best_acc    = -Inf

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n)
        rep_ctr  = zeros(Int,     n)
        for s in ensemble
            apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
        end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint; p_smart=p_smart, rng=rng)
        current_acc  = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for _ in 1:nsteps
            which = rand(rng, 1:nstate)
            ns    = rand(rng) < p_smart ? smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng) :
                                          random_pghz(nqubit, rng)

            old_state = ensemble[which]

            # Tentative swap
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                # Accept: update incremental rep at all touched indices
                update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
                update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
                ensemble[which] = ns
                current_acc     = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                # Reject: undo
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
# Random SA baseline (no goal-aware proposals)
# ---------------------------------------------------------------------------

function run_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=3, seed=42)
    run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                      n_restarts=n_restarts, p_smart=0.0, seed=seed)
end

# ---------------------------------------------------------------------------
# Improved optimizer baseline
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Main: compare improved, random SA, and goal-aware SA
# ---------------------------------------------------------------------------

function main()
    configs = [
        # (nqubit, nstate_mult, niter_improved, sa_nsteps, sa_alpha)
        (4,  2, 2_000,    200_000, 0.999),
        (6,  3, 2_000,    500_000, 0.999),
        (8,  3, 5_000,  2_000_000, 0.999),
    ]
    pzero_vals = [0.0, 0.5]
    nseeds     = 5
    n_restarts = 3
    base_seed  = 42
    outdir     = joinpath(@__DIR__, "data")
    mkpath(outdir)

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("Seeds: $seeds\n")

    outfile = joinpath(outdir, "goal_aware_comparison.h5")
    h5open(outfile, "w") do h5f

    for (nqubit, nstate_mult, niter, sa_nsteps, sa_alpha) in configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        nstate      = nstate_mult * base_nstate
        ngbits      = (3^nqubit - 1) ÷ 2

        println("="^65)
        @printf("nqubit=%d  nstate=%d (%d×)  niter=%d  SA: nsteps=%d  alpha=%.4f\n",
                nqubit, nstate, nstate_mult, niter, sa_nsteps, sa_alpha)
        @printf("%-6s  %-10s  %-10s  %-10s\n", "pzero", "improved", "rand_sa", "goal_sa")
        println("-"^42)

        gp_nq = create_group(h5f, "nqubit_$(nqubit)")
        HDF5.attributes(gp_nq)["nqubit"]      = nqubit
        HDF5.attributes(gp_nq)["nstate"]      = nstate
        HDF5.attributes(gp_nq)["nstate_mult"] = nstate_mult
        HDF5.attributes(gp_nq)["niter"]       = niter
        HDF5.attributes(gp_nq)["sa_nsteps"]   = sa_nsteps
        HDF5.attributes(gp_nq)["sa_alpha"]    = sa_alpha

        for pzero in pzero_vals
            rng_g = Random.MersenneTwister(base_seed + round(Int, pzero * 1000))
            goals = [sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits) for _ in 1:nseeds]

            imp_accs  = Float64[]
            rsa_accs  = Float64[]
            gsa_accs  = Float64[]

            for (si, seed) in enumerate(seeds)
                push!(imp_accs,  run_improved(goals[si], nqubit, nstate, niter; seed=seed))
                push!(rsa_accs,  run_sa(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                        n_restarts=n_restarts, seed=seed))
                push!(gsa_accs,  run_goal_aware_sa(goals[si], nqubit, nstate, sa_nsteps, sa_alpha;
                                                   n_restarts=n_restarts, p_smart=1.0, seed=seed))
            end

            @printf("%-6.1f  %-10.4f  %-10.4f  %-10.4f\n",
                    pzero, median(imp_accs), median(rsa_accs), median(gsa_accs))
            flush(stdout)

            gp_pz = create_group(gp_nq, "pzero_$(pzero)")
            HDF5.attributes(gp_pz)["pzero"] = pzero
            gp_pz["improved"] = imp_accs
            gp_pz["rand_sa"]  = rsa_accs
            gp_pz["goal_sa"]  = gsa_accs
        end
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
