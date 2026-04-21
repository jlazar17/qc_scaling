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
# Shared rep helpers
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

# ---------------------------------------------------------------------------
# Base optimizer (faithful stripped version)
#
# Key properties vs improved:
#   - No generator mutation in normal branch (alphas only)
#   - Full rep recompute each iteration via calculate_representation
#   - Escape fires randomly at prob 1e-3, does a single generator mutation
#   - Uses score(states, goal, cxt_master) — no cached rep
# ---------------------------------------------------------------------------

function run_base(goal, nqubit, nstate, niter; seed=42, nreplace=1)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]

    accuracies = zeros(Float64, niter)

    for iter in 1:niter
        scores = QCScaling.score(states, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]

        rep = QCScaling.calculate_representation(states)
        accuracies[iter] = accuracy(rep, goal)

        if rand() > 1e-3
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace; replace=false)
            for which in whiches
                rplc     = states[which]
                base_cxt = rplc.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                cxt      = QCScaling.Context(rplc.generator, base_cxt)
                alphas   = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                states[which] = QCScaling.PseudoGHZState(alphas..., rplc.generator)
            end
        else
            # Escape: generator mutation on one randomly chosen state
            replace_idx   = rand(1:length(states))
            new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
            new_generator = first(new_cxt.pos)
            base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
            alphas        = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
            states[replace_idx] = QCScaling.PseudoGHZState(alphas..., new_generator)
        end
    end

    return maximum(accuracies), accuracies
end

# ---------------------------------------------------------------------------
# Improved optimizer (stripped of HDF5 I/O)
# ---------------------------------------------------------------------------

function run_improved(goal, nqubit, nstate, niter; seed=42, nreplace=1,
                      p_mutate=0.3, n_same_tol=10)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]

    rep_sum = zeros(Float64, 3^nqubit)
    rep_ctr = zeros(Int,     3^nqubit)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    accuracies = zeros(Float64, niter)
    n_same = 0; last_acc = -1.0

    for iter in 1:niter
        scores = QCScaling.score(states, rep, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]

        acc = accuracy(rep, goal)
        accuracies[iter] = acc
        n_same   = (acc == last_acc) ? n_same + 1 : 0
        last_acc = acc

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

    return maximum(accuracies), accuracies
end

# ---------------------------------------------------------------------------
# SA (random state generation, with optional curve tracking)
# ---------------------------------------------------------------------------

function random_pghz(nqubit, rng)
    theta_s = rand(rng, 0:1)
    theta_z = rand(rng, 0:1)
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

function run_sa(goal, nqubit, nstate, nsteps, alpha; n_restarts=3, seed=42,
                checkpoint_every=nothing)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    tracking   = !isnothing(checkpoint_every)
    total_steps = nsteps * n_restarts
    n_ckpts    = tracking ? div(total_steps, checkpoint_every) : 0
    acc_curve  = tracking ? fill(-Inf, n_ckpts) : Float64[]
    best_acc   = -Inf
    global_step = 0

    for _ in 1:n_restarts
        ensemble = [random_pghz(nqubit, rng) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end

        T            = calibrate_T0(ensemble, rep_sum, rep_ctr, goal, nqubit, cxt_master; rng=rng)
        current_acc  = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for step in 1:nsteps
            which = rand(rng, 1:nstate); ns = random_pghz(nqubit, rng)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc
            if delta >= 0 || rand(rng) < exp(delta / T)
                ensemble[which] = ns; current_acc = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
            end
            T *= alpha

            if tracking
                global_step += 1
                if global_step % checkpoint_every == 0
                    ci = div(global_step, checkpoint_every)
                    best_acc = max(best_acc, restart_best)
                    acc_curve[ci] = best_acc
                end
            end
        end
        restart_best > best_acc && (best_acc = restart_best)
    end

    return tracking ? (best_acc, acc_curve) : (best_acc, Float64[])
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit      = 8
    nstate      = 3 * Int(ceil(3^nqubit / 2^(nqubit - 1)))   # 156
    ngbits      = (3^nqubit - 1) ÷ 2
    niter       = 5_000
    sa_nsteps   = 2_000_000
    sa_alpha    = 0.999
    n_restarts  = 3
    nseeds      = 10
    pzero_vals  = [0.0, 0.5]
    base_seed   = 42
    outdir      = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile     = joinpath(outdir, "comparison_nqubit$(nqubit).h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("nqubit=$nqubit  nstate=$nstate  niter=$niter")
    println("SA: alpha=$sa_alpha  nsteps=$sa_nsteps  n_restarts=$n_restarts")
    println("Seeds: $seeds\n")

    h5open(outfile, "w") do h5f
        # ---------------------------------------------------------------
        # Three-way comparison
        # ---------------------------------------------------------------
        for pzero in pzero_vals
            println("=== pzero=$pzero ===")
            @printf("%-12s  %-10s  %-10s  %-10s\n", "seed", "base", "improved", "sa")
            println("-"^48)

            gp = create_group(h5f, "pzero_$(pzero)")
            base_accs = Float64[]; impr_accs = Float64[]; sa_accs = Float64[]
            base_curves = Matrix{Float64}(undef, niter,   nseeds)
            impr_curves = Matrix{Float64}(undef, niter,   nseeds)

            rng_g = Random.MersenneTwister(base_seed + round(Int, pzero * 1000))
            for (si, seed) in enumerate(seeds)
                goal = sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits)

                b_best, b_curve = run_base(goal,     nqubit, nstate, niter; seed=seed)
                i_best, i_curve = run_improved(goal, nqubit, nstate, niter; seed=seed)
                s_best, _       = run_sa(goal, nqubit, nstate, sa_nsteps, sa_alpha;
                                         n_restarts=n_restarts, seed=seed)

                push!(base_accs, b_best); push!(impr_accs, i_best); push!(sa_accs, s_best)
                base_curves[:, si] = b_curve
                impr_curves[:, si] = i_curve

                @printf("%-12d  %-10.4f  %-10.4f  %-10.4f\n", seed, b_best, i_best, s_best)
                flush(stdout)
            end

            gp["base_best"]     = base_accs
            gp["improved_best"] = impr_accs
            gp["sa_best"]       = sa_accs
            gp["base_curves"]   = base_curves
            gp["impr_curves"]   = impr_curves
            attributes(gp)["niter"]    = niter
            attributes(gp)["nseeds"]   = nseeds
            attributes(gp)["sa_nsteps"] = sa_nsteps
            attributes(gp)["sa_alpha"] = sa_alpha

            println("-"^48)
            @printf("%-12s  %-10.4f  %-10.4f  %-10.4f\n",
                    "median", median(base_accs), median(impr_accs), median(sa_accs))
            println()
        end

        # ---------------------------------------------------------------
        # SA accuracy curves for different alphas (pzero=0.5, one goal)
        # ---------------------------------------------------------------
        println("=== SA curves (pzero=0.5) ===")
        ckpt_every = 20_000
        sa_alphas  = [0.999, 0.9999, 0.99999]
        rng_c      = Random.MersenneTwister(base_seed + 999)
        goal_curve = sample(rng_c, 0:1, Weights([0.5, 0.5]), ngbits)
        cgp        = create_group(h5f, "sa_curves")
        attributes(cgp)["checkpoint_every"] = ckpt_every
        attributes(cgp)["nsteps"]           = sa_nsteps
        attributes(cgp)["n_restarts"]       = n_restarts
        attributes(cgp)["nstate"]           = nstate

        for alpha in sa_alphas
            _, curve = run_sa(goal_curve, nqubit, nstate, sa_nsteps, alpha;
                              n_restarts=n_restarts, seed=base_seed,
                              checkpoint_every=ckpt_every)
            cgp["alpha_$(alpha)"] = curve
            @printf("  alpha=%.5f  final_acc=%.4f\n", alpha, curve[end])
            flush(stdout)
        end
    end

    println("\nSaved to $outfile")
end

main()
