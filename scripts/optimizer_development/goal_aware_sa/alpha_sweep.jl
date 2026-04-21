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
# Single SA run — full budget, logs trajectory of acc/rate/T every log_every
# ---------------------------------------------------------------------------

function run_gsa_full(goal, nqubit, nstate, nsteps, alpha; log_every=5000, rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fingerprint; rng=rng)
    T0          = T
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc    = current_acc

    acc_window = 10_000
    accept_buf = zeros(Bool, acc_window)
    n_accepted = 0

    traj_steps = Int[]
    traj_accs  = Float64[]
    traj_rates = Float64[]
    traj_temps = Float64[]

    for step in 1:nsteps
        which     = rand(rng, 1:nstate)
        ns        = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
        old_state = ensemble[which]

        apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta   = new_acc - current_acc

        accepted = delta >= 0 || rand(rng) < exp(delta / T)
        if accepted
            update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
            update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
            ensemble[which] = ns
            current_acc     = new_acc
            new_acc > best_acc && (best_acc = new_acc)
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end

        buf_idx = mod1(step, acc_window)
        n_accepted += Int(accepted) - Int(accept_buf[buf_idx])
        accept_buf[buf_idx] = accepted

        T *= alpha

        if step % log_every == 0
            rolling_rate = step >= acc_window ? n_accepted / acc_window : n_accepted / step
            push!(traj_steps, step)
            push!(traj_accs,  best_acc)
            push!(traj_rates, rolling_rate)
            push!(traj_temps, T)
            @printf("    step %8d  acc=%.4f  rate=%.5f  T=%.2e\n",
                    step, best_acc, rolling_rate, T)
            flush(stdout)
        end
    end

    return (
        final_acc  = best_acc,
        T0         = T0,
        traj_steps = traj_steps,
        traj_accs  = traj_accs,
        traj_rates = traj_rates,
        traj_temps = traj_temps,
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit      = 8
    nstate_mult = 3
    nsteps      = 2_000_000
    nseeds      = 10          # enough to see the pattern; fast to run
    base_seed   = 7777
    H_targets   = [0.0, 1.0]
    log_every   = 20_000      # coarser logging — 100 points per run

    # Alphas chosen so T reaches min_delta ≈ 1/ngbits at ~0.15%, 1.5%, 15%, 50% of run
    alphas = [0.999, 0.9999, 0.99999, 0.999997]

    base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
    nstate      = nstate_mult * base_nstate
    ngbits      = (3^nqubit - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "alpha_sweep.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("nqubit=$nqubit  nstate=$nstate  nsteps=$nsteps  nseeds=$nseeds")
    println("alphas: $alphas")
    println("ngbits=$ngbits  min_delta≈$(round(1/ngbits, sigdigits=2))\n")

    h5open(outfile, "w") do h5f
    HDF5.attributes(h5f)["nqubit"]    = nqubit
    HDF5.attributes(h5f)["nstate"]    = nstate
    HDF5.attributes(h5f)["nsteps"]    = nsteps
    HDF5.attributes(h5f)["log_every"] = log_every

    for H_target in H_targets
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000) + 3333)
        goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

        gp_H = create_group(h5f, @sprintf("H%.1f_k%d", H_target, k))
        HDF5.attributes(gp_H)["H_target"] = H_target
        HDF5.attributes(gp_H)["H_actual"] = H_act
        HDF5.attributes(gp_H)["k"]        = k

        println("="^70)
        @printf("H=%.1f (k=%d, H_act=%.4f)\n\n", H_target, k, H_act)

        for alpha in alphas
            efold_steps = round(Int, -1 / log(alpha))
            # Step at which T reaches min_delta (≈1/ngbits), given T0≈5e-3
            t_meaningful = round(Int, log(5e-3 / (1/ngbits)) / (-log(alpha)))

            println("-"^60)
            @printf("alpha=%.6f  e-fold=%d steps  T→min_delta at step ~%d (%.1f%%)\n",
                    alpha, efold_steps, t_meaningful, 100*t_meaningful/nsteps)

            final_accs = Float64[]
            T0s        = Float64[]
            all_traj_accs  = Vector{Vector{Float64}}()
            all_traj_rates = Vector{Vector{Float64}}()
            all_traj_temps = Vector{Vector{Float64}}()
            ref_steps      = Int[]

            for (si, seed) in enumerate(seeds)
                @printf("  seed %d/%d\n", si, nseeds); flush(stdout)
                rng_s = Random.MersenneTwister(seed)
                r = run_gsa_full(goals[si], nqubit, nstate, nsteps, alpha;
                                 log_every=log_every, rng=rng_s)
                push!(final_accs,    r.final_acc)
                push!(T0s,           r.T0)
                push!(all_traj_accs,  r.traj_accs)
                push!(all_traj_rates, r.traj_rates)
                push!(all_traj_temps, r.traj_temps)
                si == 1 && (ref_steps = r.traj_steps)
            end

            @printf("  RESULT  median=%.4f  best=%.4f  mean_T0=%.2e\n\n",
                    median(final_accs), maximum(final_accs), mean(T0s))
            flush(stdout)

            key = replace(@sprintf("alpha_%.6f", alpha), "." => "p")
            gp  = create_group(gp_H, key)
            HDF5.attributes(gp)["alpha"]          = alpha
            HDF5.attributes(gp)["efold_steps"]    = efold_steps
            HDF5.attributes(gp)["t_meaningful"]   = t_meaningful
            gp["final_accs"]  = final_accs
            gp["T0s"]         = T0s
            gp["ref_steps"]   = ref_steps
            gp["traj_accs"]   = hcat(all_traj_accs...)
            gp["traj_rates"]  = hcat(all_traj_rates...)
            gp["traj_temps"]  = hcat(all_traj_temps...)
        end
    end

    end  # h5open
    println("Saved to $outfile")
end

main()
