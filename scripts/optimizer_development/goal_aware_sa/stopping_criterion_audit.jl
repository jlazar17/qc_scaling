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
# Full-run goal-SA with stopping-criterion audit
#
# Runs for the full nsteps regardless of acceptance rate, but records:
#   stop_step    : first step where rolling acceptance rate < min_accept_rate
#                  (0 if criterion never triggered)
#   acc_at_stop  : best_acc at stop_step (NaN if never triggered)
#   final_acc    : best_acc at end of full run
#   improved_after : true if any improvement occurred after stop_step
#   traj_steps   : step indices of logged trajectory
#   traj_accs    : best_acc trajectory (monotone non-decreasing)
#   traj_rates   : rolling acceptance rate at each log point
# ---------------------------------------------------------------------------

function run_gsa_audit(goal, nqubit, nstate, nsteps, alpha;
                       min_accept_rate=0.001, acc_window=10_000,
                       log_every=2000, rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fingerprint; rng=rng)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc    = current_acc

    accept_buf = zeros(Bool, acc_window)
    n_accepted = 0

    stop_step     = 0       # 0 = criterion never triggered
    acc_at_stop   = NaN
    improved_after = false

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
            if new_acc > best_acc
                best_acc = new_acc
                stop_step > 0 && (improved_after = true)
            end
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end

        buf_idx = mod1(step, acc_window)
        n_accepted += Int(accepted) - Int(accept_buf[buf_idx])
        accept_buf[buf_idx] = accepted

        T *= alpha

        rolling_rate = step >= acc_window ? n_accepted / acc_window : n_accepted / step

        # Flag first time criterion triggers (but keep running)
        if stop_step == 0 && step >= acc_window && rolling_rate < min_accept_rate
            stop_step   = step
            acc_at_stop = best_acc
        end

        if step % log_every == 0
            push!(traj_steps, step)
            push!(traj_accs,  best_acc)
            push!(traj_rates, rolling_rate)
            push!(traj_temps, T)
            stopped_str = stop_step > 0 ? @sprintf(" [CRIT@%d]", stop_step) : ""
            @printf("  step %7d  acc=%.4f  accept_rate=%.5f  T=%.2e%s\n",
                    step, best_acc, rolling_rate, T, stopped_str)
            flush(stdout)
        end
    end

    return (
        stop_step      = stop_step,
        acc_at_stop    = acc_at_stop,
        final_acc      = best_acc,
        improved_after = improved_after,
        improvement    = isnan(acc_at_stop) ? 0.0 : best_acc - acc_at_stop,
        traj_steps     = traj_steps,
        traj_accs      = traj_accs,
        traj_rates     = traj_rates,
        traj_temps     = traj_temps,
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit      = 8
    nstate_mult = 3
    nsteps      = 2_000_000
    alpha       = 0.99999
    nseeds      = 30          # enough to get stable statistics
    base_seed   = 9999
    H_targets   = [0.0, 1.0]

    min_accept_rate = 0.001
    acc_window      = 10_000
    log_every       = 2_000

    base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
    nstate      = nstate_mult * base_nstate
    ngbits      = (3^nqubit - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "stopping_criterion_audit.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("nqubit=$nqubit  nstate=$nstate  nsteps=$nsteps  alpha=$alpha")
    println("criterion: rolling acceptance rate < $min_accept_rate over last $acc_window steps")
    println("nseeds=$nseeds  log_every=$log_every\n")

    h5open(outfile, "w") do h5f
    HDF5.attributes(h5f)["nqubit"]          = nqubit
    HDF5.attributes(h5f)["nstate"]          = nstate
    HDF5.attributes(h5f)["nsteps"]          = nsteps
    HDF5.attributes(h5f)["alpha"]           = alpha
    HDF5.attributes(h5f)["min_accept_rate"] = min_accept_rate
    HDF5.attributes(h5f)["acc_window"]      = acc_window
    HDF5.attributes(h5f)["log_every"]       = log_every

    for H_target in H_targets
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000) + 5555)
        goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

        stop_steps     = Int[]
        accs_at_stop   = Float64[]
        final_accs     = Float64[]
        improvements   = Float64[]
        improved_flags = Bool[]
        all_traj_accs  = Vector{Vector{Float64}}()
        all_traj_rates = Vector{Vector{Float64}}()
        all_traj_temps = Vector{Vector{Float64}}()
        ref_steps      = Int[]

        println("="^70)
        @printf("H=%.1f (k=%d, H_act=%.4f)\n", H_target, k, H_act)
        @printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-12s  %s\n",
                "seed", "acc@stop", "final", "improve", "stop_step",
                "frac_budget", "improved?")
        println("-"^70)

        for (si, seed) in enumerate(seeds)
            rng_s = Random.MersenneTwister(seed)
            r = run_gsa_audit(goals[si], nqubit, nstate, nsteps, alpha;
                              min_accept_rate=min_accept_rate,
                              acc_window=acc_window,
                              log_every=log_every, rng=rng_s)

            push!(stop_steps,     r.stop_step)
            push!(accs_at_stop,   isnan(r.acc_at_stop) ? r.final_acc : r.acc_at_stop)
            push!(final_accs,     r.final_acc)
            push!(improvements,   r.improvement)
            push!(improved_flags, r.improved_after)
            push!(all_traj_accs,  r.traj_accs)
            push!(all_traj_rates, r.traj_rates)
            push!(all_traj_temps, r.traj_temps)
            si == 1 && (ref_steps = r.traj_steps)

            frac = r.stop_step == 0 ? 1.0 : r.stop_step / nsteps
            @printf("%-6d  %-10.4f  %-10.4f  %-10.4f  %-10d  %-12.3f  %s\n",
                    si,
                    isnan(r.acc_at_stop) ? r.final_acc : r.acc_at_stop,
                    r.final_acc, r.improvement, r.stop_step,
                    frac, r.improved_after ? "YES" : "no")
            flush(stdout)
        end

        n_improved  = sum(improved_flags)
        n_triggered = sum(stop_steps .> 0)
        @printf("\nSUMMARY:\n")
        @printf("  criterion triggered:   %d/%d seeds\n", n_triggered, nseeds)
        @printf("  improved after stop:   %d/%d seeds (%.0f%%)\n",
                n_improved, n_triggered, 100*n_improved/max(n_triggered,1))
        @printf("  mean stop step:        %.0f (%.1f%% of budget)\n",
                mean(stop_steps[stop_steps .> 0]),
                100*mean(stop_steps[stop_steps .> 0])/nsteps)
        @printf("  mean improvement:      %.5f\n", mean(improvements))
        @printf("  max improvement:       %.5f\n", maximum(improvements))
        @printf("  median acc@stop:       %.4f\n", median(accs_at_stop))
        @printf("  median final_acc:      %.4f\n\n", median(final_accs))

        key = @sprintf("H%.1f_k%d", H_target, k)
        gp  = create_group(h5f, key)
        HDF5.attributes(gp)["H_target"]  = H_target
        HDF5.attributes(gp)["H_actual"]  = H_act
        HDF5.attributes(gp)["k"]         = k
        gp["stop_steps"]      = stop_steps
        gp["accs_at_stop"]    = accs_at_stop
        gp["final_accs"]      = final_accs
        gp["improvements"]    = improvements
        gp["improved_flags"]  = Int.(improved_flags)
        gp["ref_steps"]       = ref_steps
        gp["traj_accs"]       = hcat(all_traj_accs...)   # [n_log_pts × nseeds]
        gp["traj_rates"]      = hcat(all_traj_rates...)
        gp["traj_temps"]      = hcat(all_traj_temps...)
    end

    end  # h5open
    println("Saved to $outfile")
end

main()
