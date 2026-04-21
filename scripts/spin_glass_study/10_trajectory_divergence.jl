# Script 10: Trajectory divergence and exhaustive landscape probe.
#
# Run H=0 and H=1 SA simultaneously with shared step count.
# At regular checkpoints, probe the landscape exhaustively:
#   - For N_probe random (gen, theta_s) proposals, try ALL 64 alpha patterns.
#   - Record the best achievable delta, the picker's delta, and the full
#     distribution of deltas across all 64 patterns.
#
# Goal: identify exactly when/where the landscapes diverge and whether
# positive-delta opportunities still exist for H=1 at that point.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Run one SA step, return (new_acc, accepted).
# Modifies ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars in place.
# ---------------------------------------------------------------------------
function sa_step!(rng, ensemble, cache_idxs, cache_pars,
                  rep_sum, rep_ctr, rep, scratch_idxs, scratch_pars,
                  goal, fingerprint, cxt_master, companion, goal_idx,
                  nqubit, T, acc_fn)
    n      = 3^nqubit
    nstate = length(ensemble)

    which  = rand(rng, 1:nstate)
    gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
    ts     = rand(rng, 0:1)
    bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
    ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                           bc, companion, goal_idx, n-1)
    ns     = QCScaling.PseudoGHZState(ret..., gen)
    fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
    apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
    apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
    cur_acc = acc_fn(rep_sum, rep_ctr)
    old_acc = cur_acc  # will compute delta below

    # Recompute: we need cur_acc BEFORE the step
    apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
    apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
    old_acc2 = acc_fn(rep_sum, rep_ctr)

    # Redo properly
    apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
    apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
    new_acc = acc_fn(rep_sum, rep_ctr)
    d = new_acc - old_acc2

    if d >= 0 || rand(rng) < exp(d / T)
        update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
        update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
        copy!(cache_idxs[which], scratch_idxs)
        copy!(cache_pars[which], scratch_pars)
        ensemble[which] = ns
        return new_acc, true
    else
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        return old_acc2, false
    end
end

# ---------------------------------------------------------------------------
# Exhaustive landscape probe at a given ensemble state.
# For N_probe random (gen, theta_s, which_state), tries ALL 64 alpha patterns.
# Returns:
#   frac_pos_oracle  : fraction of proposals where BEST pattern has delta > 0
#   frac_pos_picker  : fraction of proposals where PICKER pattern has delta > 0
#   mean_best_delta  : mean of best delta per proposal (over all proposals)
#   mean_best_delta_pos : mean best delta conditional on best > 0
#   delta_distribution : histogram of best deltas (rounded to 3 decimal places)
# ---------------------------------------------------------------------------
function probe_landscape(rng_probe, ensemble, rep_sum, rep_ctr,
                         goal, fingerprint, cxt_master, companion, goal_idx,
                         nqubit, cur_acc, n_probe)
    n      = 3^nqubit
    npos   = length(cxt_master.base_even.pos)
    ntz    = 2; nalpha = 2^(nqubit-1)

    oracle_pos = 0; picker_pos = 0
    best_deltas = Float64[]
    picker_deltas = Float64[]

    scratch_sum = zeros(Int, n); scratch_ctr = zeros(Int, n)

    for _ in 1:n_probe
        which  = rand(rng_probe, 1:length(ensemble))
        gen    = QCScaling.ParityOperator(rand(rng_probe, 0:n-1), nqubit)
        ts     = rand(rng_probe, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt    = QCScaling.Context(gen, bc)

        # Picker's choice
        ret    = pick_alphas_s(cxt, goal, rep_from_cache(rep_sum, rep_ctr),
                               fingerprint, bc, companion, goal_idx, n-1)
        picker_state = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(scratch_sum, scratch_ctr, ensemble[which], cxt_master, -1)
        apply_state!(scratch_sum, scratch_ctr, picker_state, cxt_master, 1)
        picker_acc = rep_accuracy_shuffled(
            rep_sum .+ scratch_sum, rep_ctr .+ scratch_ctr,
            goal, companion, goal_idx)
        picker_d = picker_acc - cur_acc
        push!(picker_deltas, picker_d)
        picker_d > 0 && (picker_pos += 1)
        # Reset scratch
        fill!(scratch_sum, 0); fill!(scratch_ctr, 0)

        # Exhaustive: try all (tz, alpha) combos
        best_d = -Inf
        for tzi in 0:1
            for ai in 0:nalpha-1
                alphas = QCScaling.idx_to_alphas(ai, nqubit)
                ns = QCScaling.PseudoGHZState(ts, tzi, alphas, gen)
                apply_state!(scratch_sum, scratch_ctr, ensemble[which], cxt_master, -1)
                apply_state!(scratch_sum, scratch_ctr, ns, cxt_master, 1)
                trial_acc = rep_accuracy_shuffled(
                    rep_sum .+ scratch_sum, rep_ctr .+ scratch_ctr,
                    goal, companion, goal_idx)
                d = trial_acc - cur_acc
                d > best_d && (best_d = d)
                fill!(scratch_sum, 0); fill!(scratch_ctr, 0)
            end
        end
        push!(best_deltas, best_d)
        best_d > 0 && (oracle_pos += 1)
    end

    pos_best   = filter(>(0), best_deltas)
    pos_picker = filter(>(0), picker_deltas)
    # For each proposal, gap = best_delta - picker_delta
    gaps = best_deltas .- picker_deltas

    return (
        frac_pos_oracle       = oracle_pos / n_probe,
        frac_pos_picker       = picker_pos / n_probe,
        mean_best_delta       = mean(best_deltas),
        mean_best_pos         = isempty(pos_best)   ? NaN : mean(pos_best),
        mean_picker_pos       = isempty(pos_picker) ? NaN : mean(pos_picker),
        mean_gap              = mean(gaps),   # best - picker, averaged over all proposals
        mean_gap_pos_oracle   = isempty(pos_best) ? NaN :
                                mean(gaps[i] for i in eachindex(gaps) if best_deltas[i] > 0),
        best_deltas           = best_deltas,
        picker_deltas         = picker_deltas,
    )
end

function init_sa(goal, nqubit, nstate, cxt_master, fingerprint,
                 companion, goal_idx; seed=42)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    rng  = Random.MersenneTwister(seed)

    ensemble   = [QCScaling.random_state(nqubit) for _ in 1:nstate]
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
    cur_acc = acc_fn(rep_sum, rep_ctr)

    # Temperature calibration
    bad_deltas = Float64[]
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                               bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        d = acc_fn(rep_sum, rep_ctr) - cur_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        d < 0 && push!(bad_deltas, abs(d))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    return rng, ensemble, cache_idxs, cache_pars, rep_sum, rep_ctr, rep,
           scratch_idxs, scratch_pars, T, cur_acc, acc_fn
end

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000

    # Checkpoints: probe at these step counts
    checkpoint_steps = [0, 1_000, 3_000, 5_000, 10_000, 20_000, 40_000,
                        80_000, 150_000, 300_000]
    n_probe_per_checkpoint = 500  # exhaustive over 64 patterns each → expensive

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    rng0 = Random.MersenneTwister(99)
    goal0 = zeros(Int, ngbits)  # H=0

    rng1 = Random.MersenneTwister(99)
    goal1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2),
                                       zeros(Int, ngbits - ngbits ÷ 2)))  # H=1

    println("=" ^ 75)
    println("Trajectory divergence and exhaustive landscape probe (nqubit=6, nstate=45)")
    println("=" ^ 75)
    @printf("%-8s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s\n",
            "step", "H=0 acc", "H=1 acc", "H=0 oracle", "H=1 oracle",
            "H=0 picker", "H=1 picker")
    println(repeat("-", 70))

    # Run 3 seeds and average
    n_seeds = 3
    all_results = []

    for seed in 1:n_seeds
        @printf("\n--- seed=%d ---\n", seed)

        s0 = init_sa(goal0, nqubit, nstate, cxt_master, fingerprint,
                     companion, goal_idx; seed=seed)
        s1 = init_sa(goal1, nqubit, nstate, cxt_master, fingerprint,
                     companion, goal_idx; seed=seed)

        rng0, ens0, ci0, cp0, rs0, rc0, rep0, si0, sp0, T0, acc0, af0 = s0
        rng1, ens1, ci1, cp1, rs1, rc1, rep1, si1, sp1, T1, acc1, af1 = s1

        step = 0
        cp_idx = 1
        seed_results = []

        function do_checkpoint()
            rng_p = Random.MersenneTwister(seed * 10000 + step)
            p0 = probe_landscape(rng_p, ens0, rs0, rc0, goal0, fingerprint,
                                 cxt_master, companion, goal_idx, nqubit,
                                 acc0, n_probe_per_checkpoint)
            rng_p2 = Random.MersenneTwister(seed * 10000 + step + 1)
            p1 = probe_landscape(rng_p2, ens1, rs1, rc1, goal1, fingerprint,
                                 cxt_master, companion, goal_idx, nqubit,
                                 acc1, n_probe_per_checkpoint)
            push!(seed_results, (step=step, acc0=acc0, acc1=acc1,
                                 oracle0=p0.frac_pos_oracle,   oracle1=p1.frac_pos_oracle,
                                 picker0=p0.frac_pos_picker,   picker1=p1.frac_pos_picker,
                                 best_pos0=p0.mean_best_pos,   best_pos1=p1.mean_best_pos,
                                 pick_pos0=p0.mean_picker_pos, pick_pos1=p1.mean_picker_pos,
                                 gap0=p0.mean_gap_pos_oracle,  gap1=p1.mean_gap_pos_oracle))
            @printf("seed=%d step=%-6d  acc0=%.4f  acc1=%.4f  oracle0=%.3f  oracle1=%.3f  best_pos0=%.5f  best_pos1=%.5f  gap0=%.5f  gap1=%.5f\n",
                    seed, step, acc0, acc1,
                    p0.frac_pos_oracle, p1.frac_pos_oracle,
                    isnan(p0.mean_best_pos) ? 0.0 : p0.mean_best_pos,
                    isnan(p1.mean_best_pos) ? 0.0 : p1.mean_best_pos,
                    isnan(p0.mean_gap_pos_oracle) ? 0.0 : p0.mean_gap_pos_oracle,
                    isnan(p1.mean_gap_pos_oracle) ? 0.0 : p1.mean_gap_pos_oracle)
            flush(stdout)
        end

        # Step 0 probe
        do_checkpoint()
        cp_idx += 1

        for s in 1:nsteps
            acc0, _ = sa_step!(rng0, ens0, ci0, cp0, rs0, rc0, rep0, si0, sp0,
                               goal0, fingerprint, cxt_master, companion, goal_idx,
                               nqubit, T0, af0)
            acc1, _ = sa_step!(rng1, ens1, ci1, cp1, rs1, rc1, rep1, si1, sp1,
                               goal1, fingerprint, cxt_master, companion, goal_idx,
                               nqubit, T1, af1)
            T0 *= alpha_cool; T1 *= alpha_cool

            if cp_idx <= length(checkpoint_steps) && s == checkpoint_steps[cp_idx]
                step = s
                do_checkpoint()
                cp_idx += 1
            end
        end
        push!(all_results, seed_results)
    end

    # Summary across seeds
    println("\n")
    println("=" ^ 100)
    println("Summary (mean across seeds)")
    println("best_pos = mean positive delta (oracle best); gap = mean(best - picker) given oracle > 0")
    println("=" ^ 100)
    @printf("%-7s  %-6s  %-6s  %-7s  %-7s  %-9s  %-9s  %-9s  %-9s  %-8s  %-8s\n",
            "step", "acc0", "acc1", "orc0", "orc1",
            "best_pos0", "best_pos1", "pik_pos0", "pik_pos1", "gap0", "gap1")
    println(repeat("-", 105))

    n_checkpoints = length(checkpoint_steps)
    for ci in 1:n_checkpoints
        rows = [all_results[s][ci] for s in 1:n_seeds]
        nanmean(v) = isempty(filter(!isnan, v)) ? NaN : mean(filter(!isnan, v))
        @printf("%-7d  %-6.4f  %-6.4f  %-7.4f  %-7.4f  %-9.5f  %-9.5f  %-9.5f  %-9.5f  %-8.5f  %-8.5f\n",
                rows[1].step,
                mean(r.acc0 for r in rows),
                mean(r.acc1 for r in rows),
                mean(r.oracle0 for r in rows),
                mean(r.oracle1 for r in rows),
                nanmean([r.best_pos0 for r in rows]),
                nanmean([r.best_pos1 for r in rows]),
                nanmean([r.pick_pos0 for r in rows]),
                nanmean([r.pick_pos1 for r in rows]),
                nanmean([r.gap0 for r in rows]),
                nanmean([r.gap1 for r in rows]))
        flush(stdout)
    end
end

main()
