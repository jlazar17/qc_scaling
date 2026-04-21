# Script 12: Distribution of actual deltas and fingerprint scores at each
# stage of the trajectory, with markers for the picker's choice and oracle best.
#
# For a set of checkpoints along a single SA trajectory, we sample N random
# (gen, theta_s, which_state) proposals and for each:
#   - Compute all 64 actual accuracy deltas (one per alpha pattern)
#   - Compute all 64 fingerprint scores (the heuristic the picker uses)
#   - Record which pattern the picker chose (lowest fingerprint score)
#   - Record which pattern is the oracle best (highest actual delta)
#
# We plot, for H=0 and H=1 side by side at each checkpoint:
#   Left panel:  violin/histogram of all-64 actual deltas,
#                vertical line at picker's delta, star at oracle best delta
#   Right panel: violin/histogram of all-64 fingerprint scores,
#                vertical line at picker's score, marker at oracle-best pattern's score
#
# This directly tests: does the fingerprint score correctly rank the oracle-best
# pattern near the top (low score)? If not for H=1, the heuristic is broken.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))
using Plots
using StatsBase

# ---------------------------------------------------------------------------
# For a given ensemble state, compute for N_probe random proposals:
#   - all 64 actual deltas
#   - all 64 fingerprint scores
#   - picker_idx (0-based, lowest fp score)
#   - oracle_idx (0-based, highest delta)
# Returns arrays of shape (N_probe, 64).
# ---------------------------------------------------------------------------
function probe_full_distributions(rng, ensemble, rep_sum, rep_ctr,
                                  goal, fingerprint, cxt_master, companion, goal_idx,
                                  nqubit, cur_acc, n_probe)
    n      = 3^nqubit
    ntz    = 2; nalpha = 2^(nqubit-1); n_patterns = ntz * nalpha
    scratch_sum = zeros(Int, n); scratch_ctr = zeros(Int, n)
    rep = rep_from_cache(rep_sum, rep_ctr)

    all_deltas     = zeros(Float64, n_probe, n_patterns)
    all_fp_scores  = zeros(Float64, n_probe, n_patterns)
    picker_idxs    = zeros(Int, n_probe)
    oracle_idxs    = zeros(Int, n_probe)

    for pi in 1:n_probe
        which  = rand(rng, 1:length(ensemble))
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt    = QCScaling.Context(gen, bc)

        # Fingerprint scores for all 64 patterns
        fp_scores = all_scores(gen, ts, goal, rep, fingerprint,
                               cxt_master, companion, goal_idx, n)[2]
        all_fp_scores[pi, :] = fp_scores

        # Picker: picks the pattern with minimum fingerprint score
        picker_idx = argmin(fp_scores) - 1  # 0-based

        # Actual deltas for all 64 patterns
        idx = 1
        best_d = -Inf; best_idx = 0
        for tzi in 0:1, ai in 0:nalpha-1
            alphas = QCScaling.idx_to_alphas(ai, nqubit)
            ns = QCScaling.PseudoGHZState(ts, tzi, alphas, gen)
            apply_state!(scratch_sum, scratch_ctr, ensemble[which], cxt_master, -1)
            apply_state!(scratch_sum, scratch_ctr, ns, cxt_master, 1)
            d = rep_accuracy_shuffled(rep_sum .+ scratch_sum,
                                      rep_ctr .+ scratch_ctr,
                                      goal, companion, goal_idx) - cur_acc
            all_deltas[pi, idx] = d
            if d > best_d; best_d = d; best_idx = idx - 1; end
            fill!(scratch_sum, 0); fill!(scratch_ctr, 0)
            idx += 1
        end
        picker_idxs[pi]  = picker_idx
        oracle_idxs[pi]  = best_idx
    end

    return all_deltas, all_fp_scores, picker_idxs, oracle_idxs
end

function make_stage_plot(stage_label, all_deltas, all_fp_scores,
                         picker_idxs, oracle_idxs, acc)
    n_probe = size(all_deltas, 1)

    # For each proposal: picker delta, oracle delta, picker fp score, oracle fp score
    picker_deltas = [all_deltas[i, picker_idxs[i]+1] for i in 1:n_probe]
    oracle_deltas = [all_deltas[i, oracle_idxs[i]+1] for i in 1:n_probe]
    picker_fp     = [all_fp_scores[i, picker_idxs[i]+1] for i in 1:n_probe]
    oracle_fp     = [all_fp_scores[i, oracle_idxs[i]+1] for i in 1:n_probe]

    # Fraction of proposals where oracle best > 0
    frac_pos = mean(oracle_deltas .> 0)

    # Panel A: histogram of all deltas (all 64 patterns pooled), plus picker and oracle
    flat_deltas = vec(all_deltas)
    dmax = max(maximum(flat_deltas), 1e-6) * 1.1
    dmin = min(minimum(flat_deltas), -1e-6)

    pA = histogram(flat_deltas, bins=40, normalize=:probability,
                   color=:grey, alpha=0.5,
                   xlabel="Δacc", ylabel="Probability",
                   title="$(stage_label)\nacc=$(round(acc,digits=3))  frac_pos_oracle=$(round(frac_pos,digits=3))",
                   label="all 64 patterns", xlims=(dmin, dmax))
    histogram!(pA, picker_deltas, bins=40, normalize=:probability,
               color=:blue, alpha=0.5, label="picker chosen")
    histogram!(pA, oracle_deltas, bins=40, normalize=:probability,
               color=:red, alpha=0.5, label="oracle best")

    # Panel B: histogram of fingerprint scores, coloured by picker vs oracle
    flat_fp = vec(all_fp_scores)
    pB = histogram(flat_fp, bins=range(minimum(flat_fp)-0.5, maximum(flat_fp)+0.5, step=1),
                   normalize=:probability, color=:grey, alpha=0.5,
                   xlabel="Fingerprint score", ylabel="Probability",
                   title="Fingerprint score distributions",
                   label="all 64 patterns")
    histogram!(pB, picker_fp, bins=range(minimum(flat_fp)-0.5, maximum(flat_fp)+0.5, step=1),
               normalize=:probability, color=:blue, alpha=0.6, label="picker chosen")
    histogram!(pB, oracle_fp, bins=range(minimum(flat_fp)-0.5, maximum(flat_fp)+0.5, step=1),
               normalize=:probability, color=:red, alpha=0.6, label="oracle best")

    return pA, pB, (
        mean_picker_delta = mean(picker_deltas),
        mean_oracle_delta = mean(oracle_deltas),
        mean_picker_fp    = mean(picker_fp),
        mean_oracle_fp    = mean(oracle_fp),
        frac_pos_oracle   = frac_pos,
        # What fraction of time does the picker choose the oracle-best pattern?
        frac_picker_is_oracle = mean(picker_idxs .== oracle_idxs),
        # Rank of oracle-best pattern in fp score ranking (0=best rank, 63=worst)
        mean_oracle_fp_rank   = mean(Float64[count(all_fp_scores[i,:] .< oracle_fp[i])
                                             for i in 1:n_probe]),
    )
end

function main()
    nqubit     = 6
    ngbits     = (3^nqubit - 1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_probe    = 200   # per checkpoint
    seed       = 7

    checkpoints = [0, 1_000, 5_000, 20_000, 80_000]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    goal0 = zeros(Int, ngbits)
    rng1  = Random.MersenneTwister(99)
    goal1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2),
                                       zeros(Int, ngbits - ngbits ÷ 2)))

    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    acc_fn0(rs, rc) = rep_accuracy_shuffled(rs, rc, goal0, companion, goal_idx)
    acc_fn1(rs, rc) = rep_accuracy_shuffled(rs, rc, goal1, companion, goal_idx)

    function init(goal, acc_fn)
        rng = Random.MersenneTwister(seed)
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
        bad_deltas = Float64[]
        cur_acc = acc_fn(rep_sum, rep_ctr)
        for _ in 1:300
            which = rand(rng, 1:nstate)
            gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
            ts    = rand(rng, 0:1)
            bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
            ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                                  bc, companion, goal_idx, n-1)
            ns    = QCScaling.PseudoGHZState(ret..., gen)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
            d = acc_fn(rep_sum, rep_ctr) - cur_acc
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
            d < 0 && push!(bad_deltas, abs(d))
        end
        T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
        cur_acc = acc_fn(rep_sum, rep_ctr)
        si = Vector{Int}(undef, npos); sp = Vector{Int}(undef, npos)
        return rng, ensemble, cache_idxs, cache_pars, rep_sum, rep_ctr, rep,
               si, sp, Ref(T), Ref(cur_acc)
    end

    r0 = init(goal0, acc_fn0)
    r1 = init(goal1, acc_fn1)
    rng0, ens0, ci0, cp0, rs0, rc0, rep0, si0, sp0, T0, acc0 = r0
    rng1x, ens1, ci1, cp1, rs1, rc1, rep1, si1, sp1, T1, acc1 = r1

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)

    stage_plots_h0 = []; stage_plots_h1 = []
    stats_h0 = []; stats_h1 = []

    function do_checkpoint(step)
        rp0 = Random.MersenneTwister(seed * 1_000_000 + step)
        rp1 = Random.MersenneTwister(seed * 1_000_000 + step + 1)

        d0, fp0, pi0, oi0 = probe_full_distributions(rp0, ens0, rs0, rc0, goal0,
                                fingerprint, cxt_master, companion, goal_idx,
                                nqubit, acc0[], n_probe)
        d1, fp1, pi1, oi1 = probe_full_distributions(rp1, ens1, rs1, rc1, goal1,
                                fingerprint, cxt_master, companion, goal_idx,
                                nqubit, acc1[], n_probe)

        pA0, pB0, s0 = make_stage_plot("H=0  step=$(step)", d0, fp0, pi0, oi0, acc0[])
        pA1, pB1, s1 = make_stage_plot("H=1  step=$(step)", d1, fp1, pi1, oi1, acc1[])

        push!(stage_plots_h0, (pA0, pB0))
        push!(stage_plots_h1, (pA1, pB1))
        push!(stats_h0, merge(s0, (step=step, acc=acc0[])))
        push!(stats_h1, merge(s1, (step=step, acc=acc1[])))

        @printf("step=%-7d  H=0: acc=%.4f  orc_rank=%.1f  frac_picker_is_oracle=%.3f\n",
                step, acc0[], s0.mean_oracle_fp_rank, s0.frac_picker_is_oracle)
        @printf("             H=1: acc=%.4f  orc_rank=%.1f  frac_picker_is_oracle=%.3f\n",
                acc1[], s1.mean_oracle_fp_rank, s1.frac_picker_is_oracle)
        flush(stdout)
    end

    step = 0; cp_idx = 1
    do_checkpoint(0); cp_idx += 1

    for s in 1:nsteps
        # H=0 step
        which = rand(rng0, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng0, 0:n-1), nqubit)
        ts    = rand(rng0, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal0, rep0, fingerprint,
                              bc, companion, goal_idx, n-1)
        ns    = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(si0, sp0, ns, cxt_master)
        apply_state_cached!(rs0, rc0, ci0[which], cp0[which], -1)
        apply_state_cached!(rs0, rc0, si0, sp0, 1)
        new_acc = acc_fn0(rs0, rc0)
        d = new_acc - acc0[]
        if d >= 0 || rand(rng0) < exp(d / T0[])
            update_rep_at_cached!(rep0, rs0, rc0, ci0[which])
            update_rep_at_cached!(rep0, rs0, rc0, si0)
            copy!(ci0[which], si0); copy!(cp0[which], sp0)
            ens0[which] = ns; acc0[] = new_acc
        else
            apply_state_cached!(rs0, rc0, si0, sp0, -1)
            apply_state_cached!(rs0, rc0, ci0[which], cp0[which], 1)
        end
        T0[] *= alpha_cool

        # H=1 step
        which = rand(rng1x, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng1x, 0:n-1), nqubit)
        ts    = rand(rng1x, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal1, rep1, fingerprint,
                              bc, companion, goal_idx, n-1)
        ns    = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(si1, sp1, ns, cxt_master)
        apply_state_cached!(rs1, rc1, ci1[which], cp1[which], -1)
        apply_state_cached!(rs1, rc1, si1, sp1, 1)
        new_acc = acc_fn1(rs1, rc1)
        d = new_acc - acc1[]
        if d >= 0 || rand(rng1x) < exp(d / T1[])
            update_rep_at_cached!(rep1, rs1, rc1, ci1[which])
            update_rep_at_cached!(rep1, rs1, rc1, si1)
            copy!(ci1[which], si1); copy!(cp1[which], sp1)
            ens1[which] = ns; acc1[] = new_acc
        else
            apply_state_cached!(rs1, rc1, si1, sp1, -1)
            apply_state_cached!(rs1, rc1, ci1[which], cp1[which], 1)
        end
        T1[] *= alpha_cool

        if cp_idx <= length(checkpoints) && s == checkpoints[cp_idx]
            step = s; do_checkpoint(step); cp_idx += 1
        end
    end

    # -----------------------------------------------------------------------
    # Summary stats table
    # -----------------------------------------------------------------------
    println("\n--- Summary statistics ---")
    @printf("%-8s  %-5s  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
            "step","H","acc","orc_delta","pick_delta","orc_fp","pick_fp","orc_rank")
    println(repeat("-", 82))
    for (s0, s1) in zip(stats_h0, stats_h1)
        for (s, label) in [(s0,"H=0"), (s1,"H=1")]
            @printf("%-8d  %-5s  %-6.4f  %-10.5f  %-10.5f  %-10.2f  %-10.2f  %-10.2f\n",
                    s.step, label, s.acc,
                    s.mean_oracle_delta, s.mean_picker_delta,
                    s.mean_oracle_fp, s.mean_picker_fp, s.mean_oracle_fp_rank)
        end
        println()
    end

    # -----------------------------------------------------------------------
    # Plot: one row per checkpoint, two columns (delta dist | fp score dist),
    # H=0 on left, H=1 on right within each column.
    # -----------------------------------------------------------------------
    n_stages = length(checkpoints)
    rows_per_stage = 2  # one for deltas, one for fp scores
    plts = []
    for i in 1:n_stages
        pA0, pB0 = stage_plots_h0[i]
        pA1, pB1 = stage_plots_h1[i]
        push!(plts, pA0, pA1, pB0, pB1)
    end

    combined = plot(plts...,
                    layout=(n_stages * 2, 2),
                    size=(1200, 450 * n_stages * 2),
                    margin=5Plots.mm)
    plotfile = joinpath(outdir, "12_delta_distributions.png")
    savefig(combined, plotfile)
    println("Saved to $plotfile")
end

main()
