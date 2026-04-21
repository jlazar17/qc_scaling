# Script 11: Picker quality vs trajectory — many seeds, save + plot.
#
# Key finding from script 10: the picker's gap (best_available - picked)
# grows for H=1 as the trajectory progresses, while it shrinks for H=0.
# This means H=1's picker degrades in quality over time.
#
# This script runs many seeds to get a clear statistical picture and
# produces a plot of best_pos, picker_pos, and gap vs step for H=0 and H=1.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

using Plots
using Statistics
using DelimitedFiles

# ---------------------------------------------------------------------------
# Probe landscape: for N_probe random proposals, try all 64 alpha patterns
# and the picker's choice. Returns per-proposal (best_delta, picker_delta).
# ---------------------------------------------------------------------------
function probe_deltas(rng_probe, ensemble, rep_sum, rep_ctr,
                      goal, fingerprint, cxt_master, companion, goal_idx,
                      nqubit, cur_acc, n_probe)
    n      = 3^nqubit
    ntz    = 2; nalpha = 2^(nqubit-1)
    scratch_sum = zeros(Int, n); scratch_ctr = zeros(Int, n)
    rep = rep_from_cache(rep_sum, rep_ctr)

    best_deltas   = Float64[]
    picker_deltas = Float64[]

    for _ in 1:n_probe
        which  = rand(rng_probe, 1:length(ensemble))
        gen    = QCScaling.ParityOperator(rand(rng_probe, 0:n-1), nqubit)
        ts     = rand(rng_probe, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd

        # Picker's choice
        ret = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                            bc, companion, goal_idx, n-1)
        ns_pick = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(scratch_sum, scratch_ctr, ensemble[which], cxt_master, -1)
        apply_state!(scratch_sum, scratch_ctr, ns_pick, cxt_master, 1)
        d_pick = rep_accuracy_shuffled(rep_sum .+ scratch_sum,
                                       rep_ctr .+ scratch_ctr,
                                       goal, companion, goal_idx) - cur_acc
        push!(picker_deltas, d_pick)
        fill!(scratch_sum, 0); fill!(scratch_ctr, 0)

        # Exhaustive best
        best_d = -Inf
        for tzi in 0:1, ai in 0:nalpha-1
            alphas = QCScaling.idx_to_alphas(ai, nqubit)
            ns = QCScaling.PseudoGHZState(ts, tzi, alphas, gen)
            apply_state!(scratch_sum, scratch_ctr, ensemble[which], cxt_master, -1)
            apply_state!(scratch_sum, scratch_ctr, ns, cxt_master, 1)
            d = rep_accuracy_shuffled(rep_sum .+ scratch_sum,
                                      rep_ctr .+ scratch_ctr,
                                      goal, companion, goal_idx) - cur_acc
            d > best_d && (best_d = d)
            fill!(scratch_sum, 0); fill!(scratch_ctr, 0)
        end
        push!(best_deltas, best_d)
    end

    return best_deltas, picker_deltas
end

function run_one_seed(goal0, goal1, nqubit, nstate, alpha_cool, nsteps,
                      companion, goal_idx, fingerprint, cxt_master,
                      checkpoint_steps, n_probe; seed=1)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    acc_fn0(rs, rc) = rep_accuracy_shuffled(rs, rc, goal0, companion, goal_idx)
    acc_fn1(rs, rc) = rep_accuracy_shuffled(rs, rc, goal1, companion, goal_idx)

    # Initialize both ensembles from the same seed
    function make_ensemble(goal, acc_fn)
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
        cur_acc = acc_fn(rep_sum, rep_ctr)
        scratch_idxs = Vector{Int}(undef, npos)
        scratch_pars = Vector{Int}(undef, npos)
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
        return rng, ensemble, cache_idxs, cache_pars, rep_sum, rep_ctr, rep,
               scratch_idxs, scratch_pars, Ref(T), Ref(cur_acc)
    end

    r0 = make_ensemble(goal0, acc_fn0)
    r1 = make_ensemble(goal1, acc_fn1)
    rng0, ens0, ci0, cp0, rs0, rc0, rep0, si0, sp0, T0, acc0 = r0
    rng1, ens1, ci1, cp1, rs1, rc1, rep1, si1, sp1, T1, acc1 = r1

    rows = []  # (step, acc0, acc1, best_pos0, best_pos1, pick_pos0, pick_pos1, gap0, gap1)

    function checkpoint(step)
        rp0 = Random.MersenneTwister(seed * 100_000 + step)
        rp1 = Random.MersenneTwister(seed * 100_000 + step + 1)
        bd0, pd0 = probe_deltas(rp0, ens0, rs0, rc0, goal0, fingerprint,
                                cxt_master, companion, goal_idx, nqubit,
                                acc0[], n_probe)
        bd1, pd1 = probe_deltas(rp1, ens1, rs1, rc1, goal1, fingerprint,
                                cxt_master, companion, goal_idx, nqubit,
                                acc1[], n_probe)

        pos0 = bd0 .> 0; pos1 = bd1 .> 0
        best_pos0   = any(pos0) ? mean(bd0[pos0]) : NaN
        best_pos1   = any(pos1) ? mean(bd1[pos1]) : NaN
        pick_pos0   = any(pos0) ? mean(pd0[pos0]) : NaN
        pick_pos1   = any(pos1) ? mean(pd1[pos1]) : NaN
        gap0        = any(pos0) ? mean((bd0 .- pd0)[pos0]) : NaN
        gap1        = any(pos1) ? mean((bd1 .- pd1)[pos1]) : NaN
        frac_pos0   = mean(pos0)
        frac_pos1   = mean(pos1)

        push!(rows, (step=step,
                     acc0=acc0[], acc1=acc1[],
                     frac_pos0=frac_pos0, frac_pos1=frac_pos1,
                     best_pos0=best_pos0, best_pos1=best_pos1,
                     pick_pos0=pick_pos0, pick_pos1=pick_pos1,
                     gap0=gap0, gap1=gap1))
    end

    checkpoint(0)
    cp_idx = 2

    for s in 1:nsteps
        # Step H=0
        which  = rand(rng0, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng0, 0:n-1), nqubit)
        ts     = rand(rng0, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal0, rep0, fingerprint,
                               bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
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

        # Step H=1
        which  = rand(rng1, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng1, 0:n-1), nqubit)
        ts     = rand(rng1, 0:1)
        bc     = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal1, rep1, fingerprint,
                               bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(si1, sp1, ns, cxt_master)
        apply_state_cached!(rs1, rc1, ci1[which], cp1[which], -1)
        apply_state_cached!(rs1, rc1, si1, sp1, 1)
        new_acc = acc_fn1(rs1, rc1)
        d = new_acc - acc1[]
        if d >= 0 || rand(rng1) < exp(d / T1[])
            update_rep_at_cached!(rep1, rs1, rc1, ci1[which])
            update_rep_at_cached!(rep1, rs1, rc1, si1)
            copy!(ci1[which], si1); copy!(cp1[which], sp1)
            ens1[which] = ns; acc1[] = new_acc
        else
            apply_state_cached!(rs1, rc1, si1, sp1, -1)
            apply_state_cached!(rs1, rc1, ci1[which], cp1[which], 1)
        end
        T1[] *= alpha_cool

        if cp_idx <= length(checkpoint_steps) && s == checkpoint_steps[cp_idx]
            checkpoint(s)
            cp_idx += 1
        end
    end

    return rows
end

function nanmean(v)
    fv = filter(!isnan, v)
    isempty(fv) ? NaN : mean(fv)
end
function nanstd(v)
    fv = filter(!isnan, v)
    length(fv) < 2 ? NaN : std(fv)
end

function main()
    nqubit     = 6
    ngbits     = (3^nqubit - 1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 30
    n_probe    = 300   # per checkpoint; 300 × 64 patterns = 19200 SA-steps equivalent

    checkpoint_steps = [0, 500, 1_000, 2_000, 3_000, 5_000, 8_000,
                        12_000, 20_000, 30_000, 50_000, 80_000,
                        120_000, 200_000, 300_000]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    goal0 = zeros(Int, ngbits)
    rng1  = Random.MersenneTwister(99)
    goal1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2),
                                       zeros(Int, ngbits - ngbits ÷ 2)))

    println("Running $n_seeds seeds × $(length(checkpoint_steps)) checkpoints...")

    all_rows = []
    for seed in 1:n_seeds
        @printf("  seed %d/%d\n", seed, n_seeds); flush(stdout)
        rows = run_one_seed(goal0, goal1, nqubit, nstate, alpha_cool, nsteps,
                            companion, goal_idx, fingerprint, cxt_master,
                            checkpoint_steps, n_probe; seed=seed)
        push!(all_rows, rows)
    end

    # -----------------------------------------------------------------------
    # Save raw data as CSV
    # -----------------------------------------------------------------------
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    outfile = joinpath(outdir, "11_picker_quality.csv")
    open(outfile, "w") do io
        println(io, "seed,step,acc0,acc1,frac_pos0,frac_pos1,best_pos0,best_pos1,pick_pos0,pick_pos1,gap0,gap1")
        for (si, rows) in enumerate(all_rows)
            for r in rows
                @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        si, r.step, r.acc0, r.acc1,
                        r.frac_pos0, r.frac_pos1,
                        isnan(r.best_pos0) ? -1.0 : r.best_pos0,
                        isnan(r.best_pos1) ? -1.0 : r.best_pos1,
                        isnan(r.pick_pos0) ? -1.0 : r.pick_pos0,
                        isnan(r.pick_pos1) ? -1.0 : r.pick_pos1,
                        isnan(r.gap0) ? -1.0 : r.gap0,
                        isnan(r.gap1) ? -1.0 : r.gap1)
            end
        end
    end
    println("Saved raw data to $outfile")

    # -----------------------------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------------------------
    n_cp = length(checkpoint_steps)
    steps = checkpoint_steps

    agg = NamedTuple[]
    for ci in 1:n_cp
        rows = [all_rows[s][ci] for s in 1:n_seeds]
        push!(agg, (
            step      = steps[ci],
            acc0      = nanmean([r.acc0 for r in rows]),
            acc1      = nanmean([r.acc1 for r in rows]),
            fpos0     = nanmean([r.frac_pos0 for r in rows]),
            fpos1     = nanmean([r.frac_pos1 for r in rows]),
            best_pos0 = nanmean([r.best_pos0 for r in rows]),
            best_pos1 = nanmean([r.best_pos1 for r in rows]),
            best_pos0_se = nanstd([r.best_pos0 for r in rows]) / sqrt(n_seeds),
            best_pos1_se = nanstd([r.best_pos1 for r in rows]) / sqrt(n_seeds),
            pick_pos0 = nanmean([r.pick_pos0 for r in rows]),
            pick_pos1 = nanmean([r.pick_pos1 for r in rows]),
            pick_pos0_se = nanstd([r.pick_pos0 for r in rows]) / sqrt(n_seeds),
            pick_pos1_se = nanstd([r.pick_pos1 for r in rows]) / sqrt(n_seeds),
            gap0      = nanmean([r.gap0 for r in rows]),
            gap1      = nanmean([r.gap1 for r in rows]),
            gap0_se   = nanstd([r.gap0 for r in rows]) / sqrt(n_seeds),
            gap1_se   = nanstd([r.gap1 for r in rows]) / sqrt(n_seeds),
        ))
    end

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    println("\nSummary (mean ± SE across $n_seeds seeds):")
    @printf("%-7s  %-6s  %-6s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s\n",
            "step","acc0","acc1","best_p0","best_p1","pick_p0","pick_p1","gap0","gap1")
    println(repeat("-", 88))
    for a in agg
        @printf("%-7d  %-6.4f  %-6.4f  %-9.5f  %-9.5f  %-9.5f  %-9.5f  %-9.5f  %-9.5f\n",
                a.step, a.acc0, a.acc1,
                isnan(a.best_pos0) ? 0.0 : a.best_pos0,
                isnan(a.best_pos1) ? 0.0 : a.best_pos1,
                isnan(a.pick_pos0) ? 0.0 : a.pick_pos0,
                isnan(a.pick_pos1) ? 0.0 : a.pick_pos1,
                isnan(a.gap0) ? 0.0 : a.gap0,
                isnan(a.gap1) ? 0.0 : a.gap1)
    end

    # -----------------------------------------------------------------------
    # Plot — focus on steps 500–30000 where oracle > 0 and signal is rich
    # -----------------------------------------------------------------------
    outdir = joinpath(@__DIR__, "results")

    # Filter to rows where both H=0 and H=1 have meaningful oracle > 0
    # and steps in the interesting range
    active = filter(a -> a.step >= 500 && a.fpos0 > 0.01 && a.fpos1 > 0.01 &&
                         !isnan(a.best_pos0) && !isnan(a.best_pos1) &&
                         a.best_pos0 > 0 && a.best_pos1 > 0, agg)
    xs = [a.step for a in active]

    # --- Panel 1: accuracy ---
    p_acc = plot(title="Accuracy trajectory",
                 xlabel="SA step", ylabel="Accuracy",
                 xscale=:log10, legend=:bottomright,
                 xlims=(500, 3e5), ylims=(0.3, 1.0))
    plot!(p_acc, [max(1, a.step) for a in agg], [a.acc0 for a in agg],
          label="H=0", color=:blue, lw=2)
    plot!(p_acc, [max(1, a.step) for a in agg], [a.acc1 for a in agg],
          label="H=1", color=:red, lw=2)

    # --- Panel 2: fraction of proposals with positive delta ---
    p_frac = plot(title="Fraction of proposals with any positive-delta pattern",
                  xlabel="SA step", ylabel="Fraction (oracle)",
                  xscale=:log10, legend=:topright,
                  xlims=(500, 3e5))
    plot!(p_frac, [max(1, a.step) for a in agg], [a.fpos0 for a in agg],
          label="H=0", color=:blue, lw=2)
    plot!(p_frac, [max(1, a.step) for a in agg], [a.fpos1 for a in agg],
          label="H=1", color=:red, lw=2)

    # --- Panel 3: best_pos and pick_pos (solid = best oracle, dash = picker) ---
    p_delta = plot(title="Mean positive delta: oracle best vs picker choice",
                   xlabel="SA step", ylabel="Mean Δacc | Δacc > 0",
                   xscale=:log10, legend=:topright,
                   xlims=(minimum(xs), maximum(xs)))
    plot!(p_delta, xs, [a.best_pos0 for a in active],
          ribbon=[a.best_pos0_se * 1.96 for a in active],
          fillalpha=0.2, label="H=0 oracle best", color=:blue, lw=2)
    plot!(p_delta, xs, [a.best_pos1 for a in active],
          ribbon=[a.best_pos1_se * 1.96 for a in active],
          fillalpha=0.2, label="H=1 oracle best", color=:red, lw=2)
    plot!(p_delta, xs, [a.pick_pos0 for a in active],
          ribbon=[a.pick_pos0_se * 1.96 for a in active],
          fillalpha=0.15, label="H=0 picker", color=:blue, lw=2, ls=:dash)
    plot!(p_delta, xs, [a.pick_pos1 for a in active],
          ribbon=[a.pick_pos1_se * 1.96 for a in active],
          fillalpha=0.15, label="H=1 picker", color=:red, lw=2, ls=:dash)

    # --- Panel 4: gap (best - picker), only where oracle > 0 ---
    gap_active = filter(a -> a.step >= 500 && !isnan(a.gap0) && !isnan(a.gap1) &&
                             a.gap0 >= 0 && a.gap1 >= 0 &&
                             a.fpos0 > 0.01 && a.fpos1 > 0.01, agg)
    xg = [a.step for a in gap_active]
    p_gap = plot(title="Picker gap = best_oracle - picker (given oracle > 0)",
                 xlabel="SA step", ylabel="Mean gap",
                 xscale=:log10, legend=:topleft,
                 xlims=(minimum(xg), maximum(xg)))
    plot!(p_gap, xg, [a.gap0 for a in gap_active],
          ribbon=[a.gap0_se * 1.96 for a in gap_active],
          fillalpha=0.2, label="H=0 gap", color=:blue, lw=2)
    plot!(p_gap, xg, [a.gap1 for a in gap_active],
          ribbon=[a.gap1_se * 1.96 for a in gap_active],
          fillalpha=0.2, label="H=1 gap", color=:red, lw=2)

    # --- Panel 5: gap ratio H=1/H=0 ---
    gap_both = filter(a -> a.step >= 500 && !isnan(a.gap0) && !isnan(a.gap1) &&
                           a.gap0 > 0 && a.gap1 > 0 &&
                           a.fpos0 > 0.01 && a.fpos1 > 0.01, agg)
    xr = [a.step for a in gap_both]
    p_ratio = plot(title="Gap ratio H=1 / H=0 (>1 means H=1 picker is worse)",
                   xlabel="SA step", ylabel="gap₁ / gap₀",
                   xscale=:log10, legend=:topleft,
                   xlims=(minimum(xr), maximum(xr)))
    hline!(p_ratio, [1.0], color=:black, ls=:dot, label="ratio = 1")
    plot!(p_ratio, xr, [a.gap1 / a.gap0 for a in gap_both],
          color=:purple, lw=2, label="gap ratio")

    combined = plot(p_acc, p_frac, p_delta, p_gap, p_ratio,
                    layout=(3, 2), size=(1400, 1200),
                    plot_title="Picker quality vs SA trajectory  (nqubit=6, nstate=45, n=$n_seeds seeds)")
    plotfile = joinpath(outdir, "11_picker_quality.png")
    savefig(combined, plotfile)
    println("Saved plot to $plotfile")
end

main()
