# Script 15: Mechanism dissection — why are H=1 wrong pairs harder to fix?
#
# The symmetry-breaking argument from script 14 claims that companion_goal_s
# sends both members of an H=1 wrong pair the same "flip" signal.  But since
# k1 and k2 are always in DIFFERENT contexts (shuffled pairing), they can't be
# updated in the same step, and the rep cache self-corrects after one flip.
# So the mechanism, if real, must be different.  Three hypotheses:
#
#   A) OSCILLATION: correct pairs re-break frequently (fix then un-fix),
#      preventing margin buildup.  H=1 correct pairs should have lower
#      "persistence" than H=0 correct pairs.
#
#   B) FIX/BREAK IMBALANCE: H=1 has a higher break rate relative to fix rate,
#      so progress is slow even if individual fix events occur.
#
#   C) COLLATERAL DAMAGE: accepted SA steps that fix one pair simultaneously
#      break other pairs.  H=1 proposals should show worse net pair gain per
#      accepted step.
#
# All three are measured in a single trajectory with fine-grained logging.
#
# Also runs the same analysis at the PAIR MARGIN level (how quickly do correct
# pairs build up large margins in H=0 vs H=1) to test the "signal noise from
# unstable companions" hypothesis.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Classify pairs: returns BitVector of length n_pairs, true = correct.
# ---------------------------------------------------------------------------
function pair_status(rep_sum, rep_ctr, goal, companion, goal_idx, n)
    status = Dict{Int,Bool}()
    for k1 in 1:n-1
        companion[k1] == 0 && continue; k1 > companion[k1] && continue
        k2 = companion[k1]
        c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
        (c1 == 0 || c2 == 0) && continue
        (2*rep_sum[k1] == c1 || 2*rep_sum[k2] == c2) && continue
        r1 = 2*rep_sum[k1] > c1; r2 = 2*rep_sum[k2] > c2
        j  = goal_idx[k1]
        status[k1] = ((r1 ⊻ r2) == !iszero(goal[j]))
    end
    return status
end

# ---------------------------------------------------------------------------
# Margin for a specific pair (min of the two members).
# Returns -1 if either member is uncovered/tied.
# ---------------------------------------------------------------------------
function pair_margin(rep_sum, rep_ctr, k1, k2)
    c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
    (c1 == 0 || c2 == 0) && return -1
    (2*rep_sum[k1] == c1 || 2*rep_sum[k2] == c2) && return -1
    return min(abs(2*rep_sum[k1] - c1), abs(2*rep_sum[k2] - c2))
end

# ---------------------------------------------------------------------------
# Full SA with per-step logging of accepted moves.
# Returns trajectory data for analysis.
# ---------------------------------------------------------------------------
function run_sa_logged(goal, nqubit, nstate, nsteps, alpha_cool,
                       companion, goal_idx, fingerprint, cxt_master;
                       seed=42,
                       checkpoints=[1000, 5000, 20000, 80000, 300000])
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
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    # Temperature calibration
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

    # Per-step log: (step, net_pair_gain, acc_after)
    # net_pair_gain = pairs_fixed - pairs_broken in this step (accepted moves only)
    step_log = Tuple{Int,Int,Float64}[]   # (step, net_gain, acc)

    # Checkpoint snapshots: pair status and margin distributions
    cp_status  = Dict{Int, Dict{Int,Bool}}()  # step → pair status map
    cp_margins = Dict{Int, Vector{Int}}()      # step → margin vector (correct pairs only)

    cp_set = Set(checkpoints)

    for step in 1:nsteps
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                               bc, companion, goal_idx, n-1)
        ns    = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc

        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc

            # Net pair gain: accuracy * n_pairs is the count of correct pairs.
            # d * n_pairs gives the integer change in correct pairs.
            net_gain = round(Int, d * (n - 1) / 2)
            push!(step_log, (step, net_gain, cur_acc))
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool

        if step in cp_set
            cp_status[step] = pair_status(rep_sum, rep_ctr, goal, companion, goal_idx, n)
            cp_margins[step] = [pair_margin(rep_sum, rep_ctr, k1, companion[k1])
                                 for k1 in 1:n-1
                                 if companion[k1]>k1 && get(cp_status[step], k1, false)]
            filter!(x->x>0, cp_margins[step])
        end
    end
    return cur_acc, step_log, cp_status, cp_margins
end

# ---------------------------------------------------------------------------
# Fix/break rates between consecutive checkpoints (from pair-status snapshots)
# and net gain distribution from step_log.
# ---------------------------------------------------------------------------
function analyse_step_log(step_log, cp_status, checkpoints)
    cp_sorted = sort(checkpoints)

    # Fix/break rates from checkpoint pair-status comparisons
    fix_rate = Float64[]
    brk_rate = Float64[]
    windows  = String[]
    for i in 1:length(cp_sorted)-1
        lo = cp_sorted[i]; hi = cp_sorted[i+1]
        nsteps_window = hi - lo
        if haskey(cp_status, lo) && haskey(cp_status, hi)
            s_lo = cp_status[lo]; s_hi = cp_status[hi]
            common = collect(intersect(keys(s_lo), keys(s_hi)))
            fixes  = count(k -> !s_lo[k] && s_hi[k],  common)   # wrong→right
            breaks = count(k ->  s_lo[k] && !s_hi[k], common)   # right→wrong
            push!(fix_rate, fixes  / nsteps_window)
            push!(brk_rate, breaks / nsteps_window)
        else
            push!(fix_rate, NaN); push!(brk_rate, NaN)
        end
        push!(windows, "$(lo)-$(hi)")
    end

    # Net gain distribution over all accepted steps in the full run
    net_gains     = [s[2] for s in step_log]
    frac_positive = count(x->x>0, net_gains) / max(1, length(net_gains))
    frac_negative = count(x->x<0, net_gains) / max(1, length(net_gains))
    mean_net      = isempty(net_gains) ? 0.0 : mean(net_gains)

    return fix_rate, brk_rate, windows, frac_positive, frac_negative, mean_net
end

# ---------------------------------------------------------------------------
# Persistence: fraction of checkpoint-correct pairs still correct at next CP
# ---------------------------------------------------------------------------
function persistence(cp_status, checkpoints)
    cp_sorted = sort(checkpoints)
    result = Float64[]
    for i in 1:length(cp_sorted)-1
        lo = cp_sorted[i]; hi = cp_sorted[i+1]
        haskey(cp_status, lo) && haskey(cp_status, hi) || continue
        s_lo = cp_status[lo]; s_hi = cp_status[hi]
        common = intersect(keys(s_lo), keys(s_hi))
        isempty(common) && push!(result, NaN) && continue
        right_at_lo = [k for k in common if s_lo[k]]
        isempty(right_at_lo) && push!(result, NaN) && continue
        push!(result, count(k -> get(s_hi, k, false), right_at_lo) / length(right_at_lo))
    end
    return result
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    nstate     = 45
    alpha_cool = 0.9999
    nsteps     = 300_000
    n_seeds    = 15
    checkpoints = [1_000, 5_000, 20_000, 80_000, 300_000]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    # Accumulators
    H_vals = [0.0, 1.0]
    persist_acc  = Dict(H => [Float64[] for _ in 1:length(checkpoints)-1] for H in H_vals)
    fix_acc      = Dict(H => [Float64[] for _ in 1:length(checkpoints)-1] for H in H_vals)
    brk_acc      = Dict(H => [Float64[] for _ in 1:length(checkpoints)-1] for H in H_vals)
    frac_pos_acc = Dict(H => Float64[] for H in H_vals)
    frac_neg_acc = Dict(H => Float64[] for H in H_vals)
    mean_net_acc = Dict(H => Float64[] for H in H_vals)
    margin_acc   = Dict(H => Dict(cp => Float64[] for cp in checkpoints) for H in H_vals)

    for gseed in 1:n_seeds
        for H in H_vals
            k_ones = h_to_kones(H, ngbits)
            rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal     = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                       zeros(Int, ngbits - k_ones)))
            _, slog, cps, cpm =
                run_sa_logged(goal, nqubit, nstate, nsteps, alpha_cool,
                               companion, goal_idx, fingerprint, cxt_master;
                               seed=gseed * 31 + 7, checkpoints=checkpoints)

            # Persistence
            pers = persistence(cps, checkpoints)
            for (i, p) in enumerate(pers)
                isnan(p) || push!(persist_acc[H][i], p)
            end

            # Fix/break rates and net gain distribution
            fr, br, _, fp, fn, mn =
                analyse_step_log(slog, cps, checkpoints)
            for (i, (f, b)) in enumerate(zip(fr, br))
                push!(fix_acc[H][i], f)
                push!(brk_acc[H][i], b)
            end
            push!(frac_pos_acc[H], fp)
            push!(frac_neg_acc[H], fn)
            push!(mean_net_acc[H], mn)

            # Mean margin of correct pairs at each checkpoint
            for cp in checkpoints
                haskey(cpm, cp) || continue
                isempty(cpm[cp]) && continue
                push!(margin_acc[H][cp], mean(cpm[cp]))
            end
        end
        @printf("seed %2d done\n", gseed); flush(stdout)
    end

    cp_windows = ["$(checkpoints[i])-$(checkpoints[i+1])"
                  for i in 1:length(checkpoints)-1]

    println()
    println("=" ^ 72)
    println("A) PAIR PERSISTENCE  (fraction of correct pairs at CP[i] still correct at CP[i+1])")
    println("=" ^ 72)
    @printf("%-16s", "window")
    for H in H_vals; @printf("  H=%-4.1f", H); end
    println()
    println(repeat("-", 32))
    for (i, w) in enumerate(cp_windows)
        @printf("%-16s", w)
        for H in H_vals
            v = persist_acc[H][i]
            @printf("  %.4f", isempty(v) ? NaN : mean(v))
        end
        println()
    end

    println()
    println("=" ^ 72)
    println("B) FIX AND BREAK RATES  (events per SA step in window)")
    println("=" ^ 72)
    @printf("%-16s", "window")
    for H in H_vals; @printf("  H=%-4.1f fix   brk   ratio", H); end
    println()
    println(repeat("-", 72))
    for (i, w) in enumerate(cp_windows)
        @printf("%-16s", w)
        for H in H_vals
            f = isempty(fix_acc[H][i]) ? NaN : mean(fix_acc[H][i])
            b = isempty(brk_acc[H][i]) ? NaN : mean(brk_acc[H][i])
            r = b > 0 ? b/f : 0.0
            @printf("  %6.4f %6.4f %5.2f  ", f, b, r)
        end
        println()
    end

    println()
    println("=" ^ 72)
    println("C) NET PAIR GAIN PER ACCEPTED STEP  (distribution over full run)")
    println("=" ^ 72)
    @printf("%-16s", "metric")
    for H in H_vals; @printf("  H=%-4.1f", H); end
    println()
    println(repeat("-", 32))
    for (label, acc) in [("frac_positive", frac_pos_acc), ("frac_negative", frac_neg_acc),
                          ("mean_net_gain", mean_net_acc)]
        @printf("%-16s", label)
        for H in H_vals
            @printf("  %.4f", isempty(acc[H]) ? NaN : mean(acc[H]))
        end
        println()
    end

    println()
    println("=" ^ 72)
    println("D) MEAN MARGIN OF CORRECT PAIRS AT EACH CHECKPOINT")
    println("=" ^ 72)
    @printf("%-10s", "step")
    for H in H_vals; @printf("  H=%-4.1f", H); end
    println()
    println(repeat("-", 26))
    for cp in checkpoints
        @printf("%-10d", cp)
        for H in H_vals
            v = margin_acc[H][cp]
            @printf("  %.2f", isempty(v) ? NaN : mean(v))
        end
        println()
    end

    # Save CSV
    outfile = joinpath(@__DIR__, "results", "15_mechanism_dissection.csv")
    open(outfile, "w") do io
        println(io, "H,window,persistence,fix_rate,brk_rate")
        for H in H_vals, (i, w) in enumerate(cp_windows)
            p = isempty(persist_acc[H][i]) ? NaN : mean(persist_acc[H][i])
            f = isempty(fix_acc[H][i]) ? NaN : mean(fix_acc[H][i])
            b = isempty(brk_acc[H][i]) ? NaN : mean(brk_acc[H][i])
            @printf(io, "%.1f,%s,%.6f,%.6f,%.6f\n", H, w, p, f, b)
        end
    end
    println("\nSaved to $outfile")
end

main()
