# Script 13: Compare alternative picker heuristics.
#
# The current fingerprint picker (PICKER_FP) is a near-random predictor:
# the oracle-best pattern ranks ~25-30/63 in its ordering.  Two alternatives:
#
#   PICKER_MARGIN1      -- same bit-packed fp score but only counts positions
#                          with margin <= 1 (the only positions a single vote
#                          can flip)
#   PICKER_LOCAL_ORACLE -- bit-packed exact expected accuracy delta at
#                          margin-flippable positions: +1 for fixing a wrong
#                          pair, -1 for breaking a right pair
#
# We run each picker on H=0 and H=1 goals for n_seeds seeds and report:
#   - Final accuracy at nsteps=300_000
#   - Accuracy at an intermediate checkpoint (step 20_000)
#   - oracle_rank of the picked pattern (rank of picked in true-delta ordering)
#   - frac_picker_is_oracle: how often the picked pattern is the true best
#
# oracle_rank and frac are measured at step 20_000 (divergence zone) using
# n_probe=200 random proposals per seed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Measure oracle rank of the pattern chosen by a picker vs true delta rank.
# Returns (mean_oracle_rank, frac_is_oracle) over n_probe proposals.
# ---------------------------------------------------------------------------
function measure_picker_quality(picker_fn, goal, rep, rep_sum, rep_ctr,
                                 fingerprint, cxt_master, companion, goal_idx,
                                 nqubit; n_probe=200, seed=9999)
    n   = 3^nqubit
    rng = Random.MersenneTwister(seed)
    oracle_ranks = Int[]
    is_oracle    = Bool[]

    for _ in 1:n_probe
        gen  = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts   = rand(rng, 0:1)
        bc   = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt  = QCScaling.Context(gen, bc)

        # All 64 true accuracy deltas (exact oracle)
        nalpha = size(fingerprint.words, 4)
        ntz    = size(fingerprint.words, 3)
        deltas = Vector{Float64}(undef, ntz * nalpha)
        acc_fn = (rs, rc) -> rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
        base_acc = acc_fn(rep_sum, rep_ctr)
        idx = 1
        for ai in 1:nalpha, tzi in 1:ntz
            alphas = QCScaling.idx_to_alphas(ai-1, nqubit)
            ns = QCScaling.PseudoGHZState(bc.parity, tzi-1, alphas, gen)
            # probe delta by temporarily adding this state (no removal of old)
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
            deltas[idx] = acc_fn(rep_sum, rep_ctr) - base_acc
            apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
            idx += 1
        end

        # Pattern chosen by the picker
        ret = picker_fn(cxt, goal, rep, rep_sum, rep_ctr, fingerprint, bc,
                        companion, goal_idx, n-1)
        picked_tzi   = ret[2] + 1          # 1-based
        picked_alpha = sum(ret[3][i] << (i-1) for i in eachindex(ret[3])) + 1  # 1-based

        # Index of picked pattern in deltas vector
        # (ai outer loop, tzi inner loop in deltas)
        picked_idx = (picked_alpha - 1) * ntz + picked_tzi

        # Rank of picked among all deltas (0 = best = highest delta)
        oracle_rank = count(deltas .> deltas[picked_idx])
        push!(oracle_ranks, oracle_rank)
        push!(is_oracle, oracle_rank == 0)
    end
    return mean(oracle_ranks), mean(is_oracle)
end

# ---------------------------------------------------------------------------
# Run SA with checkpointed quality measurement
# ---------------------------------------------------------------------------
function run_with_checkpoint(goal, nqubit, nstate, nsteps_checkpoint,
                              nsteps_total, alpha_cool,
                              companion, goal_idx, fingerprint, cxt_master;
                              seed=42, picker=PICKER_FP,
                              n_probe=200, quality_seed=9999)
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
        ret   = picker(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
                       fingerprint, bc, companion, goal_idx, n-1)
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

    acc_checkpoint = NaN
    orank_checkpoint = NaN
    frac_checkpoint  = NaN

    for step in 1:nsteps_total
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = picker(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
                       fingerprint, bc, companion, goal_idx, n-1)
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
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool

        if step == nsteps_checkpoint
            acc_checkpoint = cur_acc
            orank_checkpoint, frac_checkpoint =
                measure_picker_quality(picker, goal, rep, rep_sum, rep_ctr,
                                       fingerprint, cxt_master, companion, goal_idx,
                                       nqubit; n_probe=n_probe, seed=quality_seed)
        end
    end
    return cur_acc, acc_checkpoint, orank_checkpoint, frac_checkpoint
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
    checkpoint = 20_000
    n_seeds    = 20
    n_probe    = 300

    H_vals   = [0.0, 1.0]
    pickers  = [("fp",           PICKER_FP),
                ("margin1",      PICKER_MARGIN1),
                ("local_oracle", PICKER_LOCAL_ORACLE),
                ("hybrid2",      PICKER_HYBRID2),
                ("hybrid4",      PICKER_HYBRID4)]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    # Storage: [picker, H, seed] for each metric
    results = Dict()
    for (pname, _) in pickers, H in H_vals
        results[(pname, H)] = (acc_final=Float64[], acc_ckpt=Float64[],
                               orank=Float64[], frac=Float64[])
    end

    for gseed in 1:n_seeds
        for H in H_vals
            k_ones = h_to_kones(H, ngbits)
            rng    = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal   = Random.shuffle!(rng, vcat(ones(Int, k_ones),
                                               zeros(Int, ngbits - k_ones)))
            for (pname, pfn) in pickers
                af, ac, or_, fr =
                    run_with_checkpoint(goal, nqubit, nstate, checkpoint, nsteps,
                                        alpha_cool, companion, goal_idx,
                                        fingerprint, cxt_master;
                                        seed=gseed * 31 + 7, picker=pfn,
                                        n_probe=n_probe,
                                        quality_seed=gseed * 997 + 13)
                r = results[(pname, H)]
                push!(r.acc_final, af); push!(r.acc_ckpt, ac)
                push!(r.orank, or_);   push!(r.frac, fr)
            end
        end
        @printf("seed %2d done\n", gseed); flush(stdout)
    end

    # Print summary table
    println()
    @printf("%-14s  %-4s  %-10s  %-10s  %-10s  %-10s\n",
            "picker", "H", "acc_final", "acc@20k", "orc_rank", "frac_oracle")
    println(repeat("-", 66))
    for H in H_vals, (pname, _) in pickers
        r = results[(pname, H)]
        @printf("%-14s  %-4.1f  %-10.4f  %-10.4f  %-10.2f  %-10.4f\n",
                pname, H,
                mean(r.acc_final), mean(r.acc_ckpt),
                mean(r.orank),     mean(r.frac))
    end

    # Save raw results to CSV
    outfile = joinpath(@__DIR__, "results", "13_heuristic_comparison.csv")
    open(outfile, "w") do io
        println(io, "picker,H,acc_final,acc_checkpoint,oracle_rank,frac_oracle")
        for H in H_vals, (pname, _) in pickers
            r = results[(pname, H)]
            for i in eachindex(r.acc_final)
                @printf(io, "%s,%.1f,%.6f,%.6f,%.4f,%.4f\n",
                        pname, H, r.acc_final[i], r.acc_ckpt[i],
                        r.orank[i], r.frac[i])
            end
        end
    end
    println("\nSaved to $outfile")
end

main()
