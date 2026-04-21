# Script 14: Phase-specific heuristics and targeted generator polishing.
#
# Two experiments:
#
# A) PHASED SA
#    The SA is split into phases using different pickers:
#    - baseline:  PICKER_FP throughout
#    - phase_mid: FP(0-5k) → LOCAL_ORACLE(5k-50k) → FP(50k-300k)
#    - phase_all: LOCAL_ORACLE throughout (already tested; repeated here for ref)
#
# B) TARGETED GENERATOR POLISHING
#    After a full SA run, run additional "repair" steps where instead of a
#    random generator, we actively select generators that cover exactly one
#    member of a wrong pair.  This directly attacks the H=1 coordination
#    failure: for H=1 wrong pairs (r1==r2), the standard fp signal pushes
#    BOTH positions to flip simultaneously, keeping the pair wrong.  A
#    generator covering only one member breaks that symmetry.
#
# Focus: H=1 (the hard case).  n_seeds=20 seeds for reliable statistics.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Precompute: for each position k, which (gen_idx, theta_s) pairs cover it?
# ---------------------------------------------------------------------------
function build_pos_to_gens(nqubit, cxt_master)
    n = 3^nqubit
    pos_to_gens = [Tuple{Int,Int}[] for _ in 1:n-1]
    for gen_idx in 0:n-1
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        for ts in 0:1
            bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, bc)
            for po in cxt.pos
                k = po.index
                (k >= 1 && k <= n-1) && push!(pos_to_gens[k], (gen_idx, ts))
            end
        end
    end
    return pos_to_gens
end

# ---------------------------------------------------------------------------
# Collect wrong pairs from current ensemble state.
# Returns Vector{Tuple{Int,Int}} of canonical (k1<k2) wrong pairs.
# ---------------------------------------------------------------------------
function wrong_pairs(rep_sum, rep_ctr, goal, companion, goal_idx, n)
    pairs = Tuple{Int,Int}[]
    for k1 in 1:n-1
        companion[k1] == 0 && continue
        k1 > companion[k1] && continue
        k2 = companion[k1]
        c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
        (c1 == 0 || c2 == 0) && continue
        (2*rep_sum[k1] == c1 || 2*rep_sum[k2] == c2) && continue  # tied
        r1 = 2*rep_sum[k1] > c1
        r2 = 2*rep_sum[k2] > c2
        j  = goal_idx[k1]
        (r1 ⊻ r2) == !iszero(goal[j]) && continue  # already correct
        push!(pairs, (k1, k2))
    end
    return pairs
end

# ---------------------------------------------------------------------------
# Targeted generator polishing.
#
# For each repair step:
#   1. Sample a random wrong pair (k1, k2).
#   2. Find generators that cover k1 but NOT k2 (or vice versa).
#   3. Use the fp picker to choose alphas for that generator.
#   4. Accept the swap if it improves (or equal) accuracy.
#
# Returns final accuracy.
# ---------------------------------------------------------------------------
function targeted_repair!(rep_sum, rep_ctr, rep, ensemble, cache_idxs, cache_pars,
                           goal, nqubit, companion, goal_idx, fingerprint, cxt_master,
                           pos_to_gens; nsteps=20_000, rng)
    n      = 3^nqubit
    nstate = length(ensemble)
    npos   = length(cxt_master.base_even.pos)
    acc_fn = (rs, rc) -> rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    n_accepted = 0
    n_exclusive_found = 0

    for step in 1:nsteps
        wp = wrong_pairs(rep_sum, rep_ctr, goal, companion, goal_idx, n)
        isempty(wp) && break

        k1, k2 = wp[rand(rng, 1:length(wp))]

        # Generators covering k1 but not k2 (exclusive to k1)
        g1 = Set(pos_to_gens[k1])
        g2 = Set(pos_to_gens[k2])
        excl1 = collect(setdiff(g1, g2))
        excl2 = collect(setdiff(g2, g1))

        # Pick whichever exclusive set is non-empty (prefer excl1)
        if !isempty(excl1)
            gen_idx, ts = excl1[rand(rng, 1:length(excl1))]
            n_exclusive_found += 1
        elseif !isempty(excl2)
            gen_idx, ts = excl2[rand(rng, 1:length(excl2))]
            n_exclusive_found += 1
        else
            # No exclusive generator found for this pair; skip
            continue
        end

        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(gen, bc)

        # Use fp picker for alpha selection
        ret = pick_alphas_s(cxt, goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns  = QCScaling.PseudoGHZState(ret..., gen)

        which    = rand(rng, 1:nstate)
        cur_acc  = acc_fn(rep_sum, rep_ctr)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)

        if new_acc >= cur_acc
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns
            n_accepted += 1
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
    end
    return acc_fn(rep_sum, rep_ctr), n_accepted, n_exclusive_found
end

# ---------------------------------------------------------------------------
# Phased SA: schedule = [(step_end_1, picker1), (step_end_2, picker2), ...]
# Step ranges are [1, schedule[1][1]], (schedule[i][1], schedule[i+1][1]], etc.
# ---------------------------------------------------------------------------
function run_sa_phased(goal, nqubit, nstate, schedule, alpha_cool,
                       companion, goal_idx, fingerprint, cxt_master; seed=42)
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

    # Build schedule lookup: sorted by step_end
    sched_sorted = sort(schedule, by=x->x[1])
    nsteps_total = sched_sorted[end][1]
    picker_fn(step) = sched_sorted[findfirst(s -> s[1] >= step, sched_sorted)][2]

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    p0 = picker_fn(1)
    for _ in 1:300
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = p0(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
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

    for step in 1:nsteps_total
        pfn   = picker_fn(step)
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pfn(QCScaling.Context(gen, bc), goal, rep, rep_sum, rep_ctr,
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
    end
    return cur_acc, ensemble, rep_sum, rep_ctr, rep
end

# ---------------------------------------------------------------------------
# Run SA and return full state (for polishing experiment)
# ---------------------------------------------------------------------------
function run_sa_returning_state(goal, nqubit, nstate, nsteps, alpha_cool,
                                 companion, goal_idx, fingerprint, cxt_master; seed=42)
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

    for _ in 1:nsteps
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
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc, ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars, rng
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
    n_seeds    = 20
    H          = 1.0   # focus on the hard case

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    # Precompute position → generator lookup for polishing
    print("Building pos_to_gens... "); flush(stdout)
    pos_to_gens = build_pos_to_gens(nqubit, cxt_master)
    # Quick diagnostic: fraction of wrong pairs that have exclusive generators
    println("done")
    sample_pair = (1, companion[1])  # just show one
    k1, k2 = sample_pair
    g1 = Set(pos_to_gens[k1]); g2 = Set(pos_to_gens[k2])
    @printf("Sample pair (%d,%d): |excl k1|=%d, |excl k2|=%d (of %d gen×ts combos each)\n",
            k1, k2, length(setdiff(g1,g2)), length(setdiff(g2,g1)), length(g1))

    # Schedules for phased SA (total = 300_000 steps)
    schedules = [
        ("fp_only",   [(300_000, PICKER_FP)]),
        ("mid_lo",    [(5_000, PICKER_FP), (50_000, PICKER_LOCAL_ORACLE), (300_000, PICKER_FP)]),
        ("mid_m1",    [(5_000, PICKER_FP), (50_000, PICKER_MARGIN1),      (300_000, PICKER_FP)]),
    ]

    # Storage
    acc_phased   = Dict(name => Float64[] for (name, _) in schedules)
    acc_pre_pol  = Float64[]
    acc_post_pol = Float64[]

    k_ones = h_to_kones(H, ngbits)

    for gseed in 1:n_seeds
        rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
        goal     = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                   zeros(Int, ngbits - k_ones)))
        sa_seed  = gseed * 31 + 7

        # A) Phased SA experiments
        for (name, sched) in schedules
            af, _, _, _, _ = run_sa_phased(goal, nqubit, nstate, sched, alpha_cool,
                                            companion, goal_idx, fingerprint, cxt_master;
                                            seed=sa_seed)
            push!(acc_phased[name], af)
        end

        # B) SA + polishing
        af_pre, ens, rs, rc, rep, ci, cp, rng_post =
            run_sa_returning_state(goal, nqubit, nstate, nsteps, alpha_cool,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=sa_seed)
        push!(acc_pre_pol, af_pre)

        n_wrong_before = length(wrong_pairs(rs, rc, goal, companion, goal_idx, n))
        af_post, n_acc, n_excl = targeted_repair!(
            rs, rc, rep, ens, ci, cp, goal, nqubit, companion, goal_idx,
            fingerprint, cxt_master, pos_to_gens;
            nsteps=20_000, rng=rng_post)
        push!(acc_post_pol, af_post)

        @printf("seed %2d  fp=%.4f  post_pol=%.4f  wrong_before=%d  n_excl=%d  n_acc=%d\n",
                gseed, af_pre, af_post, n_wrong_before, n_excl, n_acc)
        flush(stdout)
    end

    println()
    println("=" ^ 72)
    println("A) PHASED SA (H=1.0, nqubit=6, nstate=45, nsteps=300k, n=$(n_seeds) seeds)")
    println("=" ^ 72)
    @printf("%-14s  %-10s\n", "schedule", "acc_final")
    println(repeat("-", 30))
    for (name, _) in schedules
        @printf("%-14s  %.4f ± %.4f\n", name,
                mean(acc_phased[name]), std(acc_phased[name])/sqrt(n_seeds))
    end

    println()
    println("=" ^ 72)
    println("B) TARGETED GENERATOR POLISHING (20k steps after 300k SA)")
    println("=" ^ 72)
    @printf("before: %.4f ± %.4f\n",
            mean(acc_pre_pol), std(acc_pre_pol)/sqrt(n_seeds))
    @printf("after:  %.4f ± %.4f\n",
            mean(acc_post_pol), std(acc_post_pol)/sqrt(n_seeds))
    @printf("gain:   %.4f\n", mean(acc_post_pol) - mean(acc_pre_pol))

    # Save
    outfile = joinpath(@__DIR__, "results", "14_polishing.csv")
    open(outfile, "w") do io
        println(io, "experiment,seed,acc")
        for (name, _) in schedules, (i, a) in enumerate(acc_phased[name])
            @printf(io, "%s,%d,%.6f\n", name, i, a)
        end
        for (i, (ap, aa)) in enumerate(zip(acc_pre_pol, acc_post_pol))
            @printf(io, "fp_only_check,%d,%.6f\n", i, ap)
            @printf(io, "polished,%d,%.6f\n", i, aa)
        end
    end
    println("Saved to $outfile")
end

main()
