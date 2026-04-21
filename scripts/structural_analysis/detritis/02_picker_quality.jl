# Analyze picker quality for H=0 vs H=1 goals at nqubit=6.
#
# The picker finds alpha settings by minimizing mismatches between fingerprint
# entries and companion_goal values. For H=0, companion goals may cluster (all
# same value) making the picker highly effective. For H=1, companion goals may
# be perfectly balanced (50% each) making the picker no better than random.
#
# This script quantifies this "picker degeneracy" effect.
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Build rep cache from SA (simplified: use random ensemble + a few SA steps)
# ---------------------------------------------------------------------------

function build_rep_for_goal(nqubit, goal, nstate, nsteps; seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit
    npos = length(cxt_master.base_even.pos)

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end

    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))

    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    alpha_cool = 0.9999
    stag_window = round(Int, -5.0 / log(alpha_cool))
    T = 0.1
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    last_improvement = 0

    for step in 1:nsteps
        which = rand(rng, 1:nstate)
        gen_idx = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s = rand(rng, 0:1)
        base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(generator, base_cxt)
        alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
        ns = QCScaling.PseudoGHZState(alphas..., generator)

        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta = new_acc - current_acc

        if delta >= 0 || rand(rng) < exp(delta / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns
            current_acc = new_acc
            new_acc > current_acc + 1.0/((n-1)÷2) && (last_improvement = step)
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return rep, current_acc
end

# ---------------------------------------------------------------------------
# Picker quality analysis
# ---------------------------------------------------------------------------

"""
For a given rep (representation vector) and goal, compute the "picker quality"
for each possible (generator, theta_s) context:
  - quality = fraction of non-NaN companion goals matched by the BEST alpha

Returns the distribution of best-match fractions over all contexts.
"""
function compute_picker_quality(nqubit, goal, rep, cxt_master)
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    fingerprint = QCScaling.Fingerprint(nqubit)

    best_fracs = Float64[]  # best match fraction per context
    cg_entropies = Float64[]  # entropy of companion goals per context

    for gen_idx in 0:n-1
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        for theta_s in 0:1
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(gen, base_cxt)

            # Compute companion goals
            cg = QCScaling.companion_goal(cxt, goal, rep)
            valid_mask = .!isnan.(cg)
            n_valid = count(valid_mask)
            n_valid == 0 && continue

            # Fraction of valid companion goals that are 1
            frac_ones = sum(cg[valid_mask] .== 1.0) / n_valid
            # Binary entropy of companion goal distribution
            h = (frac_ones == 0 || frac_ones == 1) ? 0.0 :
                -frac_ones*log2(frac_ones) - (1-frac_ones)*log2(1-frac_ones)
            push!(cg_entropies, h)

            # Find best alpha match fraction
            parity_idx = theta_s + 1
            nalpha = size(fingerprint.a, 4)
            ntz = size(fingerprint.a, 3)
            npos = size(fingerprint.a, 1)

            best_match = 0
            for ai in 1:nalpha
                for tzi in 1:ntz
                    # Count matches between fingerprint and companion_goal
                    match = 0
                    for pi in 1:npos
                        !valid_mask[pi] && continue
                        fp_val = fingerprint.a[pi, parity_idx, tzi, ai]
                        cg_val = round(Int, cg[pi])
                        match += (fp_val == cg_val)
                    end
                    best_match = max(best_match, match)
                end
            end
            push!(best_fracs, best_match / n_valid)
        end
    end
    return best_fracs, cg_entropies
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit = 6
    n = 3^nqubit
    ngbits = (n - 1) ÷ 2
    nstate = 50
    nsteps = 30_000

    cxt_master = QCScaling.ContextMaster(nqubit)

    println("nqubit=$nqubit, ngbits=$ngbits, nstate=$nstate")
    println()

    rng_goals = Random.MersenneTwister(77)

    for H_target in [0.0, 0.5, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        goal = shuffle!(rng_goals, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

        @printf("--- H=%.2f (ones=%d / ngbits=%d) ---\n", H_target, goal_ones, ngbits)
        print("  Building SA rep (nsteps=$nsteps)...")
        flush(stdout)
        rep, acc = build_rep_for_goal(nqubit, goal, nstate, nsteps; seed=42)
        @printf(" done. acc=%.4f\n", acc)

        print("  Computing picker quality over all contexts...")
        flush(stdout)
        best_fracs, cg_entropies = compute_picker_quality(nqubit, goal, rep, cxt_master)
        println(" done.")

        println()
        println("  Companion goal entropy distribution (0=uniform, 1=balanced):")
        @printf("    mean=%.4f  median=%.4f  min=%.4f  max=%.4f\n",
                mean(cg_entropies), median(cg_entropies),
                minimum(cg_entropies), maximum(cg_entropies))
        pct_balanced = count(cg_entropies .> 0.9) / length(cg_entropies)
        pct_pure = count(cg_entropies .< 0.1) / length(cg_entropies)
        @printf("    Fraction with entropy>0.9 (nearly balanced): %.3f\n", pct_balanced)
        @printf("    Fraction with entropy<0.1 (nearly uniform):  %.3f\n", pct_pure)

        println()
        println("  Best picker match fraction (over alpha settings):")
        @printf("    mean=%.4f  median=%.4f  min=%.4f  max=%.4f\n",
                mean(best_fracs), median(best_fracs),
                minimum(best_fracs), maximum(best_fracs))
        pct_above_75 = count(best_fracs .>= 0.75) / length(best_fracs)
        pct_above_60 = count(best_fracs .>= 0.60) / length(best_fracs)
        pct_above_50 = count(best_fracs .> 0.50)  / length(best_fracs)
        @printf("    Fraction with best_frac>=0.75: %.3f\n", pct_above_75)
        @printf("    Fraction with best_frac>=0.60: %.3f\n", pct_above_60)
        @printf("    Fraction with best_frac>0.50:  %.3f\n", pct_above_50)
        println()

        # Distribution histogram
        println("  Histogram of best_match_fraction:")
        bins = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
        for i in 1:length(bins)-1
            cnt = count(bins[i] .<= best_fracs .< bins[i+1])
            pct = cnt / length(best_fracs)
            @printf("    [%.2f, %.2f): %5d  (%.1f%%)\n", bins[i], bins[i+1], cnt, 100*pct)
        end
        println()
    end

    # ---------------------------------------------------------------------------
    # Key comparison: for same rep, compare picker quality H=0 vs H=1
    # ---------------------------------------------------------------------------
    println("="^65)
    println("Picker quality using a PERFECT rep (oracle knowledge)")
    println("="^65)
    println("For each goal, assume perfect rep: rep[k1]=target, rep[k2]=target")
    println()

    rng2 = Random.MersenneTwister(13)
    for H_target in [0.0, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        goal = shuffle!(rng2, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

        # Build "oracle" rep: set rep correctly for every pair
        rep = fill(NaN, n)
        # Assign rep[k1]=0, rep[k2]=goal[j] for each pair j
        for j in 1:ngbits
            rep[2j-1] = 0.0
            rep[2j]   = Float64(goal[j])  # 0 if goal=0 (same), 1 if goal=1 (different)
        end

        best_fracs, cg_entropies = compute_picker_quality(nqubit, goal, rep, cxt_master)

        @printf("H=%.2f (oracle rep):\n", H_target)
        @printf("  CG entropy:  mean=%.4f, median=%.4f\n",
                mean(cg_entropies), median(cg_entropies))
        @printf("  Best match:  mean=%.4f, median=%.4f, frac>0.50: %.3f\n",
                mean(best_fracs), median(best_fracs),
                count(best_fracs .> 0.50)/length(best_fracs))
        println()
    end
end

main()
