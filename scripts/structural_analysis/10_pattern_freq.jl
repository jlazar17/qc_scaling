# Pattern selection frequency during SA: H=0 vs H=1
#
# Hypothesis: H=0 picker selections concentrate on a small number of the 64
# (tz, alpha) patterns, causing the same positions to accumulate repeated votes
# and build large margins. H=1 selections spread more uniformly, so margins
# stay small.
#
# For each accepted SA proposal, we record the pattern index:
#   alpha_idx = sum(alphas[i] * 2^(i-1))    (0 .. 2^(nqubit-1) - 1)
#   pattern_idx = tz * nalpha + alpha_idx    (0 .. 2*nalpha - 1 = 63 for nqubit=6)
#
# Frequencies are bucketed by current accuracy so H=0 and H=1 can be
# compared at the same accuracy level, not just at end-of-run.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Pattern index from pick_alphas_s return value (theta_s, tz, alphas)
# ---------------------------------------------------------------------------
function pattern_index(tz, alphas)
    nalpha = 1 << length(alphas)
    ai = 0
    for i in eachindex(alphas)
        ai |= alphas[i] << (i - 1)
    end
    return tz * nalpha + ai   # 0-based, 0 .. 2*nalpha-1
end

# ---------------------------------------------------------------------------
# SA with pattern logging bucketed by accuracy
# acc_bins: e.g. [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# Returns (final_acc, proposed_freq[nbin, npatterns], accepted_freq[nbin, npatterns])
# ---------------------------------------------------------------------------
function run_sa_log(goal, nqubit, nstate, nsteps, alpha_cool,
                    companion, goal_idx, fingerprint, cxt_master,
                    acc_bins; seed=42)
    n     = 3^nqubit
    npos  = length(cxt_master.base_even.pos)
    nalpha = 1 << (nqubit - 1)
    npatterns = 2 * nalpha
    nbin  = length(acc_bins) - 1

    rng        = Random.MersenneTwister(seed)
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
    scratch_idxs = Vector{Int}(undef, npos); scratch_pars = Vector{Int}(undef, npos)

    # Temperature calibration
    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        alphas_ret = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(alphas_ret..., gen)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        d = acc_fn(rep_sum, rep_ctr) - cur_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        d < 0 && push!(bad_deltas, abs(d))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    proposed_freq  = zeros(Int, nbin, npatterns)
    accepted_pos   = zeros(Int, nbin, npatterns)  # delta >= 0 (temp-independent)
    accepted_boltz = zeros(Int, nbin, npatterns)  # delta < 0, accepted via Boltzmann

    bin_idx(acc) = clamp(searchsortedlast(acc_bins, acc), 1, nbin)

    for _ in 1:nsteps
        which  = rand(rng, 1:nstate)
        gen    = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts     = rand(rng, 0:1); bc = ts==0 ? cxt_master.base_even : cxt_master.base_odd
        ret    = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint, bc, companion, goal_idx, n-1)
        ns     = QCScaling.PseudoGHZState(ret..., gen)

        pidx = pattern_index(ret[2], ret[3]) + 1   # 1-based
        bi   = bin_idx(cur_acc)
        proposed_freq[bi, pidx] += 1

        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc

        if d >= 0
            accepted_pos[bi, pidx] += 1
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs); copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        elseif rand(rng) < exp(d / T)
            accepted_boltz[bi, pidx] += 1
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs); copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc, proposed_freq, accepted_pos, accepted_boltz
end

# ---------------------------------------------------------------------------
# Distribution entropy (bits)
# ---------------------------------------------------------------------------
function dist_entropy(freq)
    total = sum(freq)
    total == 0 && return NaN
    h = 0.0
    for f in freq
        f == 0 && continue
        p = f / total
        h -= p * log2(p)
    end
    return h
end

# ---------------------------------------------------------------------------
# Acceptance/proposal ratio per pattern, as a summary scalar
# (correlation between proposed_frac and accepted_frac across patterns)
# A high correlation means the picker's top patterns are also the most accepted.
# ---------------------------------------------------------------------------
function prop_acc_correlation(prop, acc)
    total_p = sum(prop); total_a = sum(acc)
    (total_p == 0 || total_a == 0) && return NaN
    pf = prop ./ total_p
    af = acc  ./ total_a
    return cor(pf, af)
end

# ---------------------------------------------------------------------------
# Print comparison for one accuracy bin
# ---------------------------------------------------------------------------
function print_bin_comparison(prop0, pos0, boltz0, prop1, pos1, boltz1, bin_label)
    npatterns = length(prop0)
    tp0 = sum(prop0); tp1 = sum(prop1)
    (tp0 == 0 && tp1 == 0) && return

    tpos0 = sum(pos0); tpos1 = sum(pos1)
    tboltz0 = sum(boltz0); tboltz1 = sum(boltz1)

    println(bin_label)
    @printf("  proposals:          H=0=%-8d  H=1=%d\n", tp0, tp1)
    @printf("  accepted (delta>=0): H=0=%-8d  H=1=%d\n", tpos0, tpos1)
    @printf("  accepted (Boltzmann): H=0=%-8d  H=1=%d\n", tboltz0, tboltz1)

    uniform = log2(npatterns)
    h_prop0 = dist_entropy(prop0); h_prop1 = dist_entropy(prop1)
    @printf("  proposal entropy:   H=0=%.3f  H=1=%.3f  (uniform=%.3f)\n",
            h_prop0, h_prop1, uniform)

    # Correlation using only positive-delta acceptances (temp-independent)
    r0 = prop_acc_correlation(prop0, pos0)
    r1 = prop_acc_correlation(prop1, pos1)
    @printf("  prop/pos-acc corr:  H=0=%.3f  H=1=%.3f  (temp-independent)\n", r0, r1)

    # Positive-delta acceptance rate per pattern
    rate0 = tpos0 / max(1, tp0)
    rate1 = tpos1 / max(1, tp1)
    @printf("  pos-delta accept rate: H=0=%.4f  H=1=%.4f\n\n", rate0, rate1)
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

    acc_bins = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    results = Dict{String, NamedTuple}()

    for (H_label, k_ones) in [("H=0.0", 0), ("H=1.0", ngbits ÷ 2)]
        rng  = Random.MersenneTwister(99)
        goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

        println("Running SA for $H_label ...")
        final_acc, prop_freq, acc_pos, acc_boltz =
            run_sa_log(goal, nqubit, nstate, nsteps, alpha_cool,
                       companion, goal_idx, fingerprint, cxt_master,
                       acc_bins; seed=42)
        @printf("  Final accuracy: %.4f\n", final_acc)
        results[H_label] = (prop_freq=prop_freq, acc_pos=acc_pos, acc_boltz=acc_boltz, final_acc=final_acc)
    end

    println()
    println("=" ^ 70)
    println("PATTERN FREQUENCY COMPARISON BY ACCURACY BIN")
    println("=" ^ 70)

    nbin = length(acc_bins) - 1
    for bi in 1:nbin
        bin_label = @sprintf("Accuracy [%.2f, %.2f)", acc_bins[bi], acc_bins[bi+1])
        prop0  = results["H=0.0"].prop_freq[bi, :]
        pos0   = results["H=0.0"].acc_pos[bi, :]
        boltz0 = results["H=0.0"].acc_boltz[bi, :]
        prop1  = results["H=1.0"].prop_freq[bi, :]
        pos1   = results["H=1.0"].acc_pos[bi, :]
        boltz1 = results["H=1.0"].acc_boltz[bi, :]
        sum(prop0) + sum(prop1) == 0 && continue
        print_bin_comparison(prop0, pos0, boltz0, prop1, pos1, boltz1, bin_label)
    end
end

main()
