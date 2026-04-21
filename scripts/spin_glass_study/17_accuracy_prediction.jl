# Script 17: Theoretical accuracy prediction as a function of nstate and H.
#
# Model:
#   Each position k receives V ~ Poisson(lambda) votes, where
#     lambda = nstate * npos / (n - 1)
#   Each vote is a "goal-aligned" vote with probability p_goal = 1 - p_anti,
#   and an anti-goal vote with probability p_anti = min_fp / n_valid.
#
#   A position has correct majority when Binomial(V, p_goal) > V/2.
#   Tied positions (2*sum == ctr) are excluded from the accuracy count.
#   A pair is correctly decided when both members have correct (non-tied) majority.
#
#   Predicted acc = E[pair correct] / E[pair counted]
#                = P(both correct & non-tied) / P(both non-tied)
#
# We first measure p_anti (via actual_min_fp and n_valid) under converged SA
# for H in {0.0, 0.25, 0.5, 0.75, 1.0} to get p_anti(H).
#
# Then we predict acc(nstate, H) and compare against actual SA measurements.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

# ---------------------------------------------------------------------------
# Measure p_anti for a given converged (goal, rep) pair.
# p_anti = actual_min_fp / n_valid, averaged over random proposals.
# ---------------------------------------------------------------------------
function measure_p_anti(goal, rep, fingerprint, cxt_master, companion, goal_idx,
                         nqubit; n_probe=300, seed=42)
    n    = 3^nqubit
    rng  = Random.MersenneTwister(seed)
    min_fps = Float64[]; n_valids = Float64[]

    for _ in 1:n_probe
        gen = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts  = rand(rng, 0:1)
        bc  = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        cxt = QCScaling.Context(gen, bc)
        cg  = [companion_goal_s(po, goal, rep, companion, goal_idx, n-1)
               for po in cxt.pos]

        n_v = count(!isnan, cg)
        n_v == 0 && continue
        push!(n_valids, n_v)

        valid, vals = QCScaling._pack_companion_goal(cg, fingerprint.nwords)
        pi = bc.parity + 1
        min_s = typemax(Int)
        @inbounds for ai in 1:size(fingerprint.words,4),
                      tzi in 1:size(fingerprint.words,3)
            s = 0
            for w in 1:fingerprint.nwords
                s += count_ones(xor(fingerprint.words[w,pi,tzi,ai], vals[w]) & valid[w])
            end
            s < min_s && (min_s = s)
        end
        push!(min_fps, min_s)
    end
    p_anti = mean(min_fps ./ n_valids)
    return p_anti, mean(n_valids), mean(min_fps)
end

# ---------------------------------------------------------------------------
# Theoretical accuracy prediction given p_anti and lambda (mean coverage).
#
# P(position has correct majority | V votes) = P(Binomial(V, p_goal) > V/2)
# P(position is valid | V votes)             = P(2*Binomial(V, p_goal) != V)
#
# Returns (pred_acc, pred_frac_valid) where pred_acc is accuracy among
# valid pairs and pred_frac_valid is the fraction of pairs that are valid.
# ---------------------------------------------------------------------------
function predict_accuracy(p_anti, lambda; V_max=50)
    p_goal = 1.0 - p_anti
    # P(V = v) ~ Poisson(lambda)
    pois = zeros(V_max + 1)
    log_pois = -lambda
    for v in 0:V_max
        pois[v+1] = exp(log_pois)
        log_pois += log(lambda) - log(v+1)
    end
    pois ./= sum(pois)  # normalize truncation

    p_correct = 0.0     # P(correct & non-tied)
    p_nontied = 0.0     # P(non-tied)

    for v in 0:V_max
        pv = pois[v+1]
        v == 0 && continue  # uncovered: not counted

        # P(Binomial(v, p_goal) = k) for k in 0..v
        binom = zeros(v+1)
        for k in 0:v
            log_b = (k==0 ? 0.0 : sum(log(j) for j in 1:k)) +
                    ((v-k)==0 ? 0.0 : sum(log(j) for j in 1:(v-k)))
            log_fact_v = sum(log(j) for j in 1:v)
            log_b = log_fact_v - (k==0 ? 0.0 : sum(log(j) for j in 1:k)) -
                    ((v-k)==0 ? 0.0 : sum(log(j) for j in 1:(v-k))) +
                    k * log(p_goal) + (v-k) * log(max(1e-300, 1-p_goal))
            binom[k+1] = exp(log_b)
        end
        binom ./= sum(binom)

        p_win    = sum(binom[k+1] for k in 0:v if k > v/2)   # majority = correct
        p_tied   = isodd(v) ? 0.0 : binom[v÷2+1]             # exact tie
        p_nonted = 1.0 - p_tied

        p_correct += pv * p_win
        p_nontied += pv * p_nonted
    end

    # Pair: both members have non-tied majority, and both are correct
    p_pair_correct  = p_correct^2
    p_pair_nontied  = p_nontied^2

    pred_acc = p_pair_nontied > 0 ? p_pair_correct / p_pair_nontied : 0.0
    return pred_acc, p_pair_nontied
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    npos       = 33   # 2^(nqubit-1) + 1
    nsteps     = 300_000
    alpha_cool = 0.9999
    n_seeds    = 8    # seeds for measuring p_anti

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master  = QCScaling.ContextMaster(nqubit)

    # ----- Step 1: Measure p_anti for a range of H values -----
    println("Measuring p_anti across H values (n=$(n_seeds) seeds each)...")
    H_probe = [0.0, 0.25, 0.5, 0.75, 1.0]
    nstate_for_probe = 45
    p_antis   = Float64[]
    n_valids  = Float64[]
    min_fps   = Float64[]

    for H in H_probe
        k_ones = h_to_kones(H, ngbits)
        seed_p_antis = Float64[]; seed_n_valids = Float64[]; seed_min_fps = Float64[]
        for gseed in 1:n_seeds
            rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
            goal     = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                       zeros(Int, ngbits - k_ones)))
            _, _, _, _, rep =
                run_sa_full(goal, nqubit, nstate_for_probe, nsteps, alpha_cool,
                            companion, goal_idx, fingerprint, cxt_master;
                            seed=gseed * 31 + 7)
            pa, nv, mf = measure_p_anti(goal, rep, fingerprint, cxt_master,
                                         companion, goal_idx, nqubit;
                                         n_probe=200, seed=gseed*997)
            push!(seed_p_antis, pa); push!(seed_n_valids, nv); push!(seed_min_fps, mf)
        end
        push!(p_antis, mean(seed_p_antis))
        push!(n_valids, mean(seed_n_valids))
        push!(min_fps, mean(seed_min_fps))
        @printf("H=%.2f:  p_anti=%.4f  n_valid=%.2f  min_fp=%.2f\n",
                H, mean(seed_p_antis), mean(seed_n_valids), mean(seed_min_fps))
        flush(stdout)
    end

    println()

    # ----- Step 2: Predicted accuracy vs nstate for H=0 and H=1 -----
    println("=" ^ 64)
    println("Predicted accuracy vs nstate  (nqubit=6)")
    println("=" ^ 64)
    nstate_vals = [10, 20, 30, 45, 60, 90, 135, 180, 270, 360]

    # Interpolate p_anti at H=0 and H=1 (already measured directly)
    pa_h0 = p_antis[findfirst(==(0.0), H_probe)]
    pa_h1 = p_antis[findfirst(==(1.0), H_probe)]

    @printf("%-8s  %-12s  %-12s\n", "nstate", "pred H=0", "pred H=1")
    println(repeat("-", 36))
    for ns in nstate_vals
        lambda = ns * npos / (n - 1)
        pred0, _ = predict_accuracy(pa_h0, lambda)
        pred1, _ = predict_accuracy(pa_h1, lambda)
        @printf("%-8d  %-12.4f  %-12.4f\n", ns, pred0, pred1)
    end

    println()

    # ----- Step 3: Compare predictions to actual SA measurements -----
    println("=" ^ 64)
    println("Predicted vs measured accuracy  (nstate=45, nqubit=6)")
    println("=" ^ 64)
    println("Measured values from script 06:")
    println("  nstate=45:  H=0 → 0.943,  H=1 → 0.775")
    println("  nstate=90:  H=0 → 1.000,  H=1 → 0.953")
    println("  nstate=180: H=0 → 1.000,  H=1 → ~1.000")
    println()
    @printf("%-8s  %-12s  %-12s  %-12s  %-12s\n",
            "nstate", "pred H=0", "meas H=0", "pred H=1", "meas H=1")
    println(repeat("-", 60))
    measured = Dict(
        45  => (0.943, 0.775),
        90  => (1.000, 0.953),
        180 => (1.000, 1.000),
    )
    for ns in [45, 90, 180]
        lambda = ns * npos / (n - 1)
        pred0, _ = predict_accuracy(pa_h0, lambda)
        pred1, _ = predict_accuracy(pa_h1, lambda)
        m0, m1 = measured[ns]
        @printf("%-8d  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n",
                ns, pred0, m0, pred1, m1)
    end

    println()

    # ----- Step 4: Predicted accuracy vs H for nstate=45 -----
    println("=" ^ 64)
    println("Predicted accuracy vs H  (nstate=45, nqubit=6)")
    println("=" ^ 64)
    lambda_45 = 45 * npos / (n - 1)
    @printf("%-6s  %-12s  %-10s  %-10s\n", "H", "p_anti", "pred_acc", "p_valid_pair")
    println(repeat("-", 44))
    for (H, pa) in zip(H_probe, p_antis)
        pred, p_valid = predict_accuracy(pa, lambda_45)
        @printf("%-6.2f  %-12.4f  %-10.4f  %-10.4f\n", H, pa, pred, p_valid)
    end

    # Save
    outfile = joinpath(@__DIR__, "results", "17_accuracy_prediction.csv")
    open(outfile, "w") do io
        println(io, "H,p_anti,n_valid,min_fp")
        for (H, pa, nv, mf) in zip(H_probe, p_antis, n_valids, min_fps)
            @printf(io, "%.2f,%.6f,%.4f,%.4f\n", H, pa, nv, mf)
        end
    end
    println("Saved to $outfile")
end

main()
