# Script 19: min_fp and p_anti as a function of nstate, for different H values.
#
# For each (H, nstate), run SA with a fixed per-state step budget to convergence,
# then measure actual_min_fp and p_anti = min_fp / n_valid from random proposals.
# This shows whether the structural H=0 advantage (lower min_fp) persists across
# ensemble sizes, or whether it is specific to the nstate=45 regime.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

using Plots
using DelimitedFiles

# ---------------------------------------------------------------------------
# Measure min_fp and p_anti for a converged (goal, rep) pair.
# Same as script 17.
# ---------------------------------------------------------------------------
function measure_p_anti(goal, rep, fingerprint, cxt_master, companion, goal_idx,
                        nqubit; n_probe=200, seed=42)
    n   = 3^nqubit
    rng = Random.MersenneTwister(seed)
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
        @inbounds for ai in 1:size(fingerprint.words, 4),
                      tzi in 1:size(fingerprint.words, 3)
            s = 0
            for w in 1:fingerprint.nwords
                s += count_ones(xor(fingerprint.words[w, pi, tzi, ai], vals[w]) & valid[w])
            end
            s < min_s && (min_s = s)
        end
        push!(min_fps, min_s)
    end
    isempty(min_fps) && return NaN, NaN, NaN
    p_anti = mean(min_fps ./ n_valids)
    return p_anti, mean(n_valids), mean(min_fps)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit          = 6
    n               = 3^nqubit
    ngbits          = (n - 1) ÷ 2
    steps_per_state = 300_000 ÷ 45   # fixed per-state budget (~6667)
    alpha_cool      = 0.9999
    n_seeds         = 8
    n_probe         = 200

    H_vals      = [0.0, 0.25, 0.5, 0.75, 1.0]
    nstate_vals = [15, 20, 25, 30, 40, 55, 70, 90, 120, 150, 180, 240, 300]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master   = QCScaling.ContextMaster(nqubit)

    # results[H] = (mean_p_anti, mean_min_fp, mean_n_valid) per nstate
    all_p_anti  = zeros(length(H_vals), length(nstate_vals))
    all_min_fp  = zeros(length(H_vals), length(nstate_vals))
    all_n_valid = zeros(length(H_vals), length(nstate_vals))

    for (hi, H) in enumerate(H_vals)
        k_ones = h_to_kones(H, ngbits)
        println("H=$(H):")
        for (ni, nstate) in enumerate(nstate_vals)
            nsteps = nstate * steps_per_state
            alpha  = exp(log(1e-4) / nsteps)

            seed_pa = Float64[]; seed_mf = Float64[]; seed_nv = Float64[]
            for gseed in 1:n_seeds
                rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000))
                goal = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                       zeros(Int, ngbits - k_ones)))
                _, _, _, _, rep =
                    run_sa_full(goal, nqubit, nstate, nsteps, alpha,
                                companion, goal_idx, fingerprint, cxt_master;
                                seed=gseed * 31 + 7)
                pa, nv, mf = measure_p_anti(goal, rep, fingerprint, cxt_master,
                                             companion, goal_idx, nqubit;
                                             n_probe=n_probe, seed=gseed * 997)
                push!(seed_pa, pa); push!(seed_mf, mf); push!(seed_nv, nv)
            end
            all_p_anti[hi, ni]  = mean(seed_pa)
            all_min_fp[hi, ni]  = mean(seed_mf)
            all_n_valid[hi, ni] = mean(seed_nv)
            @printf("  nstate=%3d:  p_anti=%.4f  min_fp=%.2f  n_valid=%.2f\n",
                    nstate, mean(seed_pa), mean(seed_mf), mean(seed_nv))
            flush(stdout)
        end
    end

    # Save CSV
    outdir  = joinpath(@__DIR__, "results")
    mkpath(outdir)
    csvfile = joinpath(outdir, "19_panti_vs_nstate.csv")
    open(csvfile, "w") do io
        println(io, "H,nstate,p_anti,min_fp,n_valid")
        for (hi, H) in enumerate(H_vals), (ni, ns) in enumerate(nstate_vals)
            @printf(io, "%.2f,%d,%.6f,%.4f,%.4f\n",
                    H, ns, all_p_anti[hi, ni], all_min_fp[hi, ni], all_n_valid[hi, ni])
        end
    end
    println("Saved CSV to $csvfile")

    # Plot
    H_colors = [:royalblue, :deepskyblue, :seagreen, :darkorange, :firebrick]
    H_labels = ["H=0.00", "H=0.25", "H=0.50", "H=0.75", "H=1.00"]

    p_minfp = plot(title="min_fp vs nstate  (nqubit=6, n_seeds=$(n_seeds))",
                   xlabel="nstate", ylabel="mean min_fp",
                   legend=:topleft, size=(600, 420), margin=5Plots.mm)
    p_panti = plot(title="p_anti vs nstate  (nqubit=6, n_seeds=$(n_seeds))",
                   xlabel="nstate", ylabel="p_anti = min_fp / n_valid",
                   legend=:topleft, size=(600, 420), margin=5Plots.mm)

    for (hi, H) in enumerate(H_vals)
        plot!(p_minfp, nstate_vals, all_min_fp[hi, :],
              label=H_labels[hi], color=H_colors[hi], lw=2, marker=:circle, ms=5)
        plot!(p_panti, nstate_vals, all_p_anti[hi, :],
              label=H_labels[hi], color=H_colors[hi], lw=2, marker=:circle, ms=5)
    end

    minfp_file = joinpath(outdir, "19_min_fp_vs_nstate.png")
    panti_file = joinpath(outdir, "19_panti_vs_nstate.png")
    savefig(p_minfp, minfp_file)
    savefig(p_panti, panti_file)
    println("Saved plots to $minfp_file and $panti_file")
end

main()
