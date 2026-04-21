# Script 20: p_anti vs nstate for nqubit=4, 6, 8.
#
# Measures p_anti at the same nstate grid as the scaling study for each nqubit,
# using the same colors as plot_efficiency_vs_H.jl.
# Results are checkpointed to CSV as they are computed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "../spin_glass_study/sg_utils.jl"))

using Plots
using DelimitedFiles

# ---------------------------------------------------------------------------
# measure_p_anti (same as scripts 17/19)
# ---------------------------------------------------------------------------
function measure_p_anti(goal, rep, fingerprint, cxt_master, companion, goal_idx,
                        nqubit; n_probe=150, seed=42)
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
        pi    = bc.parity + 1
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
    return mean(min_fps ./ n_valids), mean(n_valids), mean(min_fps)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    csvfile = joinpath(outdir, "20_panti_scaling.csv")

    # Config: (nqubit, nsteps, n_seeds, nstate_vals)
    # nstate_vals chosen to match the scaling study grid for each nqubit.
    configs = [
        (nqubit=4, nsteps=200_000, n_seeds=3,
         nstate_vals=[11, 14, 17, 22, 33, 56, 97, 168, 220]),
        (nqubit=6, nsteps=500_000, n_seeds=3,
         nstate_vals=[23, 35, 46, 68, 118, 203, 350, 460]),
        (nqubit=8, nsteps=2_000_000, n_seeds=2,
         nstate_vals=[52, 94, 228, 742]),
    ]
    H_vals = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]

    # Load any already-computed rows from CSV.
    done = Set{Tuple{Int,Float64,Int}}()   # (nqubit, H, nstate)
    if isfile(csvfile)
        existing, hdr = readdlm(csvfile, ',', header=true)
        for row in eachrow(existing)
            push!(done, (Int(row[1]), Float64(row[2]), Int(row[3])))
        end
        println("Resuming: $(length(done)) rows already computed.")
    else
        open(csvfile, "w") do io
            println(io, "nqubit,H,nstate,p_anti,min_fp,n_valid")
        end
    end

    for cfg in configs
        nqubit    = cfg.nqubit
        nsteps    = cfg.nsteps
        n_seeds   = cfg.n_seeds
        nstate_vals = cfg.nstate_vals

        n      = 3^nqubit
        ngbits = (n - 1) ÷ 2
        alpha  = exp(log(1e-4) / nsteps)

        companion, goal_idx, _ = build_shuffled_pairing(nqubit)
        fingerprint  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
        cxt_master   = QCScaling.ContextMaster(nqubit)

        println("\n" * "="^60)
        @printf("nqubit=%d  nsteps=%d  n_seeds=%d\n", nqubit, nsteps, n_seeds)

        for H in H_vals
            k_ones = h_to_kones(H, ngbits)
            println("  H=$(H):")
            for nstate in nstate_vals
                (nqubit, H, nstate) in done && begin
                    @printf("    nstate=%4d  [resumed]\n", nstate); continue
                end

                seed_pa = Float64[]; seed_mf = Float64[]; seed_nv = Float64[]
                for gseed in 1:n_seeds
                    rng_goal = Random.MersenneTwister(gseed * 137 + round(Int, H * 1000) + nqubit * 10_000)
                    goal = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones),
                                                          zeros(Int, ngbits - k_ones)))
                    _, _, _, _, rep =
                        run_sa_full(goal, nqubit, nstate, nsteps, alpha,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=gseed * 31 + 7 + nqubit * 1000)
                    pa, nv, mf = measure_p_anti(goal, rep, fingerprint, cxt_master,
                                                 companion, goal_idx, nqubit;
                                                 n_probe=150, seed=gseed * 997)
                    push!(seed_pa, pa); push!(seed_mf, mf); push!(seed_nv, nv)
                end
                pa_mean = mean(seed_pa); mf_mean = mean(seed_mf); nv_mean = mean(seed_nv)
                @printf("    nstate=%4d  p_anti=%.4f  min_fp=%.2f  n_valid=%.2f\n",
                        nstate, pa_mean, mf_mean, nv_mean)
                flush(stdout)

                # Checkpoint
                open(csvfile, "a") do io
                    @printf(io, "%d,%.3f,%d,%.6f,%.4f,%.4f\n",
                            nqubit, H, nstate, pa_mean, mf_mean, nv_mean)
                end
                push!(done, (nqubit, H, nstate))
            end
        end
    end

    println("\nAll measurements done. Plotting...")

    # ---------------------------------------------------------------------------
    # Load CSV and plot
    # ---------------------------------------------------------------------------
    data, hdr = readdlm(csvfile, ',', header=true)
    hdr = vec(hdr)
    col(name) = findfirst(==(name), hdr)

    nqubits_all = sort(unique(Int.(data[:, col("nqubit")])))
    H_unique    = sort(unique(Float64.(data[:, col("H")])))

    # Colors matching plot_efficiency_vs_H.jl plasma colormap (sampled at 7 points)
    # plasma: dark purple → blue → cyan → green → yellow → orange → red
    H_colors_hex = [
        "#0d0887",  # H=0.000  dark violet
        "#5302a3",  # H=0.250
        "#8b0aa5",  # H=0.500
        "#b83289",  # H=0.625
        "#db5c68",  # H=0.750
        "#f48849",  # H=0.875
        "#f0f921",  # H=1.000  yellow
    ]
    H_to_color = Dict(H => H_colors_hex[i] for (i, H) in enumerate(H_unique))

    plts = []
    for (pi, nqubit) in enumerate(nqubits_all)
        rows = data[Int.(data[:, col("nqubit")]) .== nqubit, :]
        nstates_nq = sort(unique(Int.(rows[:, col("nstate")])))

        p = plot(title="nqubit = $nqubit",
                 xlabel="nstate",
                 ylabel=pi == 1 ? "p_anti" : "",
                 legend=pi == 1 ? :topright : false,
                 size=(420, 360), margin=5Plots.mm)

        for H in H_unique
            mask = (Float64.(rows[:, col("H")]) .== H)
            any(mask) || continue
            sub = rows[mask, :]
            ns  = Int.(sub[:, col("nstate")])
            pa  = Float64.(sub[:, col("p_anti")])
            idx = sortperm(ns)
            plot!(p, ns[idx], pa[idx];
                  label="H=$(H)", color=H_to_color[H], lw=2, marker=:circle, ms=5)
        end
        push!(plts, p)
    end

    fig = plot(plts..., layout=(1, length(plts)), size=(420 * length(plts), 380))
    outpng = joinpath(outdir, "20_panti_scaling.png")
    savefig(fig, outpng)
    println("Saved $outpng")
end

main()
