# Script 01: Collect optimized solutions across seeds.
#
# For each (nqubit, H_target), runs SA with n_seeds=20 at the peak nstate
# (the nstate that maximizes eta_med in the scaling study).  The full
# ensemble (generator index, theta_s, theta_z, alphas) and rep (rep_sum,
# rep_ctr) are saved to HDF5 so that scripts 02-04 can analyse solution
# uniqueness without re-running SA.
#
# HDF5 layout:
#   nq{N}/H{h}/
#     peak_nstate  (attribute)
#     seed{s}/
#       gen_indices  Int[nstate]           generator.index for each state
#       theta_s_vec  Int[nstate]
#       theta_z_vec  Int[nstate]
#       alphas_mat   Int[nstate, nqubit-1]
#       rep_sum      Int[n]
#       rep_ctr      Int[n]
#       final_acc    Float64               scalar attribute

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "../spin_glass_study/sg_utils.jl"))

using HDF5
using Printf

# ---------------------------------------------------------------------------
# Step budget per nqubit (matches script 20 / spin_glass_study configs).
# ---------------------------------------------------------------------------
const NSTEPS = Dict(4 => 200_000, 6 => 500_000, 8 => 2_000_000)

# ---------------------------------------------------------------------------
# Load peak nstate per (nqubit, H) from the scaling study HDF5.
# Returns Dict{Int, Dict{Float64, Int}}: nqubit => H_target => peak_nstate
# ---------------------------------------------------------------------------
function load_peak_nstates(h5path)
    peak = Dict{Int, Dict{Float64, Int}}()
    h5open(h5path, "r") do h5f
        for nq_key in sort(collect(keys(h5f)))
            g      = h5f[nq_key]
            nqubit = read(HDF5.attributes(g)["nqubit"])
            peak[nqubit] = Dict{Float64, Int}()
            for hkey in keys(g)
                gp = g[hkey]
                haskey(gp, "nstates") || continue
                H_t     = read(HDF5.attributes(gp)["H_target"])
                nstates = read(gp["nstates"])
                eta_med = read(gp["eta_med"])
                peak[nqubit][H_t] = nstates[argmax(eta_med)]
            end
        end
    end
    return peak
end

# ---------------------------------------------------------------------------
# Serialise one seed's output into the HDF5 group.
# ---------------------------------------------------------------------------
function save_seed!(gp, ensemble, rep_sum, rep_ctr, final_acc)
    nstate  = length(ensemble)
    nqubit  = length(ensemble[1].generator)
    nalpha  = nqubit - 1

    gen_indices = [s.generator.index for s in ensemble]
    theta_s_vec = [s.theta_s         for s in ensemble]
    theta_z_vec = [s.theta_z         for s in ensemble]
    alphas_mat  = [s.alphas[j] for s in ensemble, j in 1:nalpha]

    gp["gen_indices"] = gen_indices
    gp["theta_s_vec"] = theta_s_vec
    gp["theta_z_vec"] = theta_z_vec
    gp["alphas_mat"]  = alphas_mat
    gp["rep_sum"]     = rep_sum
    gp["rep_ctr"]     = rep_ctr
    attributes(gp)["final_acc"] = final_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    scaling_h5 = joinpath(@__DIR__, "../scaling_study/data/scaling_study_adaptive.h5")
    outfile    = joinpath(@__DIR__, "data/01_solutions.h5")

    isfile(scaling_h5) || error("Scaling study HDF5 not found: $scaling_h5")

    peak_nstates = load_peak_nstates(scaling_h5)
    nqubits      = sort(collect(keys(peak_nstates)))
    H_vals       = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]
    n_seeds      = 20

    h5open(outfile, "cw") do h5f
        for nqubit in nqubits
            haskey(NSTEPS, nqubit) || begin
                @printf("nqubit=%d: no step budget defined, skipping\n", nqubit); continue
            end
            nsteps = NSTEPS[nqubit]
            n      = 3^nqubit
            ngbits = (n - 1) ÷ 2
            alpha  = exp(log(1e-4) / nsteps)

            companion, goal_idx, _ = build_shuffled_pairing(nqubit)
            fingerprint  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
            cxt_master   = QCScaling.ContextMaster(nqubit)

            nq_key = "nq$(nqubit)"
            nq_grp = haskey(h5f, nq_key) ? h5f[nq_key] : create_group(h5f, nq_key)
            attributes(nq_grp)["nqubit"] = nqubit

            println("\n" * "="^60)
            @printf("nqubit=%d  nsteps=%d\n", nqubit, nsteps)

            for H in H_vals
                haskey(peak_nstates[nqubit], H) || begin
                    @printf("  H=%.3f: no scaling data, skipping\n", H); continue
                end
                nstate = peak_nstates[nqubit][H]
                k_ones = h_to_kones(H, ngbits)

                H_key = @sprintf("H%.3f", H)
                H_grp = haskey(nq_grp, H_key) ? nq_grp[H_key] : create_group(nq_grp, H_key)
                attributes(H_grp)["H_target"]   = H
                attributes(H_grp)["peak_nstate"] = nstate

                @printf("  H=%.3f  peak_nstate=%d\n", H, nstate)

                for s in 1:n_seeds
                    seed_key = "seed$(s)"
                    if haskey(H_grp, seed_key)
                        @printf("    seed=%2d  [resumed]\n", s)
                        continue
                    end

                    rng_goal = Random.MersenneTwister(s * 137 + round(Int, H * 1000) + nqubit * 10_000)
                    goal = Random.shuffle!(rng_goal, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

                    sa_seed = s * 31 + 7 + nqubit * 1_000
                    final_acc, ensemble, rep_sum, rep_ctr, _ =
                        run_sa_full(goal, nqubit, nstate, nsteps, alpha,
                                    companion, goal_idx, fingerprint, cxt_master;
                                    seed=sa_seed)

                    seed_grp = create_group(H_grp, seed_key)
                    attributes(seed_grp)["goal"] = goal
                    save_seed!(seed_grp, ensemble, rep_sum, rep_ctr, final_acc)

                    @printf("    seed=%2d  acc=%.4f\n", s, final_acc)
                    flush(stdout)
                end
            end
        end
    end
    println("\nDone. Saved to $outfile")
end

main()
