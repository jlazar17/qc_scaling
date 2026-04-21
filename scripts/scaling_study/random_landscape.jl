using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using HDF5
using Printf

# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N
    return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0
    H >= 1.0 && return N ÷ 2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), 0:N÷2)
    return (0:N÷2)[idx]
end

# ---------------------------------------------------------------------------
# Rep helpers
# ---------------------------------------------------------------------------

function apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

function rep_accuracy_fast(rep_sum, rep_ctr, goal)
    s = 0
    @inbounds for i in eachindex(goal)
        i1 = 2i - 1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1] / c1; p2 = rep_sum[i2] / c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    configs = [
        (4, 2),
        (6, 3),
        (8, 3),
    ]
    entropy_vals  = [0.0, 0.5, 1.0]
    n_random      = 10_000   # random ensembles per (nqubit, H) combination
    base_seed     = 42
    outdir        = joinpath(@__DIR__, "data")
    mkpath(outdir)

    # Goal_sa best accuracies from full_comparison for reference
    goal_sa_best = Dict(
        (4, 0.0) => 1.000, (4, 0.5) => 1.000, (4, 1.0) => 1.000,
        (6, 0.0) => 0.997, (6, 0.5) => 0.945, (6, 1.0) => 0.896,
        (8, 0.0) => 0.903, (8, 0.5) => 0.790, (8, 1.0) => 0.625,
    )

    outfile = joinpath(outdir, "random_landscape.h5")
    h5open(outfile, "w") do h5f
    HDF5.attributes(h5f)["n_random"]  = n_random
    HDF5.attributes(h5f)["base_seed"] = base_seed

    for (nqubit, nstate_mult) in configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        nstate      = nstate_mult * base_nstate
        ngbits      = (3^nqubit - 1) ÷ 2
        n           = 3^nqubit

        println("="^60)
        @printf("nqubit=%d  nstate=%d (%d×)\n", nqubit, nstate, nstate_mult)
        @printf("%-8s  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s\n",
                "H", "k", "mean", "std", "p50", "p90", "goal_sa")
        println("-"^60)

        gp_nq = create_group(h5f, "nqubit_$(nqubit)")
        HDF5.attributes(gp_nq)["nqubit"] = nqubit
        HDF5.attributes(gp_nq)["nstate"] = nstate

        cxt_master = QCScaling.ContextMaster(nqubit)

        for H_target in entropy_vals
            k        = k_from_entropy(H_target, ngbits)
            H_actual = hamming_entropy(k, ngbits)
            rng      = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goal     = goal_from_hamming(k, ngbits, rng)
            best_ref = get(goal_sa_best, (nqubit, H_target), NaN)

            accs = Vector{Float64}(undef, n_random)
            for t in 1:n_random
                ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
                rep_sum  = zeros(Float64, n)
                rep_ctr  = zeros(Int,     n)
                for s in ensemble
                    apply_state!(rep_sum, rep_ctr, s, cxt_master, 1)
                end
                accs[t] = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            end

            @printf("%-8.2f  %-6d  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f\n",
                    H_target, k,
                    mean(accs), std(accs),
                    quantile(accs, 0.50), quantile(accs, 0.90),
                    best_ref)
            flush(stdout)

            key = @sprintf("H%.2f_k%d", H_target, k)
            gp  = create_group(gp_nq, key)
            HDF5.attributes(gp)["H_target"] = H_target
            HDF5.attributes(gp)["H_actual"] = H_actual
            HDF5.attributes(gp)["k"]        = k
            HDF5.attributes(gp)["goal_sa_best"] = best_ref
            gp["accs"] = accs
        end
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
