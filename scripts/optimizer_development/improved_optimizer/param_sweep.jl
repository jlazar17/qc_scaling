using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using StatsBase
using Random
using HDF5
using ArgParse

include("../../utils/io_utils.jl")
include("../../utils/optimization_utils.jl")
include("improved_optimizer.jl")

function make_goal(args)
    nqubit = args["nqubit"]
    p = args["pzero"]
    goal = sample(0:1, Weights([p, 1 - p]), (3^nqubit - 1) ÷ 2)
    return goal
end

function generate_seeds(base_seed, nseeds)
    rng = Random.MersenneTwister(base_seed)
    seeds = rand(rng, UInt32, nseeds)
    while length(unique(seeds)) < nseeds
        seeds = rand(rng, UInt32, nseeds)
    end
    return Int.(seeds)
end

function existing_groups(path)
    !isfile(path) && return Set{String}()
    h5open(path, "r") do h5f
        Set{String}(keys(h5f))
    end
end

function run_sweep(outfile, fixed_args, sweep_key, sweep_values, seeds, pzero_values, multipliers, nqubit)
    base_nstate_exact = 3^nqubit / 2^(nqubit - 1)
    for val in sweep_values
        for mult in multipliers
            nstate = Int(ceil(mult * base_nstate_exact))
            for pzero in pzero_values
                for seed in seeds
                    group = "$(sweep_key)_$(val)_nstate_$(nstate)_pzero_$(pzero)_seed_$(seed)"
                    done  = existing_groups(outfile)
                    if group in done
                        println("skip: $group")
                        continue
                    end
                    println("run: $group")
                    args = merge(fixed_args, Dict(
                        sweep_key  => val,
                        "nstate"   => nstate,
                        "pzero"    => pzero,
                        "seed"     => seed,
                        "outfile"  => outfile,
                        "outgroup" => group,
                    ))
                    improved_optimization(make_goal, args)
                end
            end
        end
    end
end

function main()
    nqubit       = 8
    nseeds       = 20
    niter        = 5_000
    base_seed    = 12345
    pzero_values = [0.1, 0.5, 0.9]
    multipliers  = [2, 4, 6]
    outdir       = joinpath(@__DIR__, "data/param_sweep")
    mkpath(outdir)

    seeds = generate_seeds(base_seed, nseeds)

    fixed = Dict(
        "nqubit"     => nqubit,
        "niter"      => niter,
        "n_same_tol" => 10,
        "nreplace"   => 1,
        "p_mutate"   => 0.3,
        "savelevel"  => "best_states",
        "goalfile"   => "",
        "statefile"  => "",
        "track"      => false,
    )

    # --- Sweep 1: nreplace ---
    println("\n=== Sweeping nreplace ===")
    run_sweep(
        joinpath(outdir, "nreplace_sweep_nqubit_$(nqubit).h5"),
        fixed,
        "nreplace", [1, 2, 5, 10],
        seeds, pzero_values, multipliers, nqubit,
    )

    # --- Sweep 2: p_mutate ---
    println("\n=== Sweeping p_mutate ===")
    run_sweep(
        joinpath(outdir, "p_mutate_sweep_nqubit_$(nqubit).h5"),
        fixed,
        "p_mutate", [0.1, 0.3, 0.5, 0.7, 1.0],
        seeds, pzero_values, multipliers, nqubit,
    )

    println("\nAll sweeps complete. Results in $outdir")
end

main()
