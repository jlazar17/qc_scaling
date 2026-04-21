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
include("../../utils/base_optimizer.jl")
include("improved_optimizer.jl")

# Shared goal sampler — both optimizers use the same one so goals are
# identical for a given (seed, pzero, nqubit) triple.
function make_goal(args)
    nqubit = args["nqubit"]
    p = args["pzero"]
    goal = sample(0:1, Weights([p, 1 - p]), (3^nqubit - 1) ÷ 2)
    return goal
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--outdir"
            help = "Directory to write results into"
            arg_type = String
            required = true
        "--nqubit"
            help = "Number of qubits"
            arg_type = Int
            default = 8
        "--niter"
            help = "Iterations per run"
            arg_type = Int
            default = 5_000
        "--nseeds"
            help = "Number of random seeds to test"
            arg_type = Int
            default = 20
        "--base-seed"
            help = "Master seed used to generate per-run seeds"
            arg_type = Int
            default = 12345
        "--pzero-values"
            help = "Comma-separated pzero values to sweep"
            arg_type = String
            default = "0.1,0.3,0.5,0.7,0.9"
        "--nstate-multipliers"
            help = "Comma-separated multipliers of the base nstate"
            arg_type = String
            default = "1,2,3"
        "--nreplace"
            help = "States replaced per iteration (0 = 10% of nstate)"
            arg_type = Int
            default = 1
        "--n-same-tol"
            help = "Plateau tolerance for the improved optimizer escape"
            arg_type = Int
            default = 10
        "--p-mutate"
            help = "Generator mutation probability (improved optimizer only)"
            arg_type = Float64
            default = 0.3
        "--savelevel"
            help = "Output detail level: all | best_states | best_state | deltas"
            arg_type = String
            default = "best_states"
    end
    return parse_args(s)
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

function run_comparison()
    args = parse_commandline()
    nqubit      = args["nqubit"]
    niter       = args["niter"]
    nreplace    = args["nreplace"]
    savelevel   = args["savelevel"]
    pzero_values = parse.(Float64, split(args["pzero-values"], ","))
    multipliers  = parse.(Int, split(args["nstate-multipliers"], ","))
    seeds        = generate_seeds(args["base-seed"], args["nseeds"])

    mkpath(args["outdir"])
    base_outfile     = joinpath(args["outdir"], "base_nqubit_$(nqubit).h5")
    improved_outfile = joinpath(args["outdir"], "improved_nqubit_$(nqubit).h5")

    base_nstate_exact = 3^nqubit / 2^(nqubit - 1)

    for mult in multipliers
        nstate = Int(ceil(mult * base_nstate_exact))
        for pzero in pzero_values
            for seed in seeds
                group = "nstate_$(nstate)_pzero_$(pzero)_seed_$(seed)"

                # Shared fields for both optimizers
                common = Dict(
                    "nqubit"      => nqubit,
                    "niter"       => niter,
                    "nstate"      => nstate,
                    "pzero"       => pzero,
                    "seed"        => seed,
                    "nreplace"    => nreplace,
                    "n_same_tol"  => args["n-same-tol"],
                    "savelevel"   => savelevel,
                    "goalfile"    => "",
                    "statefile"   => "",
                    "track"       => false,
                    "outgroup"    => group,
                )

                # --- Base optimizer ---
                base_done = existing_groups(base_outfile)
                if group in base_done
                    println("base    skip: $group")
                else
                    println("base    run:  nstate=$nstate pzero=$pzero seed=$seed")
                    base_args = merge(common, Dict("outfile" => base_outfile))
                    base_optimization(make_goal, base_args)
                end

                # --- Improved optimizer ---
                improved_done = existing_groups(improved_outfile)
                if group in improved_done
                    println("improved skip: $group")
                else
                    println("improved run:  nstate=$nstate pzero=$pzero seed=$seed")
                    improved_args = merge(common, Dict(
                        "outfile"   => improved_outfile,
                        "p_mutate"  => args["p-mutate"],
                    ))
                    improved_optimization(make_goal, improved_args)
                end
            end
        end
    end

    println("\nDone. Results written to:")
    println("  base:     $base_outfile")
    println("  improved: $improved_outfile")
    println("Groups are identically named so h5 keys match 1:1 for analysis.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
