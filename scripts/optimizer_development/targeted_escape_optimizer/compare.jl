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
include("../1_improved_optimizer/improved_optimizer.jl")
include("targeted_escape_optimizer.jl")

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
            arg_type = String
            required = true
        "--nqubit"
            arg_type = Int
            default = 8
        "--niter"
            arg_type = Int
            default = 5_000
        "--nseeds"
            arg_type = Int
            default = 20
        "--base-seed"
            arg_type = Int
            default = 12345
        "--pzero-values"
            arg_type = String
            default = "0.1,0.5,0.9"
        "--nstate-multipliers"
            arg_type = String
            default = "1,2,3,4,5,6"
        "--nreplace"
            arg_type = Int
            default = 1
        "--n-same-tol"
            arg_type = Int
            default = 10
        "--p-mutate"
            arg_type = Float64
            default = 0.3
        "--savelevel"
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
    args         = parse_commandline()
    nqubit       = args["nqubit"]
    niter        = args["niter"]
    savelevel    = args["savelevel"]
    pzero_values = parse.(Float64, split(args["pzero-values"], ","))
    multipliers  = parse.(Int,     split(args["nstate-multipliers"], ","))
    seeds        = generate_seeds(args["base-seed"], args["nseeds"])

    mkpath(args["outdir"])
    outfiles = Dict(
        "improved"         => joinpath(args["outdir"], "improved_nqubit_$(nqubit).h5"),
        "targeted_escape"  => joinpath(args["outdir"], "targeted_escape_nqubit_$(nqubit).h5"),
    )
    optimizers = Dict(
        "improved"        => improved_optimization,
        "targeted_escape" => targeted_escape_optimization,
    )

    base_nstate_exact = 3^nqubit / 2^(nqubit - 1)

    for mult in multipliers
        nstate = Int(ceil(mult * base_nstate_exact))
        for pzero in pzero_values
            for seed in seeds
                group = "nstate_$(nstate)_pzero_$(pzero)_seed_$(seed)"
                common = Dict(
                    "nqubit"     => nqubit,
                    "niter"      => niter,
                    "nstate"     => nstate,
                    "pzero"      => pzero,
                    "seed"       => seed,
                    "nreplace"   => args["nreplace"],
                    "n_same_tol" => args["n-same-tol"],
                    "savelevel"  => savelevel,
                    "goalfile"   => "",
                    "statefile"  => "",
                    "track"      => false,
                    "outgroup"   => group,
                    "p_mutate"   => args["p-mutate"],
                )
                for (name, optfn) in optimizers
                    done = existing_groups(outfiles[name])
                    if group in done
                        println("$name skip: $group")
                        continue
                    end
                    println("$name run:  nstate=$nstate pzero=$pzero seed=$seed")
                    optfn(make_goal, merge(common, Dict("outfile" => outfiles[name])))
                end
            end
        end
    end

    println("\nDone. Results in $(args["outdir"]):")
    for (name, path) in outfiles
        println("  $name: $path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
