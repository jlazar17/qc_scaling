using Pkg
Pkg.activate("..")

Pkg.develop(path="../../../QCScaling")
using QCScaling

using StatsBase
using ProgressBars
using Random
using HDF5
using ArgParse

include("../io_utils.jl")
include("../optimization_utils.jl")
include("../base_optimizer.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--outfile"
            help = "where to store the output"
            arg_type = String
            required = true
        "--nqubit"
            help = "Number of qubits"
            arg_type = Int
            required = true
        "--seed"
            help = "Seed for RNG"
            arg_type = Int
            required = true
        "--nreplace"
            help = "Number of states to replace each time"
            arg_type = Int
            default = 1
        "--niter"
            help = "Number of iterations to go for"
            arg_type = Int
            default = 10_000
        "--pzero"
            help = "Probability that zero is picked"
            arg_type = Float64
            required = true
        "--goalfile"
            help = "File in which the desired binary string can be found"
            arg_type = String
            default = ""
        "--statefile"
            help = "File in which previously optimized states are stored"
            arg_type = String
            default = ""
        "--nstate"
            help = "Number of states to use"
            arg_type = Int
            default = 0
        "--n_same_tol"
            help = "Number of times we can get the same accuracy before switching"
            arg_type = Int
            default = 10
        "--track"
            help = "Track progress with a `ProgressBar`"
            action = :store_true
        "--outgroup"
            help = "Name for the group to have in outfile. Default results"
            default = "results"

    end
    return parse_args(s)
end

function make_goal(args)
    nqubit = args["nqubit"]
    p = args["pzero"]
    goal = sample(0:1, Weights([p, 1-p]), (3^nqubit - 1) ÷ 2)
    return goal
end

function main(args=nothing)
    if isnothing(args)
        args = parse_commandline()
    end
    base_optimization(make_goal, args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
