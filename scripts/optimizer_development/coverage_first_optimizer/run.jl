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
include("coverage_optimizer.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--outfile"
            arg_type = String
            required = true
        "--nqubit"
            arg_type = Int
            required = true
        "--seed"
            arg_type = Int
            required = true
        "--nreplace"
            arg_type = Int
            default = 1
        "--niter"
            arg_type = Int
            default = 10_000
        "--pzero"
            arg_type = Float64
            required = true
        "--goalfile"
            arg_type = String
            default = ""
        "--statefile"
            arg_type = String
            default = ""
        "--savelevel"
            arg_type = String
            default = "best_states"
        "--nstate"
            arg_type = Int
            default = 0
        "--n_same_tol"
            arg_type = Int
            default = 10
        "--p_mutate"
            arg_type = Float64
            default = 0.3
        "--track"
            action = :store_true
        "--outgroup"
            default = "results"
    end
    return parse_args(s)
end

function make_goal(args)
    nqubit = args["nqubit"]
    p = args["pzero"]
    goal = sample(0:1, Weights([p, 1 - p]), (3^nqubit - 1) ÷ 2)
    return goal
end

function main(args=nothing)
    if isnothing(args)
        args = parse_commandline()
    end
    coverage_optimization(make_goal, args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
