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

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--outfile"
            help = "Where to store the output"
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
            help = "Number of states to replace each iteration (0 = 10% of nstate)"
            arg_type = Int
            default = 1
        "--niter"
            help = "Number of iterations"
            arg_type = Int
            default = 10_000
        "--pzero"
            help = "Probability that zero is picked when sampling the goal"
            arg_type = Float64
            required = true
        "--goalfile"
            help = "File containing the desired binary string"
            arg_type = String
            default = ""
        "--statefile"
            help = "File containing previously optimized states"
            arg_type = String
            default = ""
        "--savelevel"
            help = "Amount of information to store: all | best_states | best_state | deltas"
            arg_type = String
            default = "best_states"
        "--nstate"
            help = "Number of states to use (0 = auto)"
            arg_type = Int
            default = 0
        "--n_same_tol"
            help = "Consecutive iterations without improvement before escape triggers"
            arg_type = Int
            default = 10
        "--p_mutate"
            help = "Probability of mutating the generator during a normal replacement"
            arg_type = Float64
            default = 0.3
        "--track"
            help = "Show a progress bar"
            action = :store_true
        "--outgroup"
            help = "HDF5 group name for results"
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
    improved_optimization(make_goal, args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
