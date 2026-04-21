using Pkg
Pkg.develop(path="../../QCScaling")
using QCScaling

Pkg.activate(".")
using StatsBase
using ProgressBars
using Random
using JLD2
using ArgParse

include("utils.jl")

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
            default = 0
        "--niter"
            help = "Number of iterations to go for"
            arg_type = Int
            default = 10_000
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
        "--fast"
            help = "Use fast mode"
            action = :store_true
        "--track"
            help = "Track progress with a `ProgressBar`"
            action = :store_true
        "--outgroup"
            help = "Name for the group to have in outfile. Default results"
            default = "results"

    end
    return parse_args(s)
end

function update_states(states, rep, nreplace, base_even, base_odd, nqubit, goal, fingerprint)
    new_gens = QCScaling.get_new_generators(states, rep, base_even, base_odd, nreplace)
    for (idx, gen) in enumerate(new_gens)
        paritybit = rand(0:1)
        base = paritybit==0 ? base_even : base_odd
        cxt = QCScaling.Context(gen, base)
        theta_s, theta_z, alphas = QCScaling.pick_new_alphas(
            cxt,
            goal,
            rep,
            fingerprint,
            base
        )
        states[idx] = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, gen)
        #states[idx] = QCScaling.PseudoGHZState(rand(0:1), rand(0:1), rand(0:1, 7), gen)
    end
    return states
end

function make_goal(args)
    if length(args["goalfile"])==0
        goal = rand(0:1, Int((3^args["nqubit"] - 1) / 2))
    else
        # TODO implement this
        throw("Loading goal not implemented yet.")
    end
    return goal
end

function make_initial_state(args)
    @assert args["nstate"]==0 || length(args["statefile"])==0
    if length(args["statefile"])==0
        # Make random states if none provided
        nstate = args["nstate"]
        if nstate==0
            nstate = 3 * Int(ceil(3^args["nqubit"] / 2^(args["nqubit"]-1)))
        end
        states = [QCScaling.random_state(args["nqubit"]) for _ in 1:nstate]
    else
        # Load in the states from a JLD2 file
        # TODO implement this
        throw("Loading states not implemented yet")
    end
end

function determine_nreplace(args, nstates)
    nreplace = args["nreplace"]
    if nreplace==0
        nreplace = Int(ceil(0.1 * nstates))
    end
    return nreplace
end

function determine_itr(args)
    itr = 1:args["niter"]
    if args["track"]
        itr = ProgressBar(itr)
    end
    return itr
end

function setup_outfile(args)
    if ~isfile(args["outfile"])
        jldopen(args["outfile"], "w") do _
        end
    end
    return args["outfile"]
end

function main(args=nothing)
    if typeof(args)==Nothing
        args = parse_commandline()
    end

    nqubit = args["nqubit"]

    Random.seed!(args["seed"])
    # Parse the args into things we need
    goal = make_goal(args)
    states = make_initial_state(args)
    nreplace = determine_nreplace(args, length(states))
    itr = determine_itr(args)
    outfile = setup_outfile(args)
    fingerprint = QCScaling.Fingerprint(nqubit)
    # Make the canonical base contexts. I think these should be packaged
    base_even = QCScaling.generate_base_context(nqubit, 0)
    base_odd = QCScaling.generate_base_context(nqubit, 1)

    # Score the first states
    rep = QCScaling.calculate_representation(states, base_even, base_odd)
    scores = QCScaling.score(states, rep, goal, base_even, base_odd)
    
    # Sort the states and scores by ascending scores
    sorter = sortperm(scores)
    scores = scores[sorter]
    states = states[sorter]
    
    # Preallocate tracking arrays
    score_tracker = zeros(Int, (args["niter"], length(states)))
    accuracy_tracker = zeros(Float64, args["niter"])

    for idx in itr
        states = update_states(
            states, 
            rep,
            nreplace,
            base_even,
            base_odd,
            nqubit,
            goal,
            fingerprint
        )
        # Update the rep
        rep = QCScaling.calculate_representation(states, base_even, base_odd)
        scores = QCScaling.score(states, rep, goal, base_even, base_odd)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]

        score_tracker[idx, :] = scores
        accuracy_tracker[idx] = accuracy(rep, goal)

    end
    jldopen(args["outfile"], "r+") do jldf
        groupname = determine_groupname(jldf; basegroupname=args["outgroup"])
        gp = JLD2.Group(jldf, groupname)
        gp["args"] = args
        gp["states"] = states
        gp["scores"] = score_tracker
        gp["accuracy"] = accuracy_tracker
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
