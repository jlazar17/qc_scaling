using Pkg
Pkg.activate(".")

Pkg.develop(path="../../QCScaling")
using QCScaling

using StatsBase
using ProgressBars
using Random
#using JLD2
using HDF5
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
            default = 1
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
        "--n_same_tol"
            help = "Number of times we can get the same accuracy before switching"
            arg_type = Int
            default = 10
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

function pick_replacement_states(
    states::Vector{QCScaling.PseudoGHZState},
    cxt_master::QCScaling.ContextMaster,
    nnew::Int
)
    counter = zeros(Int, 3^cxt_master.nqubit)
    ## Count how many times a po is covered
    for state in states
        # This should be a function
        base_cxt = ifelse(state.theta_s==0, cxt_master.base_even, cxt_master.base_odd)
        idxs =  map(x->x.index, QCScaling.Context(state.generator, base_cxt))
        counter[idxs] .+= 1
    end
    overlap = zeros(Int, size(states))
    for (idx, state) in enumerate(states)
        base_cxt = ifelse(state.theta_s==0, cxt_master.base_even, cxt_master.base_odd)
        cxt = QCScaling.Context(state.generator, base_cxt)
        idxs = map(x->x.index, cxt.pos)
        overlap[idx] = sum(counter[idxs])
    end
    weights = Weights(overlap)
    chosen_idxs = sample(1:length(states), weights, nnew, replace=false)
    return chosen_idxs
end

function make_goal(args)
    if length(args["goalfile"])==0
        goal = rand(0:1, Int((3^args["nqubit"] - 1) / 2))
    else
        # TODO implement this
        fname, key = split(args["goalfile"], ":")
        goal = h5open(fname) do jldf
            binaryify(jldf[key], args["nqubit"])
        end
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
        h5open(args["outfile"], "w") do _
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
    #base_even = QCScaling.generate_base_context(nqubit, 0)
    #base_odd = QCScaling.generate_base_context(nqubit, 1)
    cxt_master = QCScaling.ContextMaster(args["nqubit"])

    state_tracker = zeros(UInt8, (length(itr), length(states), 2nqubit + 1))
    n_same_dict = Dict()

    for idx in itr
        # Sort the states and scores by ascending score
        scores = QCScaling.score(states, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]

        for (jdx, state) in enumerate(states)
            state_tracker[idx, jdx, 1] = state.theta_s
            state_tracker[idx, jdx, 2] = state.theta_z
            state_tracker[idx, jdx, 3:1+nqubit] = state.alphas
            state_tracker[idx, jdx, end-nqubit+1:end] = state.generator.βs
        end

        # Calculate the representation
        rep = QCScaling.calculate_representation(states)
        acc = accuracy(rep, goal)
        if mod(idx, 200)==1
            display(acc)
        end
        hash_states = hash(states)
        if ~(hash_states in keys(n_same_dict))
            n_same_dict[hash_states] = 0
        end

        n_same_dict[hash_states] += 1

        if rand() > 1e-3
        #if n_same_dict[hash_states] <= args["n_same_tol"]
            ws = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace, replace=false)
            for which in whiches
                rplc_state = states[which]
                base_cxt = rplc_state.theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
                cxt = QCScaling.Context(rplc_state.generator, base_cxt)
                alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                new_state = QCScaling.PseudoGHZState(alphas..., rplc_state.generator)
                states[which] = new_state
            end
        else
            n_same_dict = Dict()
            #proposed_states, proposed_scores = copy(states), copy(scores)
            #replace_idxs = rand(1:length(states), length(new_cxts))
            #replace_idxs = pick_replacement_states(states, cxt_master, 1)
            for _ in nreplace
                new_states, new_scores = copy(states), copy(scores)
                while sum(new_scores) <= sum(scores)
                    new_states, new_scores = copy(states), copy(scores)
                    replace_idx = rand(1:length(states))
                    @show sum(new_scores), sum(scores)
                    cxt = first(QCScaling.get_new_contexts(states, cxt_master, 1))
                    base_cxt = cxt==0 ? cxt_master.base_even : cxt_master.base_odd
                    alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                    new_state = QCScaling.PseudoGHZState(alphas..., first(cxt.pos))
                    new_states[replace_idx] = new_state
                    new_scores = QCScaling.score(new_states, goal, cxt_master)
                end
                states, scores = new_states, new_scores
            end
        end


    end
    h5open(args["outfile"], "r+") do h5f
        groupname = determine_groupname(h5f; basegroupname=args["outgroup"])
        gp = create_group(h5f, groupname)
        for (k, v) in args
            attributes(gp)[k] = v
        end
        #gp["args"] = args
        gp["states"] = state_tracker
        gp["goal"] = goal
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
