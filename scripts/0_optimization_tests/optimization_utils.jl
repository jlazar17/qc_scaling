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

function accuracy(rep, goal)
    pred = abs.(rep[1:2:end-2] .- rep[2:2:end-1])
    @assert length(pred)==length(goal)
    s = 0
    for (x, y) in zip(pred, goal)
        if x!=y
            continue
        end
        s +=1
    end
    return s / length(goal)
end
