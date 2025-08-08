function base_optimization(make_goal_fxn, args)
    nqubit = args["nqubit"]

    Random.seed!(args["seed"])
    # Parse the args into things we need
    goal = make_goal_fxn(args)
    states = make_initial_state(args)
    nreplace = determine_nreplace(args, length(states))
    itr = determine_itr(args)
    outfile = setup_outfile(args)
    fingerprint = QCScaling.Fingerprint(nqubit)
    # Make the canonical base contexts. I think these should be packaged
    cxt_master = QCScaling.ContextMaster(args["nqubit"])

    state_tracker = zeros(UInt8, (length(itr), length(states), 2nqubit + 1))
    n_same_dict = Dict()
    accuracies = zeros(Float64, length(itr))

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
        accuracies[idx] = acc
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
                idx = 0
                while sum(new_scores) <= sum(scores)
                    new_states, new_scores = copy(states), copy(scores)
                    replace_idx = rand(1:length(states))
                    cxt = first(QCScaling.get_new_contexts(states, cxt_master, 1))
                    base_cxt = cxt==0 ? cxt_master.base_even : cxt_master.base_odd
                    alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                    new_state = QCScaling.PseudoGHZState(alphas..., first(cxt.pos))
                    new_states[replace_idx] = new_state
                    new_scores = QCScaling.score(new_states, goal, cxt_master)
                    idx += 1
                    if idx > 1000
                        break
                    end
                end
                states, scores = new_states, new_scores
            end
        end
    end

    output_states = make_output_states(state_tracker, accuracies, args)

    h5open(args["outfile"], "r+") do h5f
        groupname = determine_groupname(h5f; basegroupname=args["outgroup"])
        gp = create_group(h5f, groupname)
        for (k, v) in args
            attributes(gp)[k] = v
        end
        gp["states"] = output_states
        gp["goal"] = goal
        gp["accuracies"] = accuracies
    end
    println(maximum(accuracies))

end

