using ProgressMeter

# ---------------------------------------------------------------------------
# Incremental representation cache
#
# The representation is an aggregate over all states. Rather than
# recomputing it from scratch each iteration (O(nstate * context_size)),
# we maintain a running sum and count and update only the states that
# change each iteration.
# ---------------------------------------------------------------------------

function build_rep_cache(states, cxt_master)
    nqubit = cxt_master.nqubit
    rep_sum = zeros(Float64, 3^nqubit)
    rep_ctr = zeros(Int, 3^nqubit)
    for state in states
        _update_rep_cache!(rep_sum, rep_ctr, state, cxt_master, 1)
    end
    return rep_sum, rep_ctr
end

function _update_rep_cache!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

# Derive the representation from cached sums/counts.
# 0/0 (uncovered indices) naturally becomes NaN via Float64 division.
# Ambiguous entries (pref == 0.5) are also set to NaN.
function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

# Replace the state at index `which`, keeping the cache consistent.
function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    _update_rep_cache!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    _update_rep_cache!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

function improved_optimization(make_goal_fxn, args)
    nqubit = args["nqubit"]
    Random.seed!(args["seed"])

    goal       = make_goal_fxn(args)
    states     = make_initial_state(args)
    nreplace   = determine_nreplace(args, length(states))
    itr        = determine_itr(args)
    outfile    = setup_outfile(args)
    fingerprint = QCScaling.Fingerprint(nqubit)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n_same_tol = args["n_same_tol"]
    p_mutate   = args["p_mutate"]

    # Build the rep cache once; update it incrementally from here on.
    rep_sum, rep_ctr = build_rep_cache(states, cxt_master)
    rep = rep_from_cache(rep_sum, rep_ctr)

    state_tracker = zeros(UInt8, (length(itr), length(states), 2nqubit + 1))
    accuracies    = zeros(Float64, length(itr))
    sort_perms    = zeros(Int, (length(itr), length(states)))
    n_same        = 0
    last_acc      = -1.0

    @showprogress for idx in itr
        # Score using the cached rep — no redundant representation recompute.
        scores = QCScaling.score(states, rep, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]
        sort_perms[idx, :] = sorter

        for (jdx, state) in enumerate(states)
            state_tracker[idx, jdx, 1]             = state.theta_s
            state_tracker[idx, jdx, 2]             = state.theta_z
            state_tracker[idx, jdx, 3:1+nqubit]    = state.alphas
            state_tracker[idx, jdx, end-nqubit+1:end] = state.generator.βs
        end

        acc = accuracy(rep, goal)
        accuracies[idx] = acc

        # Deterministic plateau detection: count consecutive iterations
        # with no accuracy improvement instead of using a random trigger.
        if acc == last_acc
            n_same += 1
        else
            n_same  = 0
            last_acc = acc
        end

        if n_same <= n_same_tol
            # Normal replacement: weighted sample of low-scoring states.
            ws     = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace, replace=false)
            for which in whiches
                rplc_state = states[which]
                if rand() < p_mutate
                    # Generator mutation: pick a context that covers
                    # currently undefined regions of the operator space.
                    new_cxt      = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(new_cxt.pos)
                    base_cxt     = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt          = new_cxt
                else
                    # Alpha/theta-only update: keep the existing generator.
                    new_generator = rplc_state.generator
                    base_cxt     = rplc_state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt          = QCScaling.Context(new_generator, base_cxt)
                end
                alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                new_state = QCScaling.PseudoGHZState(alphas..., new_generator)
                replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
            end
        else
            # Escape: plateau detected — force generator mutation on all
            # replacements to break out of the local optimum.
            n_same = 0
            for _ in 1:nreplace
                replace_idx  = rand(1:length(states))
                new_cxt      = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                new_generator = first(new_cxt.pos)
                base_cxt     = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                alphas       = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
                new_state    = QCScaling.PseudoGHZState(alphas..., new_generator)
                replace_state!(states, replace_idx, new_state, rep_sum, rep_ctr, cxt_master)
            end
        end

        rep = rep_from_cache(rep_sum, rep_ctr)
    end

    h5open(args["outfile"], "r+") do h5f
        groupname = determine_groupname(h5f; basegroupname=args["outgroup"])
        gp = create_group(h5f, groupname)
        for (k, v) in args
            attributes(gp)[k] = v
        end
        if args["savelevel"] == "deltas"
            deltas = make_delta_states(state_tracker, sort_perms)
            attributes(gp)["niter_actual"] = size(state_tracker, 1)
            gp["initial_states"]          = deltas["initial_states"]
            gp["delta_iterations"]        = deltas["delta_iterations"]
            gp["delta_state_indices"]     = deltas["delta_state_indices"]
            gp["delta_values"]            = deltas["delta_values"]
            gp["initial_perm"]            = deltas["initial_perm"]
            gp["perm_delta_iterations"]   = deltas["perm_delta_iterations"]
            gp["perm_delta_positions"]    = deltas["perm_delta_positions"]
            gp["perm_delta_values"]       = deltas["perm_delta_values"]
        else
            output_states = make_output_states(state_tracker, accuracies, args)
            gp["states"] = output_states
        end
        gp["goal"]       = goal
        gp["accuracies"] = accuracies
    end
    println(maximum(accuracies))
end
