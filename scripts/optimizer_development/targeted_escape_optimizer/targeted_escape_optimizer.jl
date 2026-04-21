using ProgressMeter

# ---------------------------------------------------------------------------
# Incremental representation cache
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

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    _update_rep_cache!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    _update_rep_cache!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

# ---------------------------------------------------------------------------
# NaN classification — distinguish never-covered from tied
# ---------------------------------------------------------------------------

function classify_undefined(rep_sum, rep_ctr)
    uncovered = Set{Int}()
    ambiguous = Set{Int}()
    for i in eachindex(rep_ctr)
        if rep_ctr[i] == 0
            push!(uncovered, i)
        elseif rep_sum[i] / rep_ctr[i] == 0.5
            push!(ambiguous, i)
        end
    end
    return uncovered, ambiguous
end

# ---------------------------------------------------------------------------
# Coverage-targeted context selection
# ---------------------------------------------------------------------------

function get_coverage_context(target_idxs, cxt_master)
    nqubit   = cxt_master.nqubit
    base_cxt = rand() > 0.5 ? cxt_master.base_even : cxt_master.base_odd
    best_score = -1
    best_cxt   = nothing
    for idx in 0:3^nqubit - 1
        generator = QCScaling.ParityOperator(idx, nqubit)
        cxt       = QCScaling.Context(generator, base_cxt)
        s = count(po -> po.index in target_idxs, cxt.pos)
        if s > best_score
            best_score = s
            best_cxt   = cxt
        end
        best_score == length(base_cxt.pos) && break
    end
    return best_cxt
end

# ---------------------------------------------------------------------------
# Main optimizer
#
# Normal replacement: identical to the improved optimizer — p_mutate controls
# generator mutation toward undefined regions, otherwise alpha/theta only.
#
# Escape branch: coverage-targeted replacement, prioritising:
#   1. uncovered operators (rep_ctr == 0)
#   2. ambiguous operators (pref == 0.5)
#   3. general undefined (fallback to get_new_contexts)
# ---------------------------------------------------------------------------

function targeted_escape_optimization(make_goal_fxn, args)
    nqubit      = args["nqubit"]
    Random.seed!(args["seed"])

    goal        = make_goal_fxn(args)
    states      = make_initial_state(args)
    nreplace    = determine_nreplace(args, length(states))
    itr         = determine_itr(args)
    outfile     = setup_outfile(args)
    fingerprint = QCScaling.Fingerprint(nqubit)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    n_same_tol  = args["n_same_tol"]
    p_mutate    = args["p_mutate"]

    rep_sum, rep_ctr = build_rep_cache(states, cxt_master)
    rep = rep_from_cache(rep_sum, rep_ctr)

    state_tracker = zeros(UInt8, (length(itr), length(states), 2nqubit + 1))
    accuracies    = zeros(Float64, length(itr))
    sort_perms    = zeros(Int, (length(itr), length(states)))
    n_same        = 0
    last_acc      = -1.0

    @showprogress for idx in itr
        scores = QCScaling.score(states, rep, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]
        sort_perms[idx, :] = sorter

        for (jdx, state) in enumerate(states)
            state_tracker[idx, jdx, 1]                = state.theta_s
            state_tracker[idx, jdx, 2]                = state.theta_z
            state_tracker[idx, jdx, 3:1+nqubit]       = state.alphas
            state_tracker[idx, jdx, end-nqubit+1:end] = state.generator.βs
        end

        acc = accuracy(rep, goal)
        accuracies[idx] = acc

        if acc == last_acc
            n_same += 1
        else
            n_same   = 0
            last_acc = acc
        end

        if n_same <= n_same_tol
            # Normal replacement: same as improved optimizer.
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace, replace=false)
            for which in whiches
                rplc_state = states[which]
                if rand() < p_mutate
                    cxt           = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(cxt.pos)
                    base_cxt      = cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                else
                    new_generator = rplc_state.generator
                    base_cxt      = rplc_state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = QCScaling.Context(new_generator, base_cxt)
                end
                alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                new_state = QCScaling.PseudoGHZState(alphas..., new_generator)
                replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
            end
        else
            # Escape: targeted coverage replacement.
            n_same = 0
            uncovered, ambiguous = classify_undefined(rep_sum, rep_ctr)
            target = !isempty(uncovered) ? uncovered :
                     !isempty(ambiguous) ? ambiguous  : nothing

            for _ in 1:nreplace
                replace_idx = rand(1:length(states))
                cxt = isnothing(target) ?
                    first(QCScaling.get_new_contexts(states, rep, cxt_master, 1)) :
                    get_coverage_context(target, cxt_master)
                new_generator = first(cxt.pos)
                base_cxt      = cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                alphas        = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                new_state     = QCScaling.PseudoGHZState(alphas..., new_generator)
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
