using ProgressMeter

# ---------------------------------------------------------------------------
# Incremental representation cache (carried over from improved optimizer)
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
# NaN classification
#
# There are two distinct reasons an operator index can be undefined:
#   :uncovered  — rep_ctr[i] == 0, no state has ever measured this operator
#   :ambiguous  — rep_ctr[i] > 0 but pref[i] == 0.5, states disagree equally
#
# The score function skips both, but they require different fixes:
#   uncovered  → need a state whose generator reaches this operator
#   ambiguous  → need a state that breaks the tie in the correct direction
# ---------------------------------------------------------------------------

function classify_undefined(rep_sum, rep_ctr)
    uncovered = Set{Int}()
    ambiguous = Set{Int}()
    n = length(rep_ctr)
    for i in 1:n
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
#
# Finds the generator whose derived context covers the most entries from
# `target_idxs`. Used both in the coverage phase and the escape branch.
# ---------------------------------------------------------------------------

function get_coverage_context(target_idxs, cxt_master)
    nqubit = cxt_master.nqubit
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
        best_score == length(base_cxt.pos) && break  # can't do better
    end
    return best_cxt
end

# ---------------------------------------------------------------------------
# Coverage phase
#
# Before entering the main accuracy loop, greedily replace states to ensure
# every operator index has at least one state covering it (rep_ctr > 0).
# A state is "safe to replace" if every operator it covers is also covered
# by at least one other state (removing it leaves no new holes).
# If no state is safe, fall back to replacing the one with the most overlap.
# ---------------------------------------------------------------------------

function coverage_phase!(states, rep_sum, rep_ctr, cxt_master, fingerprint, goal, max_iter)
    rep = rep_from_cache(rep_sum, rep_ctr)
    for _ in 1:max_iter
        uncovered = Set(findall(==(0), rep_ctr))
        isempty(uncovered) && break

        # Find the best context for hitting uncovered operators
        cxt          = get_coverage_context(uncovered, cxt_master)
        new_generator = first(cxt.pos)
        base_cxt     = cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd

        # Find a safe state to replace: one where all its operators are
        # covered at least twice (so removing it leaves no new holes).
        replace_idx = nothing
        max_overlap = -1
        for (i, state) in enumerate(states)
            sc = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            idxs = [QCScaling.parity(state, state.generator + bp) >= 0 ?
                    (state.generator + bp).index : 0
                    for bp in sc.pos]
            own_base = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            covered_idxs = [(state.generator + bp).index for bp in own_base.pos]
            safe = all(rep_ctr[j] >= 2 for j in covered_idxs)
            overlap = sum(rep_ctr[j] for j in covered_idxs)
            if safe && overlap > max_overlap
                max_overlap  = overlap
                replace_idx  = i
            end
        end
        # If no safe state found, replace the most redundant one anyway
        if isnothing(replace_idx)
            replace_idx = argmax([
                sum((s.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd).pos .|>
                    bp -> rep_ctr[(s.generator + bp).index])
                for s in states
            ])
        end

        alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
        new_state = QCScaling.PseudoGHZState(alphas..., new_generator)
        replace_state!(states, replace_idx, new_state, rep_sum, rep_ctr, cxt_master)
        rep = rep_from_cache(rep_sum, rep_ctr)
    end
    return rep
end

# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

function coverage_optimization(make_goal_fxn, args)
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

    # Coverage phase: ensure every operator is covered before accuracy loop.
    # Budget is 10% of total iterations, capped at 500.
    coverage_budget = min(500, length(itr) ÷ 10)
    rep = coverage_phase!(states, rep_sum, rep_ctr, cxt_master, fingerprint, goal, coverage_budget)

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

        uncovered, ambiguous = classify_undefined(rep_sum, rep_ctr)

        if n_same <= n_same_tol
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace, replace=false)
            for which in whiches
                rplc_state = states[which]

                if !isempty(uncovered)
                    # Highest priority: cover operators no state has reached.
                    cxt          = get_coverage_context(uncovered, cxt_master)
                    new_generator = first(cxt.pos)
                    base_cxt     = cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                elseif rand() < p_mutate
                    # Normal generator mutation toward undefined regions
                    # (ambiguous operators or remaining NaN entries).
                    cxt          = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(cxt.pos)
                    base_cxt     = cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
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
            # Escape: prioritize uncovered > ambiguous > general undefined.
            n_same = 0
            target = !isempty(uncovered) ? uncovered : ambiguous
            for _ in 1:nreplace
                replace_idx = rand(1:length(states))
                cxt = isempty(target) ?
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
