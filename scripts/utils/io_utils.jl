function determine_groupname(h5f; basegroupname="results")
    if ~(basegroupname in keys(h5f))
        return basegroupname
    end

    idx = 1
    groupname = "$(basegroupname)_$(idx)"
    while groupname in keys(h5f)
        idx += 1
        groupname = "$(basegroupname)_$(idx)"
    end
    return groupname
end

function binaryify(a::Vector{Int}, nqubit)
    goal = zeros(Int, Int((3^nqubit-1) / 2))
    for (idx, v) in enumerate(a)
        start = (idx-1) * 4 + 1
        stop = idx * 4
        goal[start:stop] = digits(Int(round(v)); base=2, pad=4)
    end
    return goal
end

function unbinaryify(a)
    out = zeros(784)
    for (jdx, idx) in enumerate(1:4:3_136)
        n = sum([2^n for n in 0:3] .* a[idx:idx+3])
        out[jdx] = n
    end
    return out
end

function setup_outfile(args)
    if ~isfile(args["outfile"])
        h5open(args["outfile"], "w") do _
        end
    end
    return args["outfile"]
end

function make_delta_states(state_tracker, sort_perms)
    niter, nstates, nparams = size(state_tracker)
    initial_states = state_tracker[1, :, :]

    delta_iterations = Int32[]
    delta_state_indices = Int32[]
    delta_values_list = Vector{UInt8}[]

    for i in 2:niter
        for j in 1:nstates
            # sort_perms[i, j] is the pre-sort position of state j,
            # i.e. which slot in iteration i-1 this state came from
            prev_j = sort_perms[i, j]
            if state_tracker[i, j, :] != state_tracker[i-1, prev_j, :]
                push!(delta_iterations, Int32(i))
                push!(delta_state_indices, Int32(j))
                push!(delta_values_list, state_tracker[i, j, :])
            end
        end
    end

    delta_values = length(delta_values_list) > 0 ?
        reduce(hcat, delta_values_list)' |> collect :
        zeros(UInt8, 0, nparams)

    # Delta-encode sort_perms: store initial perm + changed positions
    initial_perm = Int16.(sort_perms[1, :])
    perm_delta_iterations = Int32[]
    perm_delta_positions = Int16[]
    perm_delta_values = Int16[]

    for i in 2:niter
        for j in 1:nstates
            if sort_perms[i, j] != sort_perms[i-1, j]
                push!(perm_delta_iterations, Int32(i))
                push!(perm_delta_positions, Int16(j))
                push!(perm_delta_values, Int16(sort_perms[i, j]))
            end
        end
    end

    return Dict(
        "initial_states" => initial_states,
        "delta_iterations" => delta_iterations,
        "delta_state_indices" => delta_state_indices,
        "delta_values" => delta_values,
        "initial_perm" => initial_perm,
        "perm_delta_iterations" => perm_delta_iterations,
        "perm_delta_positions" => perm_delta_positions,
        "perm_delta_values" => perm_delta_values,
    )
end

function reconstruct_states(gp)
    initial_states = read(gp["initial_states"])
    delta_iterations = read(gp["delta_iterations"])
    delta_state_indices = read(gp["delta_state_indices"])
    delta_values = read(gp["delta_values"])
    niter = Int(read(attributes(gp)["niter_actual"]))

    # Reconstruct sort_perms from deltas
    initial_perm = read(gp["initial_perm"])
    perm_delta_iterations = read(gp["perm_delta_iterations"])
    perm_delta_positions = read(gp["perm_delta_positions"])
    perm_delta_values = read(gp["perm_delta_values"])

    nstates, nparams = size(initial_states)

    # Group perm deltas by iteration
    perm_deltas_by_iter = Dict{Int, Vector{Int}}()
    for (k, iter) in enumerate(perm_delta_iterations)
        if !haskey(perm_deltas_by_iter, iter)
            perm_deltas_by_iter[iter] = Int[]
        end
        push!(perm_deltas_by_iter[iter], k)
    end

    # Group state deltas by iteration
    deltas_by_iter = Dict{Int, Vector{Int}}()
    for (k, iter) in enumerate(delta_iterations)
        if !haskey(deltas_by_iter, iter)
            deltas_by_iter[iter] = Int[]
        end
        push!(deltas_by_iter[iter], k)
    end

    state_tracker = zeros(UInt8, niter, nstates, nparams)
    state_tracker[1, :, :] = initial_states

    current_perm = Int.(initial_perm)
    for i in 2:niter
        # Update perm for this iteration
        if haskey(perm_deltas_by_iter, i)
            for k in perm_deltas_by_iter[i]
                current_perm[perm_delta_positions[k]] = perm_delta_values[k]
            end
        end
        # Apply sort permutation
        for j in 1:nstates
            state_tracker[i, j, :] = state_tracker[i-1, current_perm[j], :]
        end
        # Apply state deltas
        if haskey(deltas_by_iter, i)
            for k in deltas_by_iter[i]
                state_tracker[i, delta_state_indices[k], :] = delta_values[k, :]
            end
        end
    end

    return state_tracker
end

function make_output_states(state_tracker, accuracies, args)
    if args["savelevel"]=="all"
        return state_tracker
    elseif args["savelevel"]=="best_states"
        bestacc, idxs = 0, []
        for idx in 1:size(state_tracker)[1]
            if accuracies[idx] < bestacc
                continue
            end
            push!(idxs, idx)
            bestacc = accuracies[idx]
        end
        return state_tracker[idxs, :, :, :]
    elseif args["savelevel"]=="best_state"
        idx = argmax(accuracies)
        return state_tracker[argmax, :, :, :]
    end
        error("Unrecognized savelevel")
end
