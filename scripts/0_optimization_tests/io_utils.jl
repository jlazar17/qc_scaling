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
