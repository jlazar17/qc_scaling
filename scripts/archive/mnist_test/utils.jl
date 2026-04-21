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
