using JLD2

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

function determine_groupname(jldf; basegroupname="results")
    if ~(basegroupname in keys(jldf))
        return basegroupname
    end

    idx = 1
    groupname = "$(basegroupname)_$(idx)"
    while groupname in keys(jldf)
        idx += 1
        groupname = "$(basegroupname)_$(idx)"
    end
    return groupname
end
