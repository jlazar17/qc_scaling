using QCScaling
using Random
using StatsBase

function accuracy(rep, goal)
    pred = abs.(rep[1:2:end-2] .- rep[2:2:end-1])
    s = 0
    for (x, y) in zip(pred, goal)
        if x != y
            continue
        end
        s += 1
    end
    return s / length(goal)
end

function run_loop(niter)
    nqubit = 8
    Random.seed!(12345)

    nstate = 3 * Int(ceil(3^nqubit / 2^(nqubit-1)))
    states = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    goal = rand(0:1, (3^nqubit - 1) ÷ 2)
    nreplace = Int(ceil(0.1 * length(states)))

    fingerprint = QCScaling.Fingerprint(nqubit)
    cxt_master = QCScaling.ContextMaster(nqubit)

    for idx in 1:niter
        scores = QCScaling.score(states, goal, cxt_master)
        sorter = sortperm(scores)
        scores = scores[sorter]
        states = states[sorter]

        rep = QCScaling.calculate_representation(states)
        acc = accuracy(rep, goal)

        ws = Weights(maximum(scores) .- scores .+ 1)
        whiches = sample(1:length(scores), ws, nreplace, replace=false)
        for which in whiches
            rplc_state = states[which]
            base_cxt = rplc_state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(rplc_state.generator, base_cxt)
            alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
            new_state = QCScaling.PseudoGHZState(alphas..., rplc_state.generator)
            states[which] = new_state
        end
    end
end

# Warmup
run_loop(1)

# Timed run
println("Running 100 iterations...")
t = @elapsed run_loop(100)
println("Time for 100 iterations: $(round(t, digits=3)) s")
