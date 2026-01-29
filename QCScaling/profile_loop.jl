using QCScaling
using Random
using StatsBase
using Profile

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
run_loop(2)

# Allocation profiling
println("=== Allocation breakdown (single iteration) ===")
function run_single_iter_profiled()
    nqubit = 8
    Random.seed!(12345)
    nstate = 3 * Int(ceil(3^nqubit / 2^(nqubit-1)))
    states = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    goal = rand(0:1, (3^nqubit - 1) ÷ 2)
    nreplace = Int(ceil(0.1 * length(states)))
    fingerprint = QCScaling.Fingerprint(nqubit)
    cxt_master = QCScaling.ContextMaster(nqubit)

    println("\n-- score(states, goal, cxt_master) --")
    @time scores = QCScaling.score(states, goal, cxt_master)

    sorter = sortperm(scores)
    scores = scores[sorter]
    states = states[sorter]

    println("\n-- calculate_representation(states) --")
    @time rep = QCScaling.calculate_representation(states)

    println("\n-- pick_new_alphas ($(nreplace)x) --")
    ws = Weights(maximum(scores) .- scores .+ 1)
    whiches = sample(1:length(scores), ws, nreplace, replace=false)
    @time begin
        for which in whiches
            rplc_state = states[which]
            base_cxt = rplc_state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            cxt = QCScaling.Context(rplc_state.generator, base_cxt)
            alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
            new_state = QCScaling.PseudoGHZState(alphas..., rplc_state.generator)
            states[which] = new_state
        end
    end

    println("\n-- Context creation (single) --")
    state = first(states)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    @time for _ in 1:1000
        QCScaling.Context(state.generator, base_cxt)
    end

    println("\n-- parity(state, cxt) (single) --")
    cxt = QCScaling.Context(state.generator, base_cxt)
    @time for _ in 1:1000
        QCScaling.parity(state, cxt)
    end
end

run_single_iter_profiled()
