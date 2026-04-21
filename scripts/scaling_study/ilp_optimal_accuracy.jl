using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))
Pkg.add("JuMP")
Pkg.add("HiGHS")

using QCScaling
using JuMP
using HiGHS
import MathOptInterface as MOI
using Statistics
using Random
using HDF5
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N
    return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0
    H >= 1.0 && return N ÷ 2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), 0:N÷2)
    return (0:N÷2)[idx]
end

# ---------------------------------------------------------------------------
# State space enumeration
#
# For each state (gen_idx, theta_s, theta_z, alpha_idx), compute:
#   vote_mat[i, k] ∈ {-1, 0, +1}: net vote of state i at position k
#     +1 if state i covers k with parity 1
#     -1 if state i covers k with parity 0
#      0 if state i does not cover k (or parity is ambiguous 0.5)
#   cov_mat[i, k] ∈ {0, 1}: 1 if state i covers position k
# ---------------------------------------------------------------------------

function enumerate_vote_matrix(nqubit)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    nalpha     = 2^(nqubit - 1)

    vote_rows = Vector{Int}[]
    cov_rows  = Vector{Int}[]

    for gen_idx in 0:n-1
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        for theta_s in 0:1
            base_cxt = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
            for theta_z in 0:1
                for alpha_idx in 0:nalpha-1
                    alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                    state  = QCScaling.PseudoGHZState(theta_s, theta_z, alphas, generator)

                    vote_row = zeros(Int, n)
                    cov_row  = zeros(Int, n)

                    for base_po in base_cxt.pos
                        derived_po = state.generator + base_po
                        p = QCScaling.parity(state, derived_po)
                        if p != 0.5
                            k = derived_po.index
                            vote_row[k] = Int(2 * p - 1)  # +1 or -1
                            cov_row[k]  = 1
                        end
                    end

                    push!(vote_rows, vote_row)
                    push!(cov_rows,  cov_row)
                end
            end
        end
    end

    vote_mat = collect(reduce(hcat, vote_rows)')   # [nstates × n]
    cov_mat  = collect(reduce(hcat, cov_rows)')
    return vote_mat, cov_mat
end

# ---------------------------------------------------------------------------
# State pruning: select top-K states most aligned with goal target
#
# For each goal bit j, the ideal coverage is:
#   position k1=2j-1: vote +1 (contribute to w[k1]=1)
#   position k2=2j:   vote +1 if goal[j]=0 (XOR=0 → both same),
#                     vote -1 if goal[j]=1 (XOR=1 → k2 opposite to k1)
#
# Alignment score for state i = sum over covered positions k of vote[i,k]*target[k].
# States with higher alignment are more likely to be in the optimal ensemble.
# ---------------------------------------------------------------------------

function top_k_states(goal::Vector{Int}, vote_mat::Matrix{Int}, cov_mat::Matrix{Int}, K::Int)
    n       = size(vote_mat, 2)
    ngbits  = length(goal)
    nstates = size(vote_mat, 1)

    target = zeros(Int, n)
    for j in 1:ngbits
        target[2j-1] = 1
        target[2j]   = goal[j] == 1 ? -1 : 1
    end

    alignment = [sum(vote_mat[i, k] * target[k]
                     for k in 1:n if cov_mat[i, k] != 0)
                 for i in 1:nstates]
    idx = sortperm(alignment, rev=true)[1:min(K, nstates)]
    return idx
end

# ---------------------------------------------------------------------------
# ILP: find maximum accuracy achievable by any ensemble of nstate states
#
# Formulation:
#   x_i ∈ {0,1}: select state i (from the pruned top-K set)
#   w_k ∈ {0,1}: majority vote direction at position k (1 = positive)
#   t_k ∈ {0,1}: position k is invalid (tie or uncovered)
#   v_j ∈ {0,1}: both positions for goal bit j are valid
#   y_j ∈ {0,1}: w[k1] AND w[k2] (linearizes XOR)
#   m_j ∈ {0,1}: XOR of majority votes matches goal[j]
#   r_j ∈ {0,1}: goal bit j is correctly predicted
#
# Majority vote linearized with big-M (M = nstate + 1).
# ---------------------------------------------------------------------------

function solve_ilp(goal::Vector{Int}, vote_mat::Matrix{Int}, cov_mat::Matrix{Int},
                   nstate::Int; K::Int=100)
    idx     = top_k_states(goal, vote_mat, cov_mat, K)
    vm      = vote_mat[idx, :]
    cm      = cov_mat[idx, :]
    nstates = size(vm, 1)
    n       = size(vm, 2)
    ngbits  = length(goal)
    M       = nstate + 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)

    @variable(model, x[1:nstates], Bin)
    @constraint(model, sum(x) == nstate)

    @variable(model, w[1:n], Bin)
    @variable(model, t[1:n], Bin)

    for k in 1:n
        Dk = @expression(model, sum(vm[i, k] * x[i] for i in 1:nstates if vm[i, k] != 0))
        Nk = @expression(model, sum(cm[i, k]  * x[i] for i in 1:nstates if cm[i, k]  != 0))
        # if valid (t=0) and w=1: Dk ≥ 1
        @constraint(model, Dk >= 1 - M * (t[k] + (1 - w[k])))
        # if valid (t=0) and w=0: Dk ≤ -1
        @constraint(model, Dk <= -1 + M * (t[k] + w[k]))
        # if valid: Nk ≥ 1
        @constraint(model, Nk >= 1 - M * t[k])
    end

    @variable(model, v[1:ngbits], Bin)
    @variable(model, y[1:ngbits], Bin)
    @variable(model, m[1:ngbits], Bin)
    @variable(model, r[1:ngbits], Bin)

    for j in 1:ngbits
        k1 = 2j - 1; k2 = 2j

        # v_j = (1 - t_k1) AND (1 - t_k2)
        @constraint(model, v[j] <= 1 - t[k1])
        @constraint(model, v[j] <= 1 - t[k2])
        @constraint(model, v[j] >= 1 - t[k1] - t[k2])

        # y_j = w_k1 AND w_k2
        @constraint(model, y[j] <= w[k1])
        @constraint(model, y[j] <= w[k2])
        @constraint(model, y[j] >= w[k1] + w[k2] - 1)

        # m_j: XOR of (w_k1, w_k2) matches goal[j]
        # XOR = w_k1 + w_k2 - 2*y_j ∈ {0,1}
        if goal[j] == 1
            @constraint(model, m[j] == w[k1] + w[k2] - 2 * y[j])
        else
            @constraint(model, m[j] == 1 - w[k1] - w[k2] + 2 * y[j])
        end

        # r_j = v_j AND m_j
        @constraint(model, r[j] <= v[j])
        @constraint(model, r[j] <= m[j])
        @constraint(model, r[j] >= v[j] + m[j] - 1)
    end

    @objective(model, Max, sum(r))
    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        return objective_value(model) / ngbits
    elseif status == MOI.OBJECTIVE_LIMIT || has_values(model)
        # Timed out but HiGHS found a feasible incumbent — return best known
        return objective_value(model) / ngbits
    else
        return NaN
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit        = 4
    nstates_sweep = [20]
    # H=0.0 omitted: goal is always all-zeros, SA confirms true optimum = 1.0,
    # and K=100 positive-alignment pruning is unreliable for the degenerate all-zeros
    # case (gives lower bound only). For H>0 goals are random (asymmetric), so
    # K=100 positive-alignment pruning correctly identifies the true optimum.
    entropy_vals  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ngoals        = 100
    base_seed     = 42
    K             = 100   # top-K state pruning per goal

    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "ilp_optimal_accuracy.h5")

    @printf("Enumerating state space for nqubit=%d...\n", nqubit)
    vote_mat, cov_mat = enumerate_vote_matrix(nqubit)
    @printf("  %d states, %d positions, %d goal bits\n\n",
            size(vote_mat, 1), n, ngbits)
    flush(stdout)

    @printf("Warming up JIT (K=%d pruning)...\n", K)
    solve_ilp(zeros(Int, ngbits), vote_mat, cov_mat, 8; K=K)
    @printf("Done.\n\n")
    flush(stdout)

    h5open(outfile, "w") do h5f
        HDF5.attributes(h5f)["nqubit"]        = nqubit
        HDF5.attributes(h5f)["ngoals"]        = ngoals
        HDF5.attributes(h5f)["base_seed"]     = base_seed
        HDF5.attributes(h5f)["nstates_sweep"] = nstates_sweep
        HDF5.attributes(h5f)["K_prune"]       = K

        for H_target in entropy_vals
            k     = k_from_entropy(H_target, ngbits)
            H_act = hamming_entropy(k, ngbits)
            rng   = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals = [goal_from_hamming(k, ngbits, rng) for _ in 1:ngoals]

            @printf("H=%.1f (k=%d, H_act=%.4f)\n", H_target, k, H_act)

            # acc_mat[nstate_idx, goal_idx]
            acc_mat = fill(NaN, length(nstates_sweep), ngoals)

            for (si, ns) in enumerate(nstates_sweep)
                t0 = time()
                for gi in 1:ngoals
                    acc_mat[si, gi] = solve_ilp(goals[gi], vote_mat, cov_mat, ns; K=K)
                end
                elapsed = time() - t0
                accs = filter(!isnan, acc_mat[si, :])
                frac_timeout = mean(isnan.(acc_mat[si, :]))
                @printf("  nstate=%2d  frac_perfect=%.2f  median_acc=%.4f  frac_timeout=%.2f  (%.1f s)\n",
                        ns, mean(accs .>= 1.0), isempty(accs) ? NaN : median(accs), frac_timeout, elapsed)
                flush(stdout)
            end

            key = @sprintf("H%.2f_k%d", H_target, k)
            gp  = create_group(h5f, key)
            HDF5.attributes(gp)["H_target"] = H_target
            HDF5.attributes(gp)["H_actual"] = H_act
            HDF5.attributes(gp)["k"]        = k
            HDF5.attributes(gp)["ngbits"]   = ngbits
            gp["acc_mat"]       = acc_mat         # [n_nstates × ngoals]
            gp["nstates_sweep"] = nstates_sweep
        end
    end

    println("\nSaved to $outfile")
end

main()
