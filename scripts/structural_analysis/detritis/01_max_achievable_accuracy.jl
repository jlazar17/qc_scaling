"""
For nqubit=4, enumerate every possible PseudoGHZ state (2592 total),
compute each state's definite parity at every position, and then for
H=0 and H=1 goals find the maximum achievable majority-vote accuracy
using greedy ensemble construction.

Key question: Is there a structural ceiling below 1.0 for H=1, or
does the gap come from the optimizer not finding the right states?
"""
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

# ---------------------------------------------------------------------------
# Enumerate all states for nqubit and compute their parity arrays
# ---------------------------------------------------------------------------

function enumerate_all_states(nqubit)
    n = 3^nqubit
    nalpha = 2^(nqubit - 1)
    states = []
    for gen_idx in 0:n-1
        gen = QCScaling.ParityOperator(gen_idx, nqubit)
        for theta_s in 0:1
            for theta_z in 0:1
                for alpha_idx in 0:nalpha-1
                    alphas = QCScaling.idx_to_alphas(alpha_idx, nqubit)
                    push!(states, QCScaling.PseudoGHZState(theta_s, theta_z, alphas, gen))
                end
            end
        end
    end
    return states
end

"""
Compute the parity of state at all n positions.
Returns a vector of length n where entries are 0, 1, or -1 (indeterminate/uncovered).
"""
function compute_parity_vector(state, nqubit)
    n = 3^nqubit
    par = fill(-1, n)  # -1 = indeterminate
    cxt_master = QCScaling.ContextMaster(nqubit)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        if p != 0.5
            par[derived_po.index] = round(Int, p)
        end
    end
    return par
end

"""
Given a parity table P (n_states x n_positions, with -1 for indeterminate),
compute the accuracy of the ensemble defined by selecting states in 'idxs'.
Uses majority vote; NaN positions don't count.
"""
function ensemble_accuracy(P, idxs, goal)
    ngbits = length(goal)
    rep_sum = zeros(Int, 2 * ngbits)
    rep_ctr = zeros(Int, 2 * ngbits)
    for si in idxs
        for k in 1:2*ngbits
            v = P[si, k]
            v == -1 && continue
            rep_sum[k] += v
            rep_ctr[k] += 1
        end
    end
    return QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
end

"""
Greedy construction: at each step, add the state that maximally increases accuracy.
Returns (best_accuracy, ensemble_indices).
"""
function greedy_build(P, goal, nstate; rng=Random.MersenneTwister(42))
    n_states = size(P, 1)
    ngbits   = length(goal)
    rep_sum  = zeros(Int, 2 * ngbits)
    rep_ctr  = zeros(Int, 2 * ngbits)

    ensemble = Int[]
    best_acc = 0.0

    for _ in 1:nstate
        best_delta = -Inf
        best_s     = -1
        # Sample a random order to break ties
        order = randperm(rng, n_states)
        for si in order
            # Tentatively add state si
            for k in 1:2*ngbits
                v = P[si, k]; v == -1 && continue
                rep_sum[k] += v; rep_ctr[k] += 1
            end
            new_acc = QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta = new_acc - best_acc
            if delta > best_delta
                best_delta = delta
                best_s = si
            end
            # Undo
            for k in 1:2*ngbits
                v = P[si, k]; v == -1 && continue
                rep_sum[k] -= v; rep_ctr[k] -= 1
            end
        end
        # Actually add best state
        for k in 1:2*ngbits
            v = P[best_s, k]; v == -1 && continue
            rep_sum[k] += v; rep_ctr[k] += 1
        end
        push!(ensemble, best_s)
        best_acc += best_delta
    end
    return best_acc, ensemble
end

"""
Upper bound via LP relaxation: allow fractional state weights in [0,1],
maximize the number of pairs where the weighted majority vote is correct.
We relax: for pair j, "correct" if weighted_sum(k1) > weighted_sum(k2) when goal=0
and weighted_sum(k1) < weighted_sum(k2) when goal=1 (approximately).
Instead we compute: for each pair j and each possible sign choice,
find the maximum fraction of pairs achievable.

Simpler: for each pair j, compute for each state whether it "helps" (+1),
"hurts" (-1), or "is indifferent" (0). Then an accuracy upper bound is the
fraction of pairs where there exists a feasible assignment.
"""
function per_pair_achievability(P, goal)
    ngbits = length(goal)
    n_states = size(P, 1)

    # For each pair j, what (rep[k1], rep[k2]) combinations are achievable
    # by majority vote with unlimited states?
    # A pair is achievable if we can find states that vote 0 at k1 and 1 at k2
    # (or vice versa for goal=0).

    achievable = zeros(Bool, ngbits)
    for j in 1:ngbits
        k1 = 2j - 1; k2 = 2j
        want_xor = goal[j]  # 0 = want same, 1 = want different

        # Can we achieve rep[k1]=0, rep[k2]=0 ? (both 0, XOR=0, good for goal=0)
        # Can we achieve rep[k1]=1, rep[k2]=1 ? (both 1, XOR=0, good for goal=0)
        # Can we achieve rep[k1]=0, rep[k2]=1 ? (XOR=1, good for goal=1)
        # Can we achieve rep[k1]=1, rep[k2]=0 ? (XOR=1, good for goal=1)

        # Achievable means: there exist states with a definite 0 at k1 AND definite 1 at k2
        # (or the other combination) -- or both positions can be independently set.

        # Check: does any state cover k1 with parity 0?
        has_k1_0 = any(P[si, k1] == 0 for si in 1:n_states)
        has_k1_1 = any(P[si, k1] == 1 for si in 1:n_states)
        has_k2_0 = any(P[si, k2] == 0 for si in 1:n_states)
        has_k2_1 = any(P[si, k2] == 1 for si in 1:n_states)

        if want_xor == 0
            # Need rep[k1]=rep[k2]; achievable if both can be 0 or both can be 1
            achievable[j] = (has_k1_0 && has_k2_0) || (has_k1_1 && has_k2_1)
        else
            # Need rep[k1]≠rep[k2]; achievable if (k1=0,k2=1) or (k1=1,k2=0)
            achievable[j] = (has_k1_0 && has_k2_1) || (has_k1_1 && has_k2_0)
        end
    end
    return achievable
end

"""
Stronger check: can we achieve rep[k1]=v1 and rep[k2]=v2 SIMULTANEOUSLY using
states that only cover the "correct" positions? Checks if the requirements
are CONSISTENT (no single state forced to give wrong parity at some position).

This is a necessary condition for achievability of the joint (v1, v2) target.
"""
function joint_achievability(P, goal)
    ngbits = length(goal)
    n_states = size(P, 1)

    # For each pair, what are the achievable JOINT parity patterns?
    pair_joint = Dict{Int, Set{Tuple{Int,Int}}}()
    for j in 1:ngbits
        k1 = 2j-1; k2 = 2j
        joints = Set{Tuple{Int,Int}}()
        for si in 1:n_states
            p1 = P[si, k1]; p2 = P[si, k2]
            # If state covers both k1 and k2 with definite parities
            p1 != -1 && p2 != -1 && push!(joints, (p1, p2))
            # If state covers only k1: can be combined with any k2-only state
            # p1 != -1 && p2 == -1: single-covered k1
        end
        pair_joint[j] = joints
    end
    return pair_joint
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit = 4
    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    println("nqubit=$nqubit, n=$n, ngbits=$ngbits")
    println("Enumerating all states...")
    all_states  = enumerate_all_states(nqubit)
    n_states    = length(all_states)
    @printf("  Total states: %d\n", n_states)

    println("Computing parity table...")
    P = fill(-1, n_states, 2 * ngbits)
    for (si, state) in enumerate(all_states)
        pv = compute_parity_vector(state, nqubit)
        for k in 1:2*ngbits
            P[si, k] = pv[k]
        end
    end

    # Coverage stats
    coverage = [count(P[:, k] .!= -1) for k in 1:2*ngbits]
    @printf("  Coverage per position: min=%d, max=%d, mean=%.1f\n",
            minimum(coverage), maximum(coverage), mean(coverage))

    # For each pair, count both-covered states (definite parity at BOTH k1 and k2)
    both_cov = [count((P[:, 2j-1] .!= -1) .& (P[:, 2j] .!= -1)) for j in 1:ngbits]
    @printf("  Both-covered states per pair: min=%d, max=%d, mean=%.1f\n",
            minimum(both_cov), maximum(both_cov), mean(both_cov))

    rng_goals = Random.MersenneTwister(42)
    n_trials = 20

    println()
    println("="^70)
    println("Testing maximum achievable accuracy: greedy ensemble construction")
    println("="^70)

    for H_target in [0.0, 0.5, 1.0]
        k = round(Int, H_target * ngbits / 2)
        # For H=0: k=0 (all zeros), H=1: k=ngbits/2 (half ones)
        if H_target == 0.0
            goal_ones = 0
        elseif H_target == 1.0
            goal_ones = ngbits ÷ 2
        else
            goal_ones = ngbits ÷ 4
        end

        accs_greedy  = Float64[]
        accs_achieve = Float64[]

        for trial in 1:n_trials
            rng = Random.MersenneTwister(trial * 1000)
            goal = shuffle!(rng, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

            # --- Greedy with 40 states ---
            acc40, _ = greedy_build(P, goal, 40; rng=Random.MersenneTwister(trial))
            push!(accs_greedy, acc40)

            # --- Check marginal achievability ---
            achievable = per_pair_achievability(P, goal)
            push!(accs_achieve, mean(achievable))
        end

        @printf("\nH = %.2f (k=%d ones)\n", H_target, goal_ones)
        @printf("  Greedy(40)  acc: mean=%.4f, max=%.4f, min=%.4f\n",
                mean(accs_greedy), maximum(accs_greedy), minimum(accs_greedy))
        @printf("  Marginal achievable fraction: mean=%.4f, min=%.4f\n",
                mean(accs_achieve), minimum(accs_achieve))
    end

    # ---------------------------------------------------------------------------
    # Deep dive: for one H=1 goal, trace what limits accuracy
    # ---------------------------------------------------------------------------
    println()
    println("="^70)
    println("Deep dive: H=1 goal, why is accuracy limited?")
    println("="^70)

    goal_ones = ngbits ÷ 2
    rng = Random.MersenneTwister(999)
    goal = shuffle!(rng, vcat(ones(Int, goal_ones), zeros(Int, ngbits - goal_ones)))

    joint_info = joint_achievability(P, goal)
    println("\nJoint (parity_k1, parity_k2) patterns achievable from both-covered states:")
    xor0_pairs = [j for j in 1:ngbits if (0,0) in joint_info[j] || (1,1) in joint_info[j]]
    xor1_pairs = [j for j in 1:ngbits if (0,1) in joint_info[j] || (1,0) in joint_info[j]]
    println("  Pairs with XOR=0 achievable by a single state: $(length(xor0_pairs))")
    println("  Pairs with XOR=1 achievable by a single state: $(length(xor1_pairs))")
    println("  Pairs with ONLY XOR=0 possible (same parity only): $(length(xor0_pairs) - length(intersect(xor0_pairs, xor1_pairs)))")
    println("  Pairs with ONLY XOR=1 possible: $(length(xor1_pairs) - length(intersect(xor0_pairs, xor1_pairs)))")
    println("  Goal-1 pairs with XOR=1 achievable: $(count(goal[j]==1 && (0,1) in joint_info[j] || (1,0) in joint_info[j] for j in 1:ngbits))")

    # Check: for H=1 goal, are there pairs where the REQUIRED XOR is impossible
    # from both-covered states? (They can still be set via singly-covered states)
    println()
    println("Checking if required XOR is achievable for each pair:")
    correct_possible = 0
    for j in 1:ngbits
        k1 = 2j-1; k2 = 2j
        want = goal[j]
        # Can we achieve the required XOR via a single both-covered state?
        via_joint = (want == 0) ? ((0,0) in joint_info[j] || (1,1) in joint_info[j]) :
                                   ((0,1) in joint_info[j] || (1,0) in joint_info[j])
        # Can we achieve it via separate singly-covered states?
        has_k1_0 = any(P[si, k1] == 0 for si in 1:n_states)
        has_k1_1 = any(P[si, k1] == 1 for si in 1:n_states)
        has_k2_0 = any(P[si, k2] == 0 for si in 1:n_states)
        has_k2_1 = any(P[si, k2] == 1 for si in 1:n_states)
        if want == 0
            via_single = (has_k1_0 && has_k2_0) || (has_k1_1 && has_k2_1)
        else
            via_single = (has_k1_0 && has_k2_1) || (has_k1_1 && has_k2_0)
        end
        correct_possible += via_single
    end
    @printf("Pairs where goal XOR is achievable (via independent coverage): %d / %d\n",
            correct_possible, ngbits)

    # Now: grow ensemble to large size and see where accuracy saturates
    println()
    println("Greedy accuracy vs ensemble size for H=0 and H=1:")
    println(@sprintf("%-10s  %-12s  %-12s", "nstate", "H=0 acc", "H=1 acc"))
    println(repeat("-", 38))

    goal_H0 = zeros(Int, ngbits)
    goal_H1 = goal  # the H=1 goal we built above

    rep_sum_H0 = zeros(Int, 2*ngbits); rep_ctr_H0 = zeros(Int, 2*ngbits)
    rep_sum_H1 = zeros(Int, 2*ngbits); rep_ctr_H1 = zeros(Int, 2*ngbits)

    rng0 = Random.MersenneTwister(1)
    rng1 = Random.MersenneTwister(2)

    ensemble_H0 = Int[]
    ensemble_H1 = Int[]

    acc_H0 = 0.0
    acc_H1 = 0.0

    function greedy_step!(rep_sum, rep_ctr, goal, ensemble, rng)
        n_states = size(P, 1)
        best_delta = -Inf; best_s = -1
        curr_acc = QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
        order = randperm(rng, n_states)
        for si in order
            for k in 1:2*ngbits
                v = P[si, k]; v == -1 && continue
                rep_sum[k] += v; rep_ctr[k] += 1
            end
            new_acc = QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta = new_acc - curr_acc
            if delta > best_delta
                best_delta = delta; best_s = si
            end
            for k in 1:2*ngbits
                v = P[si, k]; v == -1 && continue
                rep_sum[k] -= v; rep_ctr[k] -= 1
            end
        end
        for k in 1:2*ngbits
            v = P[best_s, k]; v == -1 && continue
            rep_sum[k] += v; rep_ctr[k] += 1
        end
        push!(ensemble, best_s)
        return QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
    end

    checkpoints = Set([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    for t in 1:100
        acc_H0 = greedy_step!(rep_sum_H0, rep_ctr_H0, goal_H0, ensemble_H0, rng0)
        acc_H1 = greedy_step!(rep_sum_H1, rep_ctr_H1, goal_H1, ensemble_H1, rng1)
        if t in checkpoints
            @printf("%-10d  %-12.4f  %-12.4f\n", t, acc_H0, acc_H1)
        end
    end

    println()
    println("Max achievable accuracy with 100 greedy states:")
    @printf("  H=0: %.4f\n", QCScaling.rep_accuracy_fast(rep_sum_H0, rep_ctr_H0, goal_H0))
    @printf("  H=1: %.4f\n", QCScaling.rep_accuracy_fast(rep_sum_H1, rep_ctr_H1, goal_H1))
end

main()
