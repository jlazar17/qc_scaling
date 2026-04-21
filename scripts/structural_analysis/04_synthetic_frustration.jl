# Synthetic frustration experiment: measure p(acc, balance, N) directly.
#
# Goal: map the positive proposal rate p as a function of:
#   - Current accuracy (acc)
#   - Wrong-pair balance b (fraction of wrong pairs wanting agreement)
#   - Number of qubits N
#
# Method: set the goal with exactly k_ones = round((1-b)*ngbits) ones, giving
# theoretical balance b = 1 - k_ones/ngbits. Run SA to a prescribed target
# accuracy, then probe the landscape by sampling random proposed states and
# counting those with delta > 0. This decouples b from H: we can achieve any
# (acc, b) combination independently, producing the full frustration surface.
#
# Key comparison: p(acc=0.80, b, N=6) vs p(acc=0.80, b, N=8). If the
# frustration function is universal in (acc, b) space, the curves should
# collapse. If they differ, it signals N-dependent structural changes.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# SA to target accuracy
# ---------------------------------------------------------------------------

function build_ensemble_to_accuracy(goal, nqubit, nstate, target_acc, max_steps,
                                     alpha_cool, companion, goal_idx, fingerprint;
                                     seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit; n_rep = n-1
    npos = length(cxt_master.base_even.pos)
    scratch_idxs=Vector{Int}(undef,npos); scratch_pars=Vector{Int}(undef,npos)

    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs=[Vector{Int}(undef,npos) for _ in 1:nstate]
    cache_pars =[Vector{Int}(undef,npos) for _ in 1:nstate]
    for i in 1:nstate; fill_state_cache!(cache_idxs[i],cache_pars[i],ensemble[i],cxt_master); end
    rep_sum=zeros(Int,n); rep_ctr=zeros(Int,n)
    for i in 1:nstate; apply_state_cached!(rep_sum,rep_ctr,cache_idxs[i],cache_pars[i],1); end
    rep=rep_from_cache(rep_sum,rep_ctr)

    acc_fn(rs,rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)

    bad_deltas=Float64[]
    current_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:300
        which=rand(rng,1:nstate)
        gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
        theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
        cxt=QCScaling.Context(gen,base_cxt)
        alphas=pick_alphas_s(cxt,goal,rep,fingerprint,base_cxt,companion,goal_idx,n_rep)
        ns=QCScaling.PseudoGHZState(alphas...,gen)
        apply_state!(rep_sum,rep_ctr,ensemble[which],cxt_master,-1)
        apply_state!(rep_sum,rep_ctr,ns,cxt_master,1)
        delta=acc_fn(rep_sum,rep_ctr)-current_acc
        apply_state!(rep_sum,rep_ctr,ns,cxt_master,-1)
        apply_state!(rep_sum,rep_ctr,ensemble[which],cxt_master,1)
        delta<0 && push!(bad_deltas, abs(delta))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas)/log(0.8)
    current_acc = acc_fn(rep_sum, rep_ctr)

    for step in 1:max_steps
        current_acc >= target_acc && break
        which=rand(rng,1:nstate)
        gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
        theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
        cxt=QCScaling.Context(gen,base_cxt)
        alphas=pick_alphas_s(cxt,goal,rep,fingerprint,base_cxt,companion,goal_idx,n_rep)
        ns=QCScaling.PseudoGHZState(alphas...,gen)
        fill_state_cache!(scratch_idxs,scratch_pars,ns,cxt_master)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,1)
        new_acc=acc_fn(rep_sum,rep_ctr)
        delta=new_acc-current_acc
        if delta>=0||rand(rng)<exp(delta/T)
            update_rep_at_cached!(rep,rep_sum,rep_ctr,cache_idxs[which])
            update_rep_at_cached!(rep,rep_sum,rep_ctr,scratch_idxs)
            copy!(cache_idxs[which],scratch_idxs); copy!(cache_pars[which],scratch_pars)
            ensemble[which]=ns; current_acc=new_acc
        else
            apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,-1)
            apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
        end
        T*=alpha_cool
    end
    return ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars, current_acc, cxt_master
end

# ---------------------------------------------------------------------------
# Probe landscape: count fraction of proposals with delta > 0
# ---------------------------------------------------------------------------

function probe_landscape(ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars,
                          cxt_master, goal, companion, goal_idx, fingerprint,
                          nqubit, n_probes; seed=999)
    rng = Random.MersenneTwister(seed)
    n = 3^nqubit; n_rep = n-1
    npos = length(cxt_master.base_even.pos)
    scratch_idxs=Vector{Int}(undef,npos); scratch_pars=Vector{Int}(undef,npos)
    nstate = length(ensemble)

    acc_fn(rs,rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    base_acc = acc_fn(rep_sum, rep_ctr)

    pos_count = 0

    for _ in 1:n_probes
        which=rand(rng,1:nstate)
        gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
        theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
        cxt=QCScaling.Context(gen,base_cxt)
        alphas=pick_alphas_s(cxt,goal,rep,fingerprint,base_cxt,companion,goal_idx,n_rep)
        ns=QCScaling.PseudoGHZState(alphas...,gen)
        fill_state_cache!(scratch_idxs,scratch_pars,ns,cxt_master)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,1)
        new_acc=acc_fn(rep_sum,rep_ctr)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,-1)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
        delta = new_acc - base_acc
        delta > 0 && (pos_count += 1)
    end
    return pos_count / n_probes
end

# ---------------------------------------------------------------------------
# Wrong-pair balance measurement (sanity check)
# ---------------------------------------------------------------------------

function measure_balance(rep_sum, rep_ctr, goal, companion, goal_idx)
    n_agree_wrong = 0; n_disagree_wrong = 0
    for k1 in 1:length(rep_sum)-1
        companion[k1]==0 && continue; k1>companion[k1] && continue
        k2=companion[k1]; c1=rep_ctr[k1]; c2=rep_ctr[k2]
        s1=rep_sum[k1]; s2=rep_sum[k2]
        (c1==0||c2==0||2*s1==c1||2*s2==c2) && continue
        r1=2*s1>c1; r2=2*s2>c2; j=goal_idx[k1]
        is_correct = (r1 ⊻ r2) == !iszero(goal[j])
        if !is_correct
            goal[j]==0 ? (n_agree_wrong += 1) : (n_disagree_wrong += 1)
        end
    end
    n_wrong = n_agree_wrong + n_disagree_wrong
    return n_wrong, (n_wrong > 0 ? n_agree_wrong / n_wrong : NaN)
end

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

function run_sweep(nqubit, acc_targets, b_vals, n_trials, nstate, max_steps, alpha_cool, n_probes)
    n = 3^nqubit; ngbits = (n-1) ÷ 2

    println("Building shuffled pairing for nqubit=$nqubit...")
    companion, goal_idx, npairs = build_shuffled_pairing(nqubit)
    @assert npairs == ngbits
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    println("  ngbits=$ngbits, done.\n")

    # Results: (acc, b) -> mean positive proposal rate
    results = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()  # -> (mean_p, std_p)

    for b_target in b_vals
        k_ones = round(Int, (1.0 - b_target) * ngbits)
        b_actual = 1.0 - k_ones / ngbits
        @printf("--- b_target=%.2f (k_ones=%d, b_actual=%.4f) ---\n", b_target, k_ones, b_actual)
        @printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
                "target", "actual_acc", "p_pos", "bal_meas", "converged?")
        println("  " * repeat("-", 55))

        for target_acc in acc_targets
            pos_rates = Float64[]
            actual_accs = Float64[]
            bal_measured = Float64[]

            for trial in 1:n_trials
                seed_sa  = trial * 137 + round(Int, b_target * 1000) + round(Int, target_acc * 10000)
                seed_prb = trial * 997 + round(Int, b_target * 1000) + round(Int, target_acc * 10000)
                local_rng = Random.MersenneTwister(seed_sa + 7)
                goal = Random.shuffle!(local_rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

                ens, rs, rc, rep, ci, cp, actual_acc, cxt_m =
                    build_ensemble_to_accuracy(goal, nqubit, nstate, target_acc, max_steps,
                                               alpha_cool, companion, goal_idx, fingerprint;
                                               seed=seed_sa)

                # Only accept if within 3% of target (SA may not always converge)
                actual_acc < target_acc - 0.03 && continue

                p = probe_landscape(ens, rs, rc, rep, ci, cp, cxt_m, goal,
                                    companion, goal_idx, fingerprint, nqubit, n_probes;
                                    seed=seed_prb)
                n_w, bal = measure_balance(rs, rc, goal, companion, goal_idx)
                push!(pos_rates, p)
                push!(actual_accs, actual_acc)
                push!(bal_measured, isnan(bal) ? 0.5 : bal)
            end

            n_conv = length(pos_rates)
            if n_conv == 0
                @printf("  %-8.2f  %-10s  %-10s  %-10s  %d/%d\n",
                        target_acc, "N/A", "N/A", "N/A", 0, n_trials)
                continue
            end

            mean_p = mean(pos_rates)
            std_p  = length(pos_rates) > 1 ? std(pos_rates) : 0.0
            mean_acc = mean(actual_accs)
            mean_bal = mean(bal_measured)

            @printf("  %-8.2f  %-10.4f  %-10.4f  %-10.4f  %d/%d\n",
                    target_acc, mean_acc, mean_p, mean_bal, n_conv, n_trials)

            results[(target_acc, b_actual)] = (mean_p, std_p)
        end
        println()
    end

    return results, ngbits
end

function main()
    alpha_cool = 0.9999
    n_probes   = 1500
    n_trials   = 3

    # nqubit=6: can reach high accuracy
    nstate6    = 100
    acc6 = [0.70, 0.80, 0.85, 0.88, 0.90, 0.93]
    max_steps6 = 200_000

    # nqubit=8: use nstate=200 (above n*≈126 for H=0) so SA can actually make progress
    nstate8    = 200
    acc8 = [0.70, 0.75, 0.80, 0.85]
    max_steps8 = 400_000

    b_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    println("=" ^ 80)
    println("Synthetic frustration sweep: p(acc, balance, N)")
    println("nstate6=$nstate6, nstate8=$nstate8, n_probes=$n_probes, n_trials=$n_trials per cell")
    println("=" ^ 80)
    println()

    # --- nqubit = 6 ---
    println("=" ^ 80)
    println("nqubit = 6")
    println("=" ^ 80)
    res6, ngbits6 = run_sweep(6, acc6, b_vals, n_trials, nstate6, max_steps6, alpha_cool, n_probes)

    # --- nqubit = 8 ---
    println("=" ^ 80)
    println("nqubit = 8")
    println("=" ^ 80)
    res8, ngbits8 = run_sweep(8, acc8, b_vals, n_trials, nstate8, max_steps8, alpha_cool, n_probes)

    # --- Summary tables ---
    println()
    println("=" ^ 80)
    println("Summary: p_pos(acc, b) for nqubit=6")
    println("Rows = balance b (1=all want agreement, 0.5=balanced)")
    println("Cols = target accuracy")
    println("=" ^ 80)

    print_summary_table(res6, acc6, b_vals, "nqubit=6")

    println()
    println("=" ^ 80)
    println("Summary: p_pos(acc, b) for nqubit=8")
    println("=" ^ 80)

    print_summary_table(res8, acc8, b_vals, "nqubit=8")

    # --- Cross-N comparison at shared accuracy ---
    shared_accs = intersect(acc6, acc8)
    if !isempty(shared_accs)
        println()
        println("=" ^ 80)
        println("Cross-N comparison at shared accuracies")
        println("Ratio p(N=6) / p(N=8) for each (acc, b)")
        println("If ratio ~1.0: frustration function is universal in (acc,b)")
        println("If ratio != 1.0: N-dependent structural effect")
        println("=" ^ 80)
        @printf("  %-6s  %-6s  %-10s  %-10s  %-10s\n",
                "acc", "b", "p(N=6)", "p(N=8)", "ratio")
        println("  " * repeat("-", 48))
        for acc in sort(collect(shared_accs))
            for b in b_vals
                k6 = findb(b, b_vals, res6, acc)
                k8 = findb(b, b_vals, res8, acc)
                p6, p8 = k6[1], k8[1]
                ratio = (p8 > 0) ? p6 / p8 : NaN
                @printf("  %-6.2f  %-6.2f  %-10.4f  %-10.4f  %-10.3f\n",
                        acc, b, p6, p8, ratio)
            end
            println()
        end
    end
end

function findb(b_target, b_vals, results, acc)
    # Find closest b entry in results dict
    best_key = nothing; best_dist = Inf
    for (key, val) in results
        acc_k, b_k = key
        if abs(acc_k - acc) < 0.02
            dist = abs(b_k - b_target)
            if dist < best_dist; best_dist = dist; best_key = key; end
        end
    end
    return best_key === nothing ? (NaN, NaN) : results[best_key]
end

function print_summary_table(results, acc_vals, b_vals, label)
    # Header
    print("  b \\ acc  ")
    for acc in acc_vals; @printf("  %-7.2f", acc); end
    println()
    println("  " * repeat("-", 10 + 9 * length(acc_vals)))

    for b in reverse(b_vals)
        @printf("  %-8.2f", b)
        for acc in acc_vals
            # Find closest entry
            best_p = NaN; best_dist = Inf
            for (key, val) in results
                acc_k, b_k = key
                if abs(acc_k - acc) < 0.02 && abs(b_k - b) < 0.06
                    dist = abs(acc_k - acc) + abs(b_k - b)
                    if dist < best_dist; best_dist = dist; best_p = val[1]; end
                end
            end
            if isnan(best_p)
                @printf("  %-7s", "---")
            else
                @printf("  %-7.4f", best_p)
            end
        end
        println()
    end
end

main()
