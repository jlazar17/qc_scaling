# nqubit=8 synthetic frustration sweep at nstate=200.
# All function definitions self-contained (copied from 09_synthetic_frustration.jl).
# Prior run with nstate=100 failed to converge for almost all (acc, b) at nqubit=8.
# nstate=200 is above n*(N=8, H=0)≈126, so the SA should actually make progress.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Shuffled pairing utilities
# ---------------------------------------------------------------------------

function build_shuffled_pairing(nqubit)
    n = 3^nqubit
    betas = [QCScaling.to_ternary(k-1, Val(nqubit)) for k in 1:n-1]
    matched = falses(n-1); pairs = Tuple{Int,Int}[]
    for k1 in 1:n-1
        matched[k1] && continue
        for k2 in k1+1:n-1
            matched[k2] && continue
            if isodd(count(i -> betas[k1][i] != betas[k2][i], 1:nqubit))
                push!(pairs, (k1, k2)); matched[k1] = matched[k2] = true; break
            end
        end
    end
    companion = zeros(Int, n); goal_idx = zeros(Int, n)
    for (gi, (k1, k2)) in enumerate(pairs)
        companion[k1]=k2; companion[k2]=k1; goal_idx[k1]=gi; goal_idx[k2]=gi
    end
    return companion, goal_idx, length(pairs)
end

function rep_accuracy_shuffled(rep_sum, rep_ctr, goal, companion, goal_idx)
    s = 0; n = length(goal)
    for k1 in 1:length(rep_sum)-1
        companion[k1]==0 && continue; k1>companion[k1] && continue
        k2=companion[k1]; c1=rep_ctr[k1]; c2=rep_ctr[k2]
        s1=rep_sum[k1]; s2=rep_sum[k2]
        valid=(c1>0)&(c2>0)&(2*s1!=c1)&(2*s2!=c2)
        r1=2*s1>c1; r2=2*s2>c2; j=goal_idx[k1]
        s += valid & ((r1 ⊻ r2) == !iszero(goal[j]))
    end
    return s / n
end

function companion_goal_s(po, goal, rep, companion, goal_idx, n_rep)
    k=po.index; (k<=0||k>n_rep) && return NaN
    comp=companion[k]; comp==0 && return NaN; isnan(rep[comp]) && return NaN
    j=goal_idx[k]; j==0 && return NaN
    return goal[j]==1 ? 1-rep[comp] : rep[comp]
end

function pick_alphas_s(cxt, goal, rep, fp, base_cxt, companion, goal_idx, n_rep)
    cg = [companion_goal_s(po, goal, rep, companion, goal_idx, n_rep) for po in cxt.pos]
    valid, vals = QCScaling._pack_companion_goal(cg, fp.nwords)
    parity_idx=base_cxt.parity+1; nwords=fp.nwords; nalpha=size(fp.words,4); ntz=size(fp.words,3)
    best_score=typemax(Int); best_tz=1; best_alpha=1
    @inbounds for ai in 1:nalpha, tzi in 1:ntz
        score=0
        for w in 1:nwords; score+=count_ones(xor(fp.words[w,parity_idx,tzi,ai],vals[w])&valid[w]); end
        if score<best_score; best_score=score; best_tz=tzi; best_alpha=ai; end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

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
# Probe landscape
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
# Sweep
# ---------------------------------------------------------------------------

function run_sweep(nqubit, acc_targets, b_vals, n_trials, nstate, max_steps, alpha_cool, n_probes)
    n = 3^nqubit; ngbits = (n-1) ÷ 2

    println("Building shuffled pairing for nqubit=$nqubit...")
    companion, goal_idx, npairs = build_shuffled_pairing(nqubit)
    @assert npairs == ngbits
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    println("  ngbits=$ngbits, done.\n")

    results = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()

    for b_target in b_vals
        k_ones = round(Int, (1.0 - b_target) * ngbits)
        b_actual = 1.0 - k_ones / ngbits
        @printf("--- b_target=%.2f (k_ones=%d, b_actual=%.4f) ---\n", b_target, k_ones, b_actual)
        @printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
                "target", "actual_acc", "p_pos", "bal_meas", "converged?")
        println("  " * repeat("-", 55))

        for target_acc in acc_targets
            pos_rates = Float64[]; actual_accs = Float64[]; bal_measured = Float64[]

            for trial in 1:n_trials
                seed_sa  = trial * 137 + round(Int, b_target * 1000) + round(Int, target_acc * 10000)
                seed_prb = trial * 997 + round(Int, b_target * 1000) + round(Int, target_acc * 10000)
                local_rng = Random.MersenneTwister(seed_sa + 7)
                goal = Random.shuffle!(local_rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

                ens, rs, rc, rep, ci, cp, actual_acc, cxt_m =
                    build_ensemble_to_accuracy(goal, nqubit, nstate, target_acc, max_steps,
                                               alpha_cool, companion, goal_idx, fingerprint;
                                               seed=seed_sa)

                actual_acc < target_acc - 0.03 && continue

                p = probe_landscape(ens, rs, rc, rep, ci, cp, cxt_m, goal,
                                    companion, goal_idx, fingerprint, nqubit, n_probes;
                                    seed=seed_prb)
                n_w, bal = measure_balance(rs, rc, goal, companion, goal_idx)
                push!(pos_rates, p); push!(actual_accs, actual_acc)
                push!(bal_measured, isnan(bal) ? 0.5 : bal)
            end

            n_conv = length(pos_rates)
            if n_conv == 0
                @printf("  %-8.2f  %-10s  %-10s  %-10s  %d/%d\n",
                        target_acc, "N/A", "N/A", "N/A", 0, n_trials)
                continue
            end
            mean_p = mean(pos_rates); std_p = length(pos_rates) > 1 ? std(pos_rates) : 0.0
            @printf("  %-8.2f  %-10.4f  %-10.4f  %-10.4f  %d/%d\n",
                    target_acc, mean(actual_accs), mean_p, mean(bal_measured), n_conv, n_trials)
            results[(target_acc, b_actual)] = (mean_p, std_p)
        end
        println()
    end
    return results, ngbits
end

function print_summary_table(results, acc_vals, b_vals)
    print("  b \\ acc  ")
    for acc in acc_vals; @printf("  %-7.2f", acc); end
    println()
    println("  " * repeat("-", 10 + 9 * length(acc_vals)))
    for b in reverse(b_vals)
        @printf("  %-8.2f", b)
        for acc in acc_vals
            best_p = NaN; best_dist = Inf
            for (key, val) in results
                acc_k, b_k = key
                if abs(acc_k - acc) < 0.02 && abs(b_k - b) < 0.06
                    dist = abs(acc_k - acc) + abs(b_k - b)
                    if dist < best_dist; best_dist = dist; best_p = val[1]; end
                end
            end
            isnan(best_p) ? @printf("  %-7s", "---") : @printf("  %-7.4f", best_p)
        end
        println()
    end
end

function findb(b_target, results, acc)
    best_key = nothing; best_dist = Inf
    for (key, _) in results
        acc_k, b_k = key
        if abs(acc_k - acc) < 0.02
            dist = abs(b_k - b_target)
            if dist < best_dist; best_dist = dist; best_key = key; end
        end
    end
    return best_key === nothing ? (NaN, NaN) : results[best_key]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    alpha_cool = 0.9999
    n_probes   = 1500
    n_trials   = 3
    nstate     = 200
    acc8       = [0.70, 0.75, 0.80, 0.85]
    max_steps  = 400_000
    b_vals     = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    println("=" ^ 80)
    println("nqubit=8, nstate=200 synthetic frustration sweep")
    println("n_probes=$n_probes, n_trials=$n_trials per cell")
    println("=" ^ 80)
    println()

    res8, _ = run_sweep(8, acc8, b_vals, n_trials, nstate, max_steps, alpha_cool, n_probes)

    println()
    println("=" ^ 80)
    println("Summary: p_pos(acc, b) — nqubit=8, nstate=200")
    println("=" ^ 80)
    print_summary_table(res8, acc8, b_vals)

    # Hardcoded N=6 results from prior run (nstate=100)
    p6 = Dict(
        (0.70, 0.50) => 0.1913, (0.70, 0.60) => 0.1711, (0.70, 0.70) => 0.1696,
        (0.70, 0.80) => 0.2087, (0.70, 0.90) => 0.2560, (0.70, 1.00) => 0.2420,
        (0.75, 0.50) => NaN,    (0.75, 0.60) => NaN,    (0.75, 0.70) => NaN,
        (0.75, 0.80) => NaN,    (0.75, 0.90) => NaN,    (0.75, 1.00) => NaN,
        (0.80, 0.50) => 0.0389, (0.80, 0.60) => 0.0453, (0.80, 0.70) => 0.0440,
        (0.80, 0.80) => 0.0764, (0.80, 0.90) => 0.1480, (0.80, 1.00) => 0.1962,
        (0.85, 0.50) => 0.0102, (0.85, 0.60) => 0.0109, (0.85, 0.70) => 0.0140,
        (0.85, 0.80) => 0.0338, (0.85, 0.90) => 0.0707, (0.85, 1.00) => 0.2158,
    )

    println()
    println("=" ^ 80)
    println("Cross-N comparison: N=6 (nstate=100) vs N=8 (nstate=200)")
    println("If p(N=6)/p(N=8) ≈ 1: frustration function is universal in (acc, b)")
    println("If ratio >> 1: N-dependent structural scaling beyond (acc, b)")
    println("=" ^ 80)
    @printf("  %-6s  %-6s  %-10s  %-10s  %-10s\n", "acc", "b", "p(N=6)", "p(N=8)", "ratio6/8")
    println("  " * repeat("-", 50))
    for acc in acc8
        for b in b_vals
            p8_val = findb(b, res8, acc)[1]
            p6_val = get(p6, (acc, b), NaN)
            ratio = (!isnan(p6_val) && !isnan(p8_val) && p8_val > 0) ? p6_val / p8_val : NaN
            @printf("  %-6.2f  %-6.2f  %-10.4f  %-10.4f  %-10.4f\n",
                    acc, b,
                    isnan(p6_val) ? 0.0 : p6_val,
                    isnan(p8_val) ? 0.0 : p8_val,
                    ratio)
        end
        println()
    end
end

main()
