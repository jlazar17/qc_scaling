# Landscape frustration test.
#
# Script 07 showed: at acc in [0.85, 0.95), H=0 accepts 39% of proposals
# while H=1 accepts only 0.7% — a 57x ratio. The mean positive delta per
# accepted step is essentially the same for both (1.0-1.2x ratio), so the
# bottleneck is finding ANY proposal with delta > 0.
#
# This script measures the positive proposal rate (p_pos) directly:
#   1. Build a near-converged ensemble at a target accuracy
#   2. Sample many random proposed states and count those with delta > 0
#   3. Compare H=0 vs H=1 across accuracy levels to characterize the collapse

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Build optimized ensemble reaching target accuracy
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
    for _ in 1:200
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
# Probe landscape: sample N random proposals, count positive deltas
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

    pos_count = 0; zero_count = 0; neg_count = 0
    deltas = Float64[]

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
        delta=new_acc-acc_fn(rep_sum,rep_ctr)  # compare before/after
        # restore
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,-1)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
        # recompute delta correctly
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,1)
        new_acc=acc_fn(rep_sum,rep_ctr)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,-1)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
        current_acc=acc_fn(rep_sum,rep_ctr)
        delta=new_acc-current_acc
        push!(deltas, delta)
        delta>0 ? pos_count+=1 : delta==0 ? zero_count+=1 : neg_count+=1
    end
    return pos_count/n_probes, mean(deltas[deltas.>0]; init=NaN), deltas
end

# Simpler, cleaner version that correctly measures proposal delta
function probe_landscape_clean(ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars,
                                cxt_master, goal, companion, goal_idx, fingerprint,
                                nqubit, n_probes; seed=999)
    rng = Random.MersenneTwister(seed)
    n = 3^nqubit; n_rep = n-1
    npos = length(cxt_master.base_even.pos)
    scratch_idxs=Vector{Int}(undef,npos); scratch_pars=Vector{Int}(undef,npos)
    nstate = length(ensemble)

    acc_fn(rs,rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    base_acc = acc_fn(rep_sum, rep_ctr)

    pos_count = 0; neg_count = 0
    pos_deltas = Float64[]

    for _ in 1:n_probes
        which=rand(rng,1:nstate)
        gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
        theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
        cxt=QCScaling.Context(gen,base_cxt)
        alphas=pick_alphas_s(cxt,goal,rep,fingerprint,base_cxt,companion,goal_idx,n_rep)
        ns=QCScaling.PseudoGHZState(alphas...,gen)
        fill_state_cache!(scratch_idxs,scratch_pars,ns,cxt_master)
        # Temporarily swap out state 'which' for ns
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,1)
        new_acc=acc_fn(rep_sum,rep_ctr)
        # Restore
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,-1)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
        delta=new_acc-base_acc
        if delta>0; pos_count+=1; push!(pos_deltas,delta)
        else; neg_count+=1; end
    end
    return pos_count/n_probes, isempty(pos_deltas) ? NaN : mean(pos_deltas)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit=6; n=3^nqubit; ngbits=(n-1)÷2
    nstate=100; alpha_cool=0.9999; max_steps=200_000
    n_probes=2000
    n_trials=6

    println("Building shuffled pairing...")
    companion,goal_idx,npairs=build_shuffled_pairing(nqubit)
    @assert npairs==ngbits
    fingerprint=FingerprintPacked(QCScaling.Fingerprint(nqubit))

    @printf("nqubit=%d, ngbits=%d, nstate=%d, n_probes=%d per snapshot\n\n",
            nqubit, ngbits, nstate, n_probes)

    println("="^85)
    println("Landscape frustration test")
    println("Prediction: H=0 pos_proposal_rate >> H=1 at high accuracy.")
    println("="^85)
    println()

    target_accs = [0.70, 0.80, 0.88, 0.93]

    for H_target in [0.0, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        @printf("--- H=%.2f (goal_ones=%d) ---\n", H_target, goal_ones)
        @printf("  %-8s  %-10s  %-10s  %-12s\n",
                "target", "actual_acc", "pos_rate", "mean_pos_d")
        println("  " * repeat("-", 45))

        for target_acc in target_accs
            pos_rates = Float64[]; mean_deltas = Float64[]

            for trial in 1:n_trials
                rng_t = Random.MersenneTwister(trial*31 + round(Int, H_target*100) + round(Int, target_acc*1000))
                goal = Random.shuffle!(rng_t, vcat(ones(Int,goal_ones),zeros(Int,ngbits-goal_ones)))
                ens, rs, rc, rep, ci, cp, actual_acc, cxt_master =
                    build_ensemble_to_accuracy(goal, nqubit, nstate, target_acc, max_steps,
                                               alpha_cool, companion, goal_idx, fingerprint;
                                               seed=trial*200+round(Int,H_target*50))

                actual_acc < target_acc - 0.03 && continue  # didn't converge far enough

                pos_r, mean_d = probe_landscape_clean(ens, rs, rc, rep, ci, cp,
                                                       cxt_master, goal, companion, goal_idx,
                                                       fingerprint, nqubit, n_probes;
                                                       seed=trial*300)

                push!(pos_rates, pos_r); push!(mean_deltas, isnan(mean_d) ? 0.0 : mean_d)
            end

            isempty(pos_rates) && continue
            @printf("  %-8.2f  %-10.4f  %-10.4f  %-12.6f\n",
                    target_acc, target_acc,
                    mean(pos_rates), mean(filter(!isnan, mean_deltas)))
        end
        println()
    end

    println("="^85)
    println("Key comparison: positive proposal rate at target accuracies")
    println("If H=1/H=0 ratio decreases (H=0 always finds proposals, H=1 runs out)")
    println("that confirms landscape frustration.")
    println("="^85)
    println()

    # Collect summary for comparison
    summary = Dict{Tuple{Float64,Float64}, Float64}()  # (H, target) -> pos_rate
    for H_target in [0.0, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        for target_acc in target_accs
            pos_rates = Float64[]
            for trial in 1:n_trials
                rng_t = Random.MersenneTwister(trial*31 + round(Int, H_target*100) + round(Int, target_acc*1000))
                goal = Random.shuffle!(rng_t, vcat(ones(Int,goal_ones),zeros(Int,ngbits-goal_ones)))
                ens, rs, rc, rep, ci, cp, actual_acc, cxt_master =
                    build_ensemble_to_accuracy(goal, nqubit, nstate, target_acc, max_steps,
                                               alpha_cool, companion, goal_idx, fingerprint;
                                               seed=trial*200+round(Int,H_target*50))
                actual_acc < target_acc - 0.03 && continue
                pos_r, _ = probe_landscape_clean(ens, rs, rc, rep, ci, cp,
                                                  cxt_master, goal, companion, goal_idx,
                                                  fingerprint, nqubit, n_probes;
                                                  seed=trial*300)
                push!(pos_rates, pos_r)
            end
            if !isempty(pos_rates)
                summary[(H_target, target_acc)] = mean(pos_rates)
            end
        end
    end

    @printf("%-8s  %-8s  %-12s  %-10s\n",
            "target", "H", "pos_rate", "H0/H1 rate")
    println(repeat("-", 45))
    for target_acc in target_accs
        h0 = get(summary, (0.0, target_acc), NaN)
        h1 = get(summary, (1.0, target_acc), NaN)
        ratio = (isnan(h0)||isnan(h1)||h1==0) ? NaN : h0/h1
        @printf("%-8.2f  H=0     %-12.4f\n", target_acc, h0)
        @printf("%-8s  H=1     %-12.4f  ratio=%.2f\n", "", h1, ratio)
        println()
    end
end

main()
