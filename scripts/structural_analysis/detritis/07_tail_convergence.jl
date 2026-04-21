# Definitive test: measure per-wrong-pair fix rate at different accuracy levels.
#
# Hypothesis: the 2.1x convergence ratio (H=0 needs 84 states, H=1 needs 176)
# comes from TAIL BEHAVIOR near acc=1.0, not from early SA efficiency.
#
# Mechanism: when accuracy is high (few wrong pairs left):
#   - H=0: remaining wrong pairs all want the SAME correction. A single state
#     covering them can fix all of them simultaneously.
#   - H=1: remaining wrong pairs are BALANCED (half want agreement, half disagreement).
#     A single state can only fix pairs matching its theta_z choice → half at most.
#
# This tail effect gets WORSE as you approach acc=1.0, because the mix of
# wrong pairs for H=1 stays balanced throughout.
#
# Test: measure "wrong pairs fixed per accepted step" at accuracy buckets
#   [0.50, 0.70), [0.70, 0.85), [0.85, 0.95), [0.95, 1.0)
# for H=0 vs H=1. If H=1's ratio degrades near the tail, that confirms the mechanism.
#
# Uses SHUFFLED pairing to isolate from both-covered pair effects.

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
        for w in 1:nwords
            score+=count_ones(xor(fp.words[w,parity_idx,tzi,ai],vals[w])&valid[w])
        end
        if score<best_score; best_score=score; best_tz=tzi; best_alpha=ai; end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Instrumented SA: record stats bucketed by current accuracy
# ---------------------------------------------------------------------------

# For each accepted step: record
#   - the current accuracy before the step
#   - delta (gain from this step)
#   - how many wrong pairs it fixed (delta * ngbits, since acc = correct/ngbits)

function run_sa_phase_bucketed(goal, nqubit, nstate, nsteps, alpha_cool,
                                companion, goal_idx, fingerprint;
                                seed=42)
    rng = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n = 3^nqubit; ngbits = length(goal); n_rep = n-1
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
    for _ in 1:500
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

    # Bucket boundaries: [0.50, 0.70, 0.85, 0.95, 1.01)
    bucket_edges = [0.50, 0.70, 0.85, 0.95, 1.01]
    nbuckets = length(bucket_edges) - 1

    # Store deltas (when accepted and delta>0) per bucket
    pos_deltas_per_bucket = [Float64[] for _ in 1:nbuckets]
    steps_per_bucket      = zeros(Int, nbuckets)   # total steps in bucket
    accepted_per_bucket   = zeros(Int, nbuckets)

    for step in 1:nsteps
        # Determine current bucket
        bucket = searchsortedlast(bucket_edges, current_acc)
        bucket = clamp(bucket, 1, nbuckets)
        steps_per_bucket[bucket] += 1

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

        accept = delta>=0 || rand(rng)<exp(delta/T)

        if accept
            if delta > 0
                push!(pos_deltas_per_bucket[bucket], delta)
            end
            accepted_per_bucket[bucket] += 1
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

    return pos_deltas_per_bucket, steps_per_bucket, accepted_per_bucket, current_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit=6; n=3^nqubit; ngbits=(n-1)÷2
    nstate=100; nsteps=60_000; alpha_cool=0.9999

    println("Building shuffled pairing...")
    companion,goal_idx,npairs=build_shuffled_pairing(nqubit)
    @assert npairs==ngbits
    fingerprint=FingerprintPacked(QCScaling.Fingerprint(nqubit))

    n_trials=8
    bucket_edges = [0.50, 0.70, 0.85, 0.95, 1.01]
    bucket_labels = ["[0.50,0.70)", "[0.70,0.85)", "[0.85,0.95)", "[0.95,1.00)"]
    nbuckets = 4

    @printf("nqubit=%d, nstate=%d, nsteps=%d, n_trials=%d\n\n",
            nqubit, nstate, nsteps, n_trials)

    println("="^80)
    println("Per-phase SA efficiency: H=0 vs H=1")
    println("Key metric: mean positive delta per accepted step, bucketed by current accuracy")
    println("Prediction: H=0/H=1 delta ratio INCREASES near acc=1.0 (tail effect)")
    println("="^80)
    println()

    results = Dict{Float64, Vector{Vector{Float64}}}()  # H -> per-bucket positive deltas

    for H_target in [0.0, 1.0]
        goal_ones = round(Int, H_target * ngbits / 2)
        rng_goals = Random.MersenneTwister(H_target == 0.0 ? 1234 : 5678)

        all_pos_deltas  = [Float64[] for _ in 1:nbuckets]
        all_steps       = zeros(Int, nbuckets)
        all_accepted    = zeros(Int, nbuckets)
        final_accs      = Float64[]

        Threads.@threads for trial in 1:n_trials
            local_rng = Random.MersenneTwister(trial * 17 + round(Int, H_target*100))
            goal = Random.shuffle!(local_rng, vcat(ones(Int,goal_ones),zeros(Int,ngbits-goal_ones)))
            pd, steps, acc, final_acc = run_sa_phase_bucketed(goal, nqubit, nstate, nsteps, alpha_cool,
                                                               companion, goal_idx, fingerprint;
                                                               seed=trial*100+round(Int,H_target*77))
            # Thread-safe accumulation
            for b in 1:nbuckets
                append!(all_pos_deltas[b], pd[b])
            end
            all_steps .+= steps
            all_accepted .+= acc
            push!(final_accs, final_acc)
        end

        results[H_target] = all_pos_deltas

        @printf("H=%.2f (goal_ones=%d/%d), final acc: mean=%.4f max=%.4f\n",
                H_target, goal_ones, ngbits, mean(final_accs), maximum(final_accs))
        @printf("  %-18s  %8s  %8s  %8s  %12s  %12s\n",
                "acc bucket", "#steps", "#accept", "acc_rate", "mean_pos_delta", "wrong_fixed/step")
        println("  " * repeat("-", 75))
        for b in 1:nbuckets
            n_pd = length(all_pos_deltas[b])
            mean_d = n_pd > 0 ? mean(all_pos_deltas[b]) : NaN
            # wrong_fixed_per_step = mean_pos_delta * ngbits (pairs fixed per accepted step)
            # normalized by total steps in this bucket
            wrong_fixed_per_step = (n_pd > 0 && all_steps[b] > 0) ?
                                    sum(all_pos_deltas[b]) * ngbits / all_steps[b] : NaN
            @printf("  %-18s  %8d  %8d  %8.4f  %12.6f  %12.6f\n",
                    bucket_labels[b],
                    all_steps[b], all_accepted[b],
                    all_steps[b] > 0 ? all_accepted[b]/all_steps[b] : NaN,
                    mean_d,
                    wrong_fixed_per_step)
        end
        println()
    end

    println("="^80)
    println("Ratio H=0/H=1: mean positive delta per accepted step (per bucket)")
    println("If ratio is ~1.0 early but ~2.0 late, tail effect is the explanation.")
    println("="^80)
    h0_pds = results[0.0]; h1_pds = results[1.0]
    @printf("%-18s  %10s  %10s  %10s\n", "acc bucket", "H=0 mean_d", "H=1 mean_d", "H0/H1 ratio")
    println(repeat("-", 55))
    for b in 1:nbuckets
        d0 = length(h0_pds[b]) > 0 ? mean(h0_pds[b]) : NaN
        d1 = length(h1_pds[b]) > 0 ? mean(h1_pds[b]) : NaN
        ratio = (isnan(d0)||isnan(d1)||d1==0) ? NaN : d0/d1
        @printf("%-18s  %10.6f  %10.6f  %10.3f\n", bucket_labels[b], d0, d1, ratio)
    end
end

main()
