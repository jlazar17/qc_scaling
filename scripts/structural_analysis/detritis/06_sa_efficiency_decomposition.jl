# Definitive test: decompose SA efficiency for H=0 vs H=1.
#
# The 2x convergence ratio persists even with the shuffled pairing (no
# both-covered pairs), so the cause must be something about H=0 vs H=1 goal
# types themselves, not about specific pair definitions.
#
# We measure three things per SA step:
#   1. Quality of the proposed state (accuracy delta if accepted)
#   2. Fraction of steps where the proposed state is actually accepted
#   3. The per-step accuracy gain (= quality × acceptance)
#
# If H=0 shows 2x higher per-step gain, that's the rate cause.
# We further split: is the 2x from better proposals, higher acceptance, or both?
#
# We use the SHUFFLED pairing to isolate the effect from both-covered pairs.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Shuffled pairing utilities (copied from script 05)
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
# Instrumented SA: record proposal quality and acceptance per step
# ---------------------------------------------------------------------------

struct SATrace
    proposed_deltas  ::Vector{Float64}  # accuracy delta of proposed state (+ or -)
    accepted         ::Vector{Bool}     # was the proposal accepted?
    picker_match_frac::Vector{Float64}  # best match fraction of the proposed state
end

function run_sa_instrumented(goal, nqubit, nstate, nsteps, alpha_cool,
                              companion, goal_idx, fingerprint,
                              acc_fn;   # function (rep_sum, rep_ctr) -> Float64
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

    # T0 calibration
    current_acc = acc_fn(rep_sum, rep_ctr)
    bad_deltas=Float64[]
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

    proposed_deltas   = Float64[]
    accepted_flags    = Bool[]
    picker_match_fracs = Float64[]

    # Precompute fingerprint in dense form for match fraction calculation
    fp_dense = QCScaling.Fingerprint(nqubit)

    for step in 1:nsteps
        which=rand(rng,1:nstate)
        gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
        theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
        cxt=QCScaling.Context(gen,base_cxt)
        alphas=pick_alphas_s(cxt,goal,rep,fingerprint,base_cxt,companion,goal_idx,n_rep)
        ns=QCScaling.PseudoGHZState(alphas...,gen)

        # Measure picker match fraction
        cg=[companion_goal_s(po,goal,rep,companion,goal_idx,n_rep) for po in cxt.pos]
        n_valid=count(!isnan,cg)
        best_match=0
        if n_valid>0
            parity_idx=theta_s+1; nalpha=size(fp_dense.a,4); ntz=size(fp_dense.a,3); npos2=size(fp_dense.a,1)
            for ai in 1:nalpha, tzi in 1:ntz
                m=sum(!isnan(cg[pi]) && fp_dense.a[pi,parity_idx,tzi,ai]==round(Int,cg[pi]) for pi in 1:npos2)
                best_match=max(best_match,m)
            end
        end
        push!(picker_match_fracs, n_valid>0 ? best_match/n_valid : NaN)

        fill_state_cache!(scratch_idxs,scratch_pars,ns,cxt_master)
        apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
        apply_state_cached!(rep_sum,rep_ctr,scratch_idxs,scratch_pars,1)
        new_acc=acc_fn(rep_sum,rep_ctr)
        delta=new_acc-current_acc

        push!(proposed_deltas, delta)

        accept = delta>=0 || rand(rng)<exp(delta/T)
        push!(accepted_flags, accept)

        if accept
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

    return SATrace(proposed_deltas, accepted_flags, picker_match_fracs)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit=6; n=3^nqubit; ngbits=(n-1)÷2
    nstate=50; nsteps=20_000; alpha_cool=0.9999

    println("Building shuffled pairing...")
    companion,goal_idx,npairs=build_shuffled_pairing(nqubit)
    @assert npairs==ngbits
    fingerprint=FingerprintPacked(QCScaling.Fingerprint(nqubit))

    rng_goals=Random.MersenneTwister(42)
    n_trials=5

    println("nqubit=$nqubit, nstate=$nstate, nsteps=$nsteps")
    println()

    for H_target in [0.0, 1.0]
        goal_ones=round(Int, H_target*ngbits/2)
        all_proposed=Float64[]; all_accepted_delta=Float64[]
        all_accept_rate=Float64[]; all_picker=Float64[]
        all_pos_proposed=Float64[]  # fraction of proposals with delta>0

        for trial in 1:n_trials
            goal=Random.shuffle!(rng_goals, vcat(ones(Int,goal_ones),zeros(Int,ngbits-goal_ones)))
            acc_fn=(rs,rc)->rep_accuracy_shuffled(rs,rc,goal,companion,goal_idx)
            trace=run_sa_instrumented(goal,nqubit,nstate,nsteps,alpha_cool,
                                       companion,goal_idx,fingerprint,acc_fn; seed=trial)

            # Proposed deltas (what the state would contribute if accepted)
            append!(all_proposed, trace.proposed_deltas)

            # Accepted deltas (actual contribution)
            accepted_d=trace.proposed_deltas[trace.accepted]
            append!(all_accepted_delta, accepted_d)

            # Acceptance rate
            push!(all_accept_rate, mean(trace.accepted))

            # Picker match fraction
            valid_picker=trace.picker_match_frac[.!isnan.(trace.picker_match_frac)]
            append!(all_picker, valid_picker)

            # Fraction of proposals with positive delta
            push!(all_pos_proposed, mean(trace.proposed_deltas .> 0))
        end

        @printf("H=%.2f:\n", H_target)
        @printf("  Proposed delta:     mean=%+.6f  std=%.6f\n",
                mean(all_proposed), std(all_proposed))
        @printf("  Pos proposal rate:  %.4f  (frac of proposals with delta>0)\n",
                mean(all_pos_proposed))
        @printf("  Acceptance rate:    %.4f\n", mean(all_accept_rate))
        @printf("  Accepted delta:     mean=%+.6f  (avg gain per accepted step)\n",
                mean(all_accepted_delta))
        @printf("  Picker best_frac:   mean=%.4f  std=%.4f\n",
                mean(all_picker), std(all_picker))
        total_gain_per_step = mean(all_accepted_delta) * mean(all_accept_rate)
        @printf("  Total gain/step:    %.8f  (accepted_delta × accept_rate)\n",
                total_gain_per_step)
        println()
    end

    println("="^65)
    println("The ratio of total gain/step (H=0 / H=1) should be ~2x")
    println("if that explains the 2x nstate ratio.")
    println("="^65)
    println()

    # Also run with ORIGINAL pairing for comparison
    println("For comparison: same analysis with ORIGINAL (consecutive) pairing")
    println()

    for H_target in [0.0, 1.0]
        goal_ones=round(Int, H_target*ngbits/2)
        all_proposed=Float64[]; all_accepted_delta=Float64[]
        all_accept_rate=Float64[]; all_picker=Float64[]
        all_pos_proposed=Float64[]

        # Use the standard pick_new_alphas and rep_accuracy_fast
        fingerprint2=FingerprintPacked(QCScaling.Fingerprint(nqubit))
        fp_dense2=QCScaling.Fingerprint(nqubit)

        for trial in 1:n_trials
            rng=Random.MersenneTwister(trial*7+13)
            goal=Random.shuffle!(rng, vcat(ones(Int,goal_ones),zeros(Int,ngbits-goal_ones)))

            # Inline instrumented SA with original pairing
            cxt_master=QCScaling.ContextMaster(nqubit)
            npos=length(cxt_master.base_even.pos)
            ensemble=[QCScaling.random_state(nqubit) for _ in 1:nstate]
            cache_idxs=[Vector{Int}(undef,npos) for _ in 1:nstate]
            cache_pars=[Vector{Int}(undef,npos) for _ in 1:nstate]
            for i in 1:nstate; fill_state_cache!(cache_idxs[i],cache_pars[i],ensemble[i],cxt_master); end
            rep_sum=zeros(Int,n); rep_ctr=zeros(Int,n)
            for i in 1:nstate; apply_state_cached!(rep_sum,rep_ctr,cache_idxs[i],cache_pars[i],1); end
            rep=rep_from_cache(rep_sum,rep_ctr)
            current_acc=rep_accuracy_fast(rep_sum,rep_ctr,goal)

            bad_deltas=Float64[]
            scratch_idxs2=Vector{Int}(undef,npos); scratch_pars2=Vector{Int}(undef,npos)
            for _ in 1:500
                which=rand(rng,1:nstate)
                gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
                theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
                cxt=QCScaling.Context(gen,base_cxt)
                alphas=QCScaling.pick_new_alphas(cxt,goal,rep,fingerprint2,base_cxt)
                ns=QCScaling.PseudoGHZState(alphas...,gen)
                apply_state!(rep_sum,rep_ctr,ensemble[which],cxt_master,-1)
                apply_state!(rep_sum,rep_ctr,ns,cxt_master,1)
                delta=rep_accuracy_fast(rep_sum,rep_ctr,goal)-current_acc
                apply_state!(rep_sum,rep_ctr,ns,cxt_master,-1)
                apply_state!(rep_sum,rep_ctr,ensemble[which],cxt_master,1)
                delta<0 && push!(bad_deltas,abs(delta))
            end
            T=isempty(bad_deltas) ? 0.1 : -mean(bad_deltas)/log(0.8)
            current_acc=rep_accuracy_fast(rep_sum,rep_ctr,goal)

            prop_d=Float64[]; acc_d=Float64[]; acc_flags=Bool[]; picker_f=Float64[]
            for step in 1:nsteps
                which=rand(rng,1:nstate)
                gen_idx=rand(rng,0:n-1); gen=QCScaling.ParityOperator(gen_idx,nqubit)
                theta_s=rand(rng,0:1); base_cxt=theta_s==0 ? cxt_master.base_even : cxt_master.base_odd
                cxt=QCScaling.Context(gen,base_cxt)
                alphas=QCScaling.pick_new_alphas(cxt,goal,rep,fingerprint2,base_cxt)
                ns=QCScaling.PseudoGHZState(alphas...,gen)

                # Picker match
                cg=QCScaling.companion_goal(cxt,goal,rep)
                n_valid=count(!isnan,cg)
                best_match=0
                if n_valid>0
                    parity_idx=theta_s+1; nalpha=size(fp_dense2.a,4); ntz=size(fp_dense2.a,3); npos2=size(fp_dense2.a,1)
                    for ai in 1:nalpha, tzi in 1:ntz
                        m=sum(!isnan(cg[pi]) && fp_dense2.a[pi,parity_idx,tzi,ai]==round(Int,cg[pi]) for pi in 1:npos2)
                        best_match=max(best_match,m)
                    end
                    push!(picker_f, best_match/n_valid)
                end

                fill_state_cache!(scratch_idxs2,scratch_pars2,ns,cxt_master)
                apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],-1)
                apply_state_cached!(rep_sum,rep_ctr,scratch_idxs2,scratch_pars2,1)
                new_acc=rep_accuracy_fast(rep_sum,rep_ctr,goal)
                delta=new_acc-current_acc
                push!(prop_d,delta)
                accept=delta>=0||rand(rng)<exp(delta/T)
                push!(acc_flags,accept)
                if accept
                    update_rep_at_cached!(rep,rep_sum,rep_ctr,cache_idxs[which])
                    update_rep_at_cached!(rep,rep_sum,rep_ctr,scratch_idxs2)
                    copy!(cache_idxs[which],scratch_idxs2); copy!(cache_pars[which],scratch_pars2)
                    ensemble[which]=ns; current_acc=new_acc
                    push!(acc_d,delta)
                else
                    apply_state_cached!(rep_sum,rep_ctr,scratch_idxs2,scratch_pars2,-1)
                    apply_state_cached!(rep_sum,rep_ctr,cache_idxs[which],cache_pars[which],1)
                end
                T*=alpha_cool
            end
            append!(all_proposed,prop_d); append!(all_accepted_delta,acc_d)
            push!(all_accept_rate,mean(acc_flags)); append!(all_picker,picker_f)
            push!(all_pos_proposed,mean(prop_d.>0))
        end

        @printf("ORIG H=%.2f:\n", H_target)
        @printf("  Proposed delta:     mean=%+.6f\n", mean(all_proposed))
        @printf("  Pos proposal rate:  %.4f\n", mean(all_pos_proposed))
        @printf("  Acceptance rate:    %.4f\n", mean(all_accept_rate))
        @printf("  Accepted delta:     mean=%+.6f\n", mean(all_accepted_delta))
        @printf("  Picker best_frac:   mean=%.4f\n", mean(all_picker))
        total_gain=mean(all_accepted_delta)*mean(all_accept_rate)
        @printf("  Total gain/step:    %.8f\n", total_gain)
        println()
    end
end

main()
