# p_pos vs nstate sweep — accuracy checkpoint probes across H values.
#
# For each (H, nstate): run SA and probe p_pos each time accuracy first crosses
# a checkpoint in {0.70, 0.75, 0.80, 0.85, 0.90}. This gives p_pos(acc, H, nstate)
# while controlling for accuracy — allowing direct comparison across H and nstate.
#
# H values: {0, 0.25, 0.5, 0.75, 1.0} using exact k_ones from the scaling study.
# nqubit=6, shuffled pairing.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Probe p_pos from current ensemble state (non-destructive)
# ---------------------------------------------------------------------------

function probe_ppos(ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars,
                    cxt_master, goal, companion, goal_idx, fingerprint,
                    nqubit, n_probes, rng_seed)
    rng = Random.MersenneTwister(rng_seed)
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
        new_acc - base_acc > 0 && (pos_count += 1)
    end
    return pos_count / n_probes
end

# ---------------------------------------------------------------------------
# SA run with accuracy checkpoint probes
# Returns: Vector of (checkpoint_acc, actual_acc, p_pos) for each checkpoint hit
# ---------------------------------------------------------------------------

function run_sa_with_checkpoints(goal, nqubit, nstate, nsteps, alpha_cool,
                                  companion, goal_idx, fingerprint,
                                  checkpoints, n_probes; seed=42)
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

    checkpoints_remaining = sort(checkpoints)
    results = Tuple{Float64,Float64,Float64}[]  # (ckpt, actual_acc, p_pos)
    probe_seed = seed + 100_000

    for step in 1:nsteps
        while !isempty(checkpoints_remaining) && current_acc >= checkpoints_remaining[1]
            ckpt = popfirst!(checkpoints_remaining)
            p = probe_ppos(ensemble, rep_sum, rep_ctr, rep, cache_idxs, cache_pars,
                           cxt_master, goal, companion, goal_idx, fingerprint,
                           nqubit, n_probes, probe_seed)
            probe_seed += 1
            push!(results, (ckpt, current_acc, p))
        end
        isempty(checkpoints_remaining) && break

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
    return results
end

# ---------------------------------------------------------------------------
# Compute k_ones for a target binary entropy H in (0,1) given ngbits.
# Solves H(p) = H_target via bisection, returns round(Int, p * ngbits).
# ---------------------------------------------------------------------------

function h_to_kones(H_target, ngbits)
    H_target == 0.0 && return 0
    H_target == 1.0 && return ngbits ÷ 2
    lo, hi = 0.0, 0.5
    for _ in 1:60
        p = (lo + hi) / 2
        h = -p*log2(p) - (1-p)*log2(1-p)
        h < H_target ? (lo = p) : (hi = p)
    end
    return round(Int, (lo + hi)/2 * ngbits)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    alpha_cool = 0.9999
    n_probes   = 1500
    n_trials   = 4

    checkpoints = [0.70, 0.75, 0.80, 0.85, 0.90]

    H_targets = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Per-nqubit configuration
    configs = [
        (nqubit=6, nsteps=300_000,
         nstate_vals=[30, 40, 55, 70, 84, 104, 130, 176]),
        (nqubit=8, nsteps=500_000,
         nstate_vals=[100, 130, 175, 230, 300, 400, 500]),
    ]

    for cfg in configs
        nqubit      = cfg.nqubit
        nsteps      = cfg.nsteps
        nstate_vals = cfg.nstate_vals
        n           = 3^nqubit
        ngbits      = (n-1) ÷ 2

        H_entries = [(label=@sprintf("H=%.2f", H), k_ones=h_to_kones(H, ngbits))
                     for H in H_targets]

        println()
        println("=" ^ 80)
        @printf("nqubit=%d  ngbits=%d  nsteps=%d  n_probes=%d  n_trials=%d\n",
                nqubit, ngbits, nsteps, n_probes, n_trials)
        for e in H_entries
            @printf("  %s  k_ones=%d\n", e.label, e.k_ones)
        end
        println("=" ^ 80)

        println("Building shuffled pairing...")
        companion, goal_idx, _ = build_shuffled_pairing(nqubit)
        fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
        println("  done.\n")

        # all_data[H_label][nstate][ckpt] -> Vector{Float64} of p_pos values across trials
        all_data = Dict{String, Dict{Int, Dict{Float64, Vector{Float64}}}}()

        for entry in H_entries
            label  = entry.label
            k_ones = entry.k_ones
            all_data[label] = Dict()

            @printf("\n--- %s (k_ones=%d) ---\n", label, k_ones)

            for nstate in nstate_vals
                all_data[label][nstate] = Dict(c => Float64[] for c in checkpoints)

                for trial in 1:n_trials
                    seed = trial * 137 + nstate * 31 + k_ones * 7
                    rng  = Random.MersenneTwister(seed + 7)
                    goal = Random.shuffle!(rng, vcat(ones(Int, k_ones), zeros(Int, ngbits - k_ones)))

                    results = run_sa_with_checkpoints(
                        goal, nqubit, nstate, nsteps, alpha_cool,
                        companion, goal_idx, fingerprint,
                        checkpoints, n_probes; seed=seed)

                    for (ckpt, _actual_acc, p) in results
                        push!(all_data[label][nstate][ckpt], p)
                    end
                end

                # Per-nstate summary
                hit_ckpts = [c for c in checkpoints if !isempty(all_data[label][nstate][c])]
                if isempty(hit_ckpts)
                    @printf("  nstate=%-4d  (no checkpoints reached)\n", nstate)
                else
                    last_c = hit_ckpts[end]
                    v_last = all_data[label][nstate][last_c]
                    @printf("  nstate=%-4d  last_ckpt=%.2f  p_pos=%.4f  (n_trials_hit=%d)\n",
                            nstate, last_c, mean(v_last), length(v_last))
                end
            end
        end

        # ---------------------------------------------------------------------------
        # Summary tables: p_pos(acc, nstate) for each H
        # ---------------------------------------------------------------------------
        println()
        println("=" ^ 80)
        @printf("Summary (nqubit=%d): p_pos by (accuracy checkpoint, nstate) for each H\n", nqubit)
        println("Rows = accuracy checkpoint, Cols = nstate")
        println("=" ^ 80)

        for entry in H_entries
            label = entry.label
            @printf("\n%s:\n", label)
            print("  acc   ")
            for ns in nstate_vals; @printf("  %-8d", ns); end; println()
            println("  " * repeat("-", 8 + 10*length(nstate_vals)))
            for c in checkpoints
                @printf("  %.2f  ", c)
                for ns in nstate_vals
                    v = all_data[label][ns][c]
                    isempty(v) ? @printf("  %-8s", "---") : @printf("  %-8.4f", mean(v))
                end
                println()
            end
        end

        # ---------------------------------------------------------------------------
        # Cross-H comparison at each accuracy checkpoint
        # Rows = H, Cols = nstate
        # ---------------------------------------------------------------------------
        println()
        println("=" ^ 80)
        @printf("Cross-H comparison (nqubit=%d) at each accuracy checkpoint\n", nqubit)
        println("=" ^ 80)

        for c in checkpoints
            @printf("\nacc=%.2f:\n", c)
            print("  H       ")
            for ns in nstate_vals; @printf("  %-8d", ns); end; println()
            println("  " * repeat("-", 8 + 10*length(nstate_vals)))
            for entry in H_entries
                label = entry.label
                @printf("  %-8s", label)
                for ns in nstate_vals
                    v = all_data[label][ns][c]
                    isempty(v) ? @printf("  %-8s", "---") : @printf("  %-8.4f", mean(v))
                end
                println()
            end
        end

    end  # for cfg
end

main()
