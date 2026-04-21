# p_pos checkpoint sweep for all H values in the scaling study (nqubit=6).
#
# Uses the exact k_ones values from the scaling study HDF5 file:
#
#   H=0.000  k=0    n*(obs)=40
#   H=0.250  k=15   n*(obs)=52
#   H=0.500  k=40   n*(obs)=68
#   H=0.625  k=57   n*(obs)=56  (anomalous — keep but flag)
#   H=0.750  k=78   n*(obs)=79
#   H=0.875  k=107  n*(obs)=84
#   H=1.000  k=182  n*(obs)=104
#
# For each (k_ones, nstate): run SA, probe p_pos each time accuracy crosses a
# checkpoint. Then compute cumulative difficulty D(acc) = ∫ da/p and compare
# predicted vs observed n*.
#
# Key test: at the observed n*(H), is D(acc_natural, n*(H)) approximately
# constant across all H? If yes, the p_pos model fully explains the scaling.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Random
using Statistics
using Printf

include("../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Probe p_pos (non-destructive)
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
    # results: (ckpt, actual_acc, p_pos)
    results = Tuple{Float64,Float64,Float64}[]
    probe_seed = seed + 100_000

    for _ in 1:nsteps
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
    return results  # Vector of (ckpt, actual_acc, p_pos) for each checkpoint hit
end

# ---------------------------------------------------------------------------
# Trapezoidal cumulative difficulty D(acc) = ∫ da / p(a)
# Returns D evaluated at each checkpoint, given a list of (acc, p) pairs.
# Pairs must be sorted by acc ascending. Gaps with p=0 give D=Inf.
# ---------------------------------------------------------------------------

function cumulative_difficulty(acc_p_pairs)
    # acc_p_pairs: sorted Vector of (acc, p) at each checkpoint hit
    isempty(acc_p_pairs) && return Float64[]
    D = Float64[]
    cumD = 0.0
    prev_acc, prev_p = acc_p_pairs[1]
    push!(D, 0.0)  # D at first checkpoint is 0 (starting point)
    for i in 2:length(acc_p_pairs)
        acc_i, p_i = acc_p_pairs[i]
        da = acc_i - prev_acc
        avg_p = (prev_p + p_i) / 2
        if avg_p <= 0
            cumD = Inf
        else
            cumD += da / avg_p
        end
        push!(D, cumD)
        prev_acc, prev_p = acc_i, p_i
    end
    return D
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nqubit     = 6
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2   # 364

    alpha_cool = 0.9999
    n_probes   = 1500
    n_trials   = 4
    nsteps     = 300_000

    # Checkpoints: probe at each accuracy crossing
    checkpoints = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.93]

    # nstate values: bracket all observed n*(H) values and saturation n* values
    # Observed n*(H): 40, 52, 68, 56, 79, 84, 104  (efficiency n* from HDF5)
    # Saturation n*:  ~84 (H=0), ~176 (H=1)
    nstate_vals = [30, 40, 52, 68, 79, 84, 104, 130, 176]

    # H values from scaling study with exact k_ones and observed n* (efficiency)
    H_entries = [
        (label="H=0.000", k_ones=0,   nstar_obs=40),
        (label="H=0.250", k_ones=15,  nstar_obs=52),
        (label="H=0.500", k_ones=40,  nstar_obs=68),
        (label="H=0.625", k_ones=57,  nstar_obs=56,  anomalous=true),
        (label="H=0.750", k_ones=78,  nstar_obs=79),
        (label="H=0.875", k_ones=107, nstar_obs=84),
        (label="H=1.000", k_ones=182, nstar_obs=104),
    ]

    println("Building shuffled pairing for nqubit=$nqubit (ngbits=$ngbits)...")
    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    println("  done.\n")

    println("=" ^ 80)
    println("p_pos checkpoint sweep — nqubit=$nqubit, all H values from scaling study")
    println("n_probes=$n_probes, n_trials=$n_trials, nsteps=$nsteps")
    println("=" ^ 80)
    println()

    # all_data[H_label][nstate][ckpt] = Vector{Float64} of p_pos across trials
    all_data = Dict{String, Dict{Int, Dict{Float64, Vector{Float64}}}}()

    for entry in H_entries
        label   = entry.label
        k_ones  = entry.k_ones
        anomalous = get(entry, :anomalous, false)

        all_data[label] = Dict()
        @printf("\n--- %s  (k_ones=%d, n*_obs=%d%s) ---\n",
                label, k_ones, entry.nstar_obs,
                anomalous ? " ANOMALOUS" : "")
        @printf("  %-6s  %-8s  %-6s  %-8s\n", "nstate", "ckpt", "n_hit", "mean_p")
        println("  " * repeat("-", 35))

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

            # Print first and last checkpoints hit for this (label, nstate)
            first_hit = nothing; last_hit = nothing
            for c in checkpoints
                v = all_data[label][nstate][c]
                isempty(v) && continue
                first_hit === nothing && (first_hit = c)
                last_hit = c
            end
            if first_hit === nothing
                @printf("  %-6d  %-8s  %-6s\n", nstate, "none", "---")
            else
                for c in checkpoints
                    v = all_data[label][nstate][c]
                    isempty(v) && continue
                    @printf("  %-6d  %-8.2f  %-6d  %-8.4f\n",
                            nstate, c, length(v), mean(v))
                end
            end
        end
    end

    # =========================================================================
    # Analysis: cumulative difficulty D and comparison to observed n*
    # =========================================================================

    println()
    println("=" ^ 80)
    println("Analysis: cumulative difficulty D(acc) and comparison to observed n*")
    println("=" ^ 80)

    # For each (H, nstate): compute mean p_pos at each checkpoint, then D integral
    # D_table[H_label][nstate] = (checkpoints_hit, D_values, last_ckpt, D_last)
    D_table = Dict{String, Dict{Int, Any}}()

    for entry in H_entries
        label = entry.label
        D_table[label] = Dict()
        for nstate in nstate_vals
            acc_p = Tuple{Float64,Float64}[]
            for c in checkpoints
                v = all_data[label][nstate][c]
                isempty(v) && continue
                push!(acc_p, (c, mean(v)))
            end
            if length(acc_p) < 2
                D_table[label][nstate] = nothing
                continue
            end
            D_vals = cumulative_difficulty(acc_p)
            last_c, last_p = acc_p[end]
            D_table[label][nstate] = (acc_p=acc_p, D_vals=D_vals,
                                       last_ckpt=last_c, D_total=D_vals[end])
        end
    end

    # Table 1: highest checkpoint reached by each (H, nstate)
    println()
    println("Highest accuracy checkpoint reached (all $n_trials trials hit it):")
    print("  H \\ nstate   ")
    for ns in nstate_vals; @printf("  %-5d", ns); end; println()
    println("  " * repeat("-", 14 + 7*length(nstate_vals)))
    for entry in H_entries
        label = entry.label
        @printf("  %-12s", label)
        for ns in nstate_vals
            d = D_table[label][ns]
            if d === nothing
                @printf("  %-5s", "---")
            else
                last_ckpt = d.last_ckpt
                # Only show if all 4 trials hit this checkpoint
                v = all_data[label][ns][last_ckpt]
                @printf("  %-5.2f", last_ckpt)
            end
        end
        println()
    end

    # Table 2: D_total at each (H, nstate) — the cumulative difficulty up to last checkpoint
    println()
    println("Cumulative difficulty D = ∫ da/p from acc=0.70 to last checkpoint:")
    print("  H \\ nstate   ")
    for ns in nstate_vals; @printf("  %-7d", ns); end; println()
    println("  " * repeat("-", 14 + 9*length(nstate_vals)))
    for entry in H_entries
        label = entry.label
        @printf("  %-12s", label)
        for ns in nstate_vals
            d = D_table[label][ns]
            if d === nothing || isinf(d.D_total) || isnan(d.D_total)
                @printf("  %-7s", "---")
            else
                @printf("  %-7.3f", d.D_total)
            end
        end
        println()
    end

    # Table 3: p_pos at acc=0.90 by (H, nstate) — the rate near convergence
    println()
    println("p_pos at acc checkpoint 0.90:")
    print("  H \\ nstate   ")
    for ns in nstate_vals; @printf("  %-7d", ns); end; println()
    println("  " * repeat("-", 14 + 9*length(nstate_vals)))
    for entry in H_entries
        label = entry.label
        @printf("  %-12s", label)
        for ns in nstate_vals
            v = all_data[label][ns][0.90]
            isempty(v) ? @printf("  %-7s", "---") : @printf("  %-7.4f", mean(v))
        end
        println()
    end

    # Table 4: at the OBSERVED n*(H), what is p_pos at acc=0.90 and D_total?
    println()
    println("=" ^ 80)
    println("At observed n*(H): p_pos and D values")
    println("If D is constant across H: the p_pos model fully explains the scaling")
    println("=" ^ 80)
    @printf("  %-12s  %-6s  %-8s  %-8s  %-10s  %-10s\n",
            "H", "n*obs", "last_ckpt", "p@0.90", "D_total", "D/nstar")
    println("  " * repeat("-", 75))

    D_at_nstar = Float64[]
    for entry in H_entries
        label    = entry.label
        nstar    = entry.nstar_obs
        anomalous = get(entry, :anomalous, false)

        # Find closest nstate_val to nstar
        closest_ns = nstate_vals[argmin(abs.(nstate_vals .- nstar))]
        d = D_table[label][closest_ns]

        p_at_090 = let v = all_data[label][closest_ns][0.90]
            isempty(v) ? NaN : mean(v)
        end

        if d === nothing
            @printf("  %-12s  %-6d  %-8s  %-8s  %-10s  %-10s%s\n",
                    label, nstar, "---", "---", "---", "---",
                    anomalous ? " *" : "")
        else
            push!(D_at_nstar, d.D_total)
            @printf("  %-12s  %-6d  %-8.2f  %-8.4f  %-10.3f  %-10.4f%s\n",
                    label, nstar, d.last_ckpt,
                    isnan(p_at_090) ? 0.0 : p_at_090,
                    d.D_total, d.D_total / closest_ns,
                    anomalous ? " *" : "")
        end
    end

    # Table 5: predicted n* from the integral model
    # Model: n*(H) = C * D(acc*, b(H), n*(H))
    # Calibrate C from H=0 entry, then predict for others
    println()
    println("=" ^ 80)
    println("Predicted n*(H) from integral model: n* = C × D(acc*, b, n*)")
    println("Calibrated on H=0, then extrapolated")
    println("=" ^ 80)

    # Get H=0 D_total at n*(H=0)=40, closest nstate=40
    d_H0 = D_table["H=0.000"][40]
    if d_H0 !== nothing && !isinf(d_H0.D_total)
        C = 40.0 / d_H0.D_total
        @printf("  Calibration: n*(H=0)=40, D(H=0, nstate=40)=%.3f → C=%.3f\n\n",
                d_H0.D_total, C)

        @printf("  %-12s  %-6s  %-8s  %-10s  %-10s  %-8s\n",
                "H", "n*obs", "n*pred", "D_total", "ratio_pred", "error%%")
        println("  " * repeat("-", 60))

        for entry in H_entries
            label    = entry.label
            nstar    = entry.nstar_obs
            anomalous = get(entry, :anomalous, false)

            # Use D at the closest nstate to nstar
            closest_ns = nstate_vals[argmin(abs.(nstate_vals .- nstar))]
            d = D_table[label][closest_ns]
            d === nothing && continue

            nstar_pred = C * d.D_total
            err = 100 * (nstar_pred - nstar) / nstar
            @printf("  %-12s  %-6d  %-8.1f  %-10.3f  %-10.3f  %-8.1f%s\n",
                    label, nstar, nstar_pred, d.D_total, nstar_pred/40.0, err,
                    anomalous ? " *" : "")
        end
        println("\n  * H=0.625 is anomalous in the scaling study (statistical noise at N=6)")
    else
        println("  Cannot calibrate: H=0 at nstate=40 has no D data.")
    end

    println()
    println("=" ^ 80)
    println("Summary: does the p_pos model explain the n*(H) scaling?")
    println("D/n* constant → yes; D/n* varies with H → additional effects at play")
    println("=" ^ 80)
end

main()
