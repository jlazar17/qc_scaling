using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using HDF5
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    return shuffle!(rng, vcat(ones(Int, k), zeros(Int, ngbits - k)))
end

function hamming_entropy(k::Int, N::Int)
    (k == 0 || k == N) && return 0.0
    p = k / N; return -p * log2(p) - (1 - p) * log2(1 - p)
end

function k_from_entropy(H::Float64, N::Int)
    H <= 0.0 && return 0; H >= 1.0 && return N ÷ 2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), 0:N÷2)
    return (0:N÷2)[idx]
end

# ---------------------------------------------------------------------------
# Rep helpers
# ---------------------------------------------------------------------------

function apply_state!(rep_sum, rep_ctr, state, cxt_master, sign)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        p = QCScaling.parity(state, derived_po)
        rep_sum[derived_po.index] += sign * p
        rep_ctr[derived_po.index] += sign
    end
end

function rep_accuracy_fast(rep_sum, rep_ctr, goal)
    s = 0
    @inbounds for i in eachindex(goal)
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1]/c1; p2 = rep_sum[i2]/c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr; pref[pref .== 0.5] .= NaN; return round.(pref)
end

function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]
        rep[i] = c == 0 ? NaN : (v = rep_sum[i]/c; v == 0.5 ? NaN : round(v))
    end
end

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble); current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    bad_deltas = Float64[]
    for _ in 1:n_samples
        which = rand(rng, 1:nstate)
        ns    = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
        delta = rep_accuracy_fast(rep_sum, rep_ctr, goal) - current_acc
        apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
        delta < 0 && push!(bad_deltas, abs(delta))
    end
    isempty(bad_deltas) && return 0.1
    return -mean(bad_deltas) / log(target_rate)
end

# ---------------------------------------------------------------------------
# Goal-aware SA with acceptance-rate stopping criterion
#
# Stops early when the rolling acceptance rate over the last `acc_window`
# steps falls below `min_accept_rate`. Returns (best_acc, steps_used).
# ---------------------------------------------------------------------------

function run_gsa_with_stopping(goal, nqubit, nstate, nsteps_max, alpha;
                               min_accept_rate=0.001, acc_window=10_000,
                               ensemble=nothing, rep_sum=nothing,
                               rep_ctr=nothing, rep=nothing, rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit

    if isnothing(ensemble)
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n)
        rep_ctr  = zeros(Int,     n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)
    end

    T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                     cxt_master, fingerprint; rng=rng)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    best_acc    = current_acc

    # Circular buffer for rolling acceptance rate
    accept_buf  = zeros(Bool, acc_window)
    n_accepted  = 0
    steps_used  = 0

    for step in 1:nsteps_max
        which     = rand(rng, 1:nstate)
        ns        = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
        old_state = ensemble[which]

        apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
        new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        delta   = new_acc - current_acc

        accepted = delta >= 0 || rand(rng) < exp(delta / T)
        if accepted
            update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
            update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
            ensemble[which] = ns
            current_acc     = new_acc
            new_acc > best_acc && (best_acc = new_acc)
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end

        # Update rolling acceptance rate buffer
        buf_idx = mod1(step, acc_window)
        n_accepted += Int(accepted) - Int(accept_buf[buf_idx])
        accept_buf[buf_idx] = accepted

        T *= alpha
        steps_used = step

        # Check stopping criterion once the window is full
        if step >= acc_window && n_accepted / acc_window < min_accept_rate
            break
        end
    end

    return best_acc, steps_used
end

# ---------------------------------------------------------------------------
# Goal-aware SA, n_restarts independent runs (for ceiling comparison)
# Returns (best_acc, total_steps_used)
# ---------------------------------------------------------------------------

function run_gsa(goal, nqubit, nstate, nsteps_max, alpha;
                 n_restarts=1, min_accept_rate=0.001, acc_window=10_000, rng)
    best_acc    = -Inf
    total_steps = 0
    for _ in 1:n_restarts
        acc, steps = run_gsa_with_stopping(goal, nqubit, nstate, nsteps_max, alpha;
                                           min_accept_rate=min_accept_rate,
                                           acc_window=acc_window, rng=rng)
        acc > best_acc && (best_acc = acc)
        total_steps += steps
    end
    return best_acc, total_steps
end

# ---------------------------------------------------------------------------
# PT warm-up (fixed temperature, no stopping — runs full warmup budget)
# ---------------------------------------------------------------------------

function run_pt_warmup(goal, nqubit, nstate, nsteps_warmup;
                       n_chains=3, T_ratio=20.0, swap_every=500, rng)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit

    ensembles = [[QCScaling.random_state(nqubit) for _ in 1:nstate] for _ in 1:n_chains]
    rep_sums  = [zeros(Float64, n) for _ in 1:n_chains]
    rep_ctrs  = [zeros(Int,     n) for _ in 1:n_chains]
    reps      = [zeros(Float64, n) for _ in 1:n_chains]

    for c in 1:n_chains
        for s in ensembles[c]; apply_state!(rep_sums[c], rep_ctrs[c], s, cxt_master, 1); end
        reps[c] = rep_from_cache(rep_sums[c], rep_ctrs[c])
    end

    T_min = calibrate_T0(ensembles[1], rep_sums[1], rep_ctrs[1], reps[1], goal, nqubit,
                          cxt_master, fingerprint; rng=rng)
    T_min = max(T_min, 1e-6)
    T_max = T_min * T_ratio
    temps = [T_min * (T_max / T_min)^((c-1) / (n_chains-1)) for c in 1:n_chains]

    current_accs = [rep_accuracy_fast(rep_sums[c], rep_ctrs[c], goal) for c in 1:n_chains]

    for step in 1:nsteps_warmup
        for c in 1:n_chains
            which     = rand(rng, 1:nstate)
            ns        = smart_proposal(nqubit, reps[c], goal, fingerprint, cxt_master, rng)
            old_state = ensembles[c][which]

            apply_state!(rep_sums[c], rep_ctrs[c], old_state, cxt_master, -1)
            apply_state!(rep_sums[c], rep_ctrs[c], ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sums[c], rep_ctrs[c], goal)
            delta   = new_acc - current_accs[c]

            if delta >= 0 || rand(rng) < exp(delta / temps[c])
                update_rep_at!(reps[c], rep_sums[c], rep_ctrs[c], old_state, cxt_master)
                update_rep_at!(reps[c], rep_sums[c], rep_ctrs[c], ns,        cxt_master)
                ensembles[c][which] = ns
                current_accs[c]     = new_acc
            else
                apply_state!(rep_sums[c], rep_ctrs[c], ns,        cxt_master, -1)
                apply_state!(rep_sums[c], rep_ctrs[c], old_state, cxt_master,  1)
            end
        end

        if step % swap_every == 0
            for c in 1:n_chains-1
                log_prob = (current_accs[c+1] - current_accs[c]) * (1/temps[c] - 1/temps[c+1])
                if log_prob >= 0 || rand(rng) < exp(log_prob)
                    ensembles[c],    ensembles[c+1]    = ensembles[c+1],    ensembles[c]
                    rep_sums[c],     rep_sums[c+1]     = rep_sums[c+1],     rep_sums[c]
                    rep_ctrs[c],     rep_ctrs[c+1]     = rep_ctrs[c+1],     rep_ctrs[c]
                    reps[c],         reps[c+1]          = reps[c+1],         reps[c]
                    current_accs[c], current_accs[c+1] = current_accs[c+1], current_accs[c]
                end
            end
        end
    end

    return [(deepcopy(ensembles[c]), copy(rep_sums[c]), copy(rep_ctrs[c]), copy(reps[c]))
            for c in 1:n_chains]
end

# ---------------------------------------------------------------------------
# Hybrid: PT warmup → goal-SA annealing with stopping criterion per chain
# Returns (best_acc, total_steps_used)
# ---------------------------------------------------------------------------

function run_hybrid(goal, nqubit, nstate, nsteps_max, alpha;
                    n_chains=3, T_ratio=20.0, swap_every=500,
                    warmup_frac=1/3, min_accept_rate=0.001, acc_window=10_000, rng)
    nsteps_warmup = round(Int, nsteps_max * warmup_frac)
    nsteps_refine = nsteps_max - nsteps_warmup

    chain_states = run_pt_warmup(goal, nqubit, nstate, nsteps_warmup;
                                 n_chains=n_chains, T_ratio=T_ratio,
                                 swap_every=swap_every, rng=rng)

    best_acc    = -Inf
    total_steps = n_chains * nsteps_warmup   # warmup cost
    for (ensemble, rep_sum, rep_ctr, rep) in chain_states
        acc, steps = run_gsa_with_stopping(goal, nqubit, nstate, nsteps_refine, alpha;
                                           min_accept_rate=min_accept_rate,
                                           acc_window=acc_window,
                                           ensemble=ensemble, rep_sum=rep_sum,
                                           rep_ctr=rep_ctr, rep=rep, rng=rng)
        acc > best_acc && (best_acc = acc)
        total_steps += steps
    end

    return best_acc, total_steps
end

# ---------------------------------------------------------------------------
# Main — 100 seeds each at n=8, H=0 and H=1
# ---------------------------------------------------------------------------

function main()
    nqubit      = 8
    nstate_mult = 3
    nsteps_max  = 2_000_000
    alpha       = 0.999
    nseeds      = 100
    base_seed   = 1234    # different from full_comparison to get fresh seeds
    H_targets   = [0.0, 1.0]

    min_accept_rate = 0.001   # stop when <0.1% acceptance over window
    acc_window      = 10_000

    base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
    nstate      = nstate_mult * base_nstate
    ngbits      = (3^nqubit - 1) ÷ 2

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "ceiling_comparison.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    println("nqubit=$nqubit  nstate=$nstate  nsteps_max=$nsteps_max  alpha=$alpha")
    println("stopping: accept_rate < $min_accept_rate over last $acc_window steps")
    println("nseeds=$nseeds\n")

    h5open(outfile, "w") do h5f
    HDF5.attributes(h5f)["nqubit"]          = nqubit
    HDF5.attributes(h5f)["nstate"]          = nstate
    HDF5.attributes(h5f)["nsteps_max"]      = nsteps_max
    HDF5.attributes(h5f)["alpha"]           = alpha
    HDF5.attributes(h5f)["min_accept_rate"] = min_accept_rate
    HDF5.attributes(h5f)["acc_window"]      = acc_window
    HDF5.attributes(h5f)["nseeds"]          = nseeds

    for H_target in H_targets
        k     = k_from_entropy(H_target, ngbits)
        H_act = hamming_entropy(k, ngbits)
        rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000) + 7777)
        goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

        gsa_accs    = Float64[]
        hybrid_accs = Float64[]
        gsa_steps   = Int[]
        hybrid_steps_used = Int[]

        println("="^70)
        @printf("H=%.1f (k=%d, H_act=%.4f)\n", H_target, k, H_act)
        @printf("%-6s  %-10s  %-10s  %-12s  %-12s\n",
                "seed", "gsa", "hybrid", "gsa_steps", "hybrid_steps")
        println("-"^60)

        for (si, seed) in enumerate(seeds)
            rng_s = Random.MersenneTwister(seed)

            g_acc, g_steps = run_gsa(goals[si], nqubit, nstate, nsteps_max, alpha;
                                     n_restarts=1, min_accept_rate=min_accept_rate,
                                     acc_window=acc_window, rng=rng_s)

            rng_h = Random.MersenneTwister(seed)
            h_acc, h_steps = run_hybrid(goals[si], nqubit, nstate, nsteps_max, alpha;
                                        n_chains=3, warmup_frac=1/3,
                                        min_accept_rate=min_accept_rate,
                                        acc_window=acc_window, rng=rng_h)

            push!(gsa_accs,          g_acc)
            push!(hybrid_accs,       h_acc)
            push!(gsa_steps,         g_steps)
            push!(hybrid_steps_used, h_steps)

            @printf("%-6d  %-10.4f  %-10.4f  %-12d  %-12d  best: gsa=%.4f  hyb=%.4f\n",
                    si, g_acc, h_acc, g_steps, h_steps,
                    maximum(gsa_accs), maximum(hybrid_accs))
            flush(stdout)
        end

        @printf("\nFINAL  median: gsa=%.4f  hybrid=%.4f\n", median(gsa_accs), median(hybrid_accs))
        @printf("       best:   gsa=%.4f  hybrid=%.4f\n", maximum(gsa_accs), maximum(hybrid_accs))
        @printf("       mean steps used: gsa=%.0f  hybrid=%.0f\n\n",
                mean(gsa_steps), mean(hybrid_steps_used))

        key = @sprintf("H%.1f_k%d", H_target, k)
        gp  = create_group(h5f, key)
        HDF5.attributes(gp)["H_target"] = H_target
        HDF5.attributes(gp)["H_actual"] = H_act
        HDF5.attributes(gp)["k"]        = k
        HDF5.attributes(gp)["ngbits"]   = ngbits
        gp["gsa_accs"]         = gsa_accs
        gp["hybrid_accs"]      = hybrid_accs
        gp["gsa_steps"]        = gsa_steps
        gp["hybrid_steps"]     = hybrid_steps_used
    end

    end  # h5open
    println("Saved to $outfile")
end

main()
