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
# Goal generation (fixed Hamming weight)
# ---------------------------------------------------------------------------

function goal_from_hamming(k::Int, ngbits::Int, rng)
    @assert 0 <= k <= ngbits
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
    ks = 0:N÷2
    _, idx = findmin(k -> abs(hamming_entropy(k, N) - H), ks)
    return ks[idx]
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
        i1 = 2i - 1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        p1 = rep_sum[i1] / c1; p2 = rep_sum[i2] / c2
        (p1 == 0.5 || p2 == 0.5) && continue
        s += (abs(round(p1) - round(p2)) == Float64(goal[i])) ? 1 : 0
    end
    return s / length(goal)
end

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function update_rep_at!(rep, rep_sum, rep_ctr, state, cxt_master)
    base_cxt = state.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    for base_po in base_cxt.pos
        derived_po = state.generator + base_po
        i = derived_po.index; c = rep_ctr[i]
        rep[i] = c == 0 ? NaN : (v = rep_sum[i] / c; v == 0.5 ? NaN : round(v))
    end
end

function smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
    gen_idx   = rand(rng, 0:3^nqubit - 1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

function calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                      cxt_master, fingerprint; target_rate=0.8, n_samples=500, rng)
    nstate = length(ensemble)
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
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
# Goal-aware SA — returns (best_acc, final_ensemble, final_rep)
# ---------------------------------------------------------------------------

function run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                           n_restarts=3, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    n           = 3^nqubit
    best_acc    = -Inf
    best_ensemble = Vector{QCScaling.PseudoGHZState}()
    best_rep      = Float64[]

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
        rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
        for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint; rng=rng)
        current_acc  = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best = current_acc

        for _ in 1:nsteps
            which     = rand(rng, 1:nstate)
            ns        = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)
            old_state = ensemble[which]

            apply_state!(rep_sum, rep_ctr, old_state, cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master,  1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at!(rep, rep_sum, rep_ctr, old_state, cxt_master)
                update_rep_at!(rep, rep_sum, rep_ctr, ns,        cxt_master)
                ensemble[which] = ns
                current_acc     = new_acc
                new_acc > restart_best && (restart_best = new_acc)
            else
                apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
                apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
            end
            T *= alpha
        end

        if restart_best > best_acc
            best_acc      = restart_best
            best_ensemble = copy(ensemble)
            best_rep      = rep_from_cache(rep_sum, rep_ctr)
        end
    end

    return best_acc, best_ensemble, best_rep
end

# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------

state_key(s) = (s.theta_s, s.theta_z, Tuple(s.alphas), Tuple(s.generator.βs))

function jaccard(states_a, states_b)
    sa = Set(state_key(s) for s in states_a)
    sb = Set(state_key(s) for s in states_b)
    isempty(sa) && isempty(sb) && return 1.0
    return length(intersect(sa, sb)) / length(union(sa, sb))
end

function rep_similarity(rep_a, rep_b)
    mask = .!isnan.(rep_a) .& .!isnan.(rep_b)
    sum(mask) == 0 && return NaN
    return mean(rep_a[mask] .== rep_b[mask])
end

# ---------------------------------------------------------------------------
# Analysis for one (nqubit, goal) combination
# ---------------------------------------------------------------------------

function analyze_convergence(goal, nqubit, nstate, nsteps, alpha, nseeds;
                             n_restarts=3, base_seed=42)
    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    accs       = Float64[]
    all_ens    = Vector{Vector{QCScaling.PseudoGHZState}}()
    all_reps   = Vector{Vector{Float64}}()

    for seed in seeds
        acc, ens, rep = run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                                          n_restarts=n_restarts, seed=seed)
        push!(accs, acc)
        push!(all_ens, ens)
        push!(all_reps, rep)
    end

    max_acc  = maximum(accs)
    top_mask = accs .>= max_acc - 1e-9
    top_idx  = findall(top_mask)
    n        = length(accs)

    jac_all = Float64[]; jac_top = Float64[]
    rs_all  = Float64[]; rs_top  = Float64[]

    for i in 1:n, j in (i+1):n
        jv = jaccard(all_ens[i], all_ens[j])
        rv = rep_similarity(all_reps[i], all_reps[j])
        push!(jac_all, jv); push!(rs_all, rv)
        if top_mask[i] && top_mask[j]
            push!(jac_top, jv); push!(rs_top, rv)
        end
    end

    return (accs=accs, max_acc=max_acc, n_top=sum(top_mask),
            jac_all=jac_all, jac_top=jac_top,
            rep_sim_all=rs_all, rep_sim_top=rs_top)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    configs = [
        # (nqubit, nstate_mult, sa_nsteps, sa_alpha)
        (6, 3,    500_000, 0.999),
        (8, 3,  2_000_000, 0.999),
    ]
    entropy_vals = [0.0, 1.0]
    nseeds       = 50
    n_restarts   = 3
    base_seed    = 42
    outdir       = joinpath(@__DIR__, "data")
    mkpath(outdir)

    outfile = joinpath(outdir, "convergence_gsa.h5")
    h5open(outfile, "w") do h5f

    for (nqubit, nstate_mult, sa_nsteps, sa_alpha) in configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        nstate      = nstate_mult * base_nstate
        ngbits      = (3^nqubit - 1) ÷ 2

        gp_nq = create_group(h5f, "nqubit_$(nqubit)")
        HDF5.attributes(gp_nq)["nqubit"]    = nqubit
        HDF5.attributes(gp_nq)["nstate"]    = nstate
        HDF5.attributes(gp_nq)["sa_nsteps"] = sa_nsteps
        HDF5.attributes(gp_nq)["sa_alpha"]  = sa_alpha
        HDF5.attributes(gp_nq)["nseeds"]    = nseeds

        for H_target in entropy_vals
            k        = k_from_entropy(H_target, ngbits)
            H_actual = hamming_entropy(k, ngbits)
            rng_g    = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goal     = goal_from_hamming(k, ngbits, rng_g)

            @printf("nqubit=%d  nstate=%d  H=%.2f (k=%d, H_act=%.4f)  nseeds=%d  ... ",
                    nqubit, nstate, H_target, k, H_actual, nseeds)
            flush(stdout)

            r = analyze_convergence(goal, nqubit, nstate, sa_nsteps, sa_alpha, nseeds;
                                    n_restarts=n_restarts, base_seed=base_seed)

            @printf("max_acc=%.4f  n_top=%d/%d  jac_top=%.3f±%.3f  rep_sim_top=%.3f±%.3f\n",
                r.max_acc, r.n_top, nseeds,
                isempty(r.jac_top) ? NaN : mean(r.jac_top),
                isempty(r.jac_top) ? NaN : std(r.jac_top),
                isempty(r.rep_sim_top) ? NaN : mean(r.rep_sim_top),
                isempty(r.rep_sim_top) ? NaN : std(r.rep_sim_top))
            flush(stdout)

            key = @sprintf("H%.2f_k%d", H_target, k)
            gp  = create_group(gp_nq, key)
            HDF5.attributes(gp)["H_target"] = H_target
            HDF5.attributes(gp)["H_actual"] = H_actual
            HDF5.attributes(gp)["k"]        = k
            HDF5.attributes(gp)["ngbits"]   = ngbits
            HDF5.attributes(gp)["max_acc"]  = r.max_acc
            HDF5.attributes(gp)["n_top"]    = r.n_top
            gp["accs"]        = r.accs
            gp["jac_all"]     = r.jac_all
            gp["jac_top"]     = isempty(r.jac_top)     ? [-1.0] : r.jac_top
            gp["rep_sim_all"] = r.rep_sim_all
            gp["rep_sim_top"] = isempty(r.rep_sim_top) ? [-1.0] : r.rep_sim_top
        end
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
