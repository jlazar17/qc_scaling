using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using StatsBase
using Random
using HDF5
using Printf

include("../utils/optimization_utils.jl")

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
# Efficiency metric
# ---------------------------------------------------------------------------

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function efficiency(acc::Float64, nqubit::Int, nstate::Int)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    return (1 - binary_entropy(acc)) * n_classical / n_quantum
end

# ---------------------------------------------------------------------------
# Rep helpers
# ---------------------------------------------------------------------------

# apply_state!, rep_from_cache, update_rep_at!, fill_state_cache!,
# apply_state_cached!, update_rep_at_cached! are all provided by QCScaling.

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

function run_goal_aware_sa(goal, nqubit, nstate, nsteps, alpha;
                           n_restarts=3, seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2
    best_acc    = -Inf

    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha))
    npos        = length(cxt_master.base_even.pos)  # same length for even and odd

    # Scratch buffers for the proposed state — filled each step, no allocation.
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]

        # Pre-allocate per-slot cache vectors and fill from initial ensemble.
        cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
        cache_pars = [Vector{Int}(undef, npos) for _ in 1:nstate]
        for i in 1:nstate
            fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
        end

        rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
        for i in 1:nstate
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
        end
        rep = rep_from_cache(rep_sum, rep_ctr)

        T = calibrate_T0(ensemble, rep_sum, rep_ctr, rep, goal, nqubit,
                         cxt_master, fingerprint; rng=rng)
        current_acc    = rep_accuracy_fast(rep_sum, rep_ctr, goal)
        restart_best   = current_acc
        last_improvement = 0

        for step in 1:nsteps
            which = rand(rng, 1:nstate)
            ns    = smart_proposal(nqubit, rep, goal, fingerprint, cxt_master, rng)

            # Fill scratch with proposed state's cache (no allocation).
            fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)

            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs,      scratch_pars,       1)
            new_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
                update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
                # Persist proposed cache into slot — copy! reuses existing allocation.
                copy!(cache_idxs[which], scratch_idxs)
                copy!(cache_pars[which], scratch_pars)
                ensemble[which] = ns
                current_acc     = new_acc
                if new_acc > restart_best + min_delta
                    restart_best     = new_acc
                    last_improvement = step
                end
                new_acc > best_acc && (best_acc = new_acc)
            else
                apply_state_cached!(rep_sum, rep_ctr, scratch_idxs,      scratch_pars,       -1)
                apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which],   1)
            end
            T *= alpha

            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

# ---------------------------------------------------------------------------
# Evaluate a single nstate: run nseeds, return (accs, etas)
# ---------------------------------------------------------------------------

function evaluate_nstate(nstate, goals, nqubit, nsteps, alpha, n_restarts, seeds)
    n = length(seeds)
    accs = Vector{Float64}(undef, n)
    Threads.@threads for si in 1:n
        accs[si] = run_goal_aware_sa(goals[si], nqubit, nstate, nsteps, alpha;
                                     n_restarts=n_restarts, seed=seeds[si])
    end
    etas = [efficiency(a, nqubit, nstate) for a in accs]
    return accs, etas
end

# ---------------------------------------------------------------------------
# Adaptive grid refinement
# ---------------------------------------------------------------------------

function log_spaced_nstates(nstate_min, nstate_max, n_coarse)
    lo   = log(nstate_min); hi = log(nstate_max)
    vals = unique(round.(Int, exp.(range(lo, hi, length=n_coarse))))
    return filter(v -> v >= nstate_min && v <= nstate_max, vals)
end

function adaptive_grid(goals, nqubit, base_nstate, nsteps, alpha,
                       n_restarts, seeds, nstate_max, outfile, gp_nq_key, h_key;
                       n_coarse=12, n_refine=2)

    # Load any existing checkpointed nstate results.
    results = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    h5open(outfile, "r") do h5f
        haskey(h5f, gp_nq_key) || return
        haskey(h5f[gp_nq_key], h_key) || return
        gp_h = h5f[gp_nq_key][h_key]
        for k in keys(gp_h)
            startswith(k, "ns_") || continue
            ns = parse(Int, k[4:end])
            results[ns] = (read(gp_h[k]["accs"]), read(gp_h[k]["etas"]))
        end
    end
    !isempty(results) && @printf("  Resuming with %d existing nstate points: %s\n",
                                  length(results), string(sort(collect(keys(results)))))

    function run_and_checkpoint(ns)
        accs, etas = evaluate_nstate(ns, goals, nqubit, nsteps, alpha, n_restarts, seeds)
        results[ns] = (accs, etas)
        h5open(outfile, "cw") do h5f
            gp_ns = create_group(h5f[gp_nq_key][h_key], "ns_$(ns)")
            gp_ns["accs"] = accs
            gp_ns["etas"] = etas
        end
        return accs, etas
    end

    coarse = log_spaced_nstates(base_nstate, nstate_max, n_coarse)

    @printf("  Phase 1: coarse grid %s\n", string(coarse))
    for ns in coarse
        if haskey(results, ns)
            accs, etas = results[ns]
            @printf("    nstate=%-5d  acc_med=%.4f  η_med=%.4f  η_max=%.4f  [resumed]\n",
                    ns, median(accs), median(etas), maximum(etas))
            continue
        end
        accs, etas = run_and_checkpoint(ns)
        @printf("    nstate=%-5d  acc_med=%.4f  η_med=%.4f  η_max=%.4f  η_seeds=[%s]  acc_seeds=[%s]\n",
                ns, median(accs), median(etas), maximum(etas),
                join([@sprintf("%.4f", e) for e in etas], " "),
                join([@sprintf("%.4f", a) for a in accs], " "))
        flush(stdout)
    end

    for phase in 1:n_refine
        sorted_ns = sort(collect(keys(results)))
        max_etas  = [maximum(results[ns][2]) for ns in sorted_ns]
        peak_idx  = argmax(max_etas)

        lo = peak_idx > 1                ? sorted_ns[peak_idx-1] : sorted_ns[peak_idx]
        hi = peak_idx < length(max_etas) ? sorted_ns[peak_idx+1] : sorted_ns[peak_idx]

        new_ns = Int[]
        for mid in [(lo + sorted_ns[peak_idx]) ÷ 2, (sorted_ns[peak_idx] + hi) ÷ 2]
            if mid >= base_nstate && mid <= nstate_max && !haskey(results, mid)
                push!(new_ns, mid)
            end
        end

        isempty(new_ns) && break
        @printf("  Phase %d refinement: bisecting around peak (nstate=%d, η_max=%.4f) → add %s\n",
                phase+1, sorted_ns[peak_idx], maximum(results[sorted_ns[peak_idx]][2]), string(new_ns))

        for ns in new_ns
            accs, etas = run_and_checkpoint(ns)
            @printf("    nstate=%-5d  acc_med=%.4f  η_med=%.4f  η_max=%.4f  η_seeds=[%s]  acc_seeds=[%s]\n",
                    ns, median(accs), median(etas), maximum(etas),
                    join([@sprintf("%.4f", e) for e in etas], " "),
                    join([@sprintf("%.4f", a) for a in accs], " "))
            flush(stdout)
        end
    end

    sorted_ns = sort(collect(keys(results)))
    acc_mat   = collect(hcat([results[ns][1] for ns in sorted_ns]...)')  # [n_ns × nseeds]
    eta_mat   = collect(hcat([results[ns][2] for ns in sorted_ns]...)')
    return sorted_ns, acc_mat, eta_mat
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(; save_full_matrices=false)
    configs = [
        # (nqubit, nstate_max, sa_nsteps, sa_alpha)
        (4,   220,    200_000, 0.99999),
        (6,   460,    500_000, 0.99999),
        (8,  1340,  2_000_000, 0.99999),
    ]
    entropy_vals = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]
    nseeds       = 20
    n_restarts   = 3
    n_coarse     = 12
    n_refine     = 2
    base_seed    = 42

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "scaling_study_adaptive.h5")

    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))
    println("Seeds: $seeds\n")

    # Write global attributes once (no-op if file already has them).
    h5open(outfile, "cw") do h5f
        haskey(HDF5.attributes(h5f), "nseeds") && return
        HDF5.attributes(h5f)["nseeds"]    = nseeds
        HDF5.attributes(h5f)["n_coarse"]  = n_coarse
        HDF5.attributes(h5f)["n_refine"]  = n_refine
        HDF5.attributes(h5f)["base_seed"] = base_seed
    end

    for (nqubit, nstate_max, sa_nsteps, sa_alpha) in configs
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        ngbits      = (3^nqubit - 1) ÷ 2
        gp_nq_key   = "nqubit_$(nqubit)"

        println("="^70)
        @printf("nqubit=%d  base_nstate=%d  nstate_max=%d  nsteps=%d  alpha=%.5f\n",
                nqubit, base_nstate, nstate_max, sa_nsteps, sa_alpha)

        # Ensure nqubit group + attributes exist.
        h5open(outfile, "cw") do h5f
            gp_nq = haskey(h5f, gp_nq_key) ? h5f[gp_nq_key] : create_group(h5f, gp_nq_key)
            haskey(HDF5.attributes(gp_nq), "nqubit") && return
            HDF5.attributes(gp_nq)["nqubit"]      = nqubit
            HDF5.attributes(gp_nq)["base_nstate"] = base_nstate
            HDF5.attributes(gp_nq)["nstate_max"]  = nstate_max
            HDF5.attributes(gp_nq)["sa_nsteps"]   = sa_nsteps
            HDF5.attributes(gp_nq)["sa_alpha"]    = sa_alpha
        end

        for H_target in entropy_vals
            k     = k_from_entropy(H_target, ngbits)
            H_act = hamming_entropy(k, ngbits)
            key   = @sprintf("H%.2f_k%d", H_target, k)

            # Resume: skip if summary arrays already written (works for both old
            # format and new format, which also writes a "nstates" dataset).
            already_done = h5open(outfile, "r") do h5f
                haskey(h5f, gp_nq_key) && haskey(h5f[gp_nq_key], key) &&
                    haskey(h5f[gp_nq_key][key], "nstates")
            end
            if already_done
                @printf("\nSkipping H=%.3f (nqubit=%d): group '%s' already complete\n",
                        H_target, nqubit, key)
                flush(stdout)
                continue
            end

            println()
            @printf("H=%.3f (k=%d, H_act=%.4f)\n", H_target, k, H_act)

            # Ensure H group exists before entering adaptive_grid (so checkpoints
            # can be written inside it even if the run is interrupted).
            h5open(outfile, "cw") do h5f
                gp_nq = h5f[gp_nq_key]
                if !haskey(gp_nq, key)
                    gp = create_group(gp_nq, key)
                    HDF5.attributes(gp)["H_target"] = H_target
                    HDF5.attributes(gp)["H_actual"] = H_act
                    HDF5.attributes(gp)["k"]        = k
                    HDF5.attributes(gp)["ngbits"]   = ngbits
                end
            end

            rng_g = Random.MersenneTwister(base_seed + round(Int, H_target * 1000))
            goals = [goal_from_hamming(k, ngbits, rng_g) for _ in 1:nseeds]

            nstates, acc_mat, eta_mat = adaptive_grid(
                goals, nqubit, base_nstate, sa_nsteps, sa_alpha,
                n_restarts, seeds, nstate_max, outfile, gp_nq_key, key;
                n_coarse=n_coarse, n_refine=n_refine)

            best_max_idx = argmax([maximum(eta_mat[i, :]) for i in 1:length(nstates)])
            best_med_idx = argmax([median(eta_mat[i, :])  for i in 1:length(nstates)])
            @printf("  → η_max peak at nstate=%d: acc_med=%.4f  η_max=%.4f\n",
                    nstates[best_max_idx],
                    median(acc_mat[best_max_idx, :]), maximum(eta_mat[best_max_idx, :]))
            @printf("  → η_med peak at nstate=%d: acc_med=%.4f  η_med=%.4f\n\n",
                    nstates[best_med_idx],
                    median(acc_mat[best_med_idx, :]), median(eta_mat[best_med_idx, :]))

            # Write summary arrays and mark the group complete.
            h5open(outfile, "cw") do h5f
                gp = h5f[gp_nq_key][key]
                gp["nstates"] = nstates
                gp["acc_med"] = [median(acc_mat[i, :])  for i in 1:length(nstates)]
                gp["acc_max"] = [maximum(acc_mat[i, :]) for i in 1:length(nstates)]
                gp["acc_min"] = [minimum(acc_mat[i, :]) for i in 1:length(nstates)]
                gp["acc_std"] = [std(acc_mat[i, :])     for i in 1:length(nstates)]
                gp["eta_med"] = [median(eta_mat[i, :])  for i in 1:length(nstates)]
                gp["eta_max"] = [maximum(eta_mat[i, :]) for i in 1:length(nstates)]
                gp["eta_min"] = [minimum(eta_mat[i, :]) for i in 1:length(nstates)]
                gp["eta_std"] = [std(eta_mat[i, :])     for i in 1:length(nstates)]
                HDF5.attributes(gp)["nstate_best_max"] = nstates[best_max_idx]
                HDF5.attributes(gp)["nstate_best_med"] = nstates[best_med_idx]
                if save_full_matrices
                    gp["acc_mat"] = acc_mat
                    gp["eta_mat"] = eta_mat
                end
            end
            flush(stdout)
        end
    end

    println("Done. Results in $outfile")
end

main(save_full_matrices=false)
