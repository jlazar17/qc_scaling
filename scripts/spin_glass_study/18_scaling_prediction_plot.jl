# Script 18: nstate scaling study with theoretical accuracy predictions overlaid.
#
# Runs the same fixed-per-state-budget SA as script 06 (n_seeds=3 for speed),
# then overlays dotted prediction lines from the Poisson-Binomial model
# (p_anti values from 17_accuracy_prediction.csv).

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

include(joinpath(@__DIR__, "sg_utils.jl"))

using Plots
using DelimitedFiles

# ---------------------------------------------------------------------------
# Theoretical accuracy prediction (same model as script 17).
# ---------------------------------------------------------------------------
function predict_accuracy(p_anti, lambda; V_max=60)
    p_goal = 1.0 - p_anti

    # Precompute log factorials
    lfact = zeros(V_max + 2)
    for i in 1:V_max + 1
        lfact[i + 1] = lfact[i] + log(i)
    end
    lf(n) = lfact[n + 1]

    # P(V = v) ~ Poisson(lambda), truncated at V_max
    pois = [exp(-lambda + v * log(lambda) - lf(v)) for v in 0:V_max]
    pois ./= sum(pois)

    p_correct = 0.0
    p_nontied = 0.0

    for v in 0:V_max
        v == 0 && continue
        pv = pois[v + 1]

        # Binomial(v, p_goal) probabilities
        binom = [exp(lf(v) - lf(k) - lf(v - k) +
                     k * log(p_goal) + (v - k) * log(max(1e-300, 1.0 - p_goal)))
                 for k in 0:v]
        binom ./= sum(binom)

        p_win  = sum(binom[k + 1] for k in 0:v if k > v / 2)
        p_tied = isodd(v) ? 0.0 : binom[v ÷ 2 + 1]

        p_correct += pv * p_win
        p_nontied += pv * (1.0 - p_tied)
    end

    p_pair_correct = p_correct^2
    p_pair_nontied = p_nontied^2
    return p_pair_nontied > 0 ? p_pair_correct / p_pair_nontied : 0.0
end

# ---------------------------------------------------------------------------
# SA runner (same logic as script 06).
# ---------------------------------------------------------------------------
function run_sa_nstate(goal, nqubit, nstate, nsteps, alpha_cool,
                       companion, goal_idx, fingerprint, cxt_master; seed=42)
    n    = 3^nqubit
    npos = length(cxt_master.base_even.pos)
    rng  = Random.MersenneTwister(seed)

    ensemble   = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    cache_idxs = [Vector{Int}(undef, npos) for _ in 1:nstate]
    cache_pars  = [Vector{Int}(undef, npos) for _ in 1:nstate]
    for i in 1:nstate
        fill_state_cache!(cache_idxs[i], cache_pars[i], ensemble[i], cxt_master)
    end
    rep_sum = zeros(Int, n); rep_ctr = zeros(Int, n)
    for i in 1:nstate
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[i], cache_pars[i], 1)
    end
    rep = rep_from_cache(rep_sum, rep_ctr)

    acc_fn(rs, rc) = rep_accuracy_shuffled(rs, rc, goal, companion, goal_idx)
    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    bad_deltas = Float64[]
    cur_acc = acc_fn(rep_sum, rep_ctr)
    for _ in 1:min(300, nsteps)
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                              bc, companion, goal_idx, n-1)
        ns    = QCScaling.PseudoGHZState(ret..., gen)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, 1)
        d = acc_fn(rep_sum, rep_ctr) - cur_acc
        apply_state!(rep_sum, rep_ctr, ns, cxt_master, -1)
        apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, 1)
        d < 0 && push!(bad_deltas, abs(d))
    end
    T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)
    cur_acc = acc_fn(rep_sum, rep_ctr)

    for _ in 1:nsteps
        which = rand(rng, 1:nstate)
        gen   = QCScaling.ParityOperator(rand(rng, 0:n-1), nqubit)
        ts    = rand(rng, 0:1)
        bc    = ts == 0 ? cxt_master.base_even : cxt_master.base_odd
        ret   = pick_alphas_s(QCScaling.Context(gen, bc), goal, rep, fingerprint,
                              bc, companion, goal_idx, n-1)
        ns    = QCScaling.PseudoGHZState(ret..., gen)
        fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
        apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
        apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, 1)
        new_acc = acc_fn(rep_sum, rep_ctr)
        d = new_acc - cur_acc
        if d >= 0 || rand(rng) < exp(d / T)
            update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
            update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
            copy!(cache_idxs[which], scratch_idxs)
            copy!(cache_pars[which], scratch_pars)
            ensemble[which] = ns; cur_acc = new_acc
        else
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs, scratch_pars, -1)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], 1)
        end
        T *= alpha_cool
    end
    return cur_acc
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    nqubit          = 6
    n               = 3^nqubit
    ngbits          = (n - 1) ÷ 2
    npos            = length(QCScaling.ContextMaster(nqubit).base_even.pos)
    n_seeds         = 3
    steps_per_state = 300_000 ÷ 45   # fixed per-state budget

    nstate_vals = [10, 20, 30, 45, 60, 90, 120, 180]

    companion, goal_idx, _ = build_shuffled_pairing(nqubit)
    fingerprint  = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    cxt_master   = QCScaling.ContextMaster(nqubit)

    goal_h0 = zeros(Int, ngbits)
    rng1    = Random.MersenneTwister(99)
    goal_h1 = Random.shuffle!(rng1, vcat(ones(Int, ngbits ÷ 2),
                                          zeros(Int, ngbits - ngbits ÷ 2)))

    # --- Empirical scaling runs ---
    mean_h0 = Float64[]; se_h0 = Float64[]
    mean_h1 = Float64[]; se_h1 = Float64[]

    println("Running nstate scaling (n_seeds=$n_seeds)...")
    for nstate in nstate_vals
        nsteps = nstate * steps_per_state
        alpha  = exp(log(1e-4) / nsteps)
        accs0  = Float64[]; accs1 = Float64[]
        for seed in 1:n_seeds
            a0 = run_sa_nstate(goal_h0, nqubit, nstate, nsteps, alpha,
                               companion, goal_idx, fingerprint, cxt_master;
                               seed=seed * 31 + 7)
            a1 = run_sa_nstate(goal_h1, nqubit, nstate, nsteps, alpha,
                               companion, goal_idx, fingerprint, cxt_master;
                               seed=seed * 31 + 7)
            push!(accs0, a0); push!(accs1, a1)
        end
        push!(mean_h0, mean(accs0)); push!(se_h0, std(accs0) / sqrt(n_seeds))
        push!(mean_h1, mean(accs1)); push!(se_h1, std(accs1) / sqrt(n_seeds))
        @printf("nstate=%3d:  H=0=%.4f  H=1=%.4f\n", nstate, mean(accs0), mean(accs1))
        flush(stdout)
    end

    # --- Read p_anti from script 17 results ---
    pred_csv = joinpath(@__DIR__, "results", "17_accuracy_prediction.csv")
    pred_data, pred_hdr = readdlm(pred_csv, ',', header=true)
    pred_hdr = vec(pred_hdr)
    h_col  = findfirst(==("H"),      pred_hdr)
    pa_col = findfirst(==("p_anti"), pred_hdr)
    h_vals  = Float64.(pred_data[:, h_col])
    pa_vals = Float64.(pred_data[:, pa_col])
    pa_h0 = pa_vals[findfirst(==(0.0), h_vals)]
    pa_h1 = pa_vals[findfirst(==(1.0), h_vals)]
    @printf("\np_anti:  H=0=%.4f  H=1=%.4f\n", pa_h0, pa_h1)

    # --- Prediction curves (dense nstate grid) ---
    nstate_dense = collect(5:5:220)
    pred_h0 = [predict_accuracy(pa_h0, ns * npos / (n - 1)) for ns in nstate_dense]
    pred_h1 = [predict_accuracy(pa_h1, ns * npos / (n - 1)) for ns in nstate_dense]

    # --- Plot ---
    p = plot(title="Accuracy vs nstate  (nqubit=6, fixed per-state budget)",
             xlabel="nstate", ylabel="Accuracy",
             legend=:bottomright, ylims=(0.4, 1.02),
             size=(700, 500), margin=6Plots.mm)

    # Empirical lines (solid, with error ribbons)
    plot!(p, nstate_vals, mean_h0,
          ribbon=1.96 .* se_h0, fillalpha=0.2,
          label="H=0 measured", color=:blue, lw=2)
    plot!(p, nstate_vals, mean_h1,
          ribbon=1.96 .* se_h1, fillalpha=0.2,
          label="H=1 measured", color=:red, lw=2)

    # Prediction lines (dotted, same colors)
    plot!(p, nstate_dense, pred_h0,
          label="H=0 predicted", color=:blue, lw=2, ls=:dot)
    plot!(p, nstate_dense, pred_h1,
          label="H=1 predicted", color=:red, lw=2, ls=:dot)

    outfile = joinpath(@__DIR__, "results", "18_scaling_prediction.png")
    savefig(p, outfile)
    println("Saved plot to $outfile")
end

main()
