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
# Shared rep helpers
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

function rep_from_cache(rep_sum, rep_ctr)
    pref = rep_sum ./ rep_ctr
    pref[pref .== 0.5] .= NaN
    return round.(pref)
end

function replace_state!(states, which, new_state, rep_sum, rep_ctr, cxt_master)
    apply_state!(rep_sum, rep_ctr, states[which], cxt_master, -1)
    states[which] = new_state
    apply_state!(rep_sum, rep_ctr, new_state, cxt_master, 1)
end

# ---------------------------------------------------------------------------
# Improved optimizer — returns (best_acc, final_states, final_rep)
# ---------------------------------------------------------------------------

function run_improved(goal, nqubit, nstate, niter; seed=42, nreplace=1,
                      p_mutate=0.3, n_same_tol=10)
    Random.seed!(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = QCScaling.Fingerprint(nqubit)
    states      = [QCScaling.random_state(nqubit) for _ in 1:nstate]

    rep_sum = zeros(Float64, 3^nqubit); rep_ctr = zeros(Int, 3^nqubit)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    best_acc = -Inf; best_states = copy(states)
    n_same = 0; last_acc = -1.0

    for _ in 1:niter
        scores = QCScaling.score(states, rep, goal, cxt_master)
        sorter = sortperm(scores); scores = scores[sorter]; states = states[sorter]

        acc = accuracy(rep, goal)
        if acc > best_acc
            best_acc    = acc
            best_states = copy(states)
        end
        n_same = (acc == last_acc) ? n_same + 1 : 0; last_acc = acc

        if n_same <= n_same_tol
            ws      = Weights(maximum(scores) .- scores .+ 1)
            whiches = sample(1:length(scores), ws, nreplace; replace=false)
            for which in whiches
                rplc = states[which]
                if rand() < p_mutate
                    new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                    new_generator = first(new_cxt.pos)
                    base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = new_cxt
                else
                    new_generator = rplc.generator
                    base_cxt      = rplc.theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
                    cxt           = QCScaling.Context(new_generator, base_cxt)
                end
                alphas = QCScaling.pick_new_alphas(cxt, goal, rep, fingerprint, base_cxt)
                replace_state!(states, which, QCScaling.PseudoGHZState(alphas..., new_generator),
                               rep_sum, rep_ctr, cxt_master)
            end
        else
            n_same = 0
            for _ in 1:nreplace
                replace_idx   = rand(1:length(states))
                new_cxt       = first(QCScaling.get_new_contexts(states, rep, cxt_master, 1))
                new_generator = first(new_cxt.pos)
                base_cxt      = new_cxt.parity == 0 ? cxt_master.base_even : cxt_master.base_odd
                alphas        = QCScaling.pick_new_alphas(new_cxt, goal, rep, fingerprint, base_cxt)
                replace_state!(states, replace_idx, QCScaling.PseudoGHZState(alphas..., new_generator),
                               rep_sum, rep_ctr, cxt_master)
            end
        end
        rep = rep_from_cache(rep_sum, rep_ctr)
    end

    return best_acc, best_states, rep_from_cache(rep_sum, rep_ctr)
end

# ---------------------------------------------------------------------------
# State fingerprint: canonical tuple for exact comparison
# ---------------------------------------------------------------------------

state_key(s) = (s.theta_s, s.theta_z, Tuple(s.alphas), Tuple(s.generator.βs))

function jaccard(states_a, states_b)
    sa = Set(state_key(s) for s in states_a)
    sb = Set(state_key(s) for s in states_b)
    isempty(sa) && isempty(sb) && return 1.0
    return length(intersect(sa, sb)) / length(union(sa, sb))
end

# Rep vector similarity (ignoring NaN positions)
function rep_similarity(rep_a, rep_b)
    mask = .!isnan.(rep_a) .& .!isnan.(rep_b)
    sum(mask) == 0 && return NaN
    return mean(rep_a[mask] .== rep_b[mask])
end

# ---------------------------------------------------------------------------
# Analysis for one (nqubit, nstate, goal) combination
# ---------------------------------------------------------------------------

function analyze_convergence(goal, nqubit, nstate, niter, nseeds; base_seed=42)
    rng   = Random.MersenneTwister(base_seed)
    seeds = Int.(rand(rng, UInt32, nseeds))

    accs        = Float64[]
    all_states  = Vector{Vector{QCScaling.PseudoGHZState}}()
    all_reps    = Vector{Vector{Float64}}()

    for seed in seeds
        acc, sts, rep = run_improved(goal, nqubit, nstate, niter; seed=seed)
        push!(accs, acc)
        push!(all_states, sts)
        push!(all_reps, rep)
    end

    # Accuracy stats
    max_acc  = maximum(accs)
    top_mask = accs .>= max_acc - 1e-9   # runs at or near maximum

    # Pairwise Jaccard for ALL runs and for TOP runs
    n = length(accs)
    jac_all = Float64[]
    jac_top = Float64[]
    rep_sim_all = Float64[]
    rep_sim_top = Float64[]
    top_idx = findall(top_mask)

    for i in 1:n, j in (i+1):n
        j_val  = jaccard(all_states[i], all_states[j])
        rs_val = rep_similarity(all_reps[i], all_reps[j])
        push!(jac_all, j_val)
        push!(rep_sim_all, rs_val)
        if top_mask[i] && top_mask[j]
            push!(jac_top, j_val)
            push!(rep_sim_top, rs_val)
        end
    end

    return (
        accs        = accs,
        max_acc     = max_acc,
        n_top       = sum(top_mask),
        jac_all     = jac_all,
        jac_top     = jac_top,
        rep_sim_all = rep_sim_all,
        rep_sim_top = rep_sim_top,
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    niter    = 2_000
    nseeds   = 50
    outdir   = joinpath(@__DIR__, "data")
    mkpath(outdir)

    configs = [
        # (nqubit, nstate_mult, pzero)
        (4, 1, 0.0),
        (4, 1, 0.5),
        (4, 2, 0.0),
        (4, 2, 0.5),
        (6, 1, 0.5),
        (6, 3, 0.5),
        (8, 1, 0.5),
        (8, 3, 0.5),
    ]

    outfile = joinpath(outdir, "convergence_analysis.h5")
    h5open(outfile, "w") do h5f

    for (nqubit, nstate_mult, pzero) in configs
        ngbits = (3^nqubit - 1) ÷ 2
        nstate = nstate_mult * Int(ceil(3^nqubit / 2^(nqubit - 1)))
        rng_g  = Random.MersenneTwister(42 + round(Int, pzero * 1000))
        goal   = sample(rng_g, 0:1, Weights([pzero, 1-pzero]), ngbits)

        print("nqubit=$nqubit  nstate=$nstate ($(nstate_mult)×)  pzero=$pzero  ... ")
        flush(stdout)

        r = analyze_convergence(goal, nqubit, nstate, niter, nseeds)

        @printf("max_acc=%.4f  n_top=%d/%d  jac_top=%.3f±%.3f  rep_sim_top=%.3f±%.3f\n",
            r.max_acc, r.n_top, nseeds,
            isempty(r.jac_top)     ? NaN : mean(r.jac_top),
            isempty(r.jac_top)     ? NaN : std(r.jac_top),
            isempty(r.rep_sim_top) ? NaN : mean(r.rep_sim_top),
            isempty(r.rep_sim_top) ? NaN : std(r.rep_sim_top),
        )
        flush(stdout)

        key = "nq$(nqubit)_ns$(nstate)_pz$(pzero)"
        gp  = create_group(h5f, key)
        gp["accs"]        = r.accs
        gp["jac_all"]     = r.jac_all
        gp["jac_top"]     = isempty(r.jac_top) ? [-1.0] : r.jac_top
        gp["rep_sim_all"] = r.rep_sim_all
        gp["rep_sim_top"] = isempty(r.rep_sim_top) ? [-1.0] : r.rep_sim_top
        attributes(gp)["nqubit"]       = nqubit
        attributes(gp)["nstate"]       = nstate
        attributes(gp)["nstate_mult"]  = nstate_mult
        attributes(gp)["pzero"]        = pzero
        attributes(gp)["max_acc"]      = r.max_acc
        attributes(gp)["n_top"]        = r.n_top
        attributes(gp)["nseeds"]       = nseeds
    end

    end  # h5open
    println("\nSaved to $outfile")
end

main()
