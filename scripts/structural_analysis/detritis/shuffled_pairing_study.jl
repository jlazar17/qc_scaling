using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../../utils/optimization_utils.jl")

# ---------------------------------------------------------------------------
# Build non-co-occurring companion pairing
#
# Two positions co-occur in the same context iff their difference (mod 3) has
# an EVEN number of non-zero components.  We find a perfect matching where
# every companion pair has an ODD number of differing components, guaranteeing
# zero both-covered pairs.
# ---------------------------------------------------------------------------

function build_shuffled_pairing(nqubit)
    n = 3^nqubit
    betas = [QCScaling.to_ternary(k-1, Val(nqubit)) for k in 1:n-1]
    matched = falses(n-1)
    pairs = Tuple{Int,Int}[]
    for k1 in 1:n-1
        matched[k1] && continue
        for k2 in k1+1:n-1
            matched[k2] && continue
            if isodd(count(i -> betas[k1][i] != betas[k2][i], 1:nqubit))
                push!(pairs, (k1, k2))
                matched[k1] = matched[k2] = true
                break
            end
        end
    end
    @assert sum(matched) == n-1 "Pairing incomplete: $(sum(.!matched)) unmatched"
    companion = zeros(Int, n)
    goal_idx  = zeros(Int, n)
    for (gi, (k1, k2)) in enumerate(pairs)
        companion[k1] = k2; companion[k2] = k1
        goal_idx[k1]  = gi; goal_idx[k2]  = gi
    end
    return companion, goal_idx, length(pairs)
end

# ---------------------------------------------------------------------------
# Modified rep_accuracy using a custom pairing
# ---------------------------------------------------------------------------

function rep_accuracy_custom(rep_sum, rep_ctr, goal, companion, goal_idx)
    s = 0; n = length(goal)
    # Iterate only over odd-indexed positions (one representative per pair)
    for k1 in 1:length(rep_sum)-1
        companion[k1] == 0 && continue
        k1 > companion[k1] && continue   # process each pair once (k1 < k2)
        k2 = companion[k1]
        c1 = rep_ctr[k1]; c2 = rep_ctr[k2]
        s1 = rep_sum[k1]; s2 = rep_sum[k2]
        valid = (c1 > 0) & (c2 > 0) & (2*s1 != c1) & (2*s2 != c2)
        r1 = 2*s1 > c1
        r2 = 2*s2 > c2
        j = goal_idx[k1]
        s += valid & ((r1 ⊻ r2) == !iszero(goal[j]))
    end
    return s / n
end

# ---------------------------------------------------------------------------
# Modified companion_goal and pick_new_alphas using custom pairing
# ---------------------------------------------------------------------------

function companion_goal_custom(po, goal, rep, companion, goal_idx, n_rep)
    k = po.index
    (k <= 0 || k > n_rep) && return NaN
    comp = companion[k]
    comp == 0 && return NaN
    isnan(rep[comp]) && return NaN
    j = goal_idx[k]
    j == 0 && return NaN
    return goal[j] == 1 ? 1 - rep[comp] : rep[comp]
end

function pick_new_alphas_custom(cxt, goal, rep, fp::FingerprintPacked, base_cxt,
                                companion, goal_idx, n_rep)
    cg = [companion_goal_custom(po, goal, rep, companion, goal_idx, n_rep)
          for po in cxt.pos]
    valid, vals = QCScaling._pack_companion_goal(cg, fp.nwords)
    parity_idx  = base_cxt.parity + 1
    nwords      = fp.nwords
    nalpha      = size(fp.words, 4)
    ntz         = size(fp.words, 3)

    best_score = typemax(Int)
    best_tz    = 1
    best_alpha = 1
    @inbounds for ai in 1:nalpha
        for tzi in 1:ntz
            score = 0
            for w in 1:nwords
                score += count_ones(xor(fp.words[w, parity_idx, tzi, ai], vals[w]) & valid[w])
            end
            if score < best_score
                best_score = score
                best_tz    = tzi
                best_alpha = ai
            end
        end
    end
    return (base_cxt.parity, best_tz-1, QCScaling.idx_to_alphas(best_alpha-1, length(cxt.pos[1])))
end

# ---------------------------------------------------------------------------
# Goal generation / efficiency
# ---------------------------------------------------------------------------

function goal_from_hamming(k, ngbits, rng)
    shuffle!(rng, vcat(ones(Int,k), zeros(Int, ngbits-k)))
end

function k_from_entropy(H, N)
    H <= 0.0 && return 0; H >= 1.0 && return N÷2
    _, idx = findmin(k -> abs(let p=k/N; (k==0||k==N) ? 0.0 : -p*log2(p)-(1-p)*log2(1-p) end - H), 0:N÷2)
    return (0:N÷2)[idx]
end

function binary_entropy(x)
    (x <= 0.0 || x >= 1.0) && return 0.0
    -x*log2(x) - (1-x)*log2(1-x)
end

function efficiency(acc, nqubit, nstate)
    n_classical = (3^nqubit - 1) / 2
    n_quantum   = nqubit * nstate
    (1 - binary_entropy(acc)) * n_classical / n_quantum
end

# ---------------------------------------------------------------------------
# Smart proposal using custom pairing
# ---------------------------------------------------------------------------

function smart_proposal_custom(nqubit, rep, goal, fingerprint, cxt_master,
                                companion, goal_idx, n_rep, rng)
    gen_idx  = rand(rng, 0:3^nqubit-1)
    generator = QCScaling.ParityOperator(gen_idx, nqubit)
    theta_s   = rand(rng, 0:1)
    base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd
    cxt       = QCScaling.Context(generator, base_cxt)
    alphas    = pick_new_alphas_custom(cxt, goal, rep, fingerprint, base_cxt,
                                       companion, goal_idx, n_rep)
    return QCScaling.PseudoGHZState(alphas..., generator)
end

# ---------------------------------------------------------------------------
# SA inner loop — parameterised on accuracy/proposal functions
# ---------------------------------------------------------------------------

function run_sa(goal, nqubit, nstate, nsteps, alpha_cool;
                n_restarts=3, seed=42,
                acc_fn, proposal_fn)
    rng        = Random.MersenneTwister(seed)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n          = 3^nqubit
    ngbits     = (n-1) ÷ 2
    best_acc   = -Inf

    min_delta   = 1.0 / ngbits
    stag_window = round(Int, -5.0 / log(alpha_cool))
    npos        = length(cxt_master.base_even.pos)

    scratch_idxs = Vector{Int}(undef, npos)
    scratch_pars = Vector{Int}(undef, npos)

    for _ in 1:n_restarts
        ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]

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

        # Calibrate T0 using the provided accuracy function
        current_acc = acc_fn(rep_sum, rep_ctr, goal)
        bad_deltas  = Float64[]
        for _ in 1:500
            which = rand(rng, 1:nstate)
            ns    = proposal_fn(nqubit, rep, goal, cxt_master, rng)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ns,              cxt_master,  1)
            delta = acc_fn(rep_sum, rep_ctr, goal) - current_acc
            apply_state!(rep_sum, rep_ctr, ns,              cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, ensemble[which], cxt_master,  1)
            delta < 0 && push!(bad_deltas, abs(delta))
        end
        T = isempty(bad_deltas) ? 0.1 : -mean(bad_deltas) / log(0.8)

        current_acc      = acc_fn(rep_sum, rep_ctr, goal)
        restart_best     = current_acc
        last_improvement = 0

        for step in 1:nsteps
            which = rand(rng, 1:nstate)
            ns    = proposal_fn(nqubit, rep, goal, cxt_master, rng)

            fill_state_cache!(scratch_idxs, scratch_pars, ns, cxt_master)
            apply_state_cached!(rep_sum, rep_ctr, cache_idxs[which], cache_pars[which], -1)
            apply_state_cached!(rep_sum, rep_ctr, scratch_idxs,      scratch_pars,       1)
            new_acc = acc_fn(rep_sum, rep_ctr, goal)
            delta   = new_acc - current_acc

            if delta >= 0 || rand(rng) < exp(delta / T)
                update_rep_at_cached!(rep, rep_sum, rep_ctr, cache_idxs[which])
                update_rep_at_cached!(rep, rep_sum, rep_ctr, scratch_idxs)
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
            T *= alpha_cool
            step > stag_window && (step - last_improvement) >= stag_window && break
        end
    end
    return best_acc
end

# ---------------------------------------------------------------------------
# Main: compare old vs new pairing for H=0 and H=1 at nqubit=6
# ---------------------------------------------------------------------------

function main()
    nqubit   = 6
    n        = 3^nqubit
    ngbits   = (n-1) ÷ 2
    nsteps   = 50_000
    alpha    = 0.9999
    n_seeds  = 20
    n_restarts = 3

    println("Building shuffled pairing for nqubit=$nqubit...")
    companion, goal_idx, npairs = build_shuffled_pairing(nqubit)
    @assert npairs == ngbits "Expected $ngbits pairs, got $npairs"
    println("  Done: $npairs pairs, zero both-covered pairs guaranteed")

    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))

    # Closures capturing the pairing
    acc_new      = (rs, rc, g)            -> rep_accuracy_custom(rs, rc, g, companion, goal_idx)
    proposal_new = (nq, rep, g, cm, rng)  -> smart_proposal_custom(nq, rep, g, fingerprint, cm,
                                                                     companion, goal_idx, n, rng)
    acc_old      = (rs, rc, g)            -> rep_accuracy_fast(rs, rc, g)
    proposal_old = (nq, rep, g, cm, rng)  -> begin
        fp = fingerprint
        gen_idx   = rand(rng, 0:3^nq-1)
        generator = QCScaling.ParityOperator(gen_idx, nq)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cm.base_even : cm.base_odd
        cxt       = QCScaling.Context(generator, base_cxt)
        alphas    = QCScaling.pick_new_alphas(cxt, g, rep, fp, base_cxt)
        QCScaling.PseudoGHZState(alphas..., generator)
    end

    # nstate grid: coarse log-spaced
    nstate_min = Int(ceil(3^nqubit / 2^(nqubit-1)))  # = 52 for n=6... wait n=6: 729/32≈23
    # Actually for n=6: 3^6=729, 2^5=32, ceil=23
    nstate_min = Int(ceil(3^nqubit / 2^(nqubit-1)))
    nstates = unique(round.(Int, 10 .^ range(log10(nstate_min), log10(8*nstate_min), length=12)))

    rng_goals = Random.MersenneTwister(99)
    seeds = rand(Random.MersenneTwister(7), UInt32, n_seeds)

    println("\nnstate_min=$nstate_min, grid: $nstates")
    println()

    for H_target in [0.0, 1.0]
        k = k_from_entropy(H_target, ngbits)
        println("="^65)
        @printf("H = %.2f  (k=%d / %d)\n", H_target, k, ngbits)
        println("="^65)
        @printf("%-8s  %-24s  %-24s\n", "nstate",
                "OLD  eta_max  eta_med", "NEW  eta_max  eta_med")
        println(repeat("-", 65))

        for nstate in nstates
            old_accs = zeros(Float64, n_seeds); new_accs = zeros(Float64, n_seeds)

            Threads.@threads for si in 1:n_seeds
                goal = goal_from_hamming(k, ngbits, Random.MersenneTwister(seeds[si] + si))
                seed = Int(seeds[si])

                a_old = run_sa(goal, nqubit, nstate, nsteps, alpha;
                               n_restarts=n_restarts, seed=seed,
                               acc_fn=acc_old, proposal_fn=proposal_old)
                a_new = run_sa(goal, nqubit, nstate, nsteps, alpha;
                               n_restarts=n_restarts, seed=seed,
                               acc_fn=acc_new, proposal_fn=proposal_new)
                old_accs[si] = a_old
                new_accs[si] = a_new
            end

            eta_old_max = maximum(efficiency.(old_accs, nqubit, nstate))
            eta_old_med = median(efficiency.(old_accs, nqubit, nstate))
            eta_new_max = maximum(efficiency.(new_accs, nqubit, nstate))
            eta_new_med = median(efficiency.(new_accs, nqubit, nstate))

            @printf("%-8d  %-10.4f  %-10.4f    %-10.4f  %-10.4f\n",
                    nstate, eta_old_max, eta_old_med, eta_new_max, eta_new_med)
            flush(stdout)
        end
        println()
    end
end

main()
