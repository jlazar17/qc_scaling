using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

include("../utils/optimization_utils.jl")

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

function time_steps(nqubit, nstate, nsteps_warmup, nsteps_timed; seed=42)
    rng         = Random.MersenneTwister(seed)
    cxt_master  = QCScaling.ContextMaster(nqubit)
    fingerprint = FingerprintPacked(QCScaling.Fingerprint(nqubit))
    n           = 3^nqubit
    ngbits      = (n - 1) ÷ 2

    goal     = rand(rng, 0:1, ngbits)
    ensemble = [QCScaling.random_state(nqubit) for _ in 1:nstate]
    rep_sum  = zeros(Float64, n); rep_ctr = zeros(Int, n)
    for s in ensemble; apply_state!(rep_sum, rep_ctr, s, cxt_master, 1); end
    rep = rep_from_cache(rep_sum, rep_ctr)

    T = 0.01; alpha = 0.99999
    current_acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)

    # Warmup to get JIT out of the way
    for _ in 1:nsteps_warmup
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
            ensemble[which] = ns; current_acc = new_acc
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end
        T *= alpha
    end

    # Timed section
    t0 = time()
    for _ in 1:nsteps_timed
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
            ensemble[which] = ns; current_acc = new_acc
        else
            apply_state!(rep_sum, rep_ctr, ns,        cxt_master, -1)
            apply_state!(rep_sum, rep_ctr, old_state, cxt_master,  1)
        end
        T *= alpha
    end
    elapsed = time() - t0
    return elapsed / nsteps_timed  # seconds per step
end

function main()
    nsteps_warmup = 2_000
    nsteps_timed  = 5_000

    @printf("%-8s  %-10s  %-12s  %-14s  %-16s  %-16s\n",
            "nqubit", "nstate", "base_nstate", "sec/step", "2M-step run", "8M-step run")
    println("-"^80)

    for nqubit in [6, 8, 10]
        base_nstate = Int(ceil(3^nqubit / 2^(nqubit - 1)))
        nstate      = 3 * base_nstate   # mult=3 as in current studies

        sps = time_steps(nqubit, nstate, nsteps_warmup, nsteps_timed)

        t_2M = sps * 2_000_000
        t_8M = sps * 8_000_000

        @printf("%-8d  %-10d  %-12d  %-14.2e  %-16s  %-16s\n",
                nqubit, nstate, base_nstate, sps,
                @sprintf("%.1f min", t_2M/60),
                @sprintf("%.1f min", t_8M/60))
        flush(stdout)
    end
end

main()
