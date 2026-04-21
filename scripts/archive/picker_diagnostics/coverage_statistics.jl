using Pkg
Pkg.activate(joinpath(@__DIR__, "../../utils"))
Pkg.develop(path=joinpath(@__DIR__, "../../QCScaling"))

using QCScaling
using Statistics
using Random
using Printf

# ---------------------------------------------------------------------------
# Coverage statistics vs nqubit
#
# For each n, measures properties of the context (set of positions covered
# by a random state). No SA needed — pure structural measurement.
#
# Metrics:
#   context_size:         number of positions covered by a single state
#   ngbits:               total number of pairs = (3^n - 1) / 2
#   pair_coverage_frac:   fraction of pairs where at least one of k1,k2 covered
#   both_covered_frac:    fraction of pairs where BOTH k1 and k2 are covered
#   singleton_frac:       fraction of pairs where exactly ONE of k1,k2 covered
#   mean_pair_mult:       mean number of states needed to cover a random pair
#                         (approx 1 / pair_coverage_frac_per_state)
#
# The key question: does a single state cover a decreasing fraction of pairs
# as n grows? And does this decrease affect H=0 and H=1 differently?
# (H=0 only needs one member of a pair to be covered to vote;
#  H=1 is more sensitive to the PATTERN of coverage across pairs.)
# ---------------------------------------------------------------------------

function coverage_stats(nqubit, ngens, rng)
    cxt_master = QCScaling.ContextMaster(nqubit)
    n      = 3^nqubit
    ngbits = (n - 1) ÷ 2

    context_sizes    = Int[]
    pair_cov_fracs   = Float64[]
    both_cov_fracs   = Float64[]
    singleton_fracs  = Float64[]

    for _ in 1:ngens
        gen_idx   = rand(rng, 0:n-1)
        generator = QCScaling.ParityOperator(gen_idx, nqubit)
        theta_s   = rand(rng, 0:1)
        base_cxt  = theta_s == 0 ? cxt_master.base_even : cxt_master.base_odd

        covered_k = Set{Int}()
        for base_po in base_cxt.pos
            derived_po = generator + base_po
            k = derived_po.index
            k == n && continue
            push!(covered_k, k)
        end

        push!(context_sizes, length(covered_k))

        n_pair_covered = 0; n_both = 0; n_singleton = 0
        for j in 1:ngbits
            k1 = 2j - 1; k2 = 2j
            has_k1 = k1 in covered_k
            has_k2 = k2 in covered_k
            if has_k1 || has_k2
                n_pair_covered += 1
                if has_k1 && has_k2
                    n_both += 1
                else
                    n_singleton += 1
                end
            end
        end

        push!(pair_cov_fracs,  n_pair_covered / ngbits)
        push!(both_cov_fracs,  n_both         / ngbits)
        push!(singleton_fracs, n_singleton     / ngbits)
    end

    return (
        nqubit        = nqubit,
        n             = n,
        ngbits        = ngbits,
        nalpha        = 2^(nqubit - 1),
        context_size  = mean(context_sizes),
        pair_cov_frac = mean(pair_cov_fracs),
        both_cov_frac = mean(both_cov_fracs),
        singleton_frac = mean(singleton_fracs),
        # expected states to cover all pairs (1 / pair_cov_frac_per_state, roughly)
        states_to_cover = 1.0 / mean(pair_cov_fracs),
    )
end

function main()
    ngens     = 50_000
    base_seed = 42
    rng       = Random.MersenneTwister(base_seed)

    @printf("Coverage statistics vs nqubit  (ngens=%d per n)\n\n", ngens)
    @printf("%-8s  %-10s  %-8s  %-8s  %-14s  %-14s  %-14s  %-16s\n",
            "nqubit", "ngbits", "nalpha", "cxt_sz",
            "pair_cov_frac", "both_cov_frac", "sng_cov_frac", "states_to_cover")
    println("-"^100)

    for nqubit in [4, 6, 8, 10]
        s = coverage_stats(nqubit, ngens, rng)
        @printf("%-8d  %-10d  %-8d  %-8.2f  %-14.6f  %-14.6f  %-14.6f  %-16.2f\n",
                s.nqubit, s.ngbits, s.nalpha,
                s.context_size,
                s.pair_cov_frac,
                s.both_cov_frac,
                s.singleton_frac,
                s.states_to_cover)
        flush(stdout)
    end

    println()
    @printf("Interpretation:\n")
    @printf("  pair_cov_frac:   fraction of ngbits pairs covered by a single state (at least one of k1,k2)\n")
    @printf("  both_cov_frac:   fraction where both k1 AND k2 are covered (GF(2) both-covered pairs)\n")
    @printf("  sng_cov_frac:    fraction where exactly one of k1,k2 is covered\n")
    @printf("  states_to_cover: ~1/pair_cov_frac, rough lower bound on nstate for full coverage\n")
    @printf("\n")
    @printf("  If pair_cov_frac ~ 1/ngbits (each state covers ~1 pair on average as n grows),\n")
    @printf("  the problem becomes harder for all H. If it falls faster, H=1 is disproportionately\n")
    @printf("  affected since it requires specific patterning across many pairs.\n")
end

main()
