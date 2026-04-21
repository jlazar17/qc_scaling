using Test
using Random
using Statistics
using StaticArrays
using QCScaling

# All rep-cache helpers (apply_state!, rep_from_cache, update_rep_at!,
# fill_state_cache!, apply_state_cached!, update_rep_at_cached!) are now
# exported from QCScaling via rep_cache.jl.

# Local script-style rep_accuracy_fast for cross-checking the package version.
function rep_accuracy_fast(rep_sum::Vector{Int}, rep_ctr::Vector{Int}, goal)
    s = 0
    @inbounds for i in eachindex(goal)
        i1 = 2i-1; i2 = 2i
        c1 = rep_ctr[i1]; c1 == 0 && continue
        c2 = rep_ctr[i2]; c2 == 0 && continue
        s1 = rep_sum[i1]; s2 = rep_sum[i2]
        (2*s1 == c1 || 2*s2 == c2) && continue
        r1 = 2*s1 > c1; r2 = 2*s2 > c2
        s += (r1 != r2) == !iszero(goal[i]) ? 1 : 0
    end
    return s / length(goal)
end

# ---------------------------------------------------------------------------

@testset "QCScaling Tests" begin

nqubit = 4

# -------------------------------------------------------------------------
@testset "to_ternary / to_index round-trip" begin
    for x in 0:3^nqubit - 1
        βs = QCScaling.to_ternary(x, nqubit)
        @test QCScaling.to_index(βs) == x + 1  # to_index is 1-based
    end
    # Boundary values
    @test QCScaling.to_index(QCScaling.to_ternary(0, nqubit)) == 1
    @test QCScaling.to_index(QCScaling.to_ternary(3^nqubit - 1, nqubit)) == 3^nqubit
end

# -------------------------------------------------------------------------
@testset "ParityOperator" begin
    po = ParityOperator([1, 2, 0, 1])
    @test po.βs == SVector{4,Int}(1, 2, 0, 1)
    @test po.index == QCScaling.to_index([1, 2, 0, 1])

    # ParityOperator(int, nqubit) takes a 0-based index and stores it as-is.
    # ParityOperator(βs) computes and stores a 1-based index via to_index.
    # These two constructors are inconsistent in their index semantics:
    # po.index is 1-based here (from βs constructor), but ParityOperator(po.index-1, n)
    # would store po.index-1 (0-based). Test only the βs field round-trip.
    po2 = ParityOperator(po.index - 1, nqubit)
    @test po2.βs == po.βs

    # Addition mod 3
    po_a = ParityOperator([1, 0, 2, 1])
    po_b = ParityOperator([2, 1, 1, 0])
    po_sum = po_a + po_b
    @test all(po_sum.βs .== mod.([1, 0, 2, 1] .+ [2, 1, 1, 0], 3))
    @test all(x -> x in (0, 1, 2), po_sum.βs)

    # Addition is associative
    po_c = ParityOperator([0, 2, 1, 2])
    @test ((po_a + po_b) + po_c).βs == (po_a + (po_b + po_c)).βs

    # Adding zero operator is identity
    po_zero = ParityOperator(zeros(Int, nqubit))
    @test (po_a + po_zero).βs == po_a.βs

    # Subtraction uses % (not mod): can produce negative values for (0 - x) cases.
    # This is the current behavior — tests document it, not endorse it.
    po_d = ParityOperator([2, 0, 1, 2])
    po_e = ParityOperator([1, 2, 0, 1])
    diff = po_d - po_e
    @test diff.βs == SVector{4,Int}((po_d.βs .- po_e.βs) .% 3)

    # Length
    @test length(po_a) == nqubit
end

# -------------------------------------------------------------------------
@testset "Context generation" begin
    base_even = QCScaling.generate_base_context(nqubit, 0)
    base_odd  = QCScaling.generate_base_context(nqubit, 1)

    @test length(base_even) == 2^(nqubit - 1) + 1
    @test length(base_odd)  == 2^(nqubit - 1) + 1
    @test base_even.parity == 0
    @test base_odd.parity  == 1

    # First element is the zero operator
    zero_po = base_even.pos[1]
    @test all(iszero, zero_po.βs)

    # All non-zero operators in base_even have no zeros in their βs
    for po in base_even.pos[2:end]
        @test !any(iszero, po.βs)
    end

    # All operators in base_even have even parity (count of 1s is even)
    for po in base_even.pos[2:end]
        @test sum(po.βs .== 1) % 2 == 0
    end
    # All operators in base_odd have odd parity
    for po in base_odd.pos[2:end]
        @test sum(po.βs .== 1) % 2 == 1
    end

    # Derived context from generator has same length and parity
    gen = ParityOperator([1, 2, 1, 2])
    cxt = QCScaling.Context(gen, base_even)
    @test length(cxt) == length(base_even)
    @test cxt.parity == base_even.parity

    # Derived context positions = generator + base positions (mod 3)
    for (i, base_po) in enumerate(base_even.pos)
        @test cxt.pos[i].βs == (gen + base_po).βs
    end

    # Cached indices match pos indices
    @test cxt.idxs == [po.index for po in cxt.pos]

    # ContextMaster
    cm = QCScaling.ContextMaster(nqubit)
    @test cm.nqubit == nqubit
    @test cm.base_even.parity == 0
    @test cm.base_odd.parity  == 1
    @test length(cm.base_even) == 2^(nqubit-1) + 1
    @test length(cm.base_odd)  == 2^(nqubit-1) + 1
end

# -------------------------------------------------------------------------
@testset "Parity: known values" begin
    # ---- Case 1: measurement == generator → all_zero → returns sum(alphas) % 2 ----
    gen = ParityOperator([1, 1, 1, 1])
    s0 = QCScaling.PseudoGHZState(0, 0, [1, 0, 1], gen)  # sum(alphas)=2 → 0
    s1 = QCScaling.PseudoGHZState(0, 0, [1, 1, 1], gen)  # sum(alphas)=3 → 1
    @test QCScaling.parity(s0, gen) == 0.0
    @test QCScaling.parity(s1, gen) == 1.0

    # sum(alphas) is invariant to theta_s/theta_z in all_zero case
    @test QCScaling.parity(QCScaling.PseudoGHZState(1, 1, [1, 0, 1], gen), gen) == 0.0
    @test QCScaling.parity(QCScaling.PseudoGHZState(1, 0, [1, 1, 1], gen), gen) == 1.0

    # ---- Case 2: mixed zeros → 0.5 ----
    # βs_state=[1,1,1,1], βs_meas=[1,2,1,2]: positions 1,3 have bd=0
    gen2 = ParityOperator([1, 1, 1, 1])
    meas = ParityOperator([1, 2, 1, 2])
    s2   = QCScaling.PseudoGHZState(0, 0, [0, 0, 0], gen2)
    @test QCScaling.parity(s2, meas) == 0.5

    # ---- Case 3: no zeros, sum_J + theta_s odd → 0.5 ----
    # Use nqubit=2 for simplicity
    gen3  = ParityOperator([1, 2])
    meas3 = ParityOperator([2, 1])
    # bd = [mod(-1,3), mod(1,3)] = [2, 1]
    # j(bd=2) = 1 - mod(1,2) = 0;  j(bd=1) = 1 - mod(0,2) = 1
    # sum_J = 1; dot_J_alpha (i<N: i=1 only, j=0): dot=0
    # theta_s=0: sum_J+theta_s=1 → odd → 0.5
    s3 = QCScaling.PseudoGHZState(0, 0, [0], gen3)
    @test QCScaling.parity(s3, meas3) == 0.5

    # theta_s=1: sum_J+theta_s=2 → even → mod(theta_z + 1 + 0, 2)
    s4_tz0 = QCScaling.PseudoGHZState(1, 0, [0], gen3)
    s4_tz1 = QCScaling.PseudoGHZState(1, 1, [0], gen3)
    @test QCScaling.parity(s4_tz0, meas3) == 1.0
    @test QCScaling.parity(s4_tz1, meas3) == 0.0

    # ---- All parity values are in {0.0, 0.5, 1.0} ----
    Random.seed!(42)
    states     = QCScaling.random_state(nqubit, 50)
    base_even  = QCScaling.generate_base_context(nqubit, 0)
    base_odd   = QCScaling.generate_base_context(nqubit, 1)
    for state in states
        base_cxt = state.theta_s == 0 ? base_even : base_odd
        cxt = QCScaling.Context(state.generator, base_cxt)
        for po in cxt.pos
            p = QCScaling.parity(state, po)
            @test p ∈ (0.0, 0.5, 1.0)
        end
    end

    # ---- Idempotency: same inputs give same output ----
    pvec1 = QCScaling.parity.(Ref(s0), [gen, meas])
    pvec2 = QCScaling.parity.(Ref(s0), [gen, meas])
    @test pvec1 == pvec2
end

# -------------------------------------------------------------------------
@testset "Fingerprint" begin
    fp = QCScaling.Fingerprint(nqubit)

    # Shape
    @test size(fp.a) == (2^(nqubit-1) + 1, 2, 2, 2^(nqubit-1))
    # Values are binary (parity takes only 0 and 1 in base context — 0.5 resolved by design)
    @test all(x -> x ∈ (0, 1), fp.a)

    # idx_to_alphas round-trip
    for idx in 0:2^(nqubit-1) - 1
        alphas = QCScaling.idx_to_alphas(idx, nqubit)
        reconstructed = sum(alphas[i] * 2^(i-1) for i in eachindex(alphas))
        @test reconstructed == idx
        @test length(alphas) == nqubit - 1
        @test all(x -> x ∈ (0, 1), alphas)
    end

    # fp[state] has the right shape and values
    Random.seed!(99)
    state = QCScaling.random_state(nqubit)
    col   = fp[state]
    @test length(col) == 2^(nqubit-1) + 1
    @test all(x -> x ∈ (0, 1), col)

    # Critical: fp[state] agrees with direct parity computation.
    # The fingerprint is built for the zero generator; since parity depends only
    # on (βs_state - βs_meas) mod 3 and the base context positions are fixed,
    # the parity values are generator-independent.  Verify for several states.
    Random.seed!(7)
    for _ in 1:20
        s      = QCScaling.random_state(nqubit)
        base_cxt = s.theta_s == 0 ?
            QCScaling.generate_base_context(nqubit, 0) :
            QCScaling.generate_base_context(nqubit, 1)
        # Build a state with the ZERO generator to compare against fingerprint
        zero_gen = base_cxt.pos[1]
        s_zero = QCScaling.PseudoGHZState(s.theta_s, s.theta_z, s.alphas, zero_gen)
        direct = QCScaling.parity(s_zero, base_cxt)
        @test fp[s] == Int.(direct)
    end
end

# -------------------------------------------------------------------------
@testset "get_goal_index / get_companion_index" begin
    # Use βs-based construction so po.index is 1-based (consistent with to_index).
    # get_goal_index: pairs (1,2)→1, (3,4)→2, (5,6)→3, ... i.e. (index+1)÷2
    for raw_idx in 0:3^nqubit - 1
        βs = QCScaling.to_ternary(raw_idx, nqubit)
        po = ParityOperator(βs)            # po.index == raw_idx + 1 (1-based)
        gi = QCScaling.get_goal_index(po)
        @test gi == (po.index + 1) ÷ 2    # == raw_idx÷2 + 1
        @test gi >= 1
        @test gi <= (3^nqubit + 1) ÷ 2
    end

    # Companion pairs: odd index ↔ next even index
    for raw_idx in 0:3^nqubit - 2   # leave room for companion
        βs = QCScaling.to_ternary(raw_idx, nqubit)
        po  = ParityOperator(βs)
        ci  = QCScaling.get_companion_index(po)
        # Adjacent: |companion - index| == 1
        @test abs(ci - po.index) == 1
        # Round-trip: companion of companion is self
        βs2 = QCScaling.to_ternary(ci - 1, nqubit)   # ci is 1-based
        po2 = ParityOperator(βs2)
        @test QCScaling.get_companion_index(po2) == po.index
    end
end

# -------------------------------------------------------------------------
@testset "calculate_representation" begin
    Random.seed!(42)
    states = QCScaling.random_state(nqubit, 30)
    rep    = QCScaling.calculate_representation(states)

    @test length(rep) == 3^nqubit
    @test all(x -> isnan(x) || x ∈ (0.0, 1.0), rep)

    # Deterministic
    rep2 = QCScaling.calculate_representation(states)
    for i in eachindex(rep)
        isnan(rep[i]) ? @test(isnan(rep2[i])) : @test(rep[i] == rep2[i])
    end
end

# -------------------------------------------------------------------------
@testset "Incremental rep cache consistency" begin
    # build_rep_incremental via apply_state! must equal calculate_representation
    Random.seed!(55)
    nstate = 20
    states = QCScaling.random_state(nqubit, nstate)
    cm     = QCScaling.ContextMaster(nqubit)
    n      = 3^nqubit

    rep_sum = zeros(Int, n)
    rep_ctr = zeros(Int, n)
    for s in states
        apply_state!(rep_sum, rep_ctr, s, cm, +1)
    end
    rep_incremental = rep_from_cache(rep_sum, rep_ctr)
    rep_package     = QCScaling.calculate_representation(states)

    for i in eachindex(rep_incremental)
        if isnan(rep_package[i])
            @test isnan(rep_incremental[i])
        else
            @test rep_incremental[i] == rep_package[i]
        end
    end

    # Removing a state and re-adding it leaves rep unchanged
    s_test = states[1]
    apply_state!(rep_sum, rep_ctr, s_test, cm, -1)
    apply_state!(rep_sum, rep_ctr, s_test, cm, +1)
    rep_after = rep_from_cache(rep_sum, rep_ctr)
    for i in eachindex(rep_incremental)
        if isnan(rep_incremental[i])
            @test isnan(rep_after[i])
        else
            @test rep_after[i] == rep_incremental[i]
        end
    end

    # Swap: remove state A, add state B, then undo — rep must return to original
    rep_sum_copy = copy(rep_sum)
    rep_ctr_copy = copy(rep_ctr)
    s_new = QCScaling.random_state(nqubit)
    apply_state!(rep_sum, rep_ctr, s_test, cm, -1)
    apply_state!(rep_sum, rep_ctr, s_new,  cm, +1)
    apply_state!(rep_sum, rep_ctr, s_new,  cm, -1)
    apply_state!(rep_sum, rep_ctr, s_test, cm, +1)
    @test rep_sum ≈ rep_sum_copy
    @test rep_ctr == rep_ctr_copy
end

# -------------------------------------------------------------------------
@testset "rep_accuracy_fast (package vs script)" begin
    # The package now exports rep_accuracy_fast; verify it matches the
    # script-local version defined at the top of this file.
    Random.seed!(77)
    nstate = 30
    states = QCScaling.random_state(nqubit, nstate)
    cm     = QCScaling.ContextMaster(nqubit)
    ngbits = (3^nqubit - 1) ÷ 2
    n      = 3^nqubit

    rep_sum = zeros(Int, n)
    rep_ctr = zeros(Int, n)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cm, +1); end

    goal = rand(0:1, ngbits)

    acc_pkg    = QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, goal)
    acc_script = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    @test acc_pkg ≈ acc_script
    @test 0.0 <= acc_pkg <= 1.0

    # Test across many random goals
    for _ in 1:20
        g = rand(0:1, ngbits)
        @test QCScaling.rep_accuracy_fast(rep_sum, rep_ctr, g) ≈
              rep_accuracy_fast(rep_sum, rep_ctr, g)
    end
end

@testset "rep_accuracy_fast" begin
    Random.seed!(77)
    nstate = 30
    states = QCScaling.random_state(nqubit, nstate)
    cm     = QCScaling.ContextMaster(nqubit)
    ngbits = (3^nqubit - 1) ÷ 2
    n      = 3^nqubit

    rep_sum = zeros(Int, n)
    rep_ctr = zeros(Int, n)
    for s in states; apply_state!(rep_sum, rep_ctr, s, cm, +1); end

    goal = rand(0:1, ngbits)

    acc = rep_accuracy_fast(rep_sum, rep_ctr, goal)
    @test 0.0 <= acc <= 1.0

    # Brute-force reference using integer comparisons
    function brute_force_acc(rep_sum::Vector{Int}, rep_ctr::Vector{Int}, goal)
        n_correct = 0
        for i in eachindex(goal)
            i1 = 2i-1; i2 = 2i
            rep_ctr[i1] == 0 && continue
            rep_ctr[i2] == 0 && continue
            s1 = rep_sum[i1]; c1 = rep_ctr[i1]
            s2 = rep_sum[i2]; c2 = rep_ctr[i2]
            (2*s1 == c1 || 2*s2 == c2) && continue
            predicted = (2*s1 > c1) != (2*s2 > c2) ? 1 : 0
            n_correct += (predicted == goal[i]) ? 1 : 0
        end
        return n_correct / length(goal)
    end

    @test acc ≈ brute_force_acc(rep_sum, rep_ctr, goal)

    # Boundary cases with manually constructed rep_sum/rep_ctr (Int)
    n_test = 10
    rs2 = zeros(Int, n); rc2 = zeros(Int, n)
    goal2 = zeros(Int, ngbits)
    for i in 1:n_test
        i1 = 2i-1; i2 = 2i
        rs2[i1] = 1; rc2[i1] = 1   # majority = 1
        rs2[i2] = 0; rc2[i2] = 1   # majority = 0 → bit = |1-0| = 1 ≠ goal=0 → wrong
    end
    acc2 = rep_accuracy_fast(rs2, rc2, goal2)
    @test acc2 == 0.0

    # Set i2 to majority=1 as well → bit = (1 != 1) = 0 == goal=0 → correct
    rs2[2:2:2*n_test] .= 1
    acc3 = rep_accuracy_fast(rs2, rc2, goal2)
    @test acc3 ≈ n_test / ngbits
end

# -------------------------------------------------------------------------
@testset "FingerprintPacked" begin
    fp        = QCScaling.Fingerprint(nqubit)
    fp_packed = QCScaling.FingerprintPacked(fp)

    nwords_expected = cld(2^(nqubit-1) + 1, 64)
    @test fp_packed.npos   == 2^(nqubit-1) + 1
    @test fp_packed.nwords == nwords_expected
    @test size(fp_packed.words) == (nwords_expected, 2, 2, 2^(nqubit-1))

    # Packed and unpacked pick_new_alphas must agree on every call
    Random.seed!(31)
    states   = QCScaling.random_state(nqubit, 20)
    cm       = QCScaling.ContextMaster(nqubit)
    ngbits   = (3^nqubit - 1) ÷ 2
    goal     = rand(0:1, ngbits)
    rep      = QCScaling.calculate_representation(states)

    for _ in 1:30
        gen_idx  = rand(0:3^nqubit-1)
        gen      = ParityOperator(gen_idx, nqubit)
        theta_s  = rand(0:1)
        base_cxt = theta_s == 0 ? cm.base_even : cm.base_odd
        cxt      = QCScaling.Context(gen, base_cxt)

        r_orig   = QCScaling.pick_new_alphas(cxt, goal, rep, fp,        base_cxt)
        r_packed = QCScaling.pick_new_alphas(cxt, goal, rep, fp_packed, base_cxt)
        @test r_orig[1] == r_packed[1]   # theta_s
        @test r_orig[2] == r_packed[2]   # theta_z
        @test r_orig[3] == r_packed[3]   # alphas
    end
end

@testset "pick_new_alphas optimality" begin
    # The returned (theta_s, theta_z, alphas) must achieve the minimum L1 distance
    # to companion_goal over all possible (theta_z, alphas) combinations.
    Random.seed!(13)
    nstate = 20
    states = QCScaling.random_state(nqubit, nstate)
    cm     = QCScaling.ContextMaster(nqubit)
    fp     = QCScaling.Fingerprint(nqubit)
    ngbits = (3^nqubit - 1) ÷ 2
    goal   = rand(0:1, ngbits)
    rep    = QCScaling.calculate_representation(states)

    for _ in 1:10
        gen_idx = rand(0:3^nqubit-1)
        gen     = ParityOperator(gen_idx, nqubit)
        for theta_s in 0:1
            base_cxt = theta_s == 0 ? cm.base_even : cm.base_odd
            cxt      = QCScaling.Context(gen, base_cxt)
            ts_ret, tz_ret, alphas_ret = QCScaling.pick_new_alphas(cxt, goal, rep, fp, base_cxt)

            # theta_s returned must equal base_cxt.parity
            @test ts_ret == base_cxt.parity

            # Compute the L1 score for the returned choice
            cg      = QCScaling.companion_goal(cxt, goal, rep)
            s_best  = QCScaling.PseudoGHZState(ts_ret, tz_ret, alphas_ret, base_cxt.pos[1])
            fp_best = fp[s_best]
            score_returned = sum(abs(fp_best[i] - cg[i]) for i in eachindex(cg) if !isnan(cg[i]))

            # Verify no other (tz, alphas) achieves a strictly lower score
            for tz in 0:1
                for ai in 0:2^(nqubit-1)-1
                    alphas_test = QCScaling.idx_to_alphas(ai, nqubit)
                    s_test  = QCScaling.PseudoGHZState(ts_ret, tz, alphas_test, base_cxt.pos[1])
                    fp_test = fp[s_test]
                    score_test = sum(abs(fp_test[i] - cg[i]) for i in eachindex(cg) if !isnan(cg[i]))
                    @test score_returned <= score_test
                end
            end
        end
    end
end

# -------------------------------------------------------------------------
@testset "score" begin
    Random.seed!(42)
    states = QCScaling.random_state(nqubit, 30)
    cm     = QCScaling.ContextMaster(nqubit)
    goal   = rand(0:1, (3^nqubit - 1) ÷ 2)

    scores = QCScaling.score(states, goal, cm)
    @test length(scores) == length(states)
    @test all(s -> s isa Integer, scores)

    # Deterministic
    @test scores == QCScaling.score(states, goal, cm)

    # Single-threaded and multi-threaded results agree
    rep = QCScaling.calculate_representation(states)
    s_single = QCScaling.score(states, rep, goal, cm; multi_threading=false)
    s_multi  = QCScaling.score(states, rep, goal, cm; multi_threading=true)
    @test s_single == s_multi
end

# -------------------------------------------------------------------------
@testset "State cache correctness" begin
    Random.seed!(42)
    nstate = 20
    states = QCScaling.random_state(nqubit, nstate)
    cm     = QCScaling.ContextMaster(nqubit)
    n      = 3^nqubit
    npos   = length(cm.base_even.pos)

    alloc_cache() = (Vector{Int}(undef, npos), Vector{Int}(undef, npos))

    # apply_state_cached! matches apply_state! for every state, sign=+1
    for s in states
        rs_ref = zeros(Int, n); rc_ref = zeros(Int, n)
        rs_cac = zeros(Int, n); rc_cac = zeros(Int, n)
        apply_state!(rs_ref, rc_ref, s, cm, +1)
        idxs, pars = alloc_cache()
        fill_state_cache!(idxs, pars, s, cm)
        apply_state_cached!(rs_cac, rc_cac, idxs, pars, +1)
        @test rs_ref == rs_cac
        @test rc_ref == rc_cac
    end

    # sign=-1 also matches
    rs_ref = zeros(Int, n); rc_ref = zeros(Int, n)
    rs_cac = zeros(Int, n); rc_cac = zeros(Int, n)
    for s in states
        apply_state!(rs_ref, rc_ref, s, cm, +1)
        idxs, pars = alloc_cache(); fill_state_cache!(idxs, pars, s, cm)
        apply_state_cached!(rs_cac, rc_cac, idxs, pars, +1)
    end
    s0 = first(states)
    apply_state!(rs_ref, rc_ref, s0, cm, -1)
    idxs, pars = alloc_cache(); fill_state_cache!(idxs, pars, s0, cm)
    apply_state_cached!(rs_cac, rc_cac, idxs, pars, -1)
    @test rs_ref == rs_cac
    @test rc_ref == rc_cac

    # Round-trip: cached add then remove leaves rep_sum/rep_ctr unchanged
    rs = zeros(Int, n); rc = zeros(Int, n)
    for s in states
        idxs, pars = alloc_cache(); fill_state_cache!(idxs, pars, s, cm)
        apply_state_cached!(rs, rc, idxs, pars, +1)
    end
    rs_snap = copy(rs); rc_snap = copy(rc)
    idxs, pars = alloc_cache(); fill_state_cache!(idxs, pars, s0, cm)
    apply_state_cached!(rs, rc, idxs, pars, -1)
    apply_state_cached!(rs, rc, idxs, pars, +1)
    @test rs == rs_snap
    @test rc == rc_snap

    # update_rep_at_cached! matches update_rep_at! after a swap
    rs = zeros(Int, n); rc = zeros(Int, n)
    for s in states; apply_state!(rs, rc, s, cm, +1); end
    rep_ref = rep_from_cache(rs, rc)
    rep_cac = copy(rep_ref)

    s_old = first(states)
    s_new = QCScaling.random_state(nqubit)
    apply_state!(rs, rc, s_old, cm, -1)
    apply_state!(rs, rc, s_new, cm, +1)

    update_rep_at!(rep_ref, rs, rc, s_old, cm)
    update_rep_at!(rep_ref, rs, rc, s_new, cm)

    idxs_old, _ = alloc_cache(); fill_state_cache!(idxs_old, Vector{Int}(undef, npos), s_old, cm)
    idxs_new, _ = alloc_cache(); fill_state_cache!(idxs_new, Vector{Int}(undef, npos), s_new, cm)
    update_rep_at_cached!(rep_cac, rs, rc, idxs_old)
    update_rep_at_cached!(rep_cac, rs, rc, idxs_new)

    for i in eachindex(rep_ref)
        isnan(rep_ref[i]) ? @test(isnan(rep_cac[i])) : @test(rep_ref[i] == rep_cac[i])
    end

    # Accuracy computed from a fully cached ensemble matches the uncached baseline
    rs_u = zeros(Int, n); rc_u = zeros(Int, n)
    rs_c = zeros(Int, n); rc_c = zeros(Int, n)
    for s in states
        apply_state!(rs_u, rc_u, s, cm, +1)
        idxs, pars = alloc_cache(); fill_state_cache!(idxs, pars, s, cm)
        apply_state_cached!(rs_c, rc_c, idxs, pars, +1)
    end
    goal = rand(0:1, (n - 1) ÷ 2)
    @test QCScaling.rep_accuracy_fast(rs_u, rc_u, goal) ==
          QCScaling.rep_accuracy_fast(rs_c, rc_c, goal)
end

# -------------------------------------------------------------------------
@testset "Numerical regression" begin
    # Hard-coded inputs with pre-verified outputs.
    # Any change to parity logic will break these.

    # nqubit=2 cases (easy to verify by hand)
    gen_a = ParityOperator([1, 2])
    gen_b = ParityOperator([2, 1])

    # all_zero: state measured by its own generator
    state_00 = QCScaling.PseudoGHZState(0, 0, [0], gen_a)
    state_01 = QCScaling.PseudoGHZState(0, 0, [1], gen_a)
    @test QCScaling.parity(state_00, gen_a) == 0.0   # sum(alphas)=0
    @test QCScaling.parity(state_01, gen_a) == 1.0   # sum(alphas)=1

    # theta_s=1, sum_J+theta_s even → returns mod(theta_z + ..., 2)
    # gen_a=[1,2] vs gen_b=[2,1]: bd=[2,1], j=[0,1], sum_J=1
    # theta_s=1: sum_J+1=2 even; dot_J_alpha: i=1<2, j=0, so dot=0
    # returns mod(theta_z + 1 + 0, 2)
    s_tz0 = QCScaling.PseudoGHZState(1, 0, [0], gen_a)
    s_tz1 = QCScaling.PseudoGHZState(1, 1, [0], gen_a)
    @test QCScaling.parity(s_tz0, gen_b) == 1.0
    @test QCScaling.parity(s_tz1, gen_b) == 0.0

    # theta_s=0: sum_J+0=1 odd → 0.5
    @test QCScaling.parity(state_00, gen_b) == 0.5

    # nqubit=4 fixed snapshot
    po_snap   = ParityOperator([1, 1, 2, 2])
    state_snap = QCScaling.PseudoGHZState(0, 1, [1, 0, 1], ParityOperator([2, 1, 0, 2]))
    p_snap = QCScaling.parity(state_snap, po_snap)
    @test p_snap ∈ (0.0, 0.5, 1.0)
    @test p_snap == 0.5   # bd=[mod(-1,3),0,...] → n_zero>0 → 0.5
end

end # @testset "QCScaling Tests"
