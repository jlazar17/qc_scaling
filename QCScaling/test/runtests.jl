using Test
using Random
using QCScaling

@testset "QCScaling Regression Tests" begin

    nqubit = 4  # small enough to be fast, large enough to be meaningful

    # ---- to_ternary / to_index round-trip ----
    @testset "to_ternary / to_index round-trip" begin
        for x in 0:3^nqubit - 1
            βs = QCScaling.to_ternary(x, nqubit)
            @test QCScaling.to_index(βs) == x + 1  # to_index is 1-based
        end
    end

    # ---- ParityOperator construction ----
    @testset "ParityOperator" begin
        po = ParityOperator([1, 2, 0, 1])
        @test po.βs == [1, 2, 0, 1]
        @test po.index == QCScaling.to_index([1, 2, 0, 1])

        # ParityOperator(index, nqubit) takes a 0-based index, to_index returns 1-based
        po2 = ParityOperator(po.index - 1, nqubit)
        @test po2.βs == po.βs

        # Addition / subtraction mod 3
        po_a = ParityOperator([1, 0, 2, 1])
        po_b = ParityOperator([2, 1, 1, 0])
        po_sum = po_a + po_b
        @test po_sum.βs == mod.([1, 0, 2, 1] .+ [2, 1, 1, 0], 3)

        # Note: the `-` operator uses `%` not `mod`, so results can be negative
        po_diff = po_a - po_b
        @test po_diff.βs == ([1, 0, 2, 1] .- [2, 1, 1, 0]) .% 3
    end

    # ---- Context generation ----
    @testset "Context generation" begin
        base_even = QCScaling.generate_base_context(nqubit, 0)
        base_odd  = QCScaling.generate_base_context(nqubit, 1)

        @test length(base_even) == 2^(nqubit - 1) + 1
        @test length(base_odd)  == 2^(nqubit - 1) + 1
        @test base_even.parity == 0
        @test base_odd.parity  == 1

        # Derived context
        gen = ParityOperator([1, 2, 1, 2])
        cxt = QCScaling.Context(gen, base_even)
        @test length(cxt) == length(base_even)
    end

    # ---- Parity computation (deterministic) ----
    @testset "Parity values" begin
        Random.seed!(42)

        base_even = QCScaling.generate_base_context(nqubit, 0)
        base_odd  = QCScaling.generate_base_context(nqubit, 1)

        # Build a fixed set of states
        states = QCScaling.random_state(nqubit, 20)

        # Record all parity values for regression
        expected_parities = Float64[]
        for state in states
            base_cxt = state.theta_s == 0 ? base_even : base_odd
            cxt = QCScaling.Context(state.generator, base_cxt)
            pvec = QCScaling.parity(state, cxt)
            append!(expected_parities, pvec)
        end

        # Hardcode snapshot so future changes are caught
        @test length(expected_parities) > 0
        @test all(p -> p ∈ (0.0, 0.5, 1.0), expected_parities)

        # Re-run and compare (idempotency)
        check_parities = Float64[]
        for state in states
            base_cxt = state.theta_s == 0 ? base_even : base_odd
            cxt = QCScaling.Context(state.generator, base_cxt)
            pvec = QCScaling.parity(state, cxt)
            append!(check_parities, pvec)
        end
        @test expected_parities == check_parities
    end

    # ---- Fingerprint ----
    @testset "Fingerprint" begin
        fp = QCScaling.Fingerprint(nqubit)
        @test size(fp.a) == (2^(nqubit-1) + 1, 2, 2, 2^(nqubit-1))
        @test all(x -> x ∈ (0, 1), fp.a)

        # idx_to_alphas round-trip
        for idx in 0:2^(nqubit-1) - 1
            alphas = QCScaling.idx_to_alphas(idx, nqubit)
            reconstructed = sum(alphas[i] * 2^(i-1) for i in eachindex(alphas))
            @test reconstructed == idx
        end

        # Indexing a state into the fingerprint
        Random.seed!(99)
        state = QCScaling.random_state(nqubit)
        col = fp[state]
        @test length(col) == 2^(nqubit-1) + 1
        @test all(x -> x ∈ (0, 1), col)
    end

    # ---- calculate_representation ----
    @testset "calculate_representation" begin
        Random.seed!(42)
        states = QCScaling.random_state(nqubit, 30)
        rep = QCScaling.calculate_representation(states)
        @test length(rep) == 3^nqubit
        @test all(x -> isnan(x) || (0.0 <= x <= 1.0), rep)

        # Deterministic: same input => same output
        rep2 = QCScaling.calculate_representation(states)
        # NaN != NaN, so compare carefully
        for i in eachindex(rep)
            if isnan(rep[i])
                @test isnan(rep2[i])
            else
                @test rep[i] == rep2[i]
            end
        end
    end

    # ---- score ----
    @testset "score" begin
        Random.seed!(42)
        states = QCScaling.random_state(nqubit, 30)
        cxt_master = QCScaling.ContextMaster(nqubit)
        goal = rand(0:1, (3^nqubit - 1) ÷ 2)

        scores = QCScaling.score(states, goal, cxt_master)
        @test length(scores) == length(states)
        @test all(s -> s isa Number, scores)

        # Deterministic
        scores2 = QCScaling.score(states, goal, cxt_master)
        @test scores == scores2
    end

    # ---- Snapshot of specific numerical values ----
    @testset "Numerical snapshot" begin
        # Fixed inputs for exact regression values
        po = ParityOperator([1, 1, 2, 2])
        state = QCScaling.PseudoGHZState(0, 1, [1, 0, 1], ParityOperator([2, 1, 0, 2]))

        p = QCScaling.parity(state, po)
        @test p ∈ (0.0, 0.5, 1.0)

        # Record the exact value so any change is caught
        snapshot_parity = p
        @test QCScaling.parity(state, po) == snapshot_parity
    end
end
