using QCScaling
using BenchmarkTools
using Random

Random.seed!(42)

nqubit = 8
states = QCScaling.random_state(nqubit, 200)

println("=== Benchmarks ===")
println()

# 1. to_ternary
print("to_ternary(1000, 8):             ")
display(@benchmark QCScaling.to_ternary(1000, 8))
println()

# 2. to_index
βs = [1, 2, 0, 1, 2, 1, 0, 2]
print("to_index(βs):                    ")
display(@benchmark QCScaling.to_index($βs))
println()

# 3. ParityOperator construction from index
print("ParityOperator(idx, nqubit):      ")
display(@benchmark ParityOperator(1000, $nqubit))
println()

# 4. ParityOperator addition
po_a = ParityOperator([1, 0, 2, 1, 2, 0, 1, 2])
po_b = ParityOperator([2, 1, 1, 0, 0, 2, 1, 1])
print("ParityOperator + ParityOperator:  ")
display(@benchmark $po_a + $po_b)
println()

# 5. generate_base_context
print("generate_base_context(8, 0):      ")
display(@benchmark QCScaling.generate_base_context($nqubit, 0))
println()

# 6. parity (single state, single PO)
base_even = QCScaling.generate_base_context(nqubit, 0)
state = states[1]
po = base_even.pos[5]
print("parity(state, po):               ")
display(@benchmark QCScaling.parity($state, $po))
println()

# 7. parity (single state, full context)
cxt = QCScaling.Context(state.generator, base_even)
print("parity(state, cxt):              ")
display(@benchmark QCScaling.parity($state, $cxt))
println()

# 8. calculate_representation
base_odd = QCScaling.generate_base_context(nqubit, 1)
print("calculate_representation:         ")
display(@benchmark QCScaling.calculate_representation($states, $base_even, $base_odd))
println()

# 9. Fingerprint construction
print("Fingerprint(nqubit):             ")
display(@benchmark QCScaling.Fingerprint($nqubit))
println()
