includet("./0_random_variation.jl")

nqubit = 8
states = [QCScaling.random_state(nqubit) for _ in 1:200]
goal = rand(0:1, Int((3^nqubit-1)/2))
cxt_master = QCScaling.ContextMaster(nqubit)
rep = QCScaling.calculate_representation(states)

println(QCScaling.score(states, rep, goal, cxt_master))
