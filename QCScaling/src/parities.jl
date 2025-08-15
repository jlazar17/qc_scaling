function parity(
    state::PseudoGHZState,
    cxt::Context
)::Vector{Float64}
    parity.(Ref(state), cxt)
end

function parity(
    state::PseudoGHZState,
    measurement_po::ParityOperator
)::Float64
    beta_diff = mod.(state.generator.βs .- measurement_po.βs, 3)
    J = 1 .- mod.((beta_diff .- 1), 2)
    if all(iszero, beta_diff)
        return mod(sum(state.alphas), 2)
    elseif any(iszero, beta_diff)
        return 0.5
    elseif mod(sum(J) + state.theta_s, 2) == 1
        return 0.5
    else
        return @views mod(state.theta_z + (sum(J) + state.theta_s) // 2 + sum(J[1:end-1] .* state.alphas), 2)
    end
end
