function parity(
    state::PseudoGHZState,
    cxt::Context
)::Vector{Float64}
    parity.(Ref(state), cxt)
end

function parity(
    state::PseudoGHZState,
    measurement_po::ParityOperator{N}
)::Float64 where {N}
    βs_state = state.generator.βs
    βs_meas = measurement_po.βs
    alphas = state.alphas

    n_zero = 0
    all_zero = true
    sum_J = 0
    dot_J_alpha = 0

    @inbounds for i in 1:N
        bd = mod(βs_state[i] - βs_meas[i], 3)
        if bd == 0
            n_zero += 1
            if !all_zero
                # already found a non-zero, so we have mixed => any(iszero) path
                # but we need to finish counting to know if ALL are zero
            end
        else
            all_zero = false
            j = 1 - mod(bd - 1, 2)
            sum_J += j
            if i < N
                dot_J_alpha += j * alphas[i]
            end
        end
    end

    if all_zero
        return mod(sum(alphas), 2)
    elseif n_zero > 0
        return 0.5
    elseif mod(sum_J + state.theta_s, 2) == 1
        return 0.5
    else
        return mod(state.theta_z + (sum_J + state.theta_s) ÷ 2 + dot_J_alpha, 2)
    end
end
