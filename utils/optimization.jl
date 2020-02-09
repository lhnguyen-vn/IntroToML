using Statistics: norm

"""
    Loss(penalty, gradient)

A loss objective wrapper of the penalty and the gradient.
"""
struct Loss
    penalty::Function
    gradient::Function
end

Loss(penalty, gradient, params...) =
    Loss(penalty(params...), gradient(params...))

# Compute loss
(L::Loss)(diff) = sum(L.penalty, diff) / length(y)
# Compute gradient
grad(L::Loss, diff) = L.gradient.(diff) / length(y)

# Square loss
"""
    p_square(error)

Compute square penalty for a given error.
"""
p_square(x) = x^2

"""
    delta_square(error)

Compute the derivative for the square penalty at the error.
"""
delta_p_square(x) = 2 * x

square_loss = Loss(p_square, delta_p_square)

# Huber loss
"""
    p_huber(α)

Return the Huber penalty function for a given `α`.
"""
p_huber(α) = x -> (abs(x) <= α ? x^2 : α * (2abs(x) - α))

"""
    delta_huber(α)

Return the derivative function for the Huber penalty for a given `α`.
"""
delta_p_huber(α) = x -> (abs(x) <= α ? 2 * x : 2 * α * x / abs(x))

huber_loss(α) = Loss(p_huber, delta_p_huber, α)

# Log Huber loss

"""
    p_log_huber(α)

Return the log Huber penalty function for a given `α`.
"""
p_log_huber(α) = x -> (abs(x) <= α ? x^2 : α^2 * (1 - 2log(α) + log(x^2)))

"""
    delta_log_huber(α)

Return the derivative function for the log Huber penalty of a given `α`.
"""
delta_log_huber(α) = x -> (abs(x) <= α ? 2 * x : 2 * α^2 / x)

log_huber_loss(α) = Loss(p_log_huber, delta_log_huber, α)

# Gradient Descent
"""
    GD(X, y, loss[, max_iter=nothing, ϵ=1e-7, h=0.1])

Gradient descent on the input `X` and the output `y` using the loss function
provided. The algorithm stops when it reaches maximum iterations, or if the
Euclidean norm of the gradient is lower than or equal to `ϵ`.
"""
function GD(X, y, loss; max_iter=100, ϵ=1e-7, h=0.1)
    θ = randn(size(X, 2), size(y, 2))
    iter = 0

    while iter ≤ max_iter
        iter += 1

        # Compute loss gradient and break if gradient is small enough
        ∇ = X' * grad(loss, X * θ - y)
        norm(∇) ≤ ϵ && break

        # Iterate to find better θ
        while true
            tent_θ = θ - h * ∇ # tentative update to θ
            if loss(X * tent_θ - y) ≤ loss(X * θ - y)
                θ = tent_θ
                h *= 1.2
                break
            else
                h *= 0.5
            end
        end
    end

    # println("Gradient descent after $iter iterations:",
            # "\n\tFinal loss: $(loss(X * θ - y))")
    return θ
end
