using Statistics: norm

### Abstract type for loss objectives
"""
    abstract type Loss

The abstract type for loss objectives. Concrete types subtyping `Loss` must
define a `penalty` method to specify the penalty function to calculate the loss,
and a `derivative` method to compute the derivative of such penalty function.
Loss is calculated as the mean of the penalty applied to the predictions and
observations.
"""
abstract type Loss end

# Useful error messages if required methods are not specified
penalty(l::Loss) = error("No `penalty` method defined for type $(typeof(l)).")
derivative(l::Loss) =
    error("No `gradient` method defined for type $(typeof(l)).")

# Loss objective evaluation
"""
    (l::Loss)(ŷ, y)

Compute the loss objective by taking the mean of the penalty applied to `ŷ - y`.
"""
(l::Loss)(ŷ, y) = sum(penalty(l), ŷ - y) / length(y)

# Loss objective gradient
"""
    gradient(l::Loss, ŷ, y)

Compute the gradient of the loss objective with the `derivative` method.
"""
gradient(l::Loss, ŷ, y) = derivative(l).(ŷ - y) / length(y)

### Square loss
"""
    struct SquareLoss

Self-explanatory loss function by squaring the error.
"""
struct SquareLoss <: Loss end

penalty(l::SquareLoss) = x -> x^2
derivative(l::SquareLoss) = x -> 2x

### Huber loss
"""
    struct HuberLoss

Loss objective using the Huber penalty function with parameter `α`.
"""
struct HuberLoss <: Loss
    α
end

penalty(l::HuberLoss) = x -> (abs(x) <= l.α ? x^2 : l.α * (2abs(x) - l.α))
derivative(l::HuberLoss) = x -> (abs(x) <= l.α ? 2 * x : 2 * l.α * x / abs(x))

### Log Huber loss

"""
    struct LogHuberLoss

Loss objective using the log Huber penalty function with parameter `α`.
"""
struct LogHuberLoss <: Loss
    α
end

penalty(l::LogHuberLoss) =
    x -> (abs(x) <= l.α ? x^2 : l.α^2 * (1 - 2log(l.α) + log(x^2)))
derivative(l::LogHuberLoss) = x -> (abs(x) <= l.α ? 2 * x : 2 * l.α^2 / x)

### Gradient Descent
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
        ∇ = X' * gradient(loss, X * θ, y)
        norm(∇) ≤ ϵ && break

        # Iterate to find better θ
        while true
            tent_θ = θ - h * ∇ # tentative update to θ
            if loss(X * tent_θ, y) ≤ loss(X * θ, y)
                θ = tent_θ
                h *= 1.2
                break
            else
                h *= 0.5
            end
        end
    end

    return θ
end
