"""
    find_theta(X, y)

Find the θ to minimize the root-mean-squared error between ŷ = X * θ and y.
Return the θ and the error.
"""
function find_theta(X, y)
    θ = X \ y # matrix least square is implemented in Julia as left division
    ŷ = X * θ
    return θ, ŷ, RMSE(ŷ, y)
end

"""
    function find_theta_ridge(X, y, λ)

Given the input `X`, the output `y`, and a regularization parameter `λ`, perform
ridge regression and return the `θ∗`, `ŷ`, and the RMSE.
"""
function find_theta_ridge(X, y, λ, affine=false)
    Xᵀ = transpose(X)

    # If the regression is linear, then there is no adjustment to I below
    # Else, if X has a constant feature, then make I[1, 1] into a 0.
    adjustment = zeros(eltype(X), size(X, 2), size(X, 2))
    affine && (adjustment[1, 1] = one(eltype(X)))
    mod_I = I - adjustment

    θ = (Xᵀ * X + λ * mod_I)^-1 * Xᵀ * y
    ŷ = X * θ
    error = RMSE(ŷ, y)
    return θ, ŷ, error
end
