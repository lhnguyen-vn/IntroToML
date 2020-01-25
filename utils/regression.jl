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
