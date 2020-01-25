using Statistics: norm

"""
    RMSE(ŷ, y)

Compute the root-mean-squared error between `ŷ` and `y`.
"""
RMSE(ŷ, y) = norm(ŷ - y) / sqrt(length(y))
