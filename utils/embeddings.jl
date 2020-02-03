using Statistics: mean, std

"""
    polynomial_features(x, degree)

Return a polynomial embedding of degree `degree - 1` of the `n`-vector `x` and
return a `n x degree` matrix, where each column is the polynomial mapping of
each scalar in `x`.
"""
function polynomial_features(x, degree)
    embedded_x = Array{eltype(x)}(undef, length(x), degree)
    for d in 1:degree
        embedded_x[:, d] = x.^(d - 1) # Julia is column major
    end
    return embedded_x
end

"""
    zscore(U)

Take in an input matrix with feature columns, return the standardized matrix,
the means and the standard deviations of each feature.
"""
function zscore(U)
    means = mean(U, dims=1)
    stdevs = std(U, dims=1)
    X = similar(U, Float64) # set up a matrix with the same dimension

    # Loop through each feature
    for i in 1:size(U, 2)
        X[:, i] = (U[:, i] .- means[i]) / stdevs[i]
    end

    return X, means, stdevs
end
