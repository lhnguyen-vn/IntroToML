using Statistics: norm, mean
using DataStructures: SortedDict

"""
    nn_predictor(x, X, y)

Given `x`, find the nearest value to `x` in `X` and return the corresponding
`y` value.
"""
function nn_predictor(x, X, y)
    ŷ = nothing # so that ŷ is accessible after for block
    min_distance = Inf

    # Loop through X, find the nearest neighbor
    for (i, row) in enumerate(eachrow(X))
        distance = norm(x - row)
        if distance < min_distance
            min_distance = distance # type stable!
            ŷ = y[i]
        end
    end

    return ŷ
end

"""
    knn_predictor(x, X, y, k)

Given `x`, find `k` nearest neighbors to `x` in `X` and return the average of
the `k` corresponding `y`'s.
"""
function knn_predictor(x, X, y, k)
    num_data = size(X, 1)
    @assert k < num_data "Dataset has only $num_data data points"
    sd = SortedDict{Float64, Float64}() # distance will be sorted in Dict

    # Loop through X, add (distance, y) pairs to sd
    for (i, row) in enumerate(eachrow(X))
        distance = norm(x - row)
        sd[distance] = y[i]
    end

    kNN = collect(keys(sd))[1:k] # the k closest distance to x
    ŷ = mean(sd[nn] for nn in kNN)

    return ŷ
end
