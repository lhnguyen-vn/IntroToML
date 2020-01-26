using Random: randperm
using Statistics: mean, std
using Base.Iterators: partition

"""
    split_dataset(x, y, ratio=0.8)

Given the dataset input `u`, output `v`, and the train-test ratio, return
randomized train and test sets.
"""
function split_dataset(x, y, ratio=0.8)
    @assert size(x, 1) == size(y, 1) "Number of inputs must equal outputs"

    # Randomize indices
    num_data = size(x, 1)
    rand_indices = randperm(num_data)

    # Split into train and test sets
    split_index = floor(Int, num_data * ratio)
    train_indices = rand_indices[1:split_index]
    test_indices = rand_indices[split_index+1:end]

    # Return datasets as named tuples
    train_set = (input=x[train_indices], output=y[train_indices]
    test_set = (input=x[test_indices], output=y[test_indices])
    return train_set, test_set
end

"""
    cross_validation(X, y, k)

Given the `n x d` matrix `X`, the label vector `y`, and the `k` number of folds,
return the average and standard deviation of the RMSE by cross validation.
"""
function cross_validation(X, y, k)
    @assert k > 1 "There must be more than one fold."

    # Randomize data
    num_data = size(X, 1)
    random_indices = randperm(num_data)

    # Divide into folds
    fold_size = floor(Int, num_data / k) # get fold size
    fold_indices = collect(partition(random_indices[1:k*fold_size], fold_size))
    # Append the rest, if there are any, to the last fold
    if num_data % k != 0
        append!(fold_indices[end], random_indices[k*fold_size+1:end])
    end

    RMSEs = Float64[]
    for fold in fold_indices
        # Set up train set and test set
        train_indices = trues(num_data)
        train_indices[fold] .= false
        test_X, test_y = X[fold, :], y[fold]
        train_X, train_y = X[train_indices, :], y[train_indices]

        # Regression
        _, _, error = find_theta(train_X, train_y)
        push!(RMSEs, error)
    end

    return mean(RMSEs), std(RMSEs, corrected=false)
end
