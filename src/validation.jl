# This file contains code for validating/testing the model

# Root mean squared error
rmse(y_true, y_pred) = sqrt(mean((y_true .- y_pred).^2))
export rmse

# TODO: Find a better measure of accuracy that utilizes likelihood

# K-folds test based on pre-trained parameters
function kfolds_pretrained(X, Y, θ; metric=rmse, k=6, σ_n)
    n_samples = size(X,2)
    @assert n_samples % k == 0
    test_size = Int(n_samples/k)    # number of samples in the validation group
    all_ind = collect(1:n_samples)
    @show all_ind
    errors = []

    for i in 1:k 
        test_ind = all_ind[(i*test_size - test_size + 1):(i*test_size)]
        train_ind = [j for j in all_ind if j ∉ test_ind]
        @show test_ind
        @show train_ind
        for m in test_ind
        y_true = Y[:, m]
        y_pred, _ = predict_y(X[:, train_ind], X[:, m],
                           Y[:, train_ind], θ; σ_n = σ_n)
        error = metric(y_true, y_pred)
        @show error
        push!(errors, error)
        end
    end
    μ_ers = mean(errors); std_ers = std(errors)
    println("error = $(μ_ers) ± $std_ers")
    return errors
end
export kfolds_pretrained

function hot_one_out_retrained(θ_init, X, Y; metric=rmse, σ_n=0.1^2)
    n_samples = size(X,2)
    all_ind = collect(1:n_samples)
    errors = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples 
        test_ind = i
        train_ind = [j for j in all_ind if j ≠ i]
        @show train_ind
        D = cos_dis(X[:, train_ind])
        sol = opt_kernel(θ_init, D, Y[:, train_ind]; σ_n = σ_n)
        θ = sol.u

        # Only use temps greater than 200 for the comparison
        comp_ind = [m for m in 1:size(X,1) if X[m, i] > 100]

        y_true = Y[:, test_ind]
        y_pred, _ = predict_y(X[:, train_ind], X[:, test_ind],
                           Y[:, train_ind], θ; σ_n = σ_n)
        errors[i] = metric(y_true[comp_ind], y_pred[comp_ind])
    end
    μ_ers = mean(errors); std_ers = std(errors)
    println("error = $(μ_ers) ± $std_ers")
    return errors
end
export hot_one_out_retrained
