# This file contains code for validating/testing the model
using Statistics

# Root mean squared error
rmse(y_true, y_pred) = sqrt(mean((y_true .- y_pred).^2))
export rmse

function R_sq(y_true, y_pred)
    ȳ = mean(y_true)
    SS_res = sum((y_true .- y_pred).^2)
    SS_tot = sum((y_true .- ȳ).^2)
    R² = 1 .- SS_res ./ SS_tot
    return R²
end
export R_sq

function test_likelihood(μ_pred, σ_pred, y_true)
    f = map(eachindex(y_true)) do i
        1/(σ_pred[i] * sqrt(2*π)) * exp(-1/2 * ((y_true[i] - μ_pred[i])/σ_pred[i])^2)
    end
    return (mean(f), std(f))
end
export test_likelihood

function leave_one_out_retrained(θ_init, X, Y, tcpath; σ_n=1e-8,
                               features = [1], t_limit=1000)
    n_samples = size(X,2)
    all_ind = collect(1:n_samples)
    R2s = Vector{Float64}(undef, n_samples)   # coefficients of determination
    rmses = Vector{Float64}(undef, n_samples) # root mean square errors
    nlmls = Vector{Float64}(undef, n_samples) # negative log marginal likelihoods
    mean_likelihood = Vector{Float64}(undef, n_samples)
    std_likelihood = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples 
        test_ind = i
        train_ind = [j for j in all_ind if j ≠ i]

        D_XX = build_all_distance_tensors(X[:,train_ind], tcpath, features=features)
        sol = opt_kernel(θ_init, D_XX, Y[:,train_ind];
                         σ_n = σ_n, n_starts=1, t_limit = t_limit)
        θ = sol.u
        nlmls[i] = sol.objective

        # Only use temps greater than 200 for the comparison
        # This prevents the 'tail' from making the results look
        # better than they really are because most transformations
        # occur above 200 C
        comp_ind = [m for m in 1:size(X,1) if X[m, i] > 200]

        y_true = Y[:, test_ind]
        D_sX = build_all_distance_tensors(X[:, test_ind], X[:, train_ind],
                                          tcpath; features=features)
        X_star = X[:, test_ind]
        X_star = reshape(X_star, (size(X_star, 1), size(X_star, 2)))
        D_ss = build_all_distance_tensors(X_star,
                                          tcpath; features=features)
        y_pred, s_pred = gp_inference(D_XX, D_sX, D_ss, Y[:, train_ind], θ)
        σ_pred = sqrt.(abs.(s_pred))
        R2s[i] = R_sq(y_true[comp_ind], y_pred[comp_ind])
        rmses[i] = rmse(y_true[comp_ind], y_pred[comp_ind])
        mean_likelihood[i], std_likelihood[i] = test_likelihood(y_pred[comp_ind],
                                                             σ_pred[comp_ind],
                                                             y_true[comp_ind])
    end

    μ_R2s = mean(R2s); std_R2s = std(R2s)
    μ_rmses = mean(rmses); std_rmses = std(rmses)
    μ_likelihood = mean(mean_likelihood); std_μ_likelihood = std(mean_likelihood)
    μ_std_likelihood = mean(std_likelihood); std_std_likelihood = std(std_likelihood)
    println("R² = $(μ_R2s) ± $std_R2s")
    println("RMSQE = $(μ_rmses) ± $std_rmses")
    println("Likelihood = $(μ_likelihood) ± $μ_std_likelihood")
    return hcat(R2s, rmses, mean_likelihood, std_likelihood)
end
export leave_one_out_retrained

function run_all_validations(X, Y, tcpath; σ_n = 1e-8)
    feature_sets = [[1], [5], [6], [7], [1,5,6,7]]
    measure_entry = []
    full_H = []
    summary_vecs = []
    for i in eachindex(feature_sets)
        println("Running features: $(feature_sets[i])...")
        features = feature_sets[i]
        n_features = length(features)
        θi = vcat([5.0], repeat([0.1], n_features))
        H = leave_one_out_retrained(θi, X,Y, tcpath; σ_n = σ_n,
                                    features=features)
        push!(full_H, H)
        summary_vec = []
        for col in eachcol(H)
            μ = mean(col); σ = std(col)
            sum_string = "$μ ± $σ"
            push!(summary_vec, sum_string)
        end
        push!(summary_vecs, summary_vec)
    end
    return summary_vecs, full_H
end
export run_all_validations
