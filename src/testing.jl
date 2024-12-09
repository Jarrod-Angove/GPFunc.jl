using LinearAlgebra, DilPredict, Optimization, OptimizationOptimJL
using GLMakie, Statistics, SpecialFunctions

function pull_test_data(;trial_path = "../dil_data/dil_X70_2014-05-28_ualberta/",
                        T_start=1000.0, T_end=200.0, scrunch = false, check_plot=false, scaler=1)
    trial = DilPredict.get_trial(trial_path)

    # Plot the data to make sure T_start and T_end are not 
    # cutting anything off -- disabled by default but useful for 
    # debugging
    if check_plot
        f = Figure()
        ax1 = Axis(f[1,1]; xlabel="Time (s)", ylabel="Temp (C)")
        ax2 = Axis(f[1,2]; xlabel="Temp (C)", ylabel="ΔL (μm)")
        for i in 1:length(trial.runs)
            lines!(ax1, trial.runs[i].time, trial.runs[i].temp)
            lines!(ax2, trial.runs[i].temp, trial.runs[i].dL)
            hlines!(ax1, [T_start, T_end], color=:red)
            vlines!(ax2, [T_start, T_end], color=:red)
        end
        println("Cooling range is currently set to:")
        @show T_start
        @show T_end
        println("Please check the plot to make sure these are correct")
        display(f)
    end

    # Regularizing the data and pulling out features
    # For more info on how regularization is done, 
    # see `DilPredict` package
    tsteps = minimum([median(diff(i.time)) for i in trial.runs]) * scaler
    @show tsteps
    regpack = DilPredict.regularize(trial, tsteps; T_start=T_start, T_end=T_end)
    X = hcat(regpack.T...)
    Y = hcat(regpack.dL...)
    dLdT = hcat(regpack.dLdT...)
    dTdt = hcat(regpack.dTdt...)
    f = hcat(regpack.f...)
    t = regpack.t
# Make all of the ΔLs at T_start the same (equal to mean)
    if scrunch
        N = size(Y,2)
        ave_dLi = mean([Y[1, i] for i in 1:N])
        Threads.@threads for i in 1:N
            delta = ave_dLi - Y[1, i]
            @show delta
            Y[:, i] = Y[:,i] .+ delta
        end
    end

    return (X, Y, t, dTdt, dLdT, f)
end
export pull_test_data

################################################
#
# DISTANCE METRICS
#
###############################################

# Get distance matrix/tensor from data set using cosine distance metric
function cos_dis(X::AbstractMatrix{<:AbstractFloat})
    n, N = size(X)
    T = Array{Float64}(undef, N,N,n)
    T .= [X[k, i] * X[k, j] for i in 1:N, j in 1:N, k in 1:n]
    @views for k in 2:n
        T[:, :, k] .= T[:, :, k] .+ T[:, :, k-1]
    end

    # This is the slow part! Very poorly optimized
    normsX = Matrix{Float64}(undef, n,N)
    Threads.@threads for i in 1:N
        for k in 1:n
        x = @view X[1:k, i]
        normsX[k, i] = norm(x)
        end
    end
    norms = Array{Float64}(undef, N,N,n)
    norms .= [normsX[k,i] * normsX[k,j] for i in 1:N, j in 1:N, k in 1:n] 

    D = sqrt.(abs.(2 .- 2 .* (T ./ norms)))
    
    return D
end
export cos_dis

function cos_dis(X::AbstractVector{<:AbstractFloat})
    n = length(X)
    N = 1
    T = Array{Float64}(undef, N,N,n)
    T .= [X[k, i] * X[k, j] for i in 1:N, j in 1:N, k in 1:n]
    @views for k in 2:n
        T[:, :, k] .= T[:, :, k] .+ T[:, :, k-1]
    end

    # This is the slow part! Very poorly optimized
    normsX = Matrix{Float64}(undef, n,N)
    Threads.@threads for i in 1:N
        for k in 1:n
        x = @view X[1:k, i]
        normsX[k, i] = norm(x)
        end
    end
    norms = Array{Float64}(undef, N,N,n)
    norms .= [normsX[k,i] * normsX[k,j] for i in 1:N, j in 1:N, k in 1:n] 

    D = sqrt.(abs.(2 .- 2 .* (T ./ norms)))
    
    return D
end
export cos_dis

function cos_dis(X1, X2)
    if X1 == X2
        return cos_dis(X1)
    else
        n, N = size(X1)
        N2 = minimum(size(Matrix(X2')))
        T = Array{Float64}(undef, N,N2,n)
        T .= [X1[k, i] * X2[k, j] for i in 1:N, j in 1:N2, k in 1:n]
        @views for k in 2:n
            T[:, :, k] .= T[:, :, k] .+ T[:, :, k-1]
        end

        # This is the slow part! Very poorly optimized
        normsX1 = Matrix{Float64}(undef, n,N)
        Threads.@threads for i in 1:N
            for k in 1:n
            x = @view X1[1:k, i]
            normsX1[k, i] = norm(x)
            end
        end
        normsX2 = Matrix{Float64}(undef, n,N2)
        for i in 1:N2
            for k in 1:n
            x = @view X2[1:k, i]
            normsX2[k, i] = norm(x)
            end
        end
        norms = Array{Float64}(undef, N,N2,n)
        norms .= [normsX1[k,i] * normsX2[k,j] for i in 1:N, j in 1:N2, k in 1:n] 

        D = sqrt.(abs.(2 .- 2 .* (T ./ norms)))
        
        return D
    end
end
export cos_dis

# squared euclidean distance for non-historetic kernels
function T_dist_i(X_i::Vector{Float64})::Matrix{Float64}
    return [(i - j)^2 for i in X_i, j in X_i]
end
export T_dist_i

function T_dist_i(X1::Vector{Float64}, X2::Vector{Float64})::Matrix{Float64}
    return [(i - j)^2 for i in X1, j in X2]
end
export T_dist_i

function T_dist(X::Matrix{Float64})::Array{Float64, 3}
    n, N = size(X)
    D = zeros(N, N, n)
    for i in 1:n
        D[:,:, i] .= T_dist_i(X[i, :])
    end
    return D
end
export T_dist

function T_dist(X1, X2)
    n, N = size(X1)
    N2 = minimum(size(Matrix(X2')))
    D = zeros(N, N2, n)
    D .= [-(X1[k, i] - X2[k, j])^2 for i in 1:N, j in 1:N2, k in 1:n]
    return D
end
return T_dist

"""
    build_all_distance_tensors(X, feature_db_path)

Builds a vector of distance tensors from a temperature matrix.
Feature matrices are pulled from a database of pre-calculated thermodynamic
properties stored in a csv file at `feature_db_path`. See the `DilPredict`
package for more info. 
"""
function build_all_distance_tensors(X, feature_db_path; features=[])
    feature_data = DilPredict.pull_tc_data(feature_db_path)
    feature_tensor = DilPredict.temps_to_feature_tensor(X, feature_data;
                                                        features=features)

    vec_of_tensors = map(eachslice(feature_tensor, dims=3)) do k
        # Adding a small amount to prevent NaN values
        @views D_slice = cos_dis(k .+ 1e-8)
    end
    # The 4th dim contains tensors for each feature
    tensor_of_tensors = cat(vec_of_tensors..., dims=4)
    return tensor_of_tensors
end
export build_all_distance_tensors

function build_all_distance_tensors(X, X_star, feature_db_path; features = [])
    X = reshape(X, (size(X, 1), size(X, 2)))
    feature_data = DilPredict.pull_tc_data(feature_db_path)
    feature_tensor = DilPredict.temps_to_feature_tensor(
        X, feature_data; features=features)

    n_features = size(feature_tensor, 3)

    X_star_mat = reshape(X_star, (size(X_star, 1), (size(X_star, 2))))

    feature_tensor_star = temps_to_feature_tensor(
        X_star_mat, feature_data; features=features
    )
    feature_tensor_star = cat(X_star, feature_tensor_star, dims=3)

    vec_of_tensors = map(1:n_features) do i
        # Adding a small amount to prevent NaN values
        @views D_slice = cos_dis(
            (feature_tensor .+ 1e-8)[:,:,i], (feature_tensor_star .+ 1e-8)[:,:,i])
    end
    # The 4th dim contains tensors for each feature
    tensor_of_tensors = cat(vec_of_tensors..., dims=4)
    return tensor_of_tensors
end
export build_all_distance_tensors

"""
    full_K_dists(X, X_star, tcpath)

Returns all distance tensors required for the `gp_inference` function.
"""
function full_K_dists(X, X_star, tcpath; features=[])
    # Force X_star to be a matrix
    X_star = reshape(X_star, (size(X_star, 1), size(X_star, 2)))

    dist_tensor_XX = build_all_distance_tensors(X,
                                                tcpath; features=features)
    dist_tensor_sX = build_all_distance_tensors(X_star, X,
                                                tcpath; features=features)
    dist_tensor_ss = build_all_distance_tensors(X_star,
                                                tcpath; features=features)
    return (dist_tensor_XX, dist_tensor_sX, dist_tensor_ss)
end
export full_K_dists

################################################
#
# KERNEL FUNCTIONS
#
###############################################

function exponentiated_kernel(D, l)
    K = exp.(-D ./ l^2)
    return K
end
export exponentiated_kernel

# This is super buggy and painfully slow
function matern_kernel(d, l, ν)
    y = sqrt(2*ν) .* d
    (2^(1 .- ν))./(gamma(ν)) .* (y./l).^ν .* besselk.(ν, y ./ l)
end
export matern_kernel

# Assumes equal signal noise across all inputs
# This can be changed
function signal_noise(D, σ_n)
    N, _, n = size(D)
    return [i==j ? abs(σ_n) : 0 for i in 1:N, j in 1:N, k in 1:n]
end
export signal_noise

function K_joined(tensor_of_dist_tensors, σ, l_vec)
    n, _, N, H = size(tensor_of_dist_tensors)
    kernels = map(1:H) do i
        K = exponentiated_kernel(
            tensor_of_dist_tensors[:,:,:,i], l_vec[i])
    end
    # σ acts as a scaling factor to capture latent variance
    return σ^2 .* sum(kernels)
end
export K_joined

################################################
#
# TRAINING
#
###############################################

"""
    nlogp(θ, vec_of_dist_tensors, Y; σ_n = 1e-8)::Float64

Negative log marginal likelihood of GP with parameters `θ` and reponse `Y`.
`tensor_of_dist_tensors` has dimensions K by K by N by F where K is the number
of time steps, N is the number of samples, and F is the number of features.
This can be generated with `build_all_distance_tensors`, as long as the
thermodynamic data is available. 

`θ` must be entered in the following form: 

`[l_f1, β_f1, l_f2, β_f2, ..., l_fh, β_fh]`

Where `h` is the total number of features. `l` denotes the 
lengthscale parameter and `β` denotes the pre-exponential parameter (σ).

This is calculated by treating
each slice of the distance tensor as its own gaussian process.
Time steps are only connected by the historetic property of the
cosine distance metric. The final value is calculated by taking 
the average nlogp across all slices -- this makes it easier
to interpret when running optimizations on data with different
time steps. 

Individual slices are calculated according to the algorithm
proposed by K. P. Murphy in 'Probabilistic Machine learning' 
page 572. ISBN: 978-0-262-04682-4
"""
function nlogp(θ, tensor_of_dist_tensors, Y; σ_n = 1e-8)
    N,_,n, H = size(tensor_of_dist_tensors)

    σ = θ[1]
    l_vec = θ[2:end]

    K = (K_joined(tensor_of_dist_tensors, σ, l_vec)
        .+ signal_noise(tensor_of_dist_tensors[:,:,:,1], σ_n))

    nlogp_all = map(1:n) do k 
        y = @view Y[k, :]
        μ_X = mean(y)
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ (y .- μ_X))
        0.5 * ((y .- μ_X)'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
        end
    return sum(nlogp_all)/n
end
export nlogp

"""
    nlogp_threaded(θ, D_hist, Y; σ_n = 1e-8)

A mulithreaded version of the `nlogp` function. This works
by splitting the kernel into a chunks; average nlogp is 
calculated for each chunk by individual threads, then these
averages are averaged over the chunks to find the complete
average nlogp. 
"""
function nlogp_threaded(θ, tensor_of_dist_tensors, Y; σ_n = 1e-8)
    N,_, n,H = size(tensor_of_dist_tensors)
    batch_ind = Iterators.partition(1:n, round(Int, n/Threads.nthreads())) |> collect
    tasks = map(1:Threads.nthreads()) do k 
        Threads.@spawn begin
        @views nlogp(θ, tensor_of_dist_tensors[:,:,batch_ind[k], :],
                     Y[batch_ind[k],:]; σ_n = σ_n) * length(batch_ind[k])
        end
    end
    chunk_sums = fetch.(tasks)
    return sum(chunk_sums)/n
end
export nlogp_threaded


"""
    opt_kernel(θi, D_hist; σ_n = 1e-8)

Find optimial kernel parameters by minimizing 
the negative log marginal likelihood `nlogp`. 
`θi` are the initial parameters, `D_hist` is the 
distance tensor, `Y` is the response matrix, `σ_n` 
is the signal noise. 
"""
function opt_kernel(θi, tensor_of_dist_tensors, Y; σ_n = 1e-8,
                    n_starts = 10, t_limit = 100)
    N, _, n, n_f = size(tensor_of_dist_tensors)
    loss(θ, p) = nlogp_threaded(θ, tensor_of_dist_tensors, Y; σ_n = σ_n)
    lower_bounds = vcat([1e-6], repeat([0.08], n_f))
    upper_bounds = vcat([50.0], repeat([300.0], n_f))

    of = OptimizationFunction(loss, AutoForwardDiff())
    prob = OptimizationProblem(of, θi, [], lb=lower_bounds,
                               ub=upper_bounds)
    function cb(p, l)
        θ_inst = p.u
        σ = θ_inst[1]
        l_vec = θ_inst[2:end]
        println("σ: $σ")
        println("l: $l_vec")
        println("loss: $l")
        return false
    end

    # Single-start options
    if n_starts==1
        sol = solve(prob, Optim.SAMIN();
                show_trace=false, callback=cb, time_limit=t_limit,
                    maxiters=10^6)
        return sol
    end

    # Multi-start options
    n_params = length(θi)
    inits = repeat([θi], n_starts)
    rand_add = [rand(n_params) for i in 1:n_starts]
    starts = [inits[i] .+ 5 .* rand_add[i] for i in 1:n_starts]
    ensembleprob = Optimization.EnsembleProblem(prob, starts)

    sol = solve(ensembleprob, Optim.BFGS(), EnsembleThreads();
                trajectories = n_starts,
                show_trace=false, callback=cb, time_limit=t_limit/n_starts)

    best_ind = argmin(i -> sol[i].objective, 1:n_starts)
    println("Best loss: $(sol[best_ind].objective)")
    println("Optimized params: $(sol[best_ind].u)")
    return sol
end 
export opt_kernel

function getminparams(sol)
    best_ind = argmin(i -> sol[i].objective, 1:length(sol))
    return sol[best_ind].u
end
export getminparams

################################################
#
# INFERENCE
#
###############################################

function gp_inference(dist_tensor_XX, dist_tensor_sX, dist_tensor_ss,
                      Y, θ; σ_n = 1.0)

    if size(dist_tensor_sX, 3) != size(dist_tensor_XX, 3) 
            throw("Dim of X_star dne dim of X")
    end

    N, _, n, H = size(dist_tensor_XX)
    M = size(dist_tensor_ss, 2)         # M = # of inference points

    σ = θ[1]
    l_vec = θ[2:end]

    K = (K_joined(dist_tensor_XX, σ, l_vec)
        .+ signal_noise(dist_tensor_XX[:,:,:,1], σ_n))
    K_star = K_joined(dist_tensor_sX, σ, l_vec)
    k_ss = (K_joined(dist_tensor_ss, σ, l_vec))

    μ = Matrix{Float64}(undef, n, M)
    s = Matrix{Float64}(undef, n, M)
    for k in 1:n
        K_t = K[:,:,k]
        y = Y[k, :]
        μ_X = mean(y)
        K_star_t = K_star[:, :, k]
        L = cholesky(K_t)
        # Assume that μ_* = μ_X:
        μ[k, :] = μ_X .+ (K_star_t * (L.U\(L.L\(y .- μ_X))))
        ν = (L.L \ K_star_t')
        # Neglecting the covariance by taking the diagonals
        s[k, :] = diag(k_ss[k] .- (ν' * ν))
    end
    return μ, s
end
export gp_inference

"""
    build_x_star(Tstart, Tstop, rate, dt, n)

Build an input thermal history for inference using a constant cooling 
rate `rate`, time step `dt`, start and stop temperatures `Tstart` and `Tstop`,
and length `n`.
"""
function build_x_star(Tstart, Tstop, rate::AbstractFloat, dt, n)
    temps = [Tstart,] 
    crates = [rate,]
    while temps[end] > Tstop
        push!(temps, temps[end] - (rate * dt))
        push!(crates, rate)
    end

    while length(temps) < n
        push!(temps, temps[end])
        push!(crates, 0.0)
    end
    return temps, crates
end
export build_x_star

function build_x_star(Tstart, Tstop, rates::Vector{<:AbstractFloat}, dt, n)
    X_star = Matrix{Float64}(undef, n, length(rates))
    crates = Matrix{Float64}(undef, n, length(rates))
    Threads.@threads for i in eachindex(rates)
        X_star[:, i], crates[:, i] = build_x_star(
            Tstart, Tstop, rates[i], dt, n
        )
    end

    return X_star, crates
end
export build_x_star

"""
    inference_surface(crate_max, crate_min, m, X, Y, t, θ)

Generates a 3D plot of the response variable (ΔL or f usually) as a function 
of temperature and cooling rate by running inference on a GP model. `crate_max`
is the fastest cooling rate and `crate_min` is the slowest. `X` is the input 
matrix, `Y` is the reponse matrix, `t` is the time basis (used to calculate dt),
and `θ` is the vector of model parameters used to run the GP. `m` is the number 
of 'points' that will be evaluated between the cooling rates; it determines the 
number of input histories that will be evaluated. If `m = 50`, the model will 
be evaluated for 50 thermal histories between `crate_min` and `crate_max`
"""
function inference_surface(crate_max, crate_min, m, X, Y, t, θ, tcpath;
                           σ_n = 1e-8, T_start=860.0, T_end=250.0, features=[])
    GLMakie.activate!()
    n, N = size(X)
    rates = range(crate_min, crate_max; length=m) |> collect
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    dt = mean(diff(t))

    println("Building inference points...")
    xstars_T, xstars_CR = build_x_star(T_start, T_end, rates, dt, n)

    println("Calculating distance tensors...")
    dists = full_K_dists(X, xstars_T, tcpath; features=features)

    println("Running infrence...")
    means, vars = gp_inference(dists..., Y, θ)

    println("Plotting...")
    f = Figure()
    ax = Axis3(f[1,1];
               xlabel="Temp. (°C)", ylabel="CR (°C/s)", zlabel="ΔL (μm)")

    σ_high = maximum(sqrt.(abs.(vars)))
    σ_low = minimum(sqrt.(abs.(vars)))
    for i in 1:m
        y = xstars_CR[:, i]
        x = xstars_T[:, i]
        z = means[:,i]
        σ = sqrt.(abs.(vars[:,i]))
        dmat = hcat(x,y,z, σ)
        dmat = hcat([dmat[i, :] for i in 1:size(dmat,1) if dmat[i,1] > T_end]...)'
        x = dmat[:, 1]; y = dmat[:, 2]; z = dmat[:,3]; σ = dmat[:,4]
        lines!(ax,x,y,z, color=σ, linewidth=3, colorrange=(σ_low, σ_high))
    end
    Colorbar(f[1,2], limits = (σ_low, σ_high))
    return f
end
export inference_surface

function inference_surface_t(crate_max, crate_min, m, X, Y, t, θ; σ_n = 0.00011)
    n, N = size(X)
    rates = 10 .^ range(log10(crate_min), log10(crate_max); length=m)
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    xstars_T = Matrix{Float64}(undef, n, m)
    xstars_CR = Matrix{Float64}(undef, n, m)

    Threads.@threads for i in eachindex(rates)
        xstars_T[:, i], xstars_CR[:,i] = build_x_star(945, 205.0, rates[i], mean(diff(t)), n) 
        means[:, i], vars[:, i] = gp_inference(X, xstars_T[:,i], Y, θ; σ_n=σ_n)
    end
    f = Figure()
    ax = Axis3(f[1,1];
               ylabel="Temp. (°C)", xlabel="log(t) (s)", zlabel="ΔL (μm)", xreversed=false)

    σ_maxs = []
    σ_mins = []
    cte_low = minimum(means)
    cte_high = maximum(means)
    for i in 1:m
        y = xstars_CR[:, i]
        x = xstars_T[:, i]
        z = means[:,i]
        σ = sqrt.(abs.(vars[:,i]))
        push!(σ_maxs, maximum(σ))
        push!(σ_mins, minimum(σ))
        dmat = hcat(x,log10.(t),z, σ)
        dmat = hcat([dmat[i, :] for i in 1:size(dmat,1)]...)'
        x = dmat[:, 1]; y = dmat[:, 2]; z = dmat[:,3]; σ = dmat[:,4]
        lines!(ax,y,x,z, color=z, linewidth=3, colorrange=(cte_low, cte_high))
    end
    σ_high = maximum(σ_maxs)
    σ_low = minimum(σ_mins)
    Colorbar(f[1,2], limits = (cte_low, cte_high))
    return f
end
export inference_surface_t
# θi = [17.414065021040688, 0.5896643153148926, 0.04492645656636867]

function single_cr_plot(cr, X, Y, θ, σ_n, T_start, t, tcpath; T_end = 250.0, features=[])
axis_kwargs = (xminortickalign=1.0, yminortickalign=1.0, xgridvisible=false,
                 ygridvisible=false, xminorticks=IntervalsBetween(2),
                 yminorticks=IntervalsBetween(2),
                 xminorticksvisible=true, yminorticksvisible=true, 
                 xtickalign=0.5, ytickalign=0.5, 
                 xticksize=10.0, yticksize=10.0, yminorticksize=5.0,
                 xminorticksize=5.0)
    n = size(X, 1)
    xstars,_ = build_x_star(T_start, T_end, cr, mean(diff(t)), n)
    dists = full_K_dists(X, xstars, tcpath; features = features)
    means, vars = gp_inference(dists..., Y, θ; σ_n = σ_n)
    means = vec(means)
    vars = vec(vars)
    f = Figure(; size=(800,480))
    ax = Axis(f[1,1]; xlabel="Temp. (°C)", ylabel="ΔL (μm)", xreversed=true,
              axis_kwargs...)
    lower = means .- sqrt.(abs.(vars))
    upper = means .+ sqrt.(abs.(vars))
    band!(ax, xstars, lower, upper; alpha=0.3)
    lines!(ax, xstars, means; label="CR = $cr (°C/s)")
    #axislegend(ax)
    return f
end
export single_cr_plot

function single_cr_plot!(cr, X, Y, θ, σ_n, T_start, t, tcpath; T_end=250.0, features=[])
    n = size(X, 1)
    xstars,_ = build_x_star(T_start, T_end, cr, mean(diff(t)), n)
    dists = full_K_dists(X, xstars, tcpath; features = features)
    means, vars = gp_inference(dists..., Y, θ; σ_n = σ_n)
    means = vec(means)
    vars = vec(vars)
    lower = means .- sqrt.(abs.(vars))
    upper = means .+ sqrt.(abs.(vars))
    band!(xstars, lower, upper; alpha=0.3)
    lines!(xstars, means, label="CR = $cr (°C/s)")
end
export single_cr_plot!

function get_info(mypath)
    trial = get_trial(mypath)
    for i in 1:length(trial.runs)
        data = get_cooling(trial.runs[i])
        println("""
                Name: $(data.name), 
                dt = $(median(diff(data.time))),
                N = $(length(data.time))
                """)
    end
end
export get_info
