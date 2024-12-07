using LinearAlgebra, DilPredict, Optimization, OptimizationOptimJL
using GLMakie, Statistics, SpecialFunctions

function pull_test_data(;trial_path = "../dil_data/dil_X70_2014-05-28_ualberta/",
                        T_start=1000.0, T_end=200.0, scrunch = false, check_plot=true, scaler=1)
    trial = DilPredict.get_trial(trial_path)

    # Check to make sure everything lines up; can be disabled for dev work
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

################################################
#
# KERNEL FUNCTIONS
#
###############################################

function exponentiated_kernel(D, σ, l, β)
    K = σ^2 .* exp.(-D.^β./ l^2)
    return K
end
export exponentiated_kernel

function matern_kernel(d, σ, l, ν)
    y = sqrt(2*ν) .* d
    σ^2 .* (2^(1 .- ν))./(gamma(ν)) .* (y./l).^ν .* besselk.(ν, y ./ l)
end
export matern_kernel

function signal_noise(D, σ_n)
    N, _, n = size(D)
    return [i==j ? abs(σ_n) : 0 for i in 1:N, j in 1:N, k in 1:n]
end
export signal_noise

################################################
#
# TRAINING
#
###############################################

"""
    nlogp(θ, D_hist, Y; σ_n = 1e-8)::Float64

Negative log marginal likelihood. `θ` is the kernel parameters,
`D_hist` is the distance tensor, and `Y` is the reponse matrix. 
`σ_n` is the signal noise (standard deviation) -- this represents
any noise that is inherent to the measurement of the reponse variable. 
This should not be treated as an optimizer variable because it creates 
a non-identifiable system; the GP does not know how to discriminate 
between latent function variance and signal variance. The default value
of `σ_n = 1e-8` assumes that the signal noise is negligible. 

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
function nlogp(θ, D_hist, Y; σ_n = 1.0)::Float64
    N, _, n = size(D_hist)
    σ_hist = θ[1]; l_hist = θ[2]; β_hist = θ[3]
    K = (exponentiated_kernel(D_hist, σ_hist, l_hist, β_hist)
        .+ signal_noise(D_hist, σ_n))
    nlogp_all = map(1:n) do k 
        y = @view Y[k, :]
        μ_X = 1.0
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
function nlogp_threaded(θ, D_hist, Y; σ_n = 1e-8)
    n = size(D_hist, 3)
    batch_ind = collect(
        Iterators.partition(1:n, round(Int, n/Threads.nthreads())))
    #Y_batch = [Y[batch_ind[i], :] for i in 1:Threads.nthreads()]
    #D_batch = [D_hist[:, :, batch_ind[i]] for i in 1:Threads.nthreads()]
    tasks = map(1:Threads.nthreads()) do k 
        Threads.@spawn nlogp(θ, D_hist[:,:,batch_ind[k]],
                             Y[batch_ind[k],:]; σ_n = σ_n)
    end
    chunk_sums = fetch.(tasks)
    return sum(chunk_sums)/Threads.nthreads()
end
export nlogp_threaded

function nlogp_slices(θ, D_hist, D_instant, Y; σ_n = 5.0)
    N, _, n = size(D_hist)
    σ_hist = θ[1]; l_hist = θ[2];
    σ_inst = 1.0; l_inst = θ[3];
    K = (exponentiated_kernel(D_hist, σ_hist, l_hist) .*
        exponentiated_kernel(D_instant, σ_inst, l_inst)
        .+ signal_noise(D_hist, σ_n, θ[4], θ[5]))
    nlogp_all = Vector{Float64}(undef, n)
    for k in 1:n
        y = @view Y[k, :]
        μ_X = mean(y)
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ (y .- μ_X))
        nlogp_all[k] = 0.5 * ((y .- μ_X)'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
    end
    return nlogp_all
end
export nlogp_slices

"""
    opt_kernel(θi, D_hist; σ_n = 1e-8)

Find optimial kernel parameters by minimizing 
the negative log marginal likelihood `nlogp`. 
`θi` are the initial parameters, `D_hist` is the 
distance tensor, `Y` is the response matrix, `σ_n` 
is the signal noise. 
"""
function opt_kernel(θi, D_hist, Y; σ_n = 1e-8)
    loss(θ, p) = nlogp_threaded(θ, D_hist, Y; σ_n = σ_n)

    of = OptimizationFunction(loss, AutoForwardDiff())#; grad = loss_grad!)
    prob = OptimizationProblem(of, θi, [], lb=[1.0e-8, 1.0e-6, 0.1],
                               ub=[50.0, 100.0, 2.0])
    sol = solve(prob, Optim.BFGS(); show_trace=true)
    return sol
end 
export opt_kernel

################################################
#
# INFERENCE
#
###############################################

# TODO: Finish adding the inst kernel
function gp_inference(X, X_star, Y, θ; σ_n = 1.0)
    if size(X_star, 1) != size(X, 1) 
            throw("Dim of X_star dne dim of X")
    end
    n, N = size(X)
    D_hist = cos_dis(X)
    D_star_hist = cos_dis(X, X_star)
    σ_hist = θ[1]; l_hist = θ[2]; β_hist = θ[3]
    K = (exponentiated_kernel(D_hist, σ_hist, l_hist, β_hist)
        .+ signal_noise(D_hist, σ_n))
    K_star = (exponentiated_kernel.(D_star_hist, σ_hist, l_hist, β_hist))
    k_ss = vec(exponentiated_kernel(cos_dis(X_star), σ_hist, l_hist, β_hist))

    μ = Vector{Float64}(undef, n)
    s = Vector{Float64}(undef, n)
    for k in 1:n
        K_t = K[:,:,k]
        y = Y[k, :]
        μ_X = 1.0
        K_star_t = K_star[:, :, k]
        L = cholesky(K_t)
        # Assume that μ_* = μ_X:
        μ[k] = μ_X .+ (K_star_t' * (L.U\(L.L\(y .- μ_X))))[1]
        ν = (L.L \ K_star_t)
        s[k] = k_ss[k] - (ν' * ν)[1]
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
function build_x_star(Tstart, Tstop, rate, dt, n)
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
function inference_surface(crate_max, crate_min, m, X, Y, t, θ;
                           σ_n = 0.25, T_start=860.0, T_end=250.0)
    GLMakie.activate!()
    n, N = size(X)
    rates = range(crate_min, crate_max; length=m)
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    xstars_T = Matrix{Float64}(undef, n, m)
    xstars_CR = Matrix{Float64}(undef, n, m)
    dt = mean(diff(t))
    Threads.@threads for i in 1:m
        xstars_T[:, i], xstars_CR[:,i] = build_x_star(T_start, T_end, rates[i], dt, n) 
        means[:, i], vars[:, i] = gp_inference(X, xstars_T[:,i], Y, θ; σ_n=σ_n)
    end
    f = Figure()
    ax = Axis3(f[1,1];
               xlabel="Temp. (°C)", ylabel="CR (°C/s)", zlabel="ΔL (μm)")

    σ_high = maximum(sqrt.(vars))
    σ_low = minimum(sqrt.(vars))
    for i in 1:m
        y = xstars_CR[:, i]
        x = xstars_T[:, i]
        z = means[:,i]
        σ = sqrt.(vars[:,i])
        dmat = hcat(x,y,z, σ)
        dmat = hcat([dmat[i, :] for i in 1:size(dmat,1) if dmat[i,1] > T_end]...)'
        x = dmat[:, 1]; y = dmat[:, 2]; z = dmat[:,3]; σ = dmat[:,4]
        lines!(ax,x,y,z, color=σ, linewidth=3, colorrange=(σ_low, σ_high))
    end
    Colorbar(f[1,2], limits = (σ_low, σ_high))
    display(f)
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

function single_cr_plot(cr, X, Y, θ, σ_n, T_start, t; T_end = 250.0)
axis_kwargs = (xminortickalign=1.0, yminortickalign=1.0, xgridvisible=false,
                 ygridvisible=false, xminorticks=IntervalsBetween(2),
                 yminorticks=IntervalsBetween(2),
                 xminorticksvisible=true, yminorticksvisible=true, 
                 xtickalign=0.5, ytickalign=0.5, 
                 xticksize=10.0, yticksize=10.0, yminorticksize=5.0,
                 xminorticksize=5.0)
    n = size(X, 1)
    xstars,_ = build_x_star(T_start, T_end, cr, mean(diff(t)), n)
    means, vars = gp_inference(X, xstars, Y, θ; σ_n = σ_n)
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

function single_cr_plot!(cr, X, Y, θ, σ_n, T_start, t; T_end=250.0)
    n = size(X, 1)
    xstars,_ = build_x_star(T_start, T_end, cr, mean(diff(t)), n)
    means, vars = gp_inference(X, xstars, Y, θ; σ_n = σ_n)
    means, vars = gp_inference(X, xstars, Y, θ; σ_n = σ_n)
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
