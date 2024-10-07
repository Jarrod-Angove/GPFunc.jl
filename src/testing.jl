using LinearAlgebra, KernelAbstractions, DilPredict, Optimization, OptimizationOptimJL
using Enzyme, GLMakie, Statistics, SpecialFunctions

function pull_test_data(; t_cut = 0.0)
    trial_path = "../dil_data/dil_X70_2014-05-28_ualberta/"
    trial = DilPredict.get_trial(trial_path)
    tsteps = trial.runs[1].time |> diff |> median
    T, dL, t, _, dTdt, dLdT = DilPredict.regularize(trial, tsteps)
    X = hcat(T...)
    Y = hcat(dL...)
    dLdT = hcat(dLdT...)
    dTdt = hcat(dTdt...)

    if t_cut != 0.0
        ind = findall(t -> t>t_cut, t)
        t = t[ind]
        X = X[ind, :]
        Y = Y[ind, :]
        dTdt = dTdt[ind, :]
        dLdT = dLdT[ind, :]
    end
    return (X, Y, t, dTdt, dLdT)
end
export pull_test_data

# Get distance matrix/tensor from data set using cosine distance metric
function cos_dis(X)
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

    D = T ./ norms
    
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

        D = T ./ norms
        
        return D
    end
end
export cos_dis

function T_dist_i(X_i::Vector{Float64})::Matrix{Float64}
    return [-(i - j)^2 for i in X_i, j in X_i]
end
export T_dist_i

function T_dist_i(X1::Vector{Float64}, X2::Vector{Float64})::Matrix{Float64}
    return [-(i - j)^2 for i in X1, j in X2]
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

function exponentiated_kernel(D, σ, l)
    K = σ^2 .* exp.(D ./ l^2)
    return K
end
export exponentiated_kernel

function matern_kernel(d, σ, l, ν)
    d = 2 - 2 * (d - 1e-8)
    y = sqrt(2*ν) * d
    σ^2 * (2^(1-ν))/(gamma(1.5)) * (y/l)^ν * besselk(ν, y/l)
end
export matern_kernel

function exponentiated_kernel_dual(D_T, D_CR, l_T, l_CR)
    K_T = exp.(D_T ./ l_T^2)
    K_CR = exp.(D_CR ./ l_CR^2)
    return K_T + K_CR
end
export exponentiated_kernel

function signal_noise(D, σ_n)
    N, _, n = size(D)
    return [i==j ? abs(σ_n) : 0 for i in 1:N, j in 1:N, k in 1:n]
end
export signal_noise

# Returns gradient of exponentiated kernel function
# This is pretty slow because it needs to calculate the
# kernel three times... This would be a really good spot to 
# start running stuff on GPU.
function grad_expo(θ, D)
    N,_,n = size(D)
    k = exp.(D ./ θ[2]^2)
    g_σf = 2 .* θ[1] .* k
    g_l = (-2 .* D .* θ[1]^2) ./ θ[2]^3 .* k
    g_σn = repeat(diagm(ones(N)), 1, 1, n)
    return [g_σf, g_l, g_σn]
end
export grad_expo

function grad_nlogp(θ, D, Y)
    N, _, n = size(D)
    len_θ = length(θ)
    σ = θ[1]; l = θ[2]; σ_n = θ[3]
    K = exponentiated_kernel(D, σ, l) .+ signal_noise(D, σ_n)
    K_grads = grad_expo(θ, D)
    #nlogp_all = Vector{Float64}(undef, n)
    Gnlogp_tot = zeros(Float32, len_θ)
    for k in 1:n
        y = @view Y[k, :]
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ y)
        #nlogp_all[k] = 0.5 * (y'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
        @views for m in 1:len_θ
            G_t = K_grads[m][:,:,k]
            Gnlogp_tot[m] += -0.5 * tr(α * α' * G_t - L.U \ (L.L \ G_t))
        end
    end
    return Gnlogp_tot
end
export grad_nlogp

function nlogp(θ, D_hist, D_instant, Y; σ_n = 5.0)
    N, _, n = size(D_hist)
    σ_hist = θ[1]; l_hist = θ[2];
    σ_inst = 1.0; l_inst = θ[3];
    K = (exponentiated_kernel(D_hist, σ_hist, l_hist) .*
        exponentiated_kernel(D_instant, σ_inst, l_inst)
        .+ signal_noise(D_hist, θ[4]))
    #nlogp_all = Vector{Float64}(undef,n)
    nlogp_all = 0.0
    for k in 1:n
        y = @view Y[k, :]
        μ_X = mean(y)
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ (y .- μ_X))
        nlogp_all += 0.5 * ((y .- μ_X)'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
    end
    return nlogp_all
end
export nlogp

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

function opt_kernel(θi, D_hist, D_instant, Y; σ_n = 0.00011)
    loss(θ, p) = nlogp(θ, D_hist, D_instant, Y; σ_n = σ_n)
    #function loss_grad!(storage::Array{Float64}, θ::Vector{Float64}, p)::Array{Float64}
    #    storage .= grad_nlogp(θ, D, Y)
    #end
    #loss_grad!(θ, _p) = grad_nlogp(θ, D, Y)
    of = OptimizationFunction(loss, AutoForwardDiff())#; grad = loss_grad!)
    prob = OptimizationProblem(of, θi, [], lb=[1e-6, 1e-6, 1e-6, 1e-6],
                               ub=[1e8, 1e3, 1e8, 1e2])
    sol = solve(prob, Optim.BFGS(); show_trace=true)
    return sol
end 
export opt_kernel
# θi = [11.5, 0.7, 0.14]

# TODO: Finish adding the inst kernel
function predict_y(X, X_star, Y, θ; σ_n = 5.0)
    if size(X_star, 1) != size(X, 1) 
            throw("Dim of X_star dne dim of X")
    end
    n, N = size(X)
    D_instant = T_dist(X)
    D_hist = cos_dis(X)
    D_star_hist = cos_dis(X, X_star)
    D_star_inst = T_dist(X, X_star)
    σ_hist = θ[1]; l_hist = θ[2];
    σ_inst = 1.0; l_inst = θ[3];
    σ_n = θ[4];
   K = (exponentiated_kernel(D_hist, σ_hist, l_hist) .*
        exponentiated_kernel(D_instant, σ_inst, l_inst)
        .+ signal_noise(D_hist, σ_n))
    K_star = (exponentiated_kernel.(D_star_hist, σ_hist, l_hist) .*
        exponentiated_kernel(D_star_inst, σ_inst, l_inst)
        .+ signal_noise(D_star_hist, σ_n))
    k_ss = (exponentiated_kernel(1.0, σ_hist, l_hist) .*
        exponentiated_kernel(1.0, σ_inst, l_inst))

    μ = Vector{Float64}(undef, n)
    s = Vector{Float64}(undef, n)
    for k in 1:n
        K_t = K[:,:,k]
        y = Y[k, :]
        μ_X = mean(y)
        K_star_t = K_star[:, :, k]
        L = cholesky(K_t)
        # Assume that μ_* = μ_X:
        μ[k] = μ_X .+ (K_star_t' * (L.U\(L.L\(y .- μ_X))))[1]
        ν = (L.L \ K_star_t)
        s[k] = k_ss - (ν' * ν)[1]
    end
    return μ, s
end
export predict_y

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

function inference_surface(crate_max, crate_min, m, X, Y, t, θ; σ_n = 0.25)
    n, N = size(X)
    rates = range(crate_min, crate_max; length=m)
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    xstars_T = Matrix{Float64}(undef, n, m)
    xstars_CR = Matrix{Float64}(undef, n, m)
    Threads.@threads for i in 1:m
        xstars_T[:, i], xstars_CR[:,i] = build_x_star(950.0, 200.0, rates[i], 0.3, n) 
        means[:, i], vars[:, i] = predict_y(X, xstars_T[:,i], Y, θ; σ_n=σ_n)
    end
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
        dmat = hcat([dmat[i, :] for i in 1:size(dmat,1) if dmat[i,1] > 40.0]...)'
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
        xstars_T[:, i], xstars_CR[:,i] = build_x_star(950.0, 205.0, rates[i], 0.3, n) 
        means[:, i], vars[:, i] = predict_y(X, xstars_T[:,i], Y, θ; σ_n=σ_n)
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

function single_cr_plot(cr, X, Y, θ; σ_n = 0.00011)
axis_kwargs = (xminortickalign=1.0, yminortickalign=1.0, xgridvisible=false,
                 ygridvisible=false, xminorticks=IntervalsBetween(2),
                 yminorticks=IntervalsBetween(2),
                 xminorticksvisible=true, yminorticksvisible=true, 
                 xtickalign=0.5, ytickalign=0.5, 
                 xticksize=10.0, yticksize=10.0, yminorticksize=5.0,
                 xminorticksize=5.0)
    n = size(X, 1)
    xstars,_ = build_x_star(950.0, 200.0, cr, 0.3, n)
    means, vars = predict_y(X[5:end, :], xstars[5:end], Y[5:end, :], θ; σ_n = σ_n)
    f = Figure(; size=(800,480))
    ax = Axis(f[1,1]; xlabel="Temp. (°C)", ylabel="ΔL (μm)", xreversed=true,
              axis_kwargs...)
    lower = means .- sqrt.(abs.(vars))
    upper = means .+ sqrt.(abs.(vars))
    band!(ax, xstars[5:end], lower, upper; alpha=0.3)
    lines!(ax, xstars[5:end], means; label="CR = $cr (°C/s)")
    #axislegend(ax)
    return f
end
export single_cr_plot

# θ = [29.674, 3.572, 426.479]

function single_cr_plot!(cr, X, Y, θ; σ_n = 0.00011)
    n = size(X, 1)
    xstars,_ = build_x_star(950.0, 200.0, cr, 0.3, n)
    means, vars = predict_y(X[5:end, :], xstars[5:end], Y[5:end, :], θ; σ_n = σ_n)
    lower = means .- sqrt.(abs.(vars))
    upper = means .+ sqrt.(abs.(vars))
    band!(xstars[5:end], lower, upper; alpha=0.3)
    lines!(xstars[5:end], means, label="CR = $cr (°C/s)")
end
export single_cr_plot!
