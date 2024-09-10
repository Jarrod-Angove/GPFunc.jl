using LinearAlgebra, KernelAbstractions, DilPredict, Optimization, OptimizationOptimJL
using Enzyme, GLMakie

function pull_test_data()
    trial_path = "../dil_data/dil_V3010A_2014-02-17_canmet/"
    trial = DilPredict.get_trial(trial_path)
    # using 0.07 because it is slightly larger
    # than the average dt in the original data
    T, dL, t, _, dTdt, dLdT = DilPredict.regularize(trial, 0.05)
    X = hcat(T...)
    Y = hcat(dLdT...)
    dTdt = hcat(dTdt...)
    return (X, Y, t, dTdt)
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

function exponentiated_kernel(D, σ, l)
    K = σ^2 .* exp.(D ./ l^2)
    return K
end
export exponentiated_kernel

function exponentiated_kernel_dual(D_T, D_CR, l_T, l_CR)
    K_T = exp.(D_T ./ l_T^2)
    K_CR = exp.(D_CR ./ l_CR^2)
    return K_T + K_CR
end
export exponentiated_kernel

function signal_noise(D, σ_n)
    N, _, n = size(D)
    noise_mat = diagm(ones(N)) .* σ_n
    noise_T = repeat(noise_mat, 1,1, n)
    return noise_T
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

function nlogp(θ, D, Y)
    N, _, n = size(D)
    σ = θ[1]; l = θ[2]; σ_n = θ[3]
    K = exponentiated_kernel(D, σ, l) .+ signal_noise(D, σ_n)
    #nlogp_all = Vector{Float64}(undef,n)
    nlogp_all = 0.0
    for k in 1:n
        y = @view Y[k, :]
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ y)
        nlogp_all += 0.5 * (y'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
    end
    return nlogp_all
end
export nlogp

function nlogp_dual(θ, D_T, D_CR, Y)
    N, _, n = size(D_T)
    l_T = θ[1]; l_CR = θ[2]; σ_n = θ[3]

    K = exponentiated_kernel_dual(D_T, D_CR, l_T, l_CR) .+ signal_noise(D_T, σ_n)
    #nlogp_all = Vector{Float64}(undef,n)
    nlogp_all = 0.0
    for k in 1:n
        y = @view Y[k, :]
        K_t = @view K[:,:,k]
        L = cholesky(K_t) 
        α = L.U \ (L.L \ y)
        nlogp_all += 0.5 * (y'*α) + sum(log.(diag(L.L))) + N/2 * log(2π)
    end
    return nlogp_all
end
export nlogp_dual

function opt_kernel(θi, D, Y)
    loss(θ, p) = nlogp(θ, D, Y)
    #function loss_grad!(storage::Array{Float64}, θ::Vector{Float64}, p)::Array{Float64}
    #    storage .= grad_nlogp(θ, D, Y)
    #end
    #loss_grad!(θ, _p) = grad_nlogp(θ, D, Y)
    of = OptimizationFunction(loss, AutoForwardDiff())#; grad = loss_grad!)
    prob = OptimizationProblem(of, θi, [], lb=[0.0001, 0.001, 0.00001], ub=[200.0, 100.0, 0.5])
    sol = solve(prob, Optim.LBFGS(); show_trace=true)
    return sol
end 
export opt_kernel

function opt_kernel_dual(θi, D_T, D_CR, Y)
    loss(θ, p) = nlogp_dual(θ, D_T, D_CR, Y)
    of = OptimizationFunction(loss, AutoForwardDiff())#; grad = loss_grad!)
    prob = OptimizationProblem(of, θi, [], lb=[0.0001, 0.00001, 0.0000001],
                               ub=[Inf, Inf, 5.0])
    sol = solve(prob, Optim.LBFGS(); show_trace=true, maxiters=30000)
    return sol
end 
export opt_kernel_dual
# θi = [11.5, 0.7, 0.14]

function predict_y(X, X_star, Y, θ)
    if size(X_star, 1) != size(X, 1) 
            throw("Dim of X_star dne dim of X")
    end
    n, N = size(X)
    D = cos_dis(X)
    D_star = cos_dis(X, X_star)
    σ = θ[1]; l = θ[2]; σ_n = θ[3];
    K = exponentiated_kernel(D, σ, l) + signal_noise(D, σ_n)
    K_star = exponentiated_kernel(D_star, σ, l)
    k_ss = exponentiated_kernel(1.0, σ, l)

    μ = Vector{Float64}(undef, n)
    s = Vector{Float64}(undef, n)
    for k in 1:n
        K_t = K[:,:,k]
        y = Y[k, :]
        K_star_t = K_star[:, :, k]
        L = cholesky(K_t)
        μ[k] = (K_star_t' * (L.U\(L.L\y)))[1]
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

function inference_surface(crate_max, crate_min, m, X, Y, dTdt, θ)
    n, N = size(X)
    rates = range(crate_min, crate_max; length=m)
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    xstars_T = Matrix{Float64}(undef, n, m)
    xstars_CR = Matrix{Float64}(undef, n, m)
    Threads.@threads for i in eachindex(rates)
        xstars_T[:, i], xstars_CR[:,i] = build_x_star(900.0, 30.0, rates[i], 0.07, n) 
        means[:, i], vars[:, i] = predict_y(X, xstars_T[:,i], Y, θ)
    end
    f = Figure()
    ax = Axis3(f[1,1];
               xlabel="Temp. (°C)", ylabel="CR (°C/s)", zlabel="dL/dT (μm/°C)")

    for i in 1:m
        y = xstars_CR[:, i]
        x = xstars_T[:, i]
        z = means[:,i]
        σ = sqrt.(vars[:,i])
        dmat = hcat(x,y,z, σ)
        dmat = hcat([dmat[i, :] for i in 1:size(dmat,1) if dmat[i,1] > 40.0]...)'
        x = dmat[:, 1]; y = dmat[:, 2]; z = dmat[:,3]; σ = dmat[:,4]
        lines!(ax,x,y,z, color=z, linewidth=3)
    end

    #for i in 1:Na
    #    y =  (-1) .* dTdt[:,i]
    #    x = X[:, i]
    #    z = Y[:, i]
    #    lines!(ax, x, y, z; color=:black, linewidth=3.0)
    #end
    #ylims!(ax, 0.0, 21.0)
    return f
end
export inference_surface

# θi = [17.414065021040688, 0.5896643153148926, 0.04492645656636867]

function single_cr_plot(cr, X, Y, θ)
    n = size(X, 1)
    xstars,_ = build_x_star(900.0, 30.0, cr, 0.07, n)
    means, vars = predict_y(X, xstars, Y, θ)
    f = Figure()
    ax = Axis(f[1,1], xlabel="Temp. (°C)", ylabel="dL/dT (μm/°C)", xreversed=true)
    lower = means .- sqrt.(vars)
    upper = means .+ sqrt.(vars)
    band!(ax, xstars, lower, upper; alpha=0.4)
    lines!(ax, xstars, means, label="CR = $cr (°C/s)")
    return f, ax
end
export single_cr_plot

function single_cr_plot!(cr, X, Y, θ)
    n = size(X, 1)
    xstars,_ = build_x_star(900.0, 30.0, cr, 0.07, n)
    means, vars = predict_y(X, xstars, Y, θ)
    lower = means .- sqrt.(vars)
    upper = means .+ sqrt.(vars)
    band!(xstars, lower, upper; alpha=0.4)
    lines!(xstars, means, label="CR = $cr (°C/s)")
end
export single_cr_plot!

#TODO: Implement Matern Kernel
