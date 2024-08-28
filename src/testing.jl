using LinearAlgebra, KernelAbstractions, DilPredict, Optimization, OptimizationOptimJL
using Enzyme, GLMakie

function pull_test_data()
    trial_path = "../dil_data/dil_V3016_2014-02-17_canmet/"
    trial = DilPredict.get_trial(trial_path)
    # using 0.07 because it is slightly larger
    # than the average dt in the original data
    T, dL, t, _ = DilPredict.regularize(trial, 0.07)
    X = hcat(T...)
    Y = hcat(dL...)
    return (X, Y, t)
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

function exponentiated_kernel(D, θ)
    σ = θ[1]; l = θ[2]
    K = σ^2 .* exp.(D ./ l^2)
    return K
end
export exponentiated_kernel

function signal_noise(D, θ)
    N, _, n = size(D)
    noise_mat = diagm(ones(N)) .* θ[3]
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
    K = exponentiated_kernel(D, θ) .+ signal_noise(D, θ)
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
    K = exponentiated_kernel(D, θ) .+ signal_noise(D, θ)
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

function opt_kernel(θi, D, Y)
    loss(θ, p) = nlogp(θ, D, Y)
    function loss_grad!(storage::Array{Float64}, θ::Vector{Float64}, p)::Array{Float64}
        storage .= grad_nlogp(θ, D, Y)
    end
    #loss_grad!(θ, _p) = grad_nlogp(θ, D, Y)
    #println(loss(θi, []))
    of = OptimizationFunction(loss, AutoForwardDiff())#; grad = loss_grad!)
    prob = OptimizationProblem(of, θi, [], lb=[0.1, 0.1, 0.00001], ub=[100.0, 100.0, 0.1])
    sol = solve(prob, Optim.LBFGS(); show_trace=true)
    return sol
end 
export opt_kernel

# θi = [11.5, 0.7, 0.14]

function predict_y(X, X_star, Y, θ)
    n, N = size(X)
    D = cos_dis(X)
    D_star = cos_dis(X, X_star)
    K = exponentiated_kernel(D, θ) + signal_noise(D, θ)
    K_star = exponentiated_kernel(D_star, θ)
    k_ss = exponentiated_kernel(1.0, θ)

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
    end
    return temps
end
export build_x_star

function inference_surface(crate_max, crate_min, m, X, Y, t, θ)
    n, N = size(X)
    rates = range(crate_min, crate_max; length=m)
    means = Matrix{Float64}(undef, n, m)
    vars = Matrix{Float64}(undef, n, m)
    xstars = Matrix{Float64}(undef, n, m)
    Threads.@threads for i in eachindex(rates)
        xstars[:, i] = build_x_star(900.0, 30.0, rates[i], 0.07, n) 
        means[:, i], vars[:, i] = predict_y(X, xstars[:,i], Y, θ)
    end
    f = Figure()
    ax = Axis3(f[1,1];
               xlabel="Temp. (°C)", ylabel="CR (°C/s)", zlabel="ΔL (μm)")

    for i in 1:m
        y = repeat([rates[i]], n)
        x = xstars[:, i]
        z = means[:,i]
        σ = sqrt.(vars[:,i])
        lines!(ax,x,y,z, color=log10.(σ), colormap=:managua)
    end
    return f
end
export inference_surface

# θi = [17.414065021040688, 0.5896643153148926, 0.04492645656636867]

function single_cr_plot(cr, X, Y, θ)
    n = size(X, 1)
    xstars = build_x_star(900.0, 30.0, cr, 0.07, n)
    means, vars = predict_y(X, xstars, Y, θ)
    f = Figure()
    ax = Axis(f[1,1], xlabel="Temp. (°C)", ylabel="ΔL (μm)")
    lines!(ax, xstars, means)
    return f
end
export single_cr_plot
