using LinearAlgebra, AMDGPU, KernelAbstractions, DilPredict
const KA = KernelAbstractions
# This is temporary; will need to change based on the user's device:
const backend = KA.get_backend(AMDGPU.ROCArray([1]))

@kernel function simdiag_k!(M, σ)
    i,j,k = @index(Global, NTuple)
    if i == j
        M[i,j,k] = σ
    end
end
simdiag! = simdiag_k!(backend)

# Create a diagonal tensor of σ with same size as D
function diagm_d(D, σ)
    M = KA.zeros(backend, Float32, size(D)...)
    simdiag!(M, σ; ndrange=size(D))
    return M
end
export diagm_d

function grad_expo_d(θ, D)
    # Try to rewrite this as a single kerenel
    σ = θ[1]; l = θ[2];
    k = exp.(D ./ l^2)
    g_σf = (2 .* σ .* k)
    g_l = ((-2 .* D .* σ^2) ./ l^3 .* k)
    g_σn = (diagm_d(D, 1.0))
    return [Array(g_σf), Array(g_l), Array(g_σn)]
end
export grad_expo_d

function nlogp_d(θ, D_d, Y)
    N, _, n = size(D_d)
    # factorization is so much faster on CPU that it actually makes
    # sense to calculate K on GPU then transfer it back to CPU to find logp
    K = Array(exponentiated_kernel(D_d, θ) .+ diagm_d(D_d, θ[3]))
    nlogp_all = 0.0
    for k in 1:n
        y = Y[k, :]
        K_t = K[:,:,k]
        L = cholesky(K_t)
        α = L.U \ (L.L \ y)
        nlogp_all += 0.5 * (y' * α) + sum(log.(diag(L.L))) + N/2 * log(2π)
    end
    return nlogp_all
end
export nlogp_d

function grad_nlogp_d(θ, D, Y)
    N, _, n = size(D)
    len_θ = length(θ)
    K = Array(exponentiated_kernel(D, θ) .+ diagm_d(D, θ[3]))
    K_grads = grad_expo_d(θ, D)
    #nlogp_all = Vector{Float64}(undef, n)
    Gnlogp_tot = zeros(Float32, len_θ)
    @views for k in 1:n
        y = Y[k, :]
        K_t = K[:,:,k]
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
export grad_nlogp_d

