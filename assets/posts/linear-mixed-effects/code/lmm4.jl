# This file was generated, do not modify it. # hide
## libraries
# linear algebra
using LinearAlgebra, SparseArrays
# optimisation
using Optim
import Statistics

"""
Calculates log likelihood for LMM. 
Internally calculates fixed and random effects given estimates of the variance-covariance components, 
with modification of first three arguments βb, LL, rr. 

Designed for `lmm_fit`.

Args
    βb  : vector of estimates of fixed and random effects
    D   : fixed and random effect design matrices
    DtD : D'D
    Dty : D'y
    y   : outcome vector
    logθ: log of variance-covariance components
    dim : tuple of dimensions

"""
function loglik!(βb,D,DtD,Dty,y,logθ,dim)
    σ,σ_b = exp.(logθ)

    # dimensions
    Nn,n,p = dim
    N = Nn/n

    # estimation of \beta and b given theta
    diagf = diagm([repeat([0.0],p);repeat([1/σ_b^2],n)])
    LL = DtD ./ σ^2 + diagf
    βb[:] = LL \ (Dty ./ σ^2)

    # -2 log likelihood (profile likelihood)
    logdetθ = logdet(DtD[(p+1):end,(p+1):end] ./ σ^2 + diagf[(p+1):end,(p+1):end])
    nll = (1/σ^2)*sum((y - D*βb).^2) + (1/σ_b^2)*sum(βb[(p+1):end].^2) + 2*logdetθ  + n*log(σ_b^2) + Nn*log(σ^2) + n*log(2*π)
    nll ./ 2
end

"""
Estimate a LMM

Args
    X : Fixed effect design matrix
    Z : Random effect design matrix
    y : outcome
"""
function lmm_fit(X,Z,y)

    # dimensions / data
    Nn = length(y)
    n = size(Z)[2]
    p = size(X)[2]
    dim = (Nn,n,p)
    D = [X Z]
    DtD = D'D
    Dty = D'y

    # optimisation setup
    βb = zeros(n+p)
    θ0 = ones(2)

    # optimise
    opt = optimize(var -> loglik!(βb,D,DtD,Dty,y,var,dim), log.(θ0), NelderMead())
    θ = exp.(Optim.minimizer(opt))

    # output
    out = LMM(βb[1:p],θ,βb[(p+1):end],opt)
    out
end


"""
A struct to store the results of our LMM estimation
"""
struct LMM
    β
    θ
    b
    opt
end

# A small test - the output should be approx [1.0,3.0]
N, n, p = 30, 100, 10
ids = repeat(1:n,inner=N)
X = [repeat([1.0],N*n) randn(N*n,p)]
β = randn(p+1)
θ2 = 3.0
b = sqrt(θ2) .* randn(n)
Z = sparse(kron(sparse(1I, n, n),repeat([1],N)))
y = X * β + Z * b + randn(N*n);
res = lmm_fit(X,Z,y);
println("Variance components: ",round.(res.θ .^ 2,digits=3))