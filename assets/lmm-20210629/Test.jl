## libraries
# linear algebra
using LinearAlgebra, SparseArrays
# plotting
using Plots
# optimisation
import Statistics, Optim
#import NLopt

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
    θ = exp.(logθ)

    # dimensions
    Nn,n,p = dim
    N = Nn/n

    # estimation of \beta and b given theta
    LL = (θ[2]/θ[1])*DtD + I
    βb[:] = LL \ (θ[2]*Dty) # <- bottleneck, see Bates et al (2015)

    # -2 log likelihood (profile likelihood)
    logdetθ = log(1/θ[1]*N + 1)*n
    nll = (1/θ[1])*sum((y - D*βb).^2) + (1/θ[2])*sum(βb[(p+1):end].^2) + 2*logdetθ  + n*log(θ[2]) + Nn*log(θ[1])
    nll
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
    opt = Optim.optimize(var -> loglik!(βb,D,DtD,Dty,y,var,dim), log.(θ0))
    θ = exp.(Optim.minimizer(opt))

    # output
    out = LMM(βb[1:p],θ,βb[(p+1):end])
    out
end

"""
A struct to store the results of our LMM estimation
"""
struct LMM
    β
    θ
    b
end


N, n, p = 30, 1000, 100
ids = repeat(1:n,inner=N)
X = [repeat([1.0],N*n) randn(N*n,p)]
β = randn(p+1)
θ2 = 3.0
b = sqrt(θ2) .* randn(n)
Z = sparse(kron(sparse(1I, n, n),repeat([1],N)))
y = X * β + Z * b + randn(N*n);
print("number of rows: ",length(y))
@time res = lmm_fit(X,Z,y);
res.θ
scatter(res.β,β)
plot!(β,β)
res.θ
scatter(res.b,b)

