@def title = "LMMs"

# Linear mixed effect models
Oisín Fitzgerald, June 2021

\toc

@@boxed
This is an introduction to linear mixed effect models. It is based on Simon Wood's book on generalised additive models and notes and articles by Douglas Bates, listed at the end.

Please feel free to comment at the end, questions, corrections etc welcome.

**Work in progress**
@@


## Introduction 

### Mixed effect models

Multilevel or mixed effect models are useful whenever our data contains repeated samples from the "statistical units" 
that make up our data. There is no fixed definition of a unit - what matters is 
that we consider it likely that the data within each unit is correlated. For example:

* Repeated blood pressure measurements from a group of individuals. Each individual's blood pressure is likely to be highly correlated over time. 
* Assessment of student performance over several schools. Since each school has it's own set of teachers, policies, and enrolment area student performance within a school may be correlated.

What this means is that
if we have $N$ units each with $n$ measurements while our data contains $N \times n$ rows we might actually have much closer to $N$ pieces of "information"
(depending on the strength of the generally positive correlation - at the extreme, if your blood pressure never changed then clearly you only have $N$ pieces of information). Since many statistical tests/models assume independent and identically distributed (*i.i.d.*) data one important application of multilevel models is correct inference due to the clustered nature of the data. We can also answer (more interesting) questions such as:

* How much variation is there between units? And how much of the unexplained variation in the outcome can be attributed to between unit variation?
    - This is a common question in institutional comparison - for example, after controlling for patient characteristics is there still variation between hospitals in patient outcomes?
* What is the best prediction for future values of unit *j*?
    - Multilevel models both pool the data of many units together while accounting for individual units thus offering an alternative to approaches to approaches that simply correct the estimated variance (of e.g. a linear model) for the clustering and econometric approaches that focus on the individual units.

![Multilevel data.](/assets/lmm-20210629/sampling.png) 

The description above suggests you might only have one "level" of units. However, multilevel models can account for many levels of hierarchical clustering. For example, measurements within patients within medical practises.  

### Linear mixed effect models 

With the notation $x$ / $X$ refering to a vector / matrix, and $x_i$ / $X_{ij}$ the element of a matrix / vector, a linear mixed effects model (LMM) can be written as

$$y = X\beta + Z b + \epsilon, b \sim N(0,\Lambda_{\theta}), \epsilon \sim N(0,\Sigma_{\theta})$$

where  

* $\beta \in \mathcal{R}^p$ are the fixed effects, analogous to the coefficients in a standard linear model.
* $X \in \mathcal{R}^{Nn \times p}$ is the model matrix for the fixed effects containing the covariates / features.
* The random vector $b$ contains the random effects, with zero expected value and covariance matrix $\Lambda_{\theta}$  
* $Z \in \mathcal{R}^{Nn \times n}$ is the model matrix for the random effects
* $\Sigma_{\theta}$ is the residual covariance matrix. It is often assumed that $\Sigma_{\theta} = \sigma^2 I$
* $\theta$ is the variance-covariance components, a vector of the parameters of XX and YY. 

## Types of random effect

Any input feature can have a random features. However, it is common to hear people talk about random intercepts and random slopes.

<graphs>

### Motivating example 

Consider modelling 1 second forced expiratory volume (FEV1) as a function of age and height.

<graph>

As mentioned above a commpnly used form of the LMM is the random intercept model. In this situation for a single level (individual over time) the resulting 
model for individual $i$ at time $j$ is

$$y_{ij} = \beta_0 + \beta_1 \text{age}_{ij} + b_i + \epsilon_{ij}, b_i \sim N(0,\sigma_b^2), \epsilon_{ij} \sim N(0,\sigma_e^2)$$

I won't go into it it's construction but it is worth thinking about what $Z$ would look like in this case, and how it would change it we added a time random effect. Doing this will give you a sense of how the size of $Z$ can grow quite quickly while being a largely sparse (filled with zeros) matrix.

## Estimation 

Despite their similarity to linear model LMMs are trickier to estimate, largely due to the presence of the unobserved 
random effects and additional need to decompose the outcome variance into several variance-covariance parameters. 
It is worth understanding the estimation process at least superficially as it can aid in debugging (commonly "why is my model taking so long to estimate!?")
and understanding warning messages when using packages written by others.

### Theory

A common approach to estimation of LMMs is maximum likelihood estimation (MLE) or restricted MLE (REML) - 
but I'll just cover MLE here. As noted in Wood (2017) estimation of $\beta$ and $\theta$ could be based on the marginal distribution 
of the outcome $y \sim N(X\beta,Z^t\Lambda_{\theta}Z + \Sigma_{\theta})$ however this would involve the inversion of a $Nn \times Nn$ matrix $Z^t\Lambda_{\theta}Z + \Sigma_{\theta}$.
 As a result estimation is generally based on the the expression 

$$f(y) = \int f(y,b) db = \int f(y|b)f(b) db$$

It is worth listing out some the log pdf of the distributions that are going to come up in the derivation of a the final expression:

* Reminder that $\text{log}f(y,b) = \text{log}f(y|b) +  \text{log}f(b)$
* $\text{log} f(y|b) = c_{y|b} + \text{log}|\Sigma_{\theta}| - (y - X\beta - Z b)^t \Sigma_{\theta}^{-1} (y - X\beta - Z b)$
* $\text{log} f(b) = c_{b} + \text{log}|\Lambda_{\theta}| - b^t \Lambda_{\theta}^{-1} b$

Now we are ready to derive the estimation equations proceeds. Let $\hat{b}$ be the MLE of $f(y,b)$. Then utilising a Taylor expansion of $f(y,b)$ about
$\hat{b}$ on the second line we have

\begin{align}
f(y) &= \int f(y,b) db = \int \text{exp}\{\text{log} f(y,b)\}db \\
     &= \int \text{exp}\{\text{log} f(y,\hat{b}) + 0 + (b-\hat{b})^t \frac{\partial^2 \text{log} f(y,\hat{b})}{\partial b \partial b^t} (b-\hat{b})\}db \\
     &= f(y,\hat{b}) \int \text{exp}\{-(b-\hat{b})^t (Z^t\Sigma_{\theta}^{-1} Z + \Lambda_{\theta}^{-1})(b-\hat{b})/2\}db \\
\end{align}

The term inside the integral can be recognised is an un-normalised Gaussian pdf with covariance $(Z^t\Sigma_{\theta}^{-1} Z + \Lambda_{\theta}^{-1})^{-1}$. The 
normalisation constant for this pdf would be $\sqrt{|(Z^t\Sigma_{\theta}^{-1} Z + \Lambda_{\theta}^{-1})^{-1}| (2\pi)^{n}}$ and so, using the fact that $|A^{-1}| = |A|^{-1}$ the result of the integral is

\begin{align}
f(y) &= f(y|\hat{b})f(\hat{b}) |(Z^t\Sigma_{\theta}^{-1} Z + \Lambda_{\theta}^{-1})^{-1}|^{-1/2} c_y \\
\end{align}

In practise we will work with the *deviance* (minus two times the log-likelihood), inputting our expressions for $\text{log} f(y|b)$ and $\text{log} f(b)$ from above gives the 
quantity to be minimised as

$$
d(\beta,\theta) = -2l(\beta,\theta) = (y - X\beta - Zb)^t \Sigma_{\theta}^{-1} (y - X\beta - Zb) + b^t \Lambda_{\theta}^{-1} b + 
\text{log}|Z^t\Sigma_{\theta}^{-1} Z + \Lambda_{\theta}^{-1}| + c_{y|b} + c_{b} +  c_y \\
$$

Computation can be based on the observation that for a fixed $\theta$ we can get estimates of $\beta$ and $b$ using the first two terms

$$
d_1(\beta,\theta) = (y - X\beta - Zb)^t \Sigma_{\theta}^{-1} (y - X\beta - Zb) + b^t \Lambda_{\theta}^{-1} b
$$

Notice that the random effects (e.g. the individual intercept or feature effect) are shrinkage estimates of what we would get 
if we let every unit have it's own intercept or feature effect, giving the name penalised least squares. 

Then estimates of $\theta$ can be based on the profile likelihood (deviance) $d_p(\theta) = d(\hat{\beta},\theta)$. 

Some other observations:  

* The term log determinant terms can be interpreted as a complexity penalties on the 
variance-covariance parameters.
* A nicer approach is to let $b = \Gamma_{\theta} u$ where $u$ is a spherical normal variable (uncorrelated equal variance) and $\Lambda_{\theta} = \Gamma_{\theta}^t\Gamma_{\theta}$, reducing the dimension of $\theta$ by one (Bates et al, 2014).

### Algorithm 

How does estimation go in practise? Often a gradient free optimisation algorithm 
(Nelder-Mead or BOBYQA) is used for $d_p$.
(Although MixedModels.jl apparently does use the gradient $\nabla l_p$ sometimes but I couldn't see where).

0. Inputs $X$, $Z$, $y$, optimisation tolerance(s) $\tau$
1. Initialise $B^{(0)}$ = [$\beta^{(0)}$,$b^{(0)}$] = 0, $\theta^{(0)} = \bold{1}$, 
2. While $\tau$ not met:
    1. $B^{(k)}$: argmin $d_1(\beta,\theta)$
    2. $\theta^{(0)}$: argmin $d_p(\theta)$  

This is high level (but reasonable for understanding) view of how software packages like lme4 or MixedModel perform estimation for LMMs. 
See Bates et al (2015) for a detailed overview of the numerical linear algebra considerations in the implementations. 


### Julia implementation

For a more complete idea of how to code LMMs in practise see the source code for MixedModels.jl.
The code below estimates $\beta$ and the variance components $\theta$. 

```julia:lmm-20210629/code/ex1
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
res = lmm_fit(X,Z,y);
res.θ
```
\output{lmm-20210629/code/ex1}


How fast? Not very, but good learning exercise.

```julia:lmm-20210629/code/ex3
@time lmm_fit(X,Z,y)

using DataFrames, CategoricalArrays
using MixedModels
using RCall

df = DataFrame([y X[:,2:(p+1)] ids],:auto)
rename!(df, ["y"; "x".*string.(1:p);"id"])
df.id = convert.(Int64,df.id)
df.id = CategoricalVector(df.id)

fm = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + (1|id))
@time mm = fit(MixedModel, fm, df)

# vs. lme4 and mgcv
@rput df
R"""
df$id <- factor(df$id)
library(lme4)
library(mgcv)

tt_lme4 <- system.time({
     lmer(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + (1|id),data=df)
})
print(tt_lme4)

tt_mgcv <-  system.time({
     gam(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + s(id,bs="re"),data=df)
})
print(tt_mgcv)
"""
```
\output{lmm-20210629/code/ex3}

Clearly it is still worth using `MixedModels.jl` but the benefit of being able to code it yourself is the 
freedom you get to make changes in the underlying algorithm and see the effects.

### Example revisited 

Estimating the FEV1 model from above using `MixedModels.jl` gives the following results. 

<code>

## References 

This document borrows from the following: 

* Wood, S. N. (2017). Generalized additive models: an introduction with R. CRC press.  
* Bates, D., Mächler, M., Bolker, B., & Walker, S. (2014). Fitting linear mixed-effects models using lme4. arXiv preprint arXiv:1406.5823.
* MixedModels.jl: https://juliastats.org/MixedModels.jl/dev/


### Appendix: a taste of linear algebra and optimisation in Julia

Below are some examples of linear algebra and optimisation in Julia for those unfamiliar with the language.

@@boxed
Aside: 

Linear algebra in Julia  

https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/

https://docs.julialang.org/en/v1/stdlib/SparseArrays/

Optimisation in Julia  

https://julianlsolvers.github.io/Optim.jl/stable/

https://github.com/JuliaOpt/NLopt.jl
@@

The LinearAlgebra handles operations on dense matrices.

```julia
using LinearAlgebra

# log det via cholesky decomposition
A = randn(10,10)
B = A'A
cholB = cholesky(B)
B - cholA.L * cholA.L'
# calculating the log det (B)
logdet(cholB)
logdet(B)
sum(log.(abs.(diag(lu(B).U))))
2*sum(log.(diag(cholB.L)))

# solving least squares via QR
N = 100
p = 5
X = randn(N,p)
β = randn(p)
y = X*β + randn(N)
# QR decomposition
QR = qr(X)
# get beta hat
X \ y  # black box (not really, is QR)
QR.R \ (QR.Q' * y)[1:p]  # just back substition
QR \ y # again back substition (there is QR method for \)
```

For sparse matrices we use SparseArrays (dense and sparse matrices generally seem to play well together).

```julia
using SparseArrays

# sparse Cholesky is key to the MixedModels implementation, although 
# it isn't done like this. 
# See http://artadia.blogspot.com/2019/10/sparse-cholesky-decomposition-in-julia.html
# and https://github.com/JuliaLang/julia/issues/26147

A = sparse(Float64[10 0 1 0; 0 10 1 0; 1 1 10 1; 0 0 1 10])
fac = cholesky(A)

# equivalent
sparse(fac.L) ≈ cholesky(Matrix(A[fac.p,fac.p])).L

# Lower triangular factor and the permutation vector
L = LowerTriangular(sparse(fac.L))
perm = fac.p
invp = invperm(perm)

# solve a system of linear equations
b = A*Float64[1,2,3,4]
A\b ≈ (L' \ (L \ b[perm]))[invp]
```

The Optim package can perform gradient and gradient-free optimisation.

```julia
using Optim


```

{{ addcomments }}
