@def title = "Gaussian processes and linear regression"
@def showall = true

# Gaussian processes and linear regression
Oisín Fitzgerald, May 2021

@@boxed
A look at section 6.4 to 6.? of: 
 
Bishop C.M. (2006). Pattern recognition and machine learning. Springer.

[https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

Basically this post goes through (Bayesian) linear regression from a Gaussian process space point of view with some example [Julia](https://julialang.org/) code to make things concrete. The observed data $\mathfrak{Y} = \{y_1,...,y_N\}$ are a finite set of observations from a linear function $f \in F$ indexed by (the input features) $\mathfrak{X} = \{x_1,...,x_N\}$, where $F$ is a space of probabilistic linear functions.
@@

## Overview

The dominant approach to solving regression problems in machine learning today is finding the parameters $w$ of a model $M_w$ that minimise a loss function $L$ by optimally combining a set of basis vectors. These basis vectors can be the original data $x_n$ or some transformation $z_n = \phi(x_n)$ where $(y_n,x_n)$ is the $n^{th}$ output-input pair $n \in \{1,...,N\}$ and $x_n$ is length $p$ (the number of features). For example: 

* Linear regression: find the best set of weights $w$ that minimise mean square error $\left\Vert Y - X w \right\Vert$ giving us predictions $y_n = w^t x_n$. 

* Deep learning: at the other extreme of complexity we can think of deep learning as learning both the basis vectors and the weights. In a network with $L$ layers the outputs and weights of the final layer are $z_L$ and $w_L$ giving us $y_n =  w_L^t z_L(x_n)$. 

With Gaussian processes with are going to switch from thinking in terms of locating which parameters are most likely to have generated the data to considering the data a finite sample from a function that has particular properties. The parameters and function space viewpoint are not conflicting, for example for linear regression:   

1. Parameter space view: $y$ is a combination of basis functions with the weights being from a mltivariate normal distribution. 

2. Function space view: $y(x_n)$ is a sample from a family of functions where any finite sample of points $\{y_1,...,y_N\}$ follow a multivariate normal distibution. 

## From the parameter to function space view

To fully see the connection let's go from the parameter space view to the function space view for linear regression. The model is 

$$y(x_n) = w^t x_n$$

In matrix form the above is written as $Y = X w$, with each row of the $N \times p$ matrix $X$ made up of the $N$ individual observations $x^t_n$, each a vector of length $p+1$, the number of features plus one (to have an intercept term). The prior distribution on our weights $w$ reflects a lack of knowledge about the process

$$w \sim N(0,\alpha^{-1}I)$$ 

For example if there is one input we have $w = (w_0, w_1)^t$ and setting $\alpha = 1.0$ (arbitrarily) the prior looks like the graph below.

```julia:fig1
using Plots, Random, Distributions, LinearAlgebra
Random.seed!(1)
α = 1.0
d = MvNormal([0,0], (1/α)*I)
W0 = range(-1, 1, length=100)
W1 = range(-1, 1, length=100)
p_w = [pdf(d, [w0,w1]) for w0 in W0, w1 in W1]
contourf(W0, W1, p_w, color=:viridis,xlab="w0",ylab="w1",title="Prior: weight space")
savefig(joinpath(@OUTPUT, "fig1.svg")) # hide
```
\fig{fig1}

Since we treat input features (the x's) as constants this implies a prior distribution for the output 

$$y \sim N(0,\alpha X^t X)$$ 

From the function space view we can randomly sample functions at finite spacings $\mathfrak{X} = \{x_1,...,x_N\}$ from the prior.

```julia:fig2
using Plots, Random, Distributions, LinearAlgebra # hide
Random.seed!(1)
α = 1.0
x1 = range(-1, 1, length=100)
X = [repeat([1],100) x1]
d = MvNormal(repeat([0],100), (1/α)*X*transpose(X) + 1e-10*I)
p = plot(x1,rand(d),legend=false,seriestype=:line,title="Prior: function space",xlabel="x",ylabel="y")
for i in 1:20
    plot!(p,x1,rand(d),legend=false,seriestype=:line)
end
savefig(p,joinpath(@OUTPUT, "fig2.svg")) # hide
```
\fig{fig2}

The matrix $K = \text{cov}(y) = \alpha^{-1} X^t X$ is made up of elements $K_{nm} = k(x_n,x_m) = \frac{1}{\alpha}x_n^t x_m$ with $k(x,x')$ the kernel function. Notice that the kernel function $k(x,x')$ returns the variance for $x = x'$ and covariance between $x$ and $x'$ otherwise. Also that we are talking here about the covariance between *observations*, not features. $K$ is a $N \times N$ matrix and so can be quite large. There are many potential kernel functions other than $k = x^tx$ but that's for another day.   

## Modelling data with straight lines

We have a prior on $y$ and then we observe some data. Let assume there is noise in the data so we observe 

$$t_n = y(x_n) + \epsilon_n$$

with $\epsilon_n \sim N(0,\beta)$ random noise that is independent between observations and $t = \{t_1,...,t_N\}$ the observed output values for input features $x_n$. 

```julia:fig3
using Plots, Random, Distributions, LinearAlgebra # hide
Random.seed!(1)
n = 10
β = 0.01
d = MvNormal(repeat([0],n), (1/α)*X*transpose(X) + β*I)
y = rand(d) 
p = scatter(x,y,legend=false,title="Observed data",xlabel="x",ylabel="y")
savefig(p,joinpath(@OUTPUT, "fig3.svg")) # hide
```
\fig{fig3}

At this point in practise we could estimate the noise parameter $\beta$, but lets come back to that. For now assume we know that $\beta = 0.01$. It is worth remember there are no weights giving us the intercept, slope etc but we can 
sample from our distribution of $y|t$ given the observed data. Because our interest is in predicting for new observations we'd like to estimate the posterior $p(t*|t,x,x*)$ for any future input $x*$. It turns out the posterior for for any $t*$ is another
normal distribution which is coded below. 

```julia:fig4
p = scatter(x,y,legend=false,
            title="Posterior: function space",xlabel="x",ylabel="y")

# new X's over which to predict
xs = range(-1, 1, length=100)
Xs = [repeat([1],100) xs]
ys = zeros(100)

# get ready to construct posterior
σ2 = zeros(100)
C = (1/α)*X*transpose(X) + β*I
Cinv = inv(C)

# one prediction at a time 
for i in 1:100
    k = X * Xs[i,:]
    c = Xs[i,:]' * Xs[i,:] + β
    ys[i] = (k' * Cinv) * y
    σ2[i] = c - (k' * Cinv) * k
end
plot!(p,xs,ys, ribbon=(2*sqrt.(σ2),2*sqrt.(σ2)), lab="estimate")
plot!(p,xs,ys)

# noise free samples from the posterior
# all predictions at once
m = (Xs * X') * Cinv * y
CV = (Xs * Xs') - (Xs * X') * Cinv * (X * Xs')
CV = Symmetric(CV) + 1e-10*I
d = MvNormal(m, Symmetric(CV) + 1e-10*I)
for i in 1:20
    plot!(p,xs,rand(d),legend=false,seriestype=:line)
end
savefig(p,joinpath(@OUTPUT, "fig4.svg")) # hide
```

\fig{fig4}


## Estimating the hyperparameters

To estimate the hyperparameters  - in this case the noise but can also be aspects of the kernel - we can use standard approaches such as maximum likelihood or Bayesian estimation.