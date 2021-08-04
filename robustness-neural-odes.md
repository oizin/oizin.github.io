
@def title = "The robustness of neural ODEs"
@def showall = true

# The robustness of neural ODEs
Oisín Fitzgerald, May 2021

@@boxed
A quick look at:  
 
Yan, H., Du, J., Tan, V. Y., & Feng, J. (2019). On robustness of neural ordinary differential equations. arXiv preprint arXiv:1910.05513.

[https://arxiv.org/abs/1910.05513](https://arxiv.org/abs/1910.05513)
@@

## Neural ODE

In a neural ODE we use a neural network to learn the dynamics of a differential equation. f_{\theta}(z(t),t) is a neural network 
parameterised by weights $\theta$ and $z(t)$ (a vector) is the state of our system at time $t$. This neural network and the initial 
conditions describe our system

$$\frac{dz(t)}{dt} = f_{\theta}(z(t),t), z_0 = z(0)$$

If we want to make a prediction for time $T$ we would use the usual approach to solving an ODE

$$z(T) = z_0 + \int_0^T f_{\theta}(z(t),t) dt$$

Noting of course that "solving" here is means computing the integral (e.g. using Euler,... methods) and that
training using automatic differentiation is required on top of that. 

## Why might Neural ODEs be more robust than other layers?

We can incorporate neural ODEs as layer in a deep learning network. The $z$'s mentioned above could be the input from a 
previous layer (e.g. a convolution). One reason a neural ODE could be more robust (and there are numeric examples in the
paper showing it is for the data considered) than other layers is due to a basic theorem of differential equations. 
The solutions to an ODE, the integral curves, can't overlap, and so perturbation of the input with noise may shift 
a output but in a fashion that is constrained. In particular the authors show you can design the 
neural ODE to enforce strong constraint on the degree of possible perturbation.