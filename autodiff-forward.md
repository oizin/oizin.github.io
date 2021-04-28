
@def title = "Forward model autodiff"
@def showall = true

# Forward mode automatic differentiation
Oisín Fitzgerald, April 2021

@@boxed
A look at the first half (up to section 3.1) of:  
 
Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. Journal of machine learning research, 18.  

[https://www.jmlr.org/papers/volume18/17-468/17-468.pdf](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
@@

Automatic differentiation (autodiff) reminds me of Arthur C. Clarke's quote "any sufficiently advanced technology is indistinguishable from magic". Whereas computer based symbolic and numerical 
differentiation seem like natural descendants from blackboard based calculus, the first time I learnt
about autodiff (through [Pytorch](https://pytorch.org/)) I was amazed. It is not that the ideas underlying autodiff themselves
are particularly complex, indeed Bayin et al's look at the history of autodiff puts Wengert's 1964
paper entitled "A simple automatic derivative evaluation program" as a key moment in forward mode
autodiff history. (BTW the paper is only 2 pages long - well worth taking a look). For me the magic comes from autodiff being this digitally
inspired look at something as "ordinary" but so important to scientific computing and AI as differentiation.

## Differentiation
If you are unsure of what the terms symbolic or numerical differentiation mean, I'll give a a quick
overview below but would encourage you to read
the paper and it's references for a more detailed exposition of their various 
strengths and weaknesses. 

### Numeric differentiation - wiggle the input

For a function $f$ with a 1D input and output describing numeric differentiation (also known as the finite difference method) comes quite naturally from the definition of the derivative. The derivative is
$$\frac{df}{dx} = \text{lim}_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h}$$ so we approximate this expression by picking a small enough $h$ 
(there are more complex schemes). 
There are two sources of error here, the first is from approximating the infinitesimally small $h$ with a plain finitely small $h$ (*truncation error*) and
the second is from *round-off error*. 
Round-off error occurs because not every number is represented in the set of floating point numbers so for a small $h$ the difference $f(x+h)-f(x)$ can be quite unstable. Unfortunately these two source of error play against each other (see graph - on the left hand size round-off error dominates whereas on the right hand side truncation error dominates).

```julia:finite
using Plots
h = 10 .^ range(-15, -3, length=1000)
x0 = 0.2
f(x) = (64*x*(1-x)*(1-2*x)^2)*(1-8*x+8*x^2)^2
df = (f.(x0 .+ h) .- f(x0)) ./ h
plot(log10.(h),log10.(abs.((df .- 9.0660864))),
xlabel="log10(h)",ylabel="log10(|Error|)",legend=false)
savefig(joinpath(@OUTPUT, "finite.svg")) # hide
```

\fig{finite}

However, such small errors are actually not all that important in machine learning! The main issue with numeric
differentiation for machine learning is that the number of required evaluations of our function $f$
scales linearly with the number of dimension of the gradient. In contrast backpropagation (an autodiff method) can calculate the 
derivatives in "two" evaluations of our function (one forward, one back). 

### Symbolic - fancy lookup tables

Symbolic differentiation is differentiation as you learnt it in school programmed into software, all the rules
$\frac{d}{dx}\text{cos}(x) = -\text{sin}(x), \frac{d}{dx} x^p = px^{(p-1)}, \frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$ etc... are known and utilised by the software. If you evaluate the derivative of a 
function `f` using a symbolic programming language `dfdx = derivative(f,x)` the object returned `dfdx` is just whatever function the symbolic program matches as the derivative of `f` using it's internal derivative lookup and application of the rules of differentiation (chain rule etc). **It is manipulation of expressions**. 
The main issue with symbolic differentiation for ML (which anyone who has
used Mathematica for a help with a difficult problem can attest to) is expression swell, where the derivative expression is exponentially longer than the original expression and involves repeated calculations.

### Automatic differentiation - alter the program

Autodiff is the augmentation of a computer program to perform standard computations along with **calculation
of derivatives**, there is no manipulation of expressions. It takes advantage of the fact that derivative expressions can be broken down into 
elementary operations that can be combined to give the derivative of the overall 
expression. I'll be more clear about elementary operations soon but you can think of an elementary operations as being any operation you could give to a node on a computational graph of your program.

## Forward mode

To be more concrete about autodiff, let's look at forward mode. Consider evaluating $f(x_1,x_2) = x_1 x_2 + \text{log}(x_1 ^2)$. We break this into the computational graph below and associate with each elementary operation the intermediate variable 
$\dot{v}_i = \frac{\partial v_i}{\partial x}$, called the "tangent". The final "tangent" value $\dot{v}_5$, which has been calculated as the function evaluates at the input (3,5) is a derivative at the point (3,5). What derivative exactly depends on the initial values of $\dot{x_1}$ and $\dot{x_2}$. 

![Example of forward autodiff.](/assets/autodiff-forward-20210426/example.png) 

## Sketching a forward mode autodiff library

\literate{/_literate/autodiff-forward-20210426.jl}

## Conclusion

Autodiff is important in machine learning and scientific computing and (forward mode) surprisingly easy to implement. I'll look at reverse mode autodiff in another post. 

Thanks to Oscar Perez Concha who helped for discussions on the content of this post.  
