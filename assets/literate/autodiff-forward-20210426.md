<!--This file was generated, do not modify it.-->
It's surprisingly easy to implement forward mode autodiff in [Julia](http://www.julialang.org)
(at least a naive form). Below I create a forward model module that creates a new object `Dual` that
is a type of `Number`, and then proceed to overload common mathematical functions (e.g. `sin` and `*`)
to account for this new number type. Each instance of `Dual` with have a `p`rime and `t`angent slot.
If we want the derivative with respect to argument `x₁` of the function `y = f(x₁,x₂)`  we
simply set `x₁.t = 1.0` (leaving `x₂.t = 0.0`) and check the value of `y.t`.
For more see [this video](https://www.youtube.com/watch?v=vAp6nUMrKYg&t=363s)
from MIT's Alan Edelman

```julia:ex1
import Base: +,-,/,*,^,sin,cos,exp,log,convert,promote_rule,println

struct Dual <: Number
  p::Number # prime
  t::Number # tangent
end

+(x::Dual,y::Dual) = Dual(x.p + y.p, x.t + y.t)

-(x::Dual,y::Dual) = Dual(x.p - y.p, x.t - y.t)

/(x::Dual,y::Dual) = Dual(x.p/y.p, (x.t*y.p - x.p*y.t)/x.p^2)

*(x::Dual,y::Dual) = Dual(x.p*y.p, x.t*y.p + x.p*y.t)

sin(x::Dual) = Dual(sin(x.p), cos(x.p) * x.t)

cos(x::Dual) = Dual(cos(x.p), -sin(x.p) * x.t)

exp(x::Dual) = Dual(exp(x.p), exp(x.p) * x.t)

log(x::Dual) = Dual(log(x.p), (1/x.p) * x.t)

^(x::Dual,p::Int) = Dual(x.p^p,p*x.p^(p-1)* x.t)

# We can think of dual numbers analogously to complex numbers
# The epsilon term will be the derivative
println(x::Dual) = println(x.p," + ",x.t,"ϵ")

# deal with conversion, and Dual with non-Dual math
convert(::Type{Dual}, x::Number) = Dual((x,zero(x)))
promote_rule(::Type{Dual},::Type{<:Number}) = Dual;
```

Lets test on our example $f(x_1,x_2) = x_1 x_2 + \text{log}(x_1 ^2)$, the derivative
at (3,5) should be $5 \frac{2}{3}$.

```julia:ex2
x1 = Dual(3.0,1.0)
x2 = Dual(5.0,0.0)
f(x1,x2) = x1*x2 + log(x1^2)
y = f(x1,x2)
# df/dx1
println(y.t)
# direct calculation
println(x2.p + 2/x1.p)
```

