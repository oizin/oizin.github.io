# This file was generated, do not modify it. # hide
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