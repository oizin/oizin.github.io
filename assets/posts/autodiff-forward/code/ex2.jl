# This file was generated, do not modify it. # hide
x1 = Dual(3.0,1.0)
x2 = Dual(5.0,0.0)
f(x1,x2) = x1*x2 + log(x1^2)
y = f(x1,x2)
# df/dx1
println(y.t)
# direct calculation
println(x2.p + 2/x1.p)