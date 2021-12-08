# This file was generated, do not modify it. # hide
using Plots
h = 10 .^ range(-15, -3, length=1000)
x0 = 0.2
f(x) = (64*x*(1-x)*(1-2*x)^2)*(1-8*x+8*x^2)^2
df = (f.(x0 .+ h) .- f(x0)) ./ h
plot(log10.(h),log10.(abs.((df .- 9.0660864))),
xlabel="log10(h)",ylabel="log10(|Error|)",legend=false)
savefig(joinpath(@OUTPUT, "finite.svg")) # hide