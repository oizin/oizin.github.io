# This file was generated, do not modify it. # hide
p = scatter(x1,y,legend=false,
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