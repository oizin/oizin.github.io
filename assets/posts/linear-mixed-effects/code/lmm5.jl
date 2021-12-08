# This file was generated, do not modify it. # hide
df.Time = df.Year .- 1965
n = length(unique(df.Country))
N = length(unique(df.Year))
X = [repeat([1.0],size(df)[1]) df.Time df.LIncomeP df.LRPMG]
Z = sparse(kron(sparse(1I, n, n),repeat([1],N)))
y = df.LCarPCap
res = lmm_fit(X,Z,y);
println("Variance components: ",round.(res.θ .^ 2,digits=3))
println("Fixed effects: ",round.(res.β,digits=4))