# This file was generated, do not modify it. # hide
#hideall
using CairoMakie, DataFrames, Statistics, CSV
df = DataFrame(CSV.File("./_assets/lmm-20210629/Gasoline.csv"))
f = Figure(resolution = (800, 400))
ax = Axis(f[1,1], xlabel = "Year", ylabel = "Cars per capita (log scale)",
    title = "Variation at baseline and over time")
for country in unique(df.Country)
    msk = df.Country .== country
    lines!(ax,df.Year[msk],df.LCarPCap[msk],color = :lightblue)
end 
f
save(joinpath(@OUTPUT, "fig1.svg"),f) # hide