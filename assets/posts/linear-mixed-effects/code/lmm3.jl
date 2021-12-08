# This file was generated, do not modify it. # hide
f = Figure(resolution = (800, 600))
ax = Axis(f[1,1:2], xlabel = "Year", ylabel = "Cars per capita (log scale)",
    title = "Variation at baseline and over time")
for country in unique(df.Country)
    msk = df.Country .== country
    lines!(ax,df.Year[msk],df.LCarPCap[msk],color = :lightblue)
end 
ax1 = scatter(f[2, 1],df.LIncomeP,df.LCarPCap)
ax1.axis.ylabel = "Cars per capita (log scale)"
ax1.axis.xlabel = "Income per capita (log scale)"
ax2 = scatter(f[2, 2],df.LRPMG,df.LCarPCap)
ax2.axis.ylabel = "Gasoline price (log scale)"
ax2.axis.xlabel = "Income per capita (log scale)"
f
save(joinpath(@OUTPUT, "fig3.svg"),f) # hide