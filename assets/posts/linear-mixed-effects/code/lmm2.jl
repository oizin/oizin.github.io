# This file was generated, do not modify it. # hide
gdf = groupby(df,:Country)
mdf = combine(gdf, :LCarPCap => mean, :LIncomeP => mean)
df = leftjoin(df,mdf,on=:Country)
df.LIncomeP_change = df.LIncomeP - df.LIncomeP_mean
df.LCarPCap_change = df.LCarPCap - df.LCarPCap_mean
f = Figure(resolution = (800, 400))
ax1 = scatter(f[1, 1],mdf.LCarPCap_mean,mdf.LIncomeP_mean)
ax1.axis.xlabel = "Mean cars per capita (log scale)"
ax1.axis.ylabel = "Mean income per capita (log scale)"
ax1.axis.title = "Variation between"
ax2 = scatter(f[1, 2],df.LCarPCap_change,df.LIncomeP_change)
ax2.axis.xlabel = "Change in cars per capita (log scale)" 
ax2.axis.ylabel = "Change in income per capita (log scale)"
ax2.axis.title = "Variation within"
f
save(joinpath(@OUTPUT, "fig2.svg"),f) # hide