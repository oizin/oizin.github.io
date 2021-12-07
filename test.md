# test page

```julia:./code/test1
using DataFrames, RDatasets, MixedModels
df = dataset("plm", "Gasoline")
```


```julia:./code/test2
println(df[1:10,:])
```

\output{./code/test2}

```julia:./code/test3
m1 = fit(MixedModel, @formula(LCarPCap ~ 1 + Year + LIncomeP + LRPMG + (1|Country)), df)
println(m1)
```

\output{./code/test3}
