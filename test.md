# test page

```julia:./code/test1
using MixedModels,RDatasets
df = dataset("plm", "Gasoline")
x = 3
```

```julia:./code/test2
println(df[1:20,:])
```

\output{./code/test}

```julia:./code/test3
println(x)
```

\output{./code/test3}

```julia:./code/test4
m1 = fit(MixedModel, @formula(LCarPCap ~ 1 + Year + LIncomeP + LRPMG + (1|Country)), df)
println(m1)
```

\output{./code/test4}
