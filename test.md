# test page

```julia:test1
using MixedModels,RDatasets
df = dataset("plm", "Gasoline")
x = 3
```

```julia:test2
println(df[1:20,:])
```

\output{test2}

```julia:test3
y = 5
println(x)
```

\output{test3}

```julia:test4
m1 = fit(MixedModel, @formula(LCarPCap ~ 1 + Year + LIncomeP + LRPMG + (1|Country)), df)
println(m1)
```

\output{test4}
