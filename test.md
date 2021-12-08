# test page

```julia:test1
using RDatasets
```

```julia:test2
df = dataset("plm", "Gasoline")
println(df)
```

\output{test2}
