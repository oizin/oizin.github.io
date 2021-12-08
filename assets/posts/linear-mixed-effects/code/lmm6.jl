# This file was generated, do not modify it. # hide
using MixedModels
m1 = fit(MixedModel, @formula(LCarPCap ~ 1 + Time + LIncomeP + LRPMG + (1|Country)), df)
println(m1)