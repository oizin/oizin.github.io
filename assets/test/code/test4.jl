# This file was generated, do not modify it. # hide
m1 = fit(MixedModel, @formula(LCarPCap ~ 1 + Year + LIncomeP + LRPMG + (1|Country)), df)
println(m1)