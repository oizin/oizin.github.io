@def title = "[notes]"

\newcommand{\figenv}[3]{
~~~
<figure style="text-align:center;padding:0;margin:0">
<img src="!#2" style="padding:0;border:1px solid black;margin:0;#3" alt="#1"/>
<figcaption>#1</figcaption>
</figure>
~~~
}

\figenv{}{/assets/banner.png}{width:100%}

# Posts

#### [Linear mixed effect models](/posts/linear-mixed-effects)

An introduction to linear mixed effects models (LMMs) and their estimation. Aims to give an intuitive reason why we need LMMs followed by some theory and code that codes them (largely) from scratch.

#### [Linear regression from a Gaussian process point of view](/posts/gp-linear)

Gaussian processes have an aura of abstract complexity - "distributions over function space". I find that linking them to linear models helps reduce the abstractness. 

#### [Frontier of simulation-based inference](/posts/simulation-based-inference)

Some notes after reading Cranmer, Brehmer & Louppe's overview of simulation based inference.

#### [Automatic differentiation I](/posts/autodiff-forward)

Some notes and code after reading Baydin, Pearlmutter, Radul & Siskind's excellent paper on the magic that is automatic differentiation.

#### [Causal mediation: an overview](/posts/causal-mediation)

A short introduction to causal mediation analysis and the need to think carefully about potential confounding when undertaking such analyses.

#### [ARCH models and Bitcoin volatility](/posts/bitcoin-volatility)

Autoregressive conditional heteroscedasticity (ARCH) models have such a long name they must be great right!? I develop some ARCH models that attempt (badly) to predict Bitcoin / \$US price movements.   

#### [Structural nested mean models](/posts/structural-nested-mean-models)

This is a long post trying to understand structural nested mean models and their role in causal inference. 

