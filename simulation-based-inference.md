@def title = "Simulation-based inference"
@def showall = true

# Simulation-based inference
Oisín Fitzgerald, April 2021

@@boxed
A look at:  
 
Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48), 30055-30062.  

[https://arxiv.org/pdf/1911.01429](https://arxiv.org/pdf/1911.01429)
@@


## Setting

We have a model $\mathcal{M}_{\theta}$, with parameters $\theta$ of some physical or biological phenomena. This model or it's dynamics are complex enough that it is considered a black box. It is easy to run $\mathcal{M}_{\theta}$ in "forward-mode" $\mathcal{M}_{\theta}: \theta \rightarrow x_{sim}$ but difficult to do the opposite direction $\mathcal{M}^{-1}_{\theta}: x_{sim} \rightarrow \theta$. Examples of such models are:

* Particle physics: the production of a Higgs boson at the Large Hadron Collider followed by decay, instrumentation interaction etc.   
* Systems biology: number of generations we need to go back to find the common ancestor of a sample of individuals   

After performing some experiments we observe data $x_{obs}$ and (assuming $\mathcal{M}_{\theta}$ is a "useful" model in the George Box sense) we wish to estimate the $\theta$ most congruent with the experiment. However, we cannot perform traditional maximum likelihood or Bayesian estimation. The reason for this is the intractability of the likelihood. There are latent variables $z$ (e.g. intermediate states) implicit in our model and forming our likelihood by integrating over all possible latent states $l(\theta) = p(x|\theta) = \int p(x,z|\theta)dz$ is "counting all the grains of sand on the beach" too difficult. 

## Solution

A broad description of the solution is that we sample $\theta$ from a prior distribution $p(\theta)$ and use this to generate $x_{sim}$ from $\mathcal{M}_{\theta}$. We then base our inference about the most likely value of $\theta$ using some measure $\rho(x_{sim},x_{obs})$ of similarity between the simulated and observed data. (Formal statistical inference could be through forming a posterior - approximate Bayesian computation (ABC) - or frequentist approaches).

There are several interesting areas coming together to produce the next generation of simulation based inference techniques:  

* Machine learning: automatic function approximation.
* (Deep learning ->) Differentiable programming: enabling joint learning of traditional neural network and other strctures through gradient based updates and assessing sensitivity to small changes.     
* (Bayesian inference ->) Probabilistic programming: incorporate/quantify uncertainty throughout. 

## After thought

I wasn't familiar with this area before reading so this was a good high level overview. I now finally understand what part of a portion of a stochastic proccesses class was I took was about :) !

My current area of work is medical where there is not always a well established  (in a mathematical or computational sense) model of a disease process. As a result it is rare to see people building or incorporating models into their work, the hunt is often for risk factors - through correlations in datasets, or evaluating decisions - RCTs/causal analysis. In many ways this makes sense, often the process is too complex to meaningfully model with the available data or we lack ability to perform necessary experiments (due to time/ethical constraints). However, I wonder whether new hardware (e.g. continuous blood glucose monitoring) will change this through enabling better connection of long-term outcomes to personal physiology over the intervening time period.
