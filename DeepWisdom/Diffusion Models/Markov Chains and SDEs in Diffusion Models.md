---
title: Markov Chains and SDEs in Diffusion Models
type: concept
tags: [diffusion, generative-models, sde, score-matching, probability]
created: 2026-06-29
status: budding
aliases: [Discrete vs Continuous Diffusion, SDE Formulation of Diffusion, Markov Chain SDE Equivalence]
---

# Markov Chains and SDEs in Diffusion Models

[[Diffusion Models Root]] are usually introduced two different ways — as a **discrete-time Markov chain** (DDPM: $T$ fixed noising/denoising steps) or as a **continuous-time stochastic differential equation** (SDE: noise added/removed continuously over $t \in [0, T]$). These aren't two different models. The Markov chain *is* a numerical discretization of the SDE, and a single network trained once can be plugged into either interpretation. This note works out why.

> [!note] Intuition
> The discrete chain takes finitely many, fixed-size noising steps. The SDE describes what happens if you shrink each step's size toward zero while adding proportionally more of them — the same noising process, viewed at infinite resolution. Training a network to denoise one discrete step turns out to estimate the same underlying quantity (the score function) that the continuous SDE needs to be reversed. That shared quantity is the bridge between the two views.

## The discrete chain, as a recap

The [[Diffusion Models Root#The forward (noising) process|forward Markov chain]] takes a small step at each $t$:

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0,I)
$$

with $T$ discrete steps, $\beta_1 < \beta_2 < \dots < \beta_T$.

## Taking the continuum limit

Let $\Delta t = 1/T$ and treat $\beta_t \approx \beta(t)\, \Delta t$ for some smooth function $\beta(t)$ — i.e. spread the same total amount of noise over more, smaller steps as $T \to \infty$. Using $\sqrt{1-\beta_t} \approx 1 - \tfrac{1}{2}\beta_t$ for small $\beta_t$:

$$
x_t - x_{t-1} \approx -\tfrac{1}{2}\beta(t)\,\Delta t\, x_{t-1} \;+\; \sqrt{\beta(t)\,\Delta t}\; \epsilon
$$

The second term is exactly a Brownian increment, $\sqrt{\Delta t}\,\epsilon \to dw$. As $\Delta t \to 0$, this becomes the **variance-preserving (VP) SDE**:

$$
dx = -\tfrac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw
$$

This is literally what DDPM's forward chain converges to in the limit of infinitely many, infinitesimally small steps. The discrete chain is an [[Euler–Maruyama discretization|Euler-Maruyama]] approximation of this SDE — not a separate model, the same one sampled coarsely.

> [!note] Other SDE families
> The VP-SDE above corresponds to DDPM. A second family, the **variance-exploding (VE) SDE**, $dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\, dw$, corresponds instead to score-matching models with annealed noise scales (NCSN, Song & Ermon's earlier work) where variance is allowed to grow rather than stay bounded. Both are special cases of the same general framework.

## Reversing the SDE

Anderson (1982) showed that any forward diffusion SDE $dx = f(x,t)\,dt + g(t)\,dw$ has a corresponding **reverse-time SDE** that exactly undoes it:

$$
dx = \big[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\big]\, dt + g(t)\, d\bar w
$$

run backward from $t=T$ to $t=0$, where $d\bar w$ is a reverse-time Wiener process and $p_t(x)$ is the marginal density of $x_t$. The only unknown quantity is $\nabla_x \log p_t(x)$ — the **score function**. Everything else ($f$, $g$) is fixed by the forward process you chose.

This is the continuous-time analogue of the discrete reverse Markov chain $p_\theta(x_{t-1}\mid x_t)$ — same role, same destination, expressed as a differential equation instead of a sequence of conditional Gaussians.

## The equivalence: noise prediction *is* score estimation

This is the crux of why one trained network serves both views. Recall the forward marginal is Gaussian:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\right)
$$

Differentiating its log-density with respect to $x_t$ gives the score in closed form:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

So the noise $\epsilon$ that DDPM trains $\epsilon_\theta(x_t,t)$ to predict is, up to a known scale factor, **the same quantity as the score**:

$$
\nabla_{x_t} \log q(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}
$$

Train the network once on the simple denoising-MSE loss from [[Diffusion Models Root#Training objective|the DDPM objective]], and you've also trained a score estimator — no separate training run, no architecture change, just a reinterpretation of the same output through this rescaling.

## One model, three valid samplers

Because $\epsilon_\theta$ doubles as a score estimator, the trained network can be dropped into any of the following without retraining:

| Sampler                          | What it does                                                                                                                             | Relationship                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **DDPM ancestral sampling**      | Literal discrete reverse Markov chain, $T$ fixed steps                                                                                   | The original training-time discretization                            |
| **Reverse SDE (Euler-Maruyama)** | Numerically integrate the reverse-time SDE with the score plugged in                                                                     | Same stochastic process, any step count/schedule                     |
| **Probability flow ODE**         | Drop the $dw$ term, halve the score coefficient: $dx = \big[f(x,t) - \tfrac{1}{2}g(t)^2\nabla_x\log p_t(x)\big]dt$ — fully deterministic | Continuous-time generalization of [[Why Reverse Diffusion Adds Noise |

The probability flow ODE shares the *same marginal distributions* $p_t(x)$ at every $t$ as the SDE — it just removes the stochastic term, exactly the deterministic-vs-stochastic tradeoff discussed in [[Why Reverse Diffusion Adds Noise]]. Because it's an ODE rather than an SDE, off-the-shelf high-order ODE solvers (Runge-Kutta, DPM-Solver) apply directly, which is why continuous-time samplers can take far fewer steps than the original $T$ used at training time — the network only ever needs to estimate the score/noise at a given $(x,t)$; how finely you step through time afterward is a numerical-integration choice independent of training.

## Why this matters in practice

- **Step count becomes a sampling-time choice, not a training-time constraint.** Train at $T=1000$, sample with 20–50 steps via an SDE/ODE solver, because the network was estimating a continuous-time quantity (the score) all along — the $T=1000$ Markov chain was just one particular discretization.
- **Likelihood computation.** The probability flow ODE permits exact likelihood evaluation via the instantaneous change-of-variables formula (as in continuous normalizing flows) — something the discrete chain alone doesn't hand you as cleanly.
- **Unifies DDPM, score matching, and DDIM** under one framework: they're different points along the discretization/determinism spectrum of the same forward/reverse SDE pair, not unrelated methods that happen to produce similar images.

## Key papers

- Anderson, 1982 — *Reverse-time diffusion equation models* (the reverse-SDE theorem this all rests on)
- Song & Ermon, 2019 — *Generative Modeling by Estimating Gradients of the Data Distribution* (score matching, VE-SDE precursor)
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, 2020 — *Score-Based Generative Modeling through Stochastic Differential Equations* (the unifying SDE framework, probability flow ODE)

## Related

- [[Diffusion Models Root]]
- [[Why Reverse Diffusion Adds Noise]]
- [[Score-Based Generative Models]]
- [[Stochastic Differential Equations]]
