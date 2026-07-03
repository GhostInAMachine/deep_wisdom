---
title: Why Generative Models Can Assign a Probability to Data
type: concept
tags: [generative-models, probability, likelihood, density-estimation]
created: 2026-07-01
status: budding
aliases: [Likelihood in Generative Models, Implicit Density of a Generative Model, Likelihood-Based Generative Models]
---

# Why Generative Models Can Assign a Probability to Data

A phrase like "$\log p_\theta(x_0)$, the probability the model assigns to real data" (see [[Diffusion ELBO]]) sounds like it belongs to classification, where a model outputs $p(y\mid x)$ — a distribution over a small set of labels. But diffusion models, [[Variational Autoencoder|VAEs]], normalizing flows, and autoregressive models are **generative**: their job is to *produce* samples, not to *score* inputs. How can such a model "assign a probability" at all?

## The resolution: the model defines a density, not just a sampler

A likelihood-based generative model's parameters $\theta$ don't just describe a sampling procedure — they implicitly pin down a full **probability density function over the entire data space** (see [[Distribution vs Density Function]] for how a density and a distribution are the same information). Sampling and density evaluation are two different operations you can perform on that *same* underlying object:

- **Sample from it**: run the generative recipe forward (e.g. for diffusion: draw $x_T \sim \mathcal{N}(0,I)$, then denoise down to $x_0$) to produce a new point.
- **Evaluate its density at a point**: ask "how much probability mass per unit volume does this distribution place at *this specific* $x_0$?" — even an $x_0$ the model didn't generate, such as a real image from the training set.

$p_\theta(x_0)$ is the second operation. It is not a forward pass through a scoring head; it is the height of an implicit density function, evaluated at one location.

> [!note] Intuition: density as "landing frequency"
> Concretely, $p_\theta(x_0)$ answers: *if you ran the generative process infinitely many times, how "hot" would the region around $x_0$ be — i.e. how often would generated samples land near there, per unit volume?* Formally,
> $$p_\theta(x_0) = \lim_{\epsilon \to 0} \frac{P(\text{a sample from the model falls within }\epsilon\text{ of }x_0)}{\text{Vol}(\epsilon\text{-ball around }x_0)}$$
> By the law of large numbers, the empirical frequency of samples landing in a shrinking neighborhood converges to exactly this — so "how often the generator lands here" isn't just an analogy, it's the definition of density in the infinite-sample, infinitesimal-volume limit. This is also why density values can exceed 1: probability of landing in an exact single point is zero in a continuous space, so density is probability *per unit volume*, and a tight cluster of landings in a tiny volume gives a large frequency-per-volume even though total probability stays $\le 1$.
>
> This reframes maximum likelihood training intuitively: pushing $p_\theta(x_0)$ up for a real data point means "adjust $\theta$ so that, if run forever, the generator would land near this point more often" — training steers the sampler's implicit landing frequency toward the training data. It's also why this can't be checked *by* sampling for a specific point in high dimensions: confirming a point is "hot" this way would take astronomically many samples, which is exactly why the analytic/bound-based routes in the sections below exist instead.

For diffusion specifically, that density is a marginal of the full generative process over trajectories:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}, \qquad p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)
$$

Every possible latent trajectory $x_{1:T}$ that *could have* produced $x_0$ contributes to this density, which is exactly why it's intractable — see the [[Diffusion ELBO|ELBO derivation]] for how training works around that.

## Why this looks nothing like classification

| | Classification: $p(y\mid x)$ | Generative modeling: $p_\theta(x_0)$ |
|---|---|---|
| Space being distributed over | A small, fixed, discrete label set (e.g. 1000 classes) | A huge, continuous space (all possible images) |
| How you get the number | One forward pass, softmax output layer | Marginalize out latent structure the model uses internally to generate |
| Tractability | Exact, cheap | Often intractable — needs a bound (ELBO) or a restricted architecture (flows) |

A classifier's output layer *is* the distribution — there's nothing left to compute. A generative model's distribution is spread over a space too large to have an output layer for; $p_\theta(x_0)$ has to be *derived* from the generative process rather than read off directly.

## Sampling and likelihood are different uses of the same model

At generation time, a diffusion model is only ever run forward (noise $\to$ image) — $p_\theta(x_0)$ is never evaluated. At training time, maximum likelihood asks the model to make $p_\theta(x_0)$ large for real data, which pulls the *distribution the model would generate* toward the *true data distribution*. The likelihood is a training signal about the shape of the model's implicit distribution, not something computed during sampling.

> [!note] Why GANs are different
> A GAN's generator is also a sampler, but it does **not** define a tractable $p_\theta(x)$ at all — there's no way to ask "how much density does this generator place at this exact pixel value." That's precisely why GANs cannot be trained by maximum likelihood the way diffusion models, VAEs, and flows can; they need an adversarial discriminator as a substitute training signal instead.

## Can you actually get this number out of a trained model?

"The model defines a density" doesn't mean that density is cheap to query — even *after* training is done. Whether you can extract $p_\theta(x_0)$, or only an estimate of it, depends on the model family:

| Model family | Can you get $p_\theta(x_0)$? | Why |
|---|---|---|
| [[Normalizing Flows\|Normalizing flow]] | **Yes, exactly** | Built from invertible layers with a tractable Jacobian determinant — one forward pass computes the exact log-density via the change-of-variables formula. No bound needed by construction. |
| [[Autoregressive Models\|Autoregressive model]] | **Yes, exactly** | $p_\theta(x_0) = \prod_i p_\theta(x_0^{(i)}\mid x_0^{(<i)})$ by the chain rule — each factor is a cheap softmax/density, same tractability story as a classifier's output layer. |
| [[Variational Autoencoder\|VAE]] | **No — estimate only, even post-training** | The ELBO computed at test time is still just a *lower bound*, not the exact value. Tighter estimates ([[Importance Weighted Autoencoder\|IWAE]], annealed importance sampling) converge toward the true $p_\theta(x_0)$ as compute increases, but an exact closed form isn't available. |
| Diffusion model | **No exact form by default; an expensive near-exact route exists** | The ELBO (this note's [[Diffusion ELBO\|derivation]]) can be evaluated fully after training — summing all $T$ KL terms instead of sampling one $t$ per step — giving a lower-bound "bits-per-dimension" estimate. Reframing the reverse process in continuous time as the [[Score-Based Generative Models\|probability-flow ODE]] turns it into a continuous normalizing flow, whose instantaneous change-of-variables formula gives an essentially *exact* log-likelihood — at the cost of solving an ODE and estimating a trace (Song et al., 2021). |
| GAN | **No, not even as an estimate** | No tractable density is defined at all — there's nothing to query, exactly or approximately. |

So the honest answer for diffusion models: right after training, the number you cheaply have access to is a *lower bound* (the ELBO), not $p_\theta(x_0)$ itself. Getting close to the true value requires either the expensive full ELBO sum or the separate ODE-based exact-likelihood machinery — it's never a single forward pass, unlike a flow or an autoregressive model.

## Related

- [[Diffusion ELBO]]
- [[Distribution vs Density Function]]
- [[Variational Autoencoder]]
- [[Diffusion Models Root]]
- [[Score-Based Generative Models]]
- [[Normalizing Flows]]
- [[Autoregressive Models]]
- [[Importance Weighted Autoencoder]]
