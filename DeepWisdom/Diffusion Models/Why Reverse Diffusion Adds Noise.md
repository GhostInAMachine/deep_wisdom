---
title: Why Reverse Diffusion Adds Noise
type: concept
tags: [diffusion, generative-models, sampling, probability]
created: 2026-06-29
status: budding
aliases: [Stochastic Sampling in Diffusion Models, Reverse Process Noise Injection]
---

# Why Reverse Diffusion Adds Noise

At first glance it seems contradictory: the [[Diffusion Models Root|reverse process]] is supposed to *denoise* $x_t$ into something cleaner, yet the standard (DDPM / ancestral) sampling update is

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)
$$

— it predicts a denoised mean $\mu_\theta$ and then immediately mixes fresh random noise $\sigma_t z$ back in. Why reintroduce noise when the whole point is to remove it?

## The reverse step is "sample the true posterior," not "invert the forward step"

Forward diffusion is a lossy, many-to-one mapping: many different clean images $x_0$ can, after enough forward noising, land at nearly the same $x_t$. Information was destroyed along the way, so there is no exact deterministic inverse. The true reverse conditional $q(x_{t-1} \mid x_t)$ is therefore a genuine *distribution* over plausible predecessors — not a single right answer.

Conditioned on the (unknown at sampling time, but derivable in training) $x_0$, Bayes' rule gives a closed form for this posterior:

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu(x_t,x_0),\ \tilde\beta_t I\right), \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

Notice $\tilde\beta_t > 0$ whenever $\beta_t > 0$. The variance isn't an artifact the network introduces — it's a property of the math itself: the forward process really does map a *range* of $x_{t-1}$'s into roughly the same $x_t$, so the correct reverse step has to express that range. The $\sigma_t^2$ used at sampling time approximates this $\tilde\beta_t$.

So "adding noise" during sampling means **drawing from a distribution with the correct variance**, not corrupting an otherwise-clean estimate.

## What if you skip the noise and just take the mean every time?

Tempting simplification: set $x_{t-1} = \mu_\theta(x_t,t)$ deterministically at every step. It doesn't work well.

> [!warning] Mean-only updates regress to blur, not to the data distribution
> $\mu_\theta(x_t,t)$ is (roughly) the *expected value* of $x_{t-1}$, averaged over every plausible $x_0$ consistent with the current $x_t$. Always taking only the mean is like always moving toward the **average** of many possible clean images instead of **committing** to one of them. Chained over hundreds of steps, this systematically over-smooths and collapses diversity — the same failure mode any MSE-trained model exhibits, since squared error is minimized by predicting the average of plausible answers, not a sample of one.

The noise injected at each step is what lets the trajectory commit to one particular path through the data manifold rather than averaging over all of them — and it's why different random seeds $x_T$ produce visibly different, equally valid outputs instead of all collapsing toward the same blurry mean.

## Mean-only updates drift off the distribution the network was trained on

Blur is the visible symptom; the deeper mechanism is a **train/test input mismatch**. The network only ever saw one kind of input during training: $x_t$ formed by the [[Diffusion Models Root#The forward (noising) process|forward process]], i.e. genuine noisy samples drawn from the marginal $q(x_t)$. Its denoising prediction is only trustworthy on inputs that look like those — points that are *typical* under $q(x_t)$. Feed it something atypical and it is extrapolating, not interpolating.

> [!warning] The conditional mean is not a typical sample in high dimensions
> In high dimensions, samples from a distribution don't cluster at its mean — they concentrate on a thin shell some distance away from it (a [[Concentration of Measure|concentration-of-measure]] effect: for an isotropic Gaussian in $d$ dimensions, almost all the mass sits near radius $\sqrt{d}\,\sigma$, while the mean at the center is in a near-empty region). So $\mu_\theta(x_t,t)$, the center of the posterior, is *more probable per point* yet lives where $q(x_{t-1})$ has almost **no mass**. A mean-only step lands $x_{t-1}$ off the typical set — exactly the kind of input the network never trained on.

Now the error compounds. That off-distribution $x_{t-1}$ becomes the *input* to the next reverse step, where the network — never exposed to such points — makes a worse prediction, whose mean lands even further off-manifold, and so on. Each step both pulls toward the (atypical) center and inherits the previous step's drift, so the trajectory marches steadily out of the region where the model is reliable. Over hundreds of steps the accumulated covariate shift is severe; the final $x_0$ is not just blurry but can be an artifacted point the training distribution never contained. (This compounding train/test gap is sometimes called **exposure bias**, by analogy with the same problem in autoregressive models.)

> [!note] Injecting the noise is self-correcting
> Adding $\sigma_t z$ restores precisely the variance that pushes $x_{t-1}$ from the empty center back out onto the typical shell of $q(x_{t-1})$. So each step hands the next one an **in-distribution** input — a point that looks like the noised data the network was trained on — keeping every subsequent prediction inside the model's reliable regime. The stochasticity isn't just for diversity; it's what stops the chain from wandering off the manifold it learned.

## The noise shrinks the trajectory; it doesn't restore it

Crucially, the variance reinjected at each step ($\tilde\beta_t$) is smaller than the variance the mean-shift $\mu_\theta$ removes. The net effect across the trajectory is still monotonically decreasing noise — $\text{Var}(x_t) \to 0$ as $t \to 0$. Each step is a small, *correctly randomized* move toward a plausible point, not a full leap to one deterministic answer. Denoising happens in aggregate over the whole chain; the local stochasticity is what keeps each individual step sampling validly from the true conditional distribution rather than silently collapsing it to its mean.

## You can remove the noise — that's DDIM

The stochastic version isn't mandatory. [[Diffusion Models Root#Sampling|DDIM]] removes the noise term entirely and instead walks a fully deterministic path between $x_T$ and $x_0$, corresponding to the **probability flow ODE** — the noise-free counterpart of the reverse SDE (see [[Score-Based Generative Models]]). DDIM still produces valid samples because it passes through the same marginal distribution at each $t$; it just traces one deterministic path through those marginals instead of randomly sampling a path each run.

> [!warning] Deterministic DDIM is *not* the mean-only update
> Don't conflate this with the off-manifold failure above. Naive mean-only steps drift off the typical set because they ignore the marginals entirely — each step jumps to the empty center of a posterior. DDIM is deterministic *and* stays on-manifold, because the probability flow ODE is constructed precisely to keep $x_t$ distributed as $q(x_t)$ at every $t$. Removing the noise is safe only when the deterministic rule preserves the marginals; mean-only doesn't, DDIM does.

| | Stochastic (DDPM-style) | Deterministic (DDIM) |
|---|---|---|
| Output for fixed $x_T$ | Varies run to run | Always identical |
| Per-step noise injection | Yes, calibrated to $\tilde\beta_t$ | None |
| Step-skipping for speed | Limited | Works well |
| Matches true reverse posterior | Exactly (in the limit) | Approximately (ODE relaxation) |

## Takeaway

The noise reintroduced during sampling isn't undoing the model's denoising work — it's what makes the reverse process a correct **generative sampler** (drawing from a real distribution of plausible images) rather than a **single best guess** (which degenerates into blur). Once you see that the variance is a feature of the true posterior, not an error, it's a deliberate design choice you can dial down (DDIM) once you've understood why it was there in the first place.

## Related

- [[Diffusion Models Root]]
- [[Score-Based Generative Models]]
- [[Stochastic Differential Equations]]
