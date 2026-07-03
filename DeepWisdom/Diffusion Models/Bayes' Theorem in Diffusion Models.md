---
title: Bayes' Theorem in Diffusion Models
type: concept
tags: [diffusion, generative-models, probability, bayesian, training]
created: 2026-06-30
status: budding
aliases: [Diffusion Posterior, Tractable Reverse Posterior, "q(x_{t-1}|x_t,x_0)"]
---

# Bayes' Theorem in Diffusion Models

[[Bayes' Theorem]] is not a side remark in diffusion modeling — it is the structural reason the whole framework works. The [[Diffusion Models Root|forward noising process]] is fixed and known in closed form, but the thing we actually want at generation time, the **reverse transition** $q(x_{t-1}\mid x_t)$, is not directly available. Bayes is the bridge that turns the known forward kernel into the reverse direction, and it shows up in three distinct places:

1. **Deriving the training target** — the tractable posterior $q(x_{t-1}\mid x_t, x_0)$ that the network is trained to match.
2. **Justifying that a reverse process exists at all** — the true reverse $q(x_{t-1}\mid x_t)$ as a Bayes posterior over predecessors.
3. **Conditional generation** — classifier and classifier-free guidance, which are Bayes' rule applied to the *score*.

This note works through all three. The recurring theme is the one from [[Bayes' Theorem#The connection to transitions between states|the transition view of Bayes]]: **knowing the forward transition is not enough to reverse it — you also need the prior over predecessor states.**

> [!note] Why "given $x_0$" makes the posterior tractable
> The raw reverse $q(x_{t-1}\mid x_t)$ requires the unknown data prior $q(x_{t-1})$ (see [[Why Reverse Diffusion Adds Noise#The reverse step is "sample the true posterior," not "invert the forward step"|the true-posterior discussion]]). But if you additionally **condition on the clean sample** $x_0$ — which you have during training — the prior collapses into a known Gaussian, and Bayes returns a closed-form Gaussian posterior. That closed form is the supervision signal for the network.

## 1. The tractable posterior $q(x_{t-1}\mid x_t, x_0)$

### Setup

Write $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. Three facts from the [[Diffusion Models Root#The forward (noising) process|forward process]] are all we need, and each is a known Gaussian:

$$
\underbrace{q(x_t\mid x_{t-1})}_{\text{one forward step}} = \mathcal{N}\!\left(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\right)
$$

$$
\underbrace{q(x_{t-1}\mid x_0)}_{\text{closed-form marginal}} = \mathcal{N}\!\left(x_{t-1};\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ (1-\bar\alpha_{t-1})I\right)
$$

$$
\underbrace{q(x_t\mid x_0)}_{\text{closed-form marginal}} = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
$$

> [!note] What "marginal" means here
> The forward process is a chain $x_0 \to x_1 \to \cdots \to x_t$, so the *full* description of how you reach $x_t$ from $x_0$ is the joint over the whole intermediate path, $q(x_1, \dots, x_t \mid x_0)$. The **marginal** $q(x_t\mid x_0)$ is what's left after **integrating out (marginalizing over) every intermediate state** $x_1, \dots, x_{t-1}$:
> $$
> q(x_t\mid x_0) = \int q(x_1, \dots, x_t\mid x_0)\,dx_1 \cdots dx_{t-1}
> $$
> It answers "given the start $x_0$, where is $x_t$ after $t$ steps?" while *staying silent about the route taken* — the in-between states have been summed away. Contrast this with the one-step kernel $q(x_t\mid x_{t-1})$, which conditions on the immediate predecessor and describes a *single* step. The remarkable fact (from the [[Diffusion Models Root#The forward (noising) process|root note]]) is that for Gaussian noising this many-step marginal collapses to one clean Gaussian with the compounded coefficient $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ — so you never actually have to perform that integral or simulate the intermediate steps.

> [!warning] "Marginal" is relative to what stays conditioned
> These are marginals over the *intermediate path*, not over $x_0$ — they are still **conditioned on $x_0$**. The fully unconditional marginal $q(x_t) = \int q(x_t\mid x_0)\,q(x_0)\,dx_0$ additionally integrates out the starting image against the data distribution, and is the intractable object discussed in Part 2. Marginalizing over the path between two known endpoints is easy; marginalizing over the unknown data distribution is not.

### Applying Bayes' rule

We want to reverse one step *given* $x_0$. Bayes' theorem, with everything additionally conditioned on $x_0$ as a held-fixed background (see [[Bayes' Theorem with Multiple Conditions]] for why $x_0$ attaches to every term):

$$
q(x_{t-1}\mid x_t, x_0) = \frac{\overbrace{q(x_t\mid x_{t-1}, x_0)}^{\text{likelihood}}\ \overbrace{q(x_{t-1}\mid x_0)}^{\text{prior}}}{\underbrace{q(x_t\mid x_0)}_{\text{evidence}}}
$$

The likelihood simplifies because the forward chain is **Markov** ([[Markov Chains and SDEs in Diffusion Models|Markov property]]): once you know $x_{t-1}$, the earlier $x_0$ tells you nothing more about $x_t$, so $q(x_t\mid x_{t-1}, x_0) = q(x_t\mid x_{t-1})$. Every term on the right is now one of the three known Gaussians above.

### Completing the square

Substitute the three Gaussian densities and keep only the terms that depend on $x_{t-1}$ (the $x_0$-only and $x_t$-only factors fold into the normalizing constant):

$$
q(x_{t-1}\mid x_t, x_0) \propto \exp\!\left(-\tfrac{1}{2}\left[\frac{(x_t-\sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{1-\bar\alpha_{t-1}}\right]\right)
$$

This is quadratic in $x_{t-1}$, so the posterior is Gaussian. Collect the coefficient of $x_{t-1}^2$ to read off the **precision** (inverse variance):

$$
\frac{1}{\tilde\beta_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
$$

where the last step uses $\alpha_t + \beta_t = 1$ and $\alpha_t\bar\alpha_{t-1} = \bar\alpha_t$. Inverting gives the **posterior variance**:

$$
\boxed{\ \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t\ }
$$

This is exactly the $\tilde\beta_t$ that appears in [[Why Reverse Diffusion Adds Noise]] and as one of the fixed-variance choices in the [[Diffusion Models Root#The reverse (denoising) process|reverse process]]. The **posterior mean** comes from the linear term, $\tilde\mu_t = \tilde\beta_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right)$, which simplifies to:

$$
\boxed{\ \tilde\mu_t(x_t, x_0) = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0\ }
$$

> [!note] Read the mean as a weighted average
> $\tilde\mu_t$ is a convex-ish blend of **where you are** ($x_t$) and **where you're headed** ($x_0$). Early in denoising (large $t$, $\bar\alpha_t \to 0$) the weight leans on $x_t$; near the end (small $t$, $\bar\alpha_t \to 1$) it leans on $x_0$. The posterior doesn't ignore either endpoint — Bayes optimally interpolates them, weighted by how much noise separates them.

### From $x_0$ to a noise prediction

At sampling time we don't have $x_0$. Using the forward reparameterization $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ to solve $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\!\left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\right)$ and substituting into $\tilde\mu_t$, the $x_t$ terms recombine and the mean reduces to a function of $x_t$ and $\epsilon$ alone:

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)
$$

Replacing the true (unknown) $\epsilon$ with the network's prediction $\epsilon_\theta(x_t,t)$ gives **exactly the $\mu_\theta$ formula** in the [[Diffusion Models Root#The reverse (denoising) process|root note]]. So the DDPM sampler's mean is the Bayes posterior mean with the network plugged in for the one quantity it can't observe.

> [!note] This is why the loss is "predict the noise"
> The reverse model $p_\theta(x_{t-1}\mid x_t)$ is trained to match this Bayes posterior $q(x_{t-1}\mid x_t, x_0)$ (minimizing their KL divergence inside the variational bound). Since both are Gaussians with the *same* fixed variance $\tilde\beta_t$, the KL collapses to a squared distance between their means — and after the reparameterization above, that reduces to $\lVert \epsilon - \epsilon_\theta(x_t,t)\rVert^2$, the [[Diffusion Models Root#Training objective|simplified training objective]] $\mathcal{L}_{\text{simple}}$. Bayes is what turns "match a distribution" into "regress the noise."

## 2. The untractable reverse $q(x_{t-1}\mid x_t)$ — why we need $x_0$

Without conditioning on $x_0$, Bayes still applies:

$$
q(x_{t-1}\mid x_t) \propto \underbrace{q(x_t\mid x_{t-1})}_{\text{known forward kernel}}\ \underbrace{q(x_{t-1})}_{\text{unknown data prior}}
$$

The likelihood is free, but the prior $q(x_{t-1})$ is the marginal data distribution at that noise level — the thing we don't know and must **learn**. This is the [[Chess Analogy for Diffusion Models#Bayes' theorem: forward transitions are only half the story|grandmaster argument]]: a predecessor can explain the observed $x_t$ perfectly (high likelihood) yet still be improbable because its prior is tiny. The rest of this section unpacks *why* adding $x_0$ to the conditions is the fix.

### Why the unconditioned prior is intractable

The prior $q(x_{t-1})$ that Part 1's posterior secretly needs is not a simple distribution — it is a **mixture over the entire dataset**. Expand it by marginalizing over which clean image generated this noisy state:

$$
q(x_{t-1}) = \int q(x_{t-1}\mid x_0)\,\underbrace{q(x_0)}_{\text{data distribution}}\,dx_0
$$

Each ingredient $q(x_{t-1}\mid x_0)$ is a tidy Gaussian, but the integral averages them against $q(x_0)$ — the distribution of *all real images*, which is exactly the high-dimensional, unknown object the whole model exists to approximate. There is no closed form, and the result is wildly multimodal (one mode region per plausible source image). So $q(x_{t-1}\mid x_t)$, built from this prior, is itself intractable and non-Gaussian.

### Why fixing $x_0$ collapses it to a Gaussian

Conditioning on a specific $x_0$ **removes the integral** — it picks out a single term of the mixture instead of averaging over all of them:

$$
q(x_{t-1}\mid x_0) = \mathcal{N}\!\left(x_{t-1};\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ (1-\bar\alpha_{t-1})I\right)
$$

This is a known Gaussian for a simple reason: running the [[Diffusion Models Root#The forward (noising) process|forward process]] from a *fixed* starting point is just adding a known amount of Gaussian noise, so the closed-form marginal applies directly. With the prior now Gaussian and the likelihood Gaussian, Bayes returns a Gaussian — the tractable posterior of Part 1.

> [!note] $x_0$ converts "average over all images" into "this one image"
> The unconditioned prior asks *"across the whole dataset, where could this noisy state have come from?"* — an intractable question. Conditioning on $x_0$ asks *"if I already know it came from this exact image, where was it one step ago?"* — a question with a one-line Gaussian answer. $x_0$ is the single piece of information that turns an integral over the data distribution into a point evaluation.

### The asymmetry: $x_0$ exists in training but not in sampling

This is the crux. During **training** we deliberately start from a real example $x_0 \sim q(x_0)$ and noise it forward, so $x_0$ is on hand — we can compute the exact Gaussian posterior and use it as a regression target. During **sampling** we start from pure noise $x_T$ and have no $x_0$; producing one *is the goal*. So $x_0$ cannot be an input to the generator.

The resolution is that the network is trained to need **only $x_t$**: it learns to predict the noise $\epsilon_\theta(x_t,t)$ (equivalently, an estimate of $x_0$), which is exactly the missing quantity. At sampling time we plug that prediction into the posterior mean in place of the true $x_0$. The network has, in effect, absorbed the intractable data prior into its weights — predicting $\epsilon_\theta$ is equivalent to estimating the **score** $\nabla_{x_t}\log q(x_t)$ (see [[Markov Chains and SDEs in Diffusion Models#The equivalence: noise prediction *is* score estimation|the noise-score equivalence]]), the gradient of the log-prior, which is precisely the ingredient Bayes needs to bend the forward kernel backwards.

> [!note] Conditioning on $x_0$ is sanctioned by the ELBO, not a hack
> It can feel like cheating to condition on the very thing we're trying to generate. It isn't. The DDPM training objective is a variational bound (the [[Diffusion ELBO]]) that is an **expectation over $x_0 \sim q(x_0)$** of per-step KL terms $D_{\text{KL}}\!\big(q(x_{t-1}\mid x_t, x_0)\ \|\ p_\theta(x_{t-1}\mid x_t)\big)$. The $x_0$-conditioned posterior appears only *inside* this expectation, as the target distribution; the learned model $p_\theta(x_{t-1}\mid x_t)$ it is matched against conditions on $x_t$ alone. So $x_0$ is a fixed background ([[Bayes' Theorem with Multiple Conditions]]) used to construct a tractable target during training, and it legitimately drops out of the sampler. We borrow $x_0$ to write down the supervision signal, not to generate with it.

## 3. Guidance is Bayes' rule on the score

Conditional generation (e.g. text-to-image) wants to sample from $p(x\mid y)$ for a condition $y$. Take the gradient of the log of Bayes' rule $p(x\mid y) \propto p(x)\,p(y\mid x)$:

$$
\nabla_x \log p(x\mid y) = \underbrace{\nabla_x \log p(x)}_{\text{unconditional score}} + \underbrace{\nabla_x \log p(y\mid x)}_{\text{guidance term}}
$$

The normalizer $p(y)$ has no $x$-dependence, so it vanishes under $\nabla_x$ — the clean reason guidance can be added to the score without recomputing any partition function.

- **Classifier guidance** estimates $\nabla_x \log p(y\mid x)$ with a separately trained, noise-aware classifier and adds it (scaled) to the unconditional score.
- **Classifier-free guidance (CFG)** avoids the extra classifier: it rewrites $\nabla_x\log p(y\mid x) = \nabla_x\log p(x\mid y) - \nabla_x\log p(x)$ (Bayes again, rearranged) and approximates both terms with a single network trained with and without conditioning. This yields the [[Diffusion Models Root#Guidance: controlling generation|CFG combination]] $\hat\epsilon = \epsilon_\theta(x_t,t,\varnothing) + w\big(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\big)$.

Both are the same Bayesian decomposition of a posterior into prior $\times$ likelihood, just expressed in score (gradient-of-log) form.

## Takeaway

Strip diffusion modeling to its skeleton and Bayes' theorem is the load-bearing beam: it derives the closed-form training target by flipping the forward kernel given $x_0$, it explains why a network must be learned (the missing prior in the unconditioned reverse), and it produces guidance by splitting a conditional score into an unconditional score plus a likelihood gradient. The forward process hands you likelihoods for free; **everything that makes a diffusion model generative is the work of supplying the prior that Bayes requires to run the conditional backwards.**

## Related

- [[Diffusion Models Root]]
- [[Bayes' Theorem]]
- [[Diffusion ELBO]]
- [[Why Reverse Diffusion Adds Noise]]
- [[Chess Analogy for Diffusion Models]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Score-Based Generative Models]]
- [[Semicolon Notation in Probability Distributions]]
