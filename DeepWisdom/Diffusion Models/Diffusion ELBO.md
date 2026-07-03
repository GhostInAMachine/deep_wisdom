---
title: Diffusion ELBO
type: concept
tags: [diffusion, generative-models, variational-inference, training, probability]
created: 2026-06-30
status: budding
aliases: [Variational Lower Bound for Diffusion, Diffusion Variational Bound, DDPM ELBO, Evidence Lower Bound]
---

# Diffusion ELBO

Diffusion models are **likelihood-based** generative models: in principle they maximize $\log p_\theta(x_0)$, the probability the model assigns to real data. That quantity is intractable — it requires integrating over every possible noising trajectory — so [[Diffusion Models Root|DDPM]] is trained by maximizing a tractable **evidence lower bound (ELBO)** instead, exactly as a [[Variational Autoencoder|VAE]] is. This note derives that bound, shows how it decomposes into per-timestep [[Kullback-Leibler Divergence|KL]] terms, and explains why each term is computable — which is where [[Bayes' Theorem in Diffusion Models|the Bayes posterior]] $q(x_{t-1}\mid x_t, x_0)$ earns its keep.

> [!note] Why a *lower bound* at all
> The marginal likelihood $p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$ sums over all latent trajectories $x_{1:T}$ — astronomically many, with no closed form. Variational inference sidesteps this by introducing a tractable distribution $q$ over the latents and bounding $\log p_\theta(x_0)$ from below. The twist in diffusion: $q$ is **not learned** — it's the fixed forward noising process. Only the reverse model $p_\theta$ is trained.

## Notation breakdown

Everything below is built from a small vocabulary. Skim this once and the rest of the note reads as plain English.

### States and indices

| Symbol | Reads as | Meaning |
|---|---|---|
| $t$ | "timestep" | Noise level, $t = 0, 1, \dots, T$. Forward goes $0 \to T$ (add noise); reverse goes $T \to 0$ (denoise). |
| $T$ | "final step" | Total number of steps (e.g. 1000); $x_T$ is (nearly) pure noise. |
| $x_0$ | "x-zero" | A clean **data** sample (the real image). The thing we ultimately model. |
| $x_t$ | "x-t" | The sample at noise level $t$ — a partially noised version of $x_0$. |
| $x_T$ | "x-T" | The fully noised endpoint, $\approx \mathcal{N}(0,I)$. |
| $x_{0:T}$ | "x-zero through T" | The **whole trajectory** $\{x_0, x_1, \dots, x_T\}$ as one object. |
| $x_{1:T}$ | "x-one through T" | All the **latents** — the trajectory excluding the observed $x_0$. |

> [!note] Colon notation: $x_{a:b}$ is a *collection*, not a single variable
> The subscript range $a{:}b$ means "all states from index $a$ to $b$ inclusive." So $x_{1:T}$ is a set of $T$ tensors, and an integral $\int \cdot\, dx_{1:T}$ is a high-dimensional integral over *every* intermediate state at once. This is why the marginal likelihood is intractable.
>
> The same collapsing applies to the differential: $dx_{1:T}$ is shorthand for $dx_1\,dx_2\cdots dx_T$, **not** integration over a single variable called "$x_{1:T}$." So
> $$\int p_\theta(x_{0:T})\,dx_{1:T} = \int\!\!\cdots\!\!\int p_\theta(x_0,x_1,\dots,x_T)\,dx_1\,dx_2\cdots dx_T$$
> is a *multiple* integral over all $T$ latents at once — and since each individual $x_t$ is already a full-sized image, this is an integral over a space $T$ times as large as one image, with $x_0$ held fixed at the observed data point. That's the concrete source of the intractability, and exactly what the ELBO sidesteps by turning it into an expectation instead. See [[Multiple Integrals]] for the general concept. ^colon-notation

### Distributions

| Symbol | Reads as | Meaning |
|---|---|---|
| $q(\cdot)$ | "q" | The **fixed forward process** — the known Gaussian noising chain. No learnable parameters. |
| $p_\theta(\cdot)$ | "p-theta" | The **learned reverse model**, with neural-network parameters $\theta$. The thing we train. |
| $p(x_T)$ | "p of x-T" | The **fixed prior** on the endpoint, set to $\mathcal{N}(0,I)$ (no subscript $\theta$: nothing to learn). |
| $q(x_t\mid x_{t-1})$ | "q of $x_t$ given $x_{t-1}$" | One **forward** step — add a known amount of noise. |
| $p_\theta(x_{t-1}\mid x_t)$ | "p-theta of $x_{t-1}$ given $x_t$" | One **reverse** step the model performs — the denoiser. |
| $p_\theta(x_{0:T})$ | "p-theta of the whole trajectory" | The model's **joint density over an entire path** $x_0,\dots,x_T$ — not just the final data point. Built by chaining reverse steps together; see below. |
| $q(x_{t-1}\mid x_t, x_0)$ | "...given $x_t$ **and** $x_0$" | The **tractable posterior** — true reverse step *given* the clean image; the training target. See [[Bayes' Theorem in Diffusion Models]]. |
| $\mathcal{N}(\mu, \Sigma)$ | "Normal / Gaussian" | A Gaussian with mean $\mu$, covariance $\Sigma$. $I$ is the identity (isotropic noise). |

> [!warning] The bar $\mid$ vs. the comma — and why $\theta$ marks what's learned
> In $q(x_{t-1}\mid x_t, x_0)$, everything **right of the bar** is held fixed (conditioned on); the **comma** just lists two such fixed quantities ($x_t$ *and* $x_0$) — see [[Bayes' Theorem with Multiple Conditions]]. The presence or absence of the subscript $\theta$ is the fastest way to read these formulas: $p_\theta$ and $\mu_\theta$, $\epsilon_\theta$, $\Sigma_\theta$ are **learned**; anything written with $q$, $\beta_t$, $\bar\alpha_t$, or $p(x_T)$ is **fixed and known in advance**.

### Schedule coefficients (all fixed, set by the noise schedule)

| Symbol | Definition | Meaning |
|---|---|---|
| $\beta_t$ | given schedule | Variance of noise **added** at step $t$ (small, increasing). |
| $\alpha_t$ | $1 - \beta_t$ | Fraction of signal **kept** at step $t$. |
| $\bar\alpha_t$ | $\prod_{s=1}^t \alpha_s$ | Cumulative signal kept from $0$ to $t$ — lets you jump to $x_t$ in one shot. |
| $\tilde\beta_t$ | $\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ | Variance of the **true posterior** $q(x_{t-1}\mid x_t,x_0)$ (derived via Bayes). |
| $\sigma_t^2$ | $\approx \tilde\beta_t$ | Variance used in the reverse step at sampling time. |

### Means, noise, and operators

| Symbol | Reads as | Meaning |
|---|---|---|
| $\tilde\mu_t(x_t, x_0)$ | "mu-tilde" | Mean of the **true posterior** — the regression target for the model's mean. |
| $\mu_\theta(x_t, t)$ | "mu-theta" | Mean the **model** predicts for the reverse step. |
| $\Sigma_\theta(x_t, t)$ | "Sigma-theta" | Covariance the model uses (often fixed to $\sigma_t^2 I$, sometimes learned). |
| $\epsilon$ | "epsilon" | The true Gaussian noise injected, $\epsilon \sim \mathcal{N}(0,I)$. |
| $\epsilon_\theta(x_t, t)$ | "epsilon-theta" | The model's **prediction** of that noise — what the network actually outputs. |
| $\mathbb{E}_q[\cdot]$ | "expectation under $q$" | Average over samples drawn from the forward process (and over $x_0 \sim q(x_0)$). |
| $D_{\text{KL}}(a \,\|\, b)$ | "KL divergence" | How far distribution $a$ is from $b$; $\ge 0$, and $=0$ iff $a=b$. See [[Kullback-Leibler Divergence]]. |
| $\mathcal{L}$ | "the ELBO" | The evidence lower bound being maximized; $L_T, L_{t-1}, L_0$ are its component terms. |

## Setup: the two trajectory distributions

The ELBO compares two distributions over the *whole* sequence $x_{0:T}$:

- The **generative (reverse) model** factorizes as a Markov chain from noise back to data:
$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t), \qquad p(x_T) = \mathcal{N}(0, I)
$$
- The **forward (inference) process** is the fixed Gaussian noising chain — the diffusion analogue of a VAE's approximate posterior, but with no parameters to learn:
$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1})
$$

> [!note] What $p_\theta(x_{0:T})$ means
> $p_\theta(x_{0:T})$ is the joint density the model assigns to one **entire specific path** $(x_0, x_1, \dots, x_T)$ — every intermediate noisy state included, not just the clean endpoint $x_0$. It's what you'd get by asking, per [[Why Generative Models Can Assign a Probability to Data|the landing-frequency intuition]]: *if you ran the full reverse sampling chain (start from noise, denoise step by step) infinitely many times, how often would you land on — or near — this exact sequence of states?*
>
> The **factorization** is just the [[Chain Rule of Probability|chain rule of probability]], made cheap by the Markov structure: sampling a path means first drawing $x_T \sim p(x_T)$, then drawing each $x_{t-1}$ conditioned only on $x_t$ (not on the whole history), so the joint density of the path is the product of the prior density and every step's conditional density along the way:
> $$
> p_\theta(x_{0:T}) = p(x_T)\,p_\theta(x_{T-1}\mid x_T)\,p_\theta(x_{T-2}\mid x_{T-1})\cdots p_\theta(x_0\mid x_1)
> $$
> Each factor is a tractable Gaussian density (mean $\mu_\theta$, covariance $\Sigma_\theta$) — so, unlike $p_\theta(x_0)$, $p_\theta(x_{0:T})$ is cheap to *evaluate* for one given path. What's expensive is $p_\theta(x_0)$, which requires **marginalizing** $p_\theta(x_{0:T})$ — summing this joint density over every possible intermediate path $x_{1:T}$ that could have led to the same $x_0$:
> $$
> p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}
> $$
> That's the intractable integral this whole note exists to work around.

## Deriving the bound

Start from the log-likelihood and insert $q$ (multiply and divide), then apply [[Jensen's Inequality|Jensen's inequality]] to pull the log inside the expectation (valid because $\log$ is concave):

$$
\log p_\theta(x_0) = \log \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] \;\ge\; \underbrace{\mathbb{E}_{q}\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]}_{\text{ELBO } \mathcal{L}}
$$

### Where the first equality comes from

The opening "$=$" is the only non-obvious algebra; the rest is Jensen. It is the standard **multiply-and-divide by $q$** trick. Start by writing the marginal likelihood as an integral over the latents, then multiply the integrand by $\frac{q(x_{1:T}\mid x_0)}{q(x_{1:T}\mid x_0)} = 1$:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T} = \int q(x_{1:T}\mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T} = \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]
$$

The last step uses the definition of expectation: $\int q(z)\,f(z)\,dz = \mathbb{E}_{q}[f(z)]$. So the intractable integral is recast as an **average of a ratio**, taken over trajectories sampled from the forward process — and *that* is something we can estimate by sampling, which is the whole point.

### Reading every piece of the bound

| Piece | Name / role | What it is |
|---|---|---|
| $\log p_\theta(x_0)$ | **the target** | Log-probability the model assigns to a real data point — what we wish we could maximize directly, but can't. See [[Why Generative Models Can Assign a Probability to Data]] for what "assigning a probability" means for a generative (not classification) model. |
| $\dfrac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}$ | **importance weight** | Ratio of the model's probability of a *whole trajectory* to the forward process's probability of having produced it. High when the model explains a forward sample well. |
| $p_\theta(x_{0:T})$ | numerator | Generative model's probability of the full path $x_0,\dots,x_T$ — factorizes as $p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t)$. |
| $q(x_{1:T}\mid x_0)$ | denominator / **proposal** | Fixed forward process — the distribution we actually sample trajectories from. It plays the role of an importance-sampling proposal. |
| $\mathbb{E}_{q}[\cdot]$ | **the averaging** | Expectation over forward trajectories $x_{1:T}\sim q(\cdot\mid x_0)$ (and over $x_0\sim q(x_0)$ when training over the dataset). Estimated by Monte Carlo — draw noised samples and average. |
| $\log(\cdot)$ inside vs. outside | **the Jensen move** | Outside the expectation ($\log\mathbb{E}$) the quantity is exact but intractable; pulling it inside ($\mathbb{E}\log$) is what makes it computable, at the cost of turning "$=$" into "$\ge$". |
| $\mathcal{L} = \mathbb{E}_q\!\big[\log\frac{p_\theta}{q}\big]$ | **the ELBO** | The tractable lower bound we actually optimize. No log-of-an-integral remains. |

> [!note] Why divide by $q$ rather than just integrate?
> Two reasons. (1) **Sampling**: an expectation under $q$ can be approximated by drawing forward-process samples — exactly what we can do cheaply — whereas the raw integral cannot. (2) **Jensen needs an expectation**: the inequality $\log\mathbb{E}[Y]\ge\mathbb{E}[\log Y]$ only applies once the marginal is written *as* an expectation. The $q$ in the denominator is what makes $Y = p_\theta/q$ a well-defined random variable to average over. Any $q$ with matching support gives a valid bound; the diffusion choice (the fixed forward process) is what also makes the resulting terms tractable Gaussians.

Training **maximizes $\mathcal{L}$** (equivalently minimizes $-\mathcal{L}$, the *variational upper bound* on the negative log-likelihood).

> [!note] The bound is exactly the likelihood minus a KL gap
> The slack in the inequality is not mysterious:
> $$\log p_\theta(x_0) - \mathcal{L} = D_{\text{KL}}\!\big(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0)\big) \ge 0$$
> The bound is tight exactly when the fixed forward process matches the model's true reverse posterior. This is the same prior/posterior story as Bayes — see [[Bayes' Theorem]] — applied to whole trajectories.

## The key rewrite: where Bayes enters

Expanding $\mathcal{L}$ naively gives a sum involving $q(x_t\mid x_{t-1})$, whose reverse direction is unknown. The decisive step (Sohl-Dickstein et al., 2015; Ho et al., 2020) is to rewrite each forward step **conditioned on $x_0$** and flip it with Bayes' rule:

$$
q(x_t\mid x_{t-1}) = q(x_t\mid x_{t-1}, x_0) = \frac{q(x_{t-1}\mid x_t, x_0)\,q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}
$$

The first equality is the [[Markov Chains and SDEs in Diffusion Models|Markov property]] (knowing $x_{t-1}$ makes $x_0$ redundant for $x_t$); the second is [[Bayes' Theorem with Multiple Conditions|Bayes with multiple conditions]], treating $x_0$ as a fixed background. The term $q(x_{t-1}\mid x_t, x_0)$ it introduces is precisely the **tractable Gaussian posterior** derived in [[Bayes' Theorem in Diffusion Models]]. Substituting it and collecting terms, the bound telescopes into:

$$
-\mathcal{L} = \mathbb{E}_q\Big[\ \underbrace{D_{\text{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T} \;+\; \sum_{t=2}^{T}\underbrace{D_{\text{KL}}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \;\underbrace{-\,\log p_\theta(x_0\mid x_1)}_{L_0}\ \Big]
$$

## Reading the three groups of terms

| Term | Form | Role | Trainable? |
|---|---|---|---|
| $L_T$ | $D_{\text{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)$ | **Prior matching** — does the fully-noised data match $\mathcal{N}(0,I)$? | No (no parameters; $\approx 0$ by schedule design) |
| $L_{t-1}$ | $D_{\text{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)$ | **Denoising matching** — model's reverse step vs. true posterior | Yes — this is the bulk of training |
| $L_0$ | $-\log p_\theta(x_0\mid x_1)$ | **Reconstruction** — final decode from $x_1$ to clean data | Yes |

Every $L_{t-1}$ is a KL between **two Gaussians** — the tractable posterior $q(x_{t-1}\mid x_t,x_0)=\mathcal{N}(\tilde\mu_t,\tilde\beta_t I)$ and the model $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta,\Sigma_\theta)$ — so it has a **closed form**. That tractability is exactly what conditioning on $x_0$ bought us; without it, $q(x_{t-1}\mid x_t)$ is an intractable mixture (see [[Bayes' Theorem in Diffusion Models|Part 2]]).

> [!note] This is why conditioning on $x_0$ is not a hack
> $x_0$ appears only *inside* the expectation $\mathbb{E}_{q}$, which is itself an expectation over $x_0 \sim q(x_0)$. It is used solely to construct the **target** distribution $q(x_{t-1}\mid x_t,x_0)$ that the model $p_\theta(x_{t-1}\mid x_t)$ — which conditions on $x_t$ alone — is trained to match. So borrowing $x_0$ to write the loss is mathematically sanctioned by the ELBO; it legitimately drops out at sampling time. See [[Bayes' Theorem in Diffusion Models#2. The untractable reverse $q(x_{t-1}\mid x_t)$ — why we need $x_0$|the asymmetry discussion]].

## From KL terms to the simple noise loss

With the variance fixed (not learned) and equal between the two Gaussians, the KL between them reduces to a scaled squared distance between their **means**:

$$
L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\,\big\lVert \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\right] + C
$$

Substituting the noise parameterization $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\!\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)$ (and the matching expression for $\tilde\mu_t$), the difference of means collapses to a difference of **noise predictions**:

$$
L_{t-1} = \mathbb{E}_{x_0,\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}\,\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\right]
$$

Ho et al. (2020) found that **dropping the $t$-dependent weight** in front trains better — it up-weights the harder, higher-noise steps. That gives the [[Diffusion Models Root#Training objective|simplified training objective]] $\mathcal{L}_{\text{simple}}$ actually used in practice:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,x_0,\,\epsilon}\!\left[\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\right]
$$

> [!warning] $\mathcal{L}_{\text{simple}}$ is no longer a strict bound on the likelihood
> Reweighting the terms means $\mathcal{L}_{\text{simple}}$ optimizes a *different* objective than the exact ELBO — better samples, but it forfeits the clean likelihood interpretation. Work that reports competitive log-likelihoods (e.g. *Improved DDPM*) reinstates the proper weighting and additionally **learns** $\Sigma_\theta$, since the variance terms the simple loss ignores do matter for likelihood. So there's a genuine tension: the ELBO is the principled training signal, but sample quality favors the reweighted surrogate.

## Connections

- The ELBO is the same object maximized by a [[Variational Autoencoder|VAE]]; diffusion is, in one view, a deep hierarchical VAE whose encoder ($q$) is fixed rather than learned.
- The continuous-time limit of this bound connects to the score-matching objective and the probability-flow ODE — see [[Score-Based Generative Models]] and [[Markov Chains and SDEs in Diffusion Models]].
- The per-step target $q(x_{t-1}\mid x_t,x_0)$ and its variance $\tilde\beta_t$ are derived in [[Bayes' Theorem in Diffusion Models]]; why that variance is reintroduced at sampling time is [[Why Reverse Diffusion Adds Noise]].

## Key papers

- Sohl-Dickstein et al., 2015 — *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (original variational bound for diffusion)
- Ho, Jain, Abbeel, 2020 — *Denoising Diffusion Probabilistic Models* (the KL decomposition and $\mathcal{L}_{\text{simple}}$)
- Nichol & Dhariwal, 2021 — *Improved Denoising Diffusion Probabilistic Models* (learned variance, proper-weighting likelihood)

## Related

- [[Diffusion Models Root]]
- [[Why Generative Models Can Assign a Probability to Data]]
- [[Chain Rule of Probability]]
- [[Multiple Integrals]]
- [[Bayes' Theorem in Diffusion Models]]
- [[Bayes' Theorem with Multiple Conditions]]
- [[Jensen's Inequality]]
- [[Why Reverse Diffusion Adds Noise]]
- [[Variational Autoencoder]]
- [[Score-Based Generative Models]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Kullback-Leibler Divergence]]
