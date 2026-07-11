---
title: D3PM - Discrete Denoising Diffusion
type: paper
tags: [diffusion, generative-models, discrete-data, nlp, paper]
created: 2026-07-05
status: seedling
aliases: [D3PM, Discrete Denoising Diffusion Probabilistic Models, Structured Denoising Diffusion, Discrete Diffusion]
authors: [Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg]
year: 2021
venue: NeurIPS 2021
arxiv: https://arxiv.org/abs/2107.03006
---

# D3PM — Structured Denoising Diffusion Models in Discrete State-Spaces

**D3PM** (Austin et al., 2021) generalizes [[Diffusion Models Root|DDPM]] from continuous $\mathbb{R}^d$ to **discrete, categorical data** — tokens, characters, quantized pixels, segmentation labels. It replaces the Gaussian forward kernel with a **Markov transition matrix** over a finite state space, and shows that the entire DDPM machinery (the [[Diffusion ELBO|variational bound]], the [[Bayes' Theorem in Diffusion Models|Bayes posterior]], the closed-form multi-step marginal) carries over unchanged once "add Gaussian noise" is swapped for "apply a stochastic matrix."

The paper's central contribution — and the focus of this note — is that **the choice of transition matrix is a design knob that interpolates between qualitatively different diffusion processes**, with continuous Gaussian diffusion recoverable as one special case. This is what makes D3PM a *bridge*: continuous and discrete diffusion are not separate model families but two points on one axis parameterized by $Q_t$.

> [!note] The one-line idea
> Continuous DDPM corrupts data by a Gaussian kernel $q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)$. D3PM corrupts data by a **row-stochastic matrix**: $q(x_t\mid x_{t-1})=\text{Cat}(x_t;\, x_{t-1}Q_t)$. Everything else — the ELBO, the tractable posterior, sampling by reverse-denoising — is structurally identical.

## The discrete forward process

Represent a single discrete variable with $K$ categories as a **one-hot row vector** $x \in \{0,1\}^K$. The forward step is a categorical distribution whose parameters are obtained by a vector–matrix product:

$$
q(x_t \mid x_{t-1}) = \text{Cat}\!\left(x_t;\ p = x_{t-1} Q_t\right), \qquad [Q_t]_{ij} = q(x_t = j \mid x_{t-1} = i)
$$

$Q_t$ is a $K\times K$ **Markov transition matrix**: row $i$ is the probability distribution over next-states given current state $i$, so each row is non-negative and sums to 1. Multiplying the one-hot $x_{t-1}$ by $Q_t$ simply selects that row.

Because the composition of Markov steps is again a matrix product, the **$t$-step marginal has a closed form** — the discrete analogue of DDPM's $\bar\alpha_t$ shortcut:

$$
q(x_t \mid x_0) = \text{Cat}\!\left(x_t;\ x_0 \bar Q_t\right), \qquad \bar Q_t = Q_1 Q_2 \cdots Q_t
$$

So you can jump to any noise level $t$ with a single matrix multiply — no need to simulate the chain, exactly as the [[Diffusion Models Root#The forward (noising) process|Gaussian closed form]] lets you sample $x_t$ from $x_0$ directly.

## The tractable posterior — Bayes, in matrix form

The training target is the same object as in continuous diffusion: the reverse step *conditioned on the clean data* $x_0$. For categorical variables it is computed by [[Bayes' Theorem in Diffusion Models|Bayes' rule]], but every density is now a discrete probability vector and the computation is elementwise:

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
= \text{Cat}\!\left(x_{t-1};\ \frac{(x_t Q_t^\top)\odot(x_0 \bar Q_{t-1})}{x_0 \bar Q_t x_t^\top}\right)
$$

where $\odot$ is elementwise product and the denominator is a scalar normalizer. This is finite and exact — a $K$-vector, no integrals — which is precisely why the per-step [[Kullback-Leibler Divergence|KL]] terms of the [[Diffusion ELBO|ELBO]] stay tractable. Conditioning on $x_0$ buys tractability here for the same reason it does in the Gaussian case (see [[Bayes' Theorem in Diffusion Models|Part 2]]): the unconditioned reverse $q(x_{t-1}\mid x_t)$ is an intractable mixture over all $x_0$.

## The bridge: transition matrices as a design axis

The forward process is *entirely determined by the family of $Q_t$*. Different structural choices give different diffusion behaviors — and this is where continuous and discrete diffusion meet.

| Transition matrix $Q_t$ | Stationary distribution | Behavior / analogy |
|---|---|---|
| **Uniform** | Uniform over $K$ states | Each token, with prob. $\beta_t$, jumps to a uniformly random category. Doubly stochastic. This is *multinomial diffusion* (Hoogeboom et al., 2021). |
| **Absorbing ([MASK])** | Point mass on `[MASK]` | Each token either stays or is replaced by a special absorbing `[MASK]` token, never to leave. Generalizes **BERT-style masked language modeling**. |
| **Discretized Gaussian** | Roughly uniform | Transition probability falls off with a Gaussian in *category distance* — nearby ordinal states mix first. **The discrete image of continuous Gaussian diffusion.** |
| **Nearest-neighbor / embedding** | data-dependent | Transitions favor states close in a learned embedding space — semantic corruption. |

### Discretized Gaussian → the continuous limit

For **ordinal** data (quantized pixel intensities $0\text{–}255$, audio samples), the categories carry a natural metric: category $127$ is "close to" $128$ and "far from" $3$. The discretized-Gaussian $Q_t$ places transition mass according to that metric:

$$
[Q_t]_{ij} \;\propto\; \exp\!\left(-\frac{(i-j)^2}{2\,\sigma_t^2}\right) \quad (\text{normalized per row, with a stay-probability on the diagonal})
$$

A token is far more likely to hop to an *adjacent* intensity than to a distant one — so the discrete process **locally mimics Gaussian diffusion on the real line**. As the number of categories $K\to\infty$ and the per-step variance $\sigma_t^2$ shrinks appropriately, this categorical chain converges to continuous DDPM. That is the concrete sense in which D3PM *contains* Gaussian diffusion: continuous diffusion is the fine-grained-ordinal limit of the discretized-Gaussian D3PM.

> [!note] Two ways to be "isotropic"
> The **uniform** matrix and the **discretized-Gaussian** matrix are the two natural discrete counterparts of continuous noise. Uniform corruption ignores any ordering of the states (right for *nominal* categories like text tokens); discretized-Gaussian respects an ordering (right for *ordinal* categories like pixels). Continuous DDPM only ever sees the ordinal case — the real line is inherently ordered — so it corresponds to the discretized-Gaussian branch. D3PM's contribution is exposing the *other* branch, corruption that has no continuous analogue at all.

### Absorbing state → masked language models

The absorbing-`[MASK]` choice reveals a bridge in the *opposite* direction — toward discrete-only models that continuous diffusion cannot express. Under it, the forward process progressively masks tokens, and the reverse process learns to unmask them. A single-step version of this is exactly **BERT's masked-token prediction**; D3PM-absorbing is its multi-step, generative generalization, and it later became the template for discrete diffusion language models. This branch has *no* Gaussian counterpart — there is no "absorbing state" in $\mathbb{R}^d$ — which is precisely the expressive range D3PM adds beyond continuous diffusion.

## Training objective

D3PM optimizes the **same variational bound** as [[Diffusion ELBO|DDPM's ELBO]] — the $L_T + \sum_t L_{t-1} + L_0$ decomposition into per-step KLs — with Gaussian KLs replaced by KLs between categorical distributions. The network is given the **$x_0$-parameterization**: it predicts a distribution $\tilde p_\theta(x_0\mid x_t)$ over the clean token, and the reverse step is reconstructed by summing the known posterior against that prediction,

$$
p_\theta(x_{t-1}\mid x_t) \;\propto\; \sum_{x_0} q(x_{t-1}\mid x_t, x_0)\,\tilde p_\theta(x_0\mid x_t)
$$

which conveniently guarantees the reverse process shares the posterior's structure and lands on valid categories. In practice they train with an **auxiliary cross-entropy term** added to the variational loss:

$$
\mathcal{L}_\lambda = \mathcal{L}_{\text{vb}} + \lambda\,\mathbb{E}_{q}\!\left[-\log \tilde p_\theta(x_0\mid x_t)\right]
$$

The second term directly encourages accurate $x_0$ prediction (the discrete echo of DDPM's $\epsilon$-prediction reweighting — see [[Diffusion ELBO#From KL terms to the simple noise loss|why reweighting helps]]), and empirically improves both likelihood and sample quality.

## Why it matters

- **Unification.** Gaussian DDPM, multinomial diffusion, and masked-token models are shown to be one framework differing only in $Q_t$ — the same ELBO, the same Bayes posterior, the same sampling loop.
- **Native discrete generation.** Text, code, molecules (graphs/SMILES), and segmentation maps can be diffused *in their own state space*, without the awkwardness of embedding discrete symbols into $\mathbb{R}^d$ and rounding — a recurring pain point for applying continuous diffusion to language.
- **Foundation for discrete diffusion LMs.** The absorbing-state variant seeded a line of non-autoregressive text and code generators, and connects to the continuous-time Markov chain (CTMC) view developed in later work.

## Key results (as reported)

- Competitive with continuous diffusion on **CIFAR-10** using the discretized-Gaussian $Q_t$ on quantized pixels.
- Strong **character-level and token-level text** generation on `text8` and `LM1B`, with the absorbing (`[MASK]`) transition generally outperforming uniform for language.

## Relation to the rest of the vault

D3PM is best read *after* the continuous story: it is DDPM with the Gaussian swapped for a matrix. If a formula here looks unfamiliar, its Gaussian original is in these notes:

- [[Diffusion Models Root]] — the continuous DDPM this generalizes.
- [[Diffusion ELBO]] — the variational bound D3PM inherits verbatim (only the KLs change from Gaussian to categorical).
- [[Bayes' Theorem in Diffusion Models]] — the posterior trick, whose matrix form appears above.
- [[Kullback-Leibler Divergence]] — the per-step loss; here between categorical rather than Gaussian distributions.
- [[Markov Chains and SDEs in Diffusion Models]] — the continuous-time / SDE view; the discrete-time-Markov-chain analogue is what D3PM's continuous-time limit connects to.

## Key papers

- Austin, Johnson, Ho, Tarlow, van den Berg, 2021 — *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM).
- Hoogeboom et al., 2021 — *Argmax Flows and Multinomial Diffusion* (the uniform-transition special case D3PM subsumes).
- Sohl-Dickstein et al., 2015 — *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (introduced diffusion, including a binomial/discrete variant).
- Ho, Jain, Abbeel, 2020 — *Denoising Diffusion Probabilistic Models* (the continuous framework D3PM mirrors).

## Related

- [[Diffusion Models Root]]
- [[Diffusion ELBO]]
- [[Bayes' Theorem in Diffusion Models]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Kullback-Leibler Divergence]]
- [[Generative Models]]
- [[Variational Autoencoder]]
