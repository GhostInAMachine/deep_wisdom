---
title: VITS
type: architecture
tags: [tts, architecture, generative-models, normalizing-flows, vae, gan, speech]
created: 2026-07-03
status: budding
aliases: [VITS, VITS Architecture, Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech]
---

# VITS

**VITS** (Kim, Kong & Son, 2021, ICML) is an end-to-end text-to-speech model that generates a raw waveform directly from a phoneme sequence in a single jointly-trained network. It combines three ideas that are usually studied separately: a conditional [[Variational Autoencoder|VAE]] that links text and audio through a shared latent `z`, a [[Normalizing Flows|normalizing flow]] that makes the text-conditioned prior expressive enough to match a complex audio posterior, and a [[Generative Adversarial Network|GAN]]-based vocoder (borrowed from [[HiFi-GAN]]) for high-fidelity waveform generation. See [[VITS Architecture Diagram]] for the full computation-graph diagram this note walks through.

> [!note] Intuition
> Older TTS systems are two stages trained separately: an acoustic model turns text into a mel-spectrogram, then a vocoder turns that spectrogram into audio — errors in the first stage cascade into the second, and the deterministic mel intermediate throws away information about prosody. VITS instead trains *one* model that goes straight from text to waveform, using a VAE to route information through a rich latent variable so the vocoder isn't limited to reconstructing from a lossy mel bottleneck, and using adversarial training so the output sounds natural rather than merely "close in L1 distance."

## Why end-to-end?

Traditional pipelines (e.g. Tacotron 2 or FastSpeech 2 feeding a separate neural vocoder) train the text-to-spectrogram model and the spectrogram-to-waveform model independently, usually on different objectives. This creates a mismatch: the vocoder is trained on ground-truth spectrograms but run at inference on the acoustic model's *predicted* spectrograms, and the spectrogram intermediate is a lossy, deterministic summary that can't represent the natural one-to-many variability of speech (the same sentence can be spoken with different rhythm, pitch, and emphasis). VITS removes the intermediate representation and the two-stage mismatch by training a single model, with a stochastic duration model to explicitly capture that one-to-many variability.

## The conditional VAE backbone

VITS is framed as a conditional VAE: maximize a lower bound on the log-likelihood of the waveform `x`, conditioned on the text `c`:

$$
\log p_\theta(x \mid c) \ge \mathbb{E}_{q_\phi(z \mid x)}\Big[\log p_\theta(x \mid z) - \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid c)}\Big]
$$

which is the usual VAE decomposition into a **reconstruction term** (how well `z` lets the decoder rebuild the audio) and a **KL term** (how close the audio-derived posterior `q_φ(z|x)` is to the text-derived prior `p_θ(z|c)`). In VITS this shows up as three trained pieces: the posterior encoder, the prior (text encoder + flow), and the decoder.

### Posterior encoder

Non-causal WaveNet-style residual convolutional blocks consume the **linear-scale spectrogram** of the ground-truth waveform and output the mean/variance of the posterior `q_φ(z | x_lin)`. A latent `z` is drawn from it via the reparameterization trick. This module only exists during training — it needs real audio, which isn't available at inference time.

### Prior: text encoder + normalizing flow

A transformer encoder (relative-position multi-head attention + feed-forward blocks, following [[Glow-TTS]]'s design) runs over the phoneme sequence to produce hidden states `h_text`. A linear projection of `h_text` gives the mean and variance of a simple, per-phoneme Gaussian prior over a latent `z_p`.

A plain Gaussian conditioned on text is quite limited, so VITS warps it through a **normalizing flow** `f_θ`: a stack of invertible affine coupling layers (alternated with channel flips, in the style of Glow/WaveGlow). The prior actually used is

$$
p_\theta(z \mid c) = \mathcal{N}\big(f_\theta(z);\ \mu_\theta(c),\ \sigma_\theta(c)\big)\,\left|\det \frac{\partial f_\theta(z)}{\partial z}\right|
$$

Because `f_θ` is invertible, this lets a simple conditional Gaussian be reshaped into a much more complex distribution over `z` while keeping an exact, tractable likelihood — which is what makes it possible to train the whole thing with a KL term at all.

> [!note] What is the flow buying us, given the posterior is also Gaussian?
> The posterior encoder's `q_φ(z|x_lin)` is itself only a diagonal Gaussian, so for a *single* utterance a Gaussian prior could match it exactly — the flow isn't fixing a single-datapoint mismatch. The point is that the prior is conditioned on **text** `c`, not on the specific audio `x`, and one sentence has many valid renditions (the one-to-many nature of speech), each producing a differently-centered Gaussian posterior. The distribution the text prior must cover is therefore the **aggregate** of all those posteriors — a mixture of Gaussians, multimodal and non-Gaussian, even though each component is Gaussian. (For a fixed encoder the ELBO-optimal prior is exactly that aggregate posterior.) A plain Gaussian prior can't represent the mixture; the flow-warped prior can. Equivalently on the generative side: at inference VITS samples the prior and inverts the flow, so without the flow the decoder would only ever be fed Gaussian-distributed latents instead of the richer distribution it was trained on. See [[Normalizing Flows#Flows inside a VAE|flows inside a VAE]] for the general version of this argument.

### Where the determinant term comes from, and how it's computed

The Jacobian determinant is exactly the change-of-variables correction: `f_θ` maps the complicated audio latent `z` toward a space where a plain Gaussian applies, and `|det ∂f_θ(z)/∂z|` accounts for how much `f_θ` locally stretches or compresses volume while doing so. The full change-of-variables derivation, why a naive determinant costs `O(d³)`, and how **affine coupling layers** collapse it to an `O(d)` sum of log-scales `s(x_a)` are covered in [[Normalizing Flows]] — VITS uses exactly that construction (coupling layers alternated with channel flips / invertible 1×1 convolutions, in the style of Glow/WaveGlow).

The one fact the KL derivation below needs is the payoff of that construction: because `f_θ = f_L ∘ ⋯ ∘ f_1` is a composition of coupling layers, its whole-stack log-determinant reduces to a single accumulated sum of every layer's log-scale outputs,

$$
\log\left|\det \frac{\partial f_\theta(z)}{\partial z}\right| = \sum_{l=1}^{L}\sum_i s_l(\cdot)_i
$$

That single number is what appears as `log|det ∂f_θ(z)/∂z|` in the KL derivation below — and, by the identical construction, as the determinant term for the duration flow `g_θ` further down this note. VITS specifically wants **coupling** layers (rather than autoregressive flows like IAF/MAF) because both `f_θ` and `f_θ⁻¹` are a single parallel forward pass, so running the flow forward at training and inverted at inference costs the same either way — see the [[Normalizing Flows#Why the determinant is the bottleneck|coupling-vs-autoregressive comparison in the flows note]].

## Aligning text and audio: Monotonic Alignment Search

The phoneme sequence and the frame sequence of `z` have different lengths, so the KL term needs to know which text prior parameters line up with which latent frames. VITS reuses **Monotonic Alignment Search (MAS)** from [[Glow-TTS]]: a dynamic-programming search (no gradients, no learned parameters) that finds the monotonic, non-skipping alignment `A` between phonemes and latent frames that maximizes the log-likelihood of `z_p` under the (expanded) text prior. This alignment does double duty:
- it tells you how to expand the per-phoneme prior stats `(μ_θ, σ_θ)` to per-frame stats, so the KL loss can be computed frame-by-frame;
- summing how many frames each phoneme was aligned to gives **ground-truth durations**, the supervision target for the duration predictor below.

> [!warning] The expanded prior is piecewise-constant — is that a problem?
> Expanding `(μ_θ, σ_θ)` by `A` means literally *repeating* one phoneme's prior stats — copying the same pair of numbers, unchanged, into every one of the frame slots that phoneme was aligned to ("broadcasting" here is just that: duplication, not any kind of interpolation or smoothing). Real audio isn't piecewise-constant within a phoneme — formants glide, pitch drifts, coarticulation bleeds across boundaries — so naively this looks like a built-in distribution mismatch the KL term has to pay for.
>
> It mostly isn't a problem, because the normalizing flow isn't frame-independent: its coupling-layer networks are dilated convolutions over the *whole* latent sequence (the same WaveNet-style receptive field as the posterior encoder), so `z_p` at a given frame can depend on a window of neighboring frames of `z`. That lets the flow absorb legitimate frame-to-frame acoustic variation into how it transforms `z`, landing close to the flat target in `z_p`-space without forcing `z` — or the decoded audio — to actually be constant. `L_kl` is a regularizer on the latent, not a constraint the decoder has to satisfy directly; `L_recon`/`L_adv`/`L_fm` act on the decoded waveform itself and don't care that the prior was piecewise-constant.
>
> Where it can still bite:
> - **Long spans** (held vowels, silence) ask the flow to explain a lot of intra-phoneme variation against one flat target — `L_kl` becomes a weaker regularizer there, though this shows up as reduced training signal rather than corrupted output.
> - **Phone boundaries** are hard, discrete jumps in the target distribution (MAS gives a monotonic *hard* alignment, not soft attention), while real coarticulation is smooth — a limitation inherited from [[Glow-TTS]]'s alignment design, and a plausible source of minor boundary artifacts independent of anything the flow can fix.
>
> Glow-TTS already used this same broadcast-expansion trick with no flow at all between its prior and the data, and got reasonable results — evidence that broadcasting itself isn't the weak link; it's what motivates giving VITS's flow cross-frame context in the first place, rather than a flaw the flow is merely tolerating.

## Stochastic Duration Predictor

A deterministic duration predictor (as in FastSpeech-style models) collapses the natural one-to-many mapping from text to rhythm into a single point estimate — every synthesis of a sentence comes out with identical timing. VITS instead models a **distribution** over durations using a small flow-based generative model conditioned on `h_text`. It's trained via a variational lower bound against the ground-truth durations extracted from MAS (durations are transformed into a form flows can operate on, since normalizing flows need continuous, dimension-matched inputs). At inference, sampling noise and running the flow in reverse produces a duration for each phoneme — different noise samples give different, still-plausible, rhythms for the same sentence.

## Decoder: HiFi-GAN-style generator

The decoder takes the latent `z` (at the audio-frame rate) and generates the raw waveform directly, using the [[HiFi-GAN]] generator architecture: transposed convolutions progressively upsample to the audio sample rate, interleaved with **multi-receptive-field fusion (MRF)** blocks — residual blocks that sum the outputs of several dilated-convolution stacks with different kernel sizes/dilations, so the model captures periodic structure at multiple timescales. For efficiency, training only decodes random, small windows of `z` at a time ("windowed generator training") rather than the full utterance.

## Adversarial training

Following [[HiFi-GAN]], VITS trains the decoder as a GAN generator against a discriminator (a multi-period discriminator, optionally combined with a multi-scale discriminator) that judges short windows of real vs. generated waveform. This adds:
- an **adversarial loss** (least-squares GAN formulation) pushing the generator to fool the discriminator, and
- a **feature-matching loss**, an L1 penalty between the discriminator's intermediate-layer activations on real vs. generated audio, which stabilizes GAN training and sharpens perceptual detail beyond what a spectrogram-reconstruction loss alone achieves.

## Training objective

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}
$$

- **`L_recon`** — L1 loss between the mel-spectrogram of the generated waveform and the mel-spectrogram of the ground truth (mel, not raw waveform, better matches perceptual similarity).
- **`L_kl`** — [[Kullback-Leibler Divergence|KL divergence]] between the posterior `q_φ(z|x_lin)` and the flow-warped, MAS-aligned prior `p_θ(z_p|c)` (derived in detail below).
- **`L_dur`** — negative variational lower bound of the Stochastic Duration Predictor's likelihood of the MAS-derived ground-truth durations (derived in detail below).
- **`L_adv`, `L_fm`** — the [[Generative Adversarial Network|adversarial]] and feature-matching losses from the discriminator (the discriminator has its own adversarial loss trained alongside).

### How the KL term is computed

In one line: `L_kl` is the (averaged, per-frame) divergence between the audio encoder's posterior over `z` and the text encoder's prior over `z` — after that prior has been reshaped by the normalizing flow and expanded to frame-rate by MAS's alignment — with the audio encoder itself discarded at inference. (That frame-rate expansion broadcasts each phoneme's prior stats across all of its aligned frames — see the callout in the MAS section above for why that piecewise-constant target doesn't end up hurting audio quality.)

This is a case of the general problem of [[Estimating KL Divergence via Sampling|estimating a KL divergence by sampling]] rather than by closed form, since the flow-warped prior below has no analytic integral against the posterior.

Expanding the KL term from the ELBO:

$$
\mathcal{L}_{\text{kl}} = \log q_\phi(z \mid x_{\text{lin}}) - \log p_\theta(z \mid c)
$$

Both log-densities are known in closed form from how the posterior encoder and the prior were defined above, so this can be evaluated *exactly* at a single sampled point — only the outer expectation over `z ~ q_φ` needs a Monte Carlo estimate, and it reuses the very same `z` that reparameterization already drew for the reconstruction term (no extra sampling cost).

**Posterior side.** `q_φ(z|x_lin) = N(z; μ_φ, σ_φ)` is the diagonal Gaussian the posterior encoder outputs directly, so

$$
\log q_\phi(z \mid x_{\text{lin}}) = \sum_i \left[-\log \sigma_{\phi,i} - \tfrac12\log(2\pi) - \frac{(z_i - \mu_{\phi,i})^2}{2\sigma_{\phi,i}^2}\right]
$$

**Prior side.** Recall from the prior section above that

$$
p_\theta(z \mid c) = \mathcal N\big(f_\theta(z);\ \mu_\theta(c,A),\ \sigma_\theta(c,A)\big)\,\left|\det \frac{\partial f_\theta(z)}{\partial z}\right|
$$

where `(μ_θ, σ_θ)` are the per-phoneme prior stats expanded to the frame rate using the MAS alignment `A`. Taking the log turns the Jacobian determinant into an additive correction term:

$$
\log p_\theta(z \mid c) = \sum_i\left[-\log \sigma_{\theta,i} - \tfrac12\log(2\pi) - \frac{(z_{p,i} - \mu_{\theta,i})^2}{2\sigma_{\theta,i}^2}\right] + \log\left|\det \frac{\partial f_\theta(z)}{\partial z}\right|
$$

with `z_p = f_θ(z)` — the same flow-mapped latent used everywhere else in this note.

**Putting it together.** The `-½log(2π)` constants are identical on both sides and cancel, leaving

$$
\mathcal{L}_{\text{kl}} = \sum_i\left[\log\sigma_{\theta,i} - \log\sigma_{\phi,i} + \frac{(z_i-\mu_{\phi,i})^2}{2\sigma_{\phi,i}^2} - \frac{(z_{p,i}-\mu_{\theta,i})^2}{2\sigma_{\theta,i}^2}\right] - \log\left|\det\frac{\partial f_\theta(z)}{\partial z}\right|
$$

With the reparameterization `z = μ_φ + σ_φ ⊙ ε`, `ε ~ N(0, I)`, the posterior term collapses to just `ε_i²/2`, so essentially all of the expression's complexity lives on the prior side — which is exactly where the normalizing flow's expressiveness (and its log-determinant bookkeeping) earns its keep.

> [!warning] Why not the textbook closed-form Gaussian KL?
> If the prior were a plain Gaussian (no flow), `L_kl` would reduce to the standard analytic `KL(N(μ_φ,σ_φ) ‖ N(μ_θ,σ_θ))` formula used in most VAEs — no sampling needed. Once a normalizing flow reshapes the prior, the posterior and prior are no longer both Gaussian *in the same space*, so no closed-form KL integral exists. VITS sidesteps this the same way any VAE with a non-conjugate prior/posterior pair does: evaluate both log-densities at one sampled point and treat that as a single-sample Monte Carlo estimate of the expectation — exactly analogous to how the reconstruction term is estimated.

### How the duration loss is computed

Modeling a per-phoneme duration `d` as a flow-based distribution runs into two obstacles that don't come up when flowing the audio latent `z`:
- **Discreteness** — a real duration is a positive integer count of frames, but a normalizing flow needs a continuous, invertible mapping; there's no useful derivative for a jump between integers.
- **Dimensionality** — `d` is a single scalar per phoneme, but coupling-layer flows (as used elsewhere in VITS) split their input in half at every layer, which needs more than one dimension to do anything.

The Stochastic Duration Predictor handles both with two tricks, wrapped in a *nested* variational lower bound — the same ELBO pattern as the main model, just one level deeper:

$$
\log p_\theta(d \mid h_{\text{text}}) \ge \mathbb{E}_{q_\phi(u,\nu \mid d,\, h_{\text{text}})}\Big[\log p_\theta(d-u,\ \nu \mid h_{\text{text}}) - \log q_\phi(u,\nu \mid d,\, h_{\text{text}})\Big]
$$

and `L_dur` is the negative of that bound.

- **Variational dequantization** replaces the discrete `d` with a continuous surrogate `d - u`, where `u ∈ [0,1)` is a small sampled noise variable — the same trick used to let flow-based models handle discrete pixel values. `d - u` is continuous, so a flow can assign it a smooth density, while still recovering `d` by rounding.
- **Variational data augmentation** introduces a second continuous latent `ν` at the same time resolution as `d` but with extra channels, purely so the flow has enough dimensions to build invertible coupling layers on. `ν` carries no meaning of its own and is discarded at inference — it just gives the flow room to represent a more flexible distribution over `d`.

Both `u` and `ν` are produced by a small flow-based **posterior** network `q_φ`, conditioned on `(d, h_text)`, mapping simple noise into `(u, ν)` via the reparameterization trick (`u` is squashed into `[0,1)`, e.g. with a sigmoid). The **prior** side, `p_θ(d-u, ν | h_text)`, is a second, separate invertible flow `g_θ` conditioned on `h_text`, evaluated with exactly the same change-of-variables formula used for the audio prior:

$$
p_\theta(d-u,\ \nu \mid h_{\text{text}}) = \mathcal N\big(g_\theta^{-1}(d-u,\nu);\ 0,\ I\big)\left|\det\frac{\partial g_\theta^{-1}(d-u,\nu)}{\partial(d-u,\nu)}\right|
$$

So `L_dur` has the same "log-likelihood-minus-entropy" shape as `L_kl` above — except here *both* sides are learned flows rather than one flow and one fixed Gaussian encoder, which is why duration modeling needs its own nested ELBO instead of a single KL term.

**At inference** there's no ground-truth `d` to condition on, so `q_φ` is skipped entirely: sample noise from `N(0, I)` at the same augmented dimensionality, run it through the *inverse* of `g_θ` conditioned on `h_text`, keep the duration channel and discard `ν`, then round up (`ceil`) to get a valid positive-integer frame count `d̂` for each phoneme. Different noise samples give different, equally plausible durations for the same sentence — this is exactly where VITS's variation in rhythm and pacing across repeated synthesis comes from.

> [!warning] Why not just regress duration directly?
> A deterministic predictor (duration via MSE regression from `h_text`, as in FastSpeech) needs only a single forward pass and a simple loss — but it can only ever produce one duration per phoneme, so the same sentence always comes out identically paced. Modeling `p_θ(d|h_text)` as a distribution — however roundabout the dequantization-plus-augmentation machinery needed to make a flow work on a 1-D discrete value — is what buys back speech's natural one-to-many timing variability.

## Training loop, step by step

One training iteration (a single batch) touches essentially every module described above, in this order:

1. **Load a batch.** Sample paired `(phonemes, waveform)` examples. Compute the ground-truth waveform's linear-scale spectrogram `x_lin` (feeds the posterior encoder) and its mel-spectrogram (used later for the reconstruction loss).
2. **Encode the prior (text side).** Run the phonemes through the **text encoder** → `h_text`, then project to per-phoneme prior stats `(μ_θ, σ_θ)`.
3. **Encode the posterior (audio side).** Run `x_lin` through the **posterior encoder** → per-frame posterior stats `(μ_φ, σ_φ)`, then sample `z = μ_φ + σ_φ ⊙ ε` via the reparameterization trick.
4. **Push the sample through the flow.** Apply the **normalizing flow** forward, `z_p = f_θ(z)`, mapping the audio-space sample into the same space as the text-derived prior.
5. **Align text to audio.** Run **Monotonic Alignment Search** between `z_p` and the prior stats to get the alignment `A`. This step is gradient-free — a dynamic-programming search over the current model's own likelihood, treated as fixed for the rest of the step. Use `A` to expand `(μ_θ, σ_θ)` from per-phoneme to per-frame resolution, and to sum per-phoneme frame counts into ground-truth durations `d`.
6. **Compute the KL loss.** With `z`, `z_p`, and the now frame-expanded `(μ_θ, σ_θ)`, evaluate `L_kl` as derived above.
7. **Train the duration predictor.** Feed `(d, h_text)` into the Stochastic Duration Predictor's posterior/prior flows to get `L_dur` (the nested ELBO derived above) — this branch only needs `h_text` and the durations from step 5, not the audio path directly.
8. **Decode a window and reconstruct.** Slice a short, random contiguous window out of `z` (and the matching window of the ground-truth waveform `y`), and run it through the **decoder** to get a generated segment `ŷ`. Compute `L_recon` as the mel-spectrogram L1 distance between `ŷ` and `y`. Decoding only a window, not the full utterance, is what keeps this step memory-tractable.
9. **Run the discriminator.** Feed both `ŷ` and `y` through the **discriminator**. From its output on `ŷ`, compute the generator's adversarial loss `L_adv` and feature-matching loss `L_fm`; from its output on both `ŷ` and `y`, compute the discriminator's own adversarial loss.
10. **Two backward passes.** Update the discriminator on its own loss, using its own optimizer. Separately, update the text encoder, posterior encoder, flow, duration predictor, and decoder jointly in a single backward pass on the generator-side total:
   $$
   \mathcal{L}_G = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}
   $$
   This is the standard alternating-optimizer pattern from GAN training, just with a VAE and a flow riding along inside the generator's update.
11. **Repeat.** Loop over batches/epochs until convergence.

Everything except the posterior encoder, MAS, and the discriminator (steps 3, 5, and 9) also has to run at inference — which is exactly why those three are the ones marked training-only throughout this note.

## Inference procedure

1. Phonemes → **text encoder** → `h_text` and prior stats `(μ_θ, σ_θ)`.
2. `h_text` → **Stochastic Duration Predictor** (sampling noise, running its flow in reverse) → predicted per-phoneme durations `d̂`.
3. Expand `(μ_θ, σ_θ)` along the time axis according to `d̂`, then sample `z_p ~ N(μ_θ, σ_θ)`.
4. Apply the **inverse** normalizing flow, `z_p → z`.
5. `z` → **decoder** → synthesized waveform.

Only four modules run at inference — text encoder, duration predictor, flow, decoder — all reused unchanged from training. The posterior encoder, MAS, and discriminator exist purely to train those four; they never run at inference.

## Multi-speaker extension

The paper also describes a multi-speaker variant: a learned speaker embedding is injected into the flow's coupling layers, the duration predictor, and the decoder, letting a single shared model produce multiple voices.

## Comparison to related TTS designs

| | Two-stage (Tacotron 2 / FastSpeech 2 + vocoder) | [[Glow-TTS]] | VITS |
|---|---|---|---|
| Stages | 2, trained separately | acoustic model (1 stage) + separate vocoder | fully end-to-end, jointly trained |
| Text–audio alignment | attention (Tacotron 2) or external aligner (FastSpeech 2) | Monotonic Alignment Search | Monotonic Alignment Search |
| Duration modeling | implicit (attention) or deterministic regression | deterministic | **stochastic** (flow-based) |
| Output of the "TTS" model | mel-spectrogram (needs a vocoder) | mel-spectrogram (needs a vocoder) | raw waveform, directly |
| Training signal | supervised regression (+ optional adversarial vocoder finetuning) | maximum likelihood (flow) | VAE + flow + adversarial, jointly |

## Key papers

- Kim, J., Kong, J., & Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.* ICML.
- Kim, J., Kim, S., Kong, J., & Yoon, S. (2020). *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search.* NeurIPS. — origin of MAS and the text-encoder/flow prior design VITS builds on.
- Kong, J., Kim, J., & Bae, J. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* NeurIPS. — source of VITS's decoder and discriminator architecture.
- Ren, Y. et al. (2021). *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.* — deterministic duration prediction baseline VITS contrasts with.

## Related

- [[VITS Architecture Diagram]] — canvas diagram of the full training/inference computation graph
- [[Normalizing Flows]]
- [[Variational Autoencoder]]
- [[Generative Adversarial Network]]
- [[Glow-TTS]]
- [[HiFi-GAN]]
- [[Monotonic Alignment Search]]
- [[Kullback-Leibler Divergence]]
- [[Estimating KL Divergence via Sampling]] — the general sampling-based estimator `L_kl` is a specific case of
