# Weights as Memory

### A field guide from linear attention to In-Place TTT

> **The one idea this whole doc is about:** every efficient sequence model — linear attention, Mamba, DeltaNet, TTT, LaCT, E2E TTT — maintains a fixed-size memory that is literally a **weight matrix (or small network) being trained online** as the sequence streams by. The models differ only in *what loss*, *what optimizer*, *what forgetting policy*, and *what granularity* they use for that online training — and in **where the fast weights live**. Our work, **In-Place TTT**, takes the last axis to its endpoint: the memory is not a new module at all, but the backbone's **own MLP down-projection**, updated in place.

---

## Table of contents

1. [Why we're here: the KV-cache problem](#1-why-were-here-the-kv-cache-problem)
2. [Linear attention: memory as a matrix](#2-linear-attention-memory-as-a-matrix)
3. [Gating and decay: RetNet and GLA](#3-gating-and-decay-retnet-and-gla)
4. [SSMs and Mamba: a parallel lineage that converged](#4-ssms-and-mamba-a-parallel-lineage-that-converged)
5. [The delta rule: DeltaNet and the bridge to TTT](#5-the-delta-rule-deltanet-and-the-bridge-to-ttt)
6. [The update-rule zoo](#6-the-update-rule-zoo)
7. [TTT: the hidden state is a model](#7-ttt-the-hidden-state-is-a-model)
8. [LaCT: test-time training done right](#8-lact-test-time-training-done-right)
9. [TTT-E2E: the inner loss is the actual LM loss](#9-ttt-e2e-the-inner-loss-is-the-actual-lm-loss)
10. [In-Place TTT: our approach — the backbone's own weights as memory](#10-in-place-ttt-our-approach)
11. [Incorporating fancier update rules into the weights](#11-incorporating-fancier-update-rules-into-the-weights)
12. [The unified view](#12-the-unified-view)
13. [References](#13-references)

**Notation, fixed once.** Throughout, the memory state is a matrix $S_t \in \mathbb{R}^{d_v \times d_k}$ (or the weights $W_t$ of a small network), keys/queries $k_t, q_t \in \mathbb{R}^{d_k}$, values $v_t \in \mathbb{R}^{d_v}$ are columns, updates look like $S_t = S_{t-1} + v_t k_t^\top$, and reads look like $o_t = S_t q_t$. Papers differ on row-vs-column conventions (GLA writes $S_t = S_{t-1} + k_t^\top v_t$, $o_t = q_t S_t$); everything here is the same operator up to transposition.

---

## 1. Why we're here: the KV-cache problem

Softmax attention computes, for each position,

$$o_t = \frac{\sum_{i \le t} \exp(q_t^\top k_i)\, v_i}{\sum_{i \le t} \exp(q_t^\top k_i)}.$$

The "state" it conditions on is the entire set of past keys and values — the **KV cache**. That state grows linearly with context: $O(t)$ memory, $O(t)$ work per new token, $O(T^2)$ total prefill. This is what makes attention nearly lossless at recall, and also what makes 128K–1M contexts painful.

Every model in this doc replaces the growing cache with a **fixed-size state**. The question that organizes the whole field is: *given a fixed budget of state, what is the best rule for writing into it?* The deep observation — which took the field several years to converge on — is that the best way to think about writing into a fixed-size state is as **online learning**: the state is a set of weights, the sequence is a training set, and the update rule is an optimizer.

---

## 2. Linear attention: memory as a matrix

**Paper:** Katharopoulos, Vyas, Pappas, Fleuret, *Transformers are RNNs* (ICML 2020, [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)).

Replace $\exp(q^\top k)$ with a factorizable similarity $\phi(q)^\top \phi(k)$ (they used $\phi = \mathrm{elu}+1$). Then associativity lets you regroup the computation:

$$o_t = \frac{\sum_{i\le t} \phi(q_t)^\top \phi(k_i)\, v_i}{\sum_{i\le t} \phi(q_t)^\top\phi(k_i)} = \frac{\left(\sum_{i\le t} v_i\, \phi(k_i)^\top\right)\phi(q_t)}{\left(\sum_{i\le t}\phi(k_i)\right)^\top \phi(q_t)}.$$

The two sums are shared across all queries, so total cost is $O(T)$ — and they define a recurrence:

$$S_t = S_{t-1} + v_t\, \phi(k_t)^\top, \qquad z_t = z_{t-1} + \phi(k_t), \qquad o_t = \frac{S_t\, \phi(q_t)}{z_t^\top \phi(q_t)}.$$

A Transformer without softmax **is an RNN** whose hidden state is the $d_v \times d_k$ matrix $S_t$. Generation became up to ~4,000× faster in their image-generation benchmarks because there's no cache to re-scan.

### 2.1 The fast-weight interpretation

**Paper:** Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (ICML 2021, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)).

This paper supplied the lens we use for everything else. The update $S_t = S_{t-1} + v_t k_t^\top$ is exactly Schmidhuber's 1991 **Fast Weight Programmer**: a slow network (trained by backprop) emits keys and values that program the weights of a fast network ($S$) via additive Hebbian outer-product writes. The state matrix *is a one-layer linear network* $k \mapsto v$, rewritten on the fly. Reading memory ($S_t q_t$) is a forward pass of that network.

Two consequences fall out immediately:

1. **Capacity.** Only $d_k$ keys can be mutually orthogonal in $\mathbb{R}^{d_k}$. Query with $k_j$ and you retrieve

   $$S\,k_j = v_j + \sum_{i \ne j}(k_i^\top k_j)\, v_i,$$

   where the second term is **retrieval interference** — crosstalk from every stored association whose key isn't orthogonal to yours. Past $d_k$ stored pairs, errors are unavoidable. As Songlin Yang's DeltaNet blog quotes (via Eagleman): *"The enemy of memory is not time; it's other memories."*

2. **The additive write can't overwrite.** If the context says `A → red` and later `A → blue`, a Hebbian memory stores the *sum* of both. There's no mechanism to deallocate. (Schlag et al. proposed the fix — the delta rule — back in 2021; it became scalable in 2024, see §5.)

One more modernization worth knowing: the normalizer $z_t$ turned out to be a source of instability, and the field dropped it. Modern linear attention is **unnormalized** ($\phi = $ identity) with a per-head RMSNorm/GroupNorm on the *output* instead.

---

## 3. Gating and decay: RetNet and GLA

A memory that can't erase eventually drowns. The first fix was **decay**.

**RetNet** ([arXiv:2307.08621](https://arxiv.org/abs/2307.08621)) uses a fixed, data-independent scalar decay per head:

$$S_t = \gamma\, S_{t-1} + v_t k_t^\top, \qquad o_t = S_t q_t,$$

with per-head $\gamma$'s spread across scales. Crucially, it popularized the **three equivalent computational forms** that every model in this family now ships with: a *recurrent form* ($O(1)$/token inference), a *parallel form* ($(QK^\top \odot D)V$ with decay matrix $D_{nm} = \gamma^{n-m}$, for training), and a **chunkwise form** that interpolates — process the sequence in chunks, carry the state across chunks recurrently, handle within-chunk interactions with an attention-like matmul.

**Gated Linear Attention (GLA)** (Yang, Wang, Shen, Panda, Kim, ICML 2024, [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)) made the decay **data-dependent and per-channel**:

$$S_t = S_{t-1}\,\mathrm{Diag}(\alpha_t) + v_t k_t^\top, \qquad \alpha_t = \sigma(\text{low-rank proj of } x_t)^{1/\tau} \in (0,1)^{d_k}.$$

The model decides, per token and per channel, how much old memory to keep. This one template subsumes a surprising amount of the field (Mamba-2, mLSTM, RWKV-6, HGRN-2 are all special cases of $S_t = G_t \odot S_{t-1} + v_t k_t^\top$ for different gate structures $G_t$).

GLA's equally important contribution is **FlashLinearAttention**: the I/O-aware chunkwise algorithm (chunk size $C$, e.g. 64):

$$S_{[i+1]} = S_{[i]} + V_{[i]}^\top K_{[i]} \;(\text{+ decay bookkeeping}), \qquad O_{[i]} = \underbrace{Q_{[i]} S_{[i]}^\top}_{\text{inter-chunk}} + \underbrace{(Q_{[i]}K_{[i]}^\top \odot M)\,V_{[i]}}_{\text{intra-chunk}}.$$

Everything is a tensor-core matmul; cost is $O(\frac{L}{C}(C^2 d + C d^2))$; $C = L$ recovers the parallel form, $C = 1$ the recurrence. **This chunkwise template is the load-bearing systems idea of the entire field** — DeltaNet, Gated DeltaNet, KDA, and (in modified form) every TTT method below all train this way.

---

## 4. SSMs and Mamba: a parallel lineage that converged

State space models arrived at the same place from control theory.

**S4** (Gu, Goel, Ré, ICLR 2022, [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)) starts from a continuous linear system, discretized with step size $\Delta$:

$$h'(t) = A h(t) + B x(t),\; y = C h(t) \quad\xrightarrow{\text{discretize}}\quad h_t = \bar{A} h_{t-1} + \bar{B} x_t,\; y_t = C h_t.$$

(S4 uses the bilinear rule $\bar{A} = (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A)$; Mamba later uses zero-order hold, $\bar{A} = \exp(\Delta A)$.) The $A$ matrix is initialized by **HiPPO** theory so the state optimally summarizes input history onto a polynomial basis. Because the parameters are **time-invariant (LTI)**, unrolling the recurrence is a single long convolution, $y = x * \bar{K}$ with $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \dots)$, computable in $O(L\log L)$ by FFT. S4 was the first model to crack Path-X (length 16,384) on Long Range Arena.

**Mamba** (Gu & Dao, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) diagnosed LTI as the limitation: constant dynamics cannot *select* what to remember based on content. Mamba makes $\Delta_t, B_t, C_t$ **functions of the input** ("selective SSM"). The cost: no more fixed convolution kernel — training now requires a **hardware-aware parallel scan** executed in SRAM with kernel fusion and backward-pass recomputation. The payoff: Mamba's selectivity theorem shows the $\Delta_t$ mechanism *is* a generalized gate ($\Delta_t \to \infty$ resets state and latches the current input; $\Delta_t \to 0$ ignores it); induction heads trained at length 256 extrapolate to 1M tokens; Mamba-1.4B beat Pythia-1.4B by ~4.5 points average zero-shot with 5× generation throughput.

**Mamba-2 / SSD** (Dao & Gu, ICML 2024, [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)) is where the two lineages formally merged. Restrict the state transition to a scalar times identity, $A_t = a_t I$, and the SSM recurrence becomes — written in our notation —

$$S_t = a_t\, S_{t-1} + v_t k_t^\top \qquad (B_t \leftrightarrow k,\; C_t \leftrightarrow q,\; x_t \leftrightarrow v),$$

which is **exactly linear attention with a data-dependent scalar decay** — GLA with a scalar gate, RetNet with a learned $\gamma_t$. That's *State Space Duality*: the same model has a recurrent/SSM form and a masked-attention form ($(QK^\top \odot L)V$ where $L$ is the 1-semiseparable decay mask), and the chunkwise algorithm in between is matmul-rich — 2–8× faster than Mamba-1's scan, enabling 8–16× larger states.

> **Reading order for the whole field, compressed:** Mamba = gated linear attention reached via control theory; Mamba-2 = the admission, plus better kernels. From here on, "SSM vs linear attention" is a distinction without a difference; what matters is the **update rule**.

---

## 5. The delta rule: DeltaNet and the bridge to TTT

**The single most important conceptual step in this doc.** Sources: Songlin Yang's three-part DeltaNet blog ([part 1](https://sustcsonglin.github.io/blog/2024/deltanet-1/), [part 2](https://sustcsonglin.github.io/blog/2024/deltanet-2/), [part 3](https://sustcsonglin.github.io/blog/2024/deltanet-3/)) and Yang, Wang, Zhang, Shen, Kim, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)).

### 5.1 The update

Gates erase *indiscriminately* — old decay applies to every association whether stale or precious. The delta rule erases *surgically*. Before writing, **read what the memory currently predicts for this key**, and write only the correction:

$$v_t^{\text{old}} = S_{t-1} k_t, \qquad S_t = S_{t-1} + \beta_t\,(v_t - v_t^{\text{old}})\,k_t^\top,$$

equivalently

$$\boxed{\;S_t = S_{t-1}\big(I - \beta_t\, k_t k_t^\top\big) + \beta_t\, v_t k_t^\top\;}$$

with learned write strength $\beta_t = \sigma(W_\beta x_t) \in (0,1)$. If the context says `A → red` then `A → blue`, the delta rule retrieves `red`, subtracts it, writes `blue`. Associative recall stops degrading: DeltaNet hits 100% on the hardest MQAR setting where additive linear attention collapses.

### 5.2 The TTT bridge: the delta rule *is* online gradient descent

Define a per-token memory loss — "how badly does my memory currently map $k_t$ to $v_t$":

$$\mathcal{L}_t(S) = \tfrac{1}{2}\,\|S k_t - v_t\|^2.$$

One SGD step at $S_{t-1}$ with learning rate $\beta_t$:

$$S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t(S_{t-1}) = S_{t-1} - \beta_t (S_{t-1}k_t - v_t)k_t^\top$$

— **exactly the delta rule.** And vanilla linear attention is one gradient step on the *linear* loss $\mathcal{L}_t(S) = -\langle S k_t, v_t\rangle$. So:

> Linear attention = online GD on an inner-product memory loss.
> DeltaNet = online GD on a squared-error memory loss.
> TTT (§7) = online GD on a *meta-learned* loss, with an arbitrary inner network.

Once you see this, the entire design space opens up: choose a loss, choose an optimizer, choose a retention policy, choose a memory parameterization — every cell of that grid is a (potential) paper. §6 is that grid filled in.

### 5.3 Making it trainable: WY representation + chunkwise form

The catch: the delta rule's transition matrix $(I - \beta_t k_t k_t^\top)$ is **not diagonal** (it's a generalized Householder matrix), so the cumsum/scan tricks of GLA/Mamba don't apply, and naively composing the transition matrices densifies — $O(L \log L\, d^3)$ work and $O(L d^2)$ memory. This is why the delta rule sat unused at scale from 2021 to 2024.

The fix imports 1980s numerical linear algebra: products of Householder-like matrices admit a compact **WY representation** (Bischof & Van Loan, 1985/87) — they stay *identity-minus-low-rank*:

$$\prod_{i=1}^{t}(I - \beta_i k_i k_i^\top) = I - \sum_{i=1}^{t} w_i k_i^\top, \qquad S_t = \sum_{i=1}^t u_i k_i^\top,$$

with cheap vector recurrences for $w_i, u_i$; the **UT transform** (Joffrain et al., 2006) then batches those recurrences into one small triangular solve per chunk, so everything becomes tensor-core matmuls in the standard chunkwise template:

$$S_{[i+1]} = S_{[i]} + (U_{[i]} - W_{[i]}S_{[i]}^\top)^\top K_{[i]}, \qquad O_{[i]} = Q_{[i]}S_{[i]}^\top + (Q_{[i]}K_{[i]}^\top \odot M)(U_{[i]} - W_{[i]}S_{[i]}^\top).$$

Result: DeltaNet trains only slightly slower than GLA. At 1.3B/100B tokens (SlimPajama, Mistral tokenizer): pure DeltaNet 16.87 WikiText ppl vs Transformer++ 16.85, Mamba 17.06, GLA 17.22 — and with 2 global-attention layers, 16.55, beating the Transformer.

### 5.4 Gated DeltaNet: decay + delta are complementary

**Paper:** Yang, Kautz, Hatamizadeh, *Gated Delta Networks* (ICLR 2025, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)).

$$\boxed{\;S_t = S_{t-1}\,\alpha_t\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top\;}$$

$\alpha_t \in (0,1)$ is data-dependent global decay ("weight decay on the fast weights"); $\beta_t$ is the targeted correction. $\beta_t \to 0$ recovers Mamba-2; $\alpha_t \to 1$ recovers DeltaNet. Decay erases stale memory wholesale; the delta rule replaces individual associations — you want both. At 1.3B/100B tokens (FineWeb-Edu, Llama-2 tokenizer — *not comparable to the SlimPajama numbers above*): Gated DeltaNet 16.42 WikiText ppl vs Mamba-2 16.56, DeltaNet 17.71. The parallelization is the same WY machinery, extended with cumulative decay products.

---

## 6. The update-rule zoo

The "fancy update rules" of 2024–2025, each in one breath. Every one of these is an online optimizer of an associative-memory objective; the table at the end of this section is arguably the most useful artifact in the doc.

**RWKV-7 "Goose"** ([arXiv:2503.14456](https://arxiv.org/abs/2503.14456)) — the *generalized* delta rule: per-channel **vector** decay $w_t$, per-channel **vector** in-context learning rate $a_t$, and an erase-key $\hat\kappa_t$ **decoupled** from the write-key:

$$S_t = S_{t-1}\big(\mathrm{diag}(w_t) - \hat\kappa_t (a_t \odot \hat\kappa_t)^\top\big) + v_t \tilde{k}_t^\top.$$

Which channels learn, which forget, and at what rate are all independent learned signals. Its transitions admit negative eigenvalues, pushing expressivity beyond $\mathsf{TC}^0$ (it can track $S_5$ group state in one layer — something diagonal-transition models provably cannot).

**Titans** (Behrouz, Zhong, Mirrokni, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) — make the memory a **deep MLP** and the optimizer **SGD with momentum + weight decay**:

$$M_t = (1-\alpha_t)M_{t-1} + S_t, \qquad S_t = \eta_t S_{t-1} - \theta_t \nabla\ell(M_{t-1}; x_t), \qquad \ell = \|M(k_t) - v_t\|^2.$$

The gradient is read as "momentary surprise," momentum as decaying past surprise. Strong long-context results (RULER NIAH 80.2% at 16K where Mamba-2 drops to ~0).

**Longhorn** (Liu et al., [arXiv:2407.14207](https://arxiv.org/abs/2407.14207)) — don't take a gradient *step*, take the **closed-form solution** of the proximal/implicit online-learning problem:

$$S_t = \arg\min_S \|S - S_{t-1}\|^2 + \beta_t\|S k_t - v_t\|^2 \;\Rightarrow\; S_t = S_{t-1}(I - \Delta_t k_t k_t^\top) + \Delta_t v_t k_t^\top,\;\; \Delta_t = \frac{\beta_t}{1+\beta_t k_t^\top k_t}.$$

A delta rule whose step size is automatically stabilized — no explicit forget gate needed. Philosophy: *declare the per-step objective and let its solution be the recurrence.*

**Mesa layer / MesaNet** (von Oswald et al., [arXiv:2506.05233](https://arxiv.org/abs/2506.05233)) — stop approximating: solve the full ridge-regression memory problem **exactly at every step**. Keep accumulators $H_t = \gamma_t H_{t-1} + \beta_t k_t k_t^\top$ and $G_t = \gamma_t G_{t-1} + \beta_t v_t k_t^\top$, output $o_t = G_t\,\mathrm{linsolve}(H_t + \Lambda, q_t)$ via conjugate gradient at test time. Locally *optimal* test-time training, paying inference compute for it; each CG iteration reduces to GLA-form chunkwise matmuls.

**DeltaProduct** (Siems et al., [arXiv:2502.10297](https://arxiv.org/abs/2502.10297)) — take $n_h$ delta micro-steps per token, making the transition a product of $n_h$ Householders: interpolates between diagonal ($n_h{=}0$) and dense orthogonal transitions, buying state-tracking expressivity (solves $S_4$ word problems in one layer at $n_h=2$) for ~linear extra cost via DeltaNet's own machinery.

**Atlas** (Behrouz et al., [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)) — the **Omega rule**: optimize the memory against a *sliding window* of the last $c$ tokens (not just the current one), with a Muon-style orthogonalized update and polynomial/exponential feature maps on keys to raise capacity:

$$M_t = \alpha_t M_{t-1} - \eta_t\,\mathrm{NewtonSchulz}_k(S_t), \qquad S_t = \theta_t S_{t-1} + \nabla\!\!\sum_{i=t-c+1}^{t}\!\gamma_i^{(t)}\|M(\phi(k_i)) - v_i\|^2.$$

"Memorize the context, not the token." At 1.3B/100B FineWeb tokens: Atlas 14.97 / Atlas++ 14.40 WikiText ppl vs 18.53 Transformer++ (baseline carried from prior work); >80% BABILong accuracy at 10M context.

**Kimi Delta Attention (Kimi Linear)** (Kimi Team, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) — Gated DeltaNet with **per-channel diagonal decay** at delta-rule cost:

$$S_t = S_{t-1}\,\mathrm{Diag}(\alpha_t)\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,$$

a constrained-DPLR transition engineered for ~2× operator speedup over the general form. Deployed at 48B-param MoE scale (3:1 KDA:MLA hybrid, 5.7T tokens): beats the full-attention baseline across tasks, 75% KV-cache reduction, up to 6× decode throughput at 1M context.

### The grid

| Rule | Inner loss | Optimizer step | Retention | Memory | Parallelization |
|---|---|---|---|---|---|
| Linear attention | $-\langle Sk_t, v_t\rangle$ | GD, lr 1 | none | matrix | chunkwise (trivial) |
| RetNet | same | GD | fixed scalar $\gamma$ | matrix | chunkwise |
| GLA | same | GD | learned diagonal $\alpha_t$ | matrix | chunkwise + log-space gates |
| Mamba-2 | same | GD | learned scalar $a_t$ | matrix | chunkwise (SSD) |
| DeltaNet | $\tfrac12\|Sk_t - v_t\|^2$ | GD, lr $\beta_t$ | none | matrix | WY + UT transform |
| Gated DeltaNet | same | GD, lr $\beta_t$ | scalar decay $\alpha_t$ | matrix | decay-extended WY |
| RWKV-7 | same (decoupled erase key) | GD, **vector lr** | **vector decay** | matrix | chunkwise matrix products |
| Longhorn | proximal objective | **closed-form implicit** | implicit | matrix (diag approx) | parallel scan |
| Titans | $\|M(k_t)-v_t\|^2$ | GD + **momentum** | weight decay $\alpha_t$ | **deep MLP** | chunked mini-batch GD |
| MesaNet | full ridge regression | **exact solve** (CG) | gates on statistics | matrix | CG over GLA kernels |
| DeltaProduct | per micro-step | $n_h$ GD steps/token | optional | matrix | WY on $n_h L$ virtual tokens |
| Atlas | **windowed** regression | **Muon** + momentum | decay + window weights | deep MLP + feature maps | sliding-window-masked chunks |
| KDA / Kimi | $\tfrac12\|S k_t - v_t\|^2$ | GD, lr $\beta_t$ | **per-channel diag** | matrix | constrained-DPLR WY |
| TTT-Linear/MLP (§7) | meta-learned reconstruction | GD, learned lr $\eta(x_t)$ | (LN + residual) | linear / MLP | mini-batch + dual form |
| LaCT (§8) | $-f_W(k)^\top v$ | GD or **Muon**, per-chunk | weight L2-normalize | SwiGLU MLP | large chunks (2K–1M) |
| TTT-E2E (§9) | **actual NTP cross-entropy** | SGD, $\eta{=}1$, clipped | none (meta-learned $W_0$) | the model's own MLPs | chunks of 1K + frozen prefix |
| **In-Place TTT (§10, ours)** | $-\langle ZW^\top, \hat V\rangle_F$, **LM-aligned target** | GD (closed-form additive) | reset at doc boundary (+ clip) | **the backbone's own $W_{\text{down}}$** | **prefix-sum scan, CP-native** |

---

## 7. TTT: the hidden state is a model

**Paper:** Yu Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* ([arXiv:2407.04620](https://arxiv.org/abs/2407.04620)).

This is the paper that named the paradigm. Take the fast-weight view literally and generalize it: the hidden state is the weights $W$ of **any** inner model $f$, and the update rule is **a step of self-supervised learning on the current token** — even at test time, hence *Test-Time Training layers*. Each sequence is its own dataset.

**Inner loop.** Three learnable low-rank projections deliberately mirror K/V/Q:

$$\ell(W; x_t) = \big\|\, f(\theta_K x_t;\, W) - \theta_V x_t \,\big\|^2 \qquad \text{(reconstruct the label view from the training view)}$$

$$W_t = W_{t-1} - \eta(x_t)\, \nabla \ell(W_{t-1}; x_t), \qquad \eta(x_t) = \eta_{\text{base}}\,\sigma(\theta_{lr}\cdot x_t)$$

$$z_t = f(\theta_Q x_t;\, W_t) \qquad \text{(output rule: read memory with the test view)}.$$

**Outer loop.** Everything that *parameterizes* the inner loop — $\theta_K, \theta_V, \theta_Q$, the input-dependent learning rate, the initialization $W_0$, and a LayerNorm + residual wrapped around $f$ — is trained by ordinary next-token prediction, differentiating **through** the inner gradient step (gradients-of-gradients, standard meta-learning). So the self-supervised task isn't hand-designed; it's *learned* to be whatever most helps language modeling. This inner/outer structure is the conceptual core of everything from here down.

**Two instantiations.**
- **TTT-Linear:** $f$ is linear. The paper's Theorem 1: with batch GD (all gradients at $W_0 = 0$, $\eta = 1/2$), this *is* unnormalized linear attention. And with online GD at batch size 1, no LN/residual, it *is* DeltaNet — the equivalence of §5.2, acknowledged in both papers' related work.
- **TTT-MLP:** $f$ is a 2-layer MLP (4× hidden, GELU) — a *nonlinear* memory that no fixed kernelized attention can express. More capacity at long context, harder on memory I/O.

**Making it fast: mini-batch TTT + dual form.** Online GD is sequential in $t$. Fix one: group tokens into inner mini-batches of $b{=}16$ and take all gradients at the batch-start weights — parallel within a batch, sequential across. Fix two: the **dual form** never materializes per-token $W_t$'s; per mini-batch it computes only the batch-end weights and outputs directly via matmuls with a causal mask on the $b \times b$ Gram matrix — structurally the same trick as chunkwise linear attention ("more than 5× faster" in their JAX implementation). The choice $b{=}16$ is an explicit expressivity↔parallelism trade: the ablation chain at 125M on Pile reads linear attention 15.91 → + LN/residual 14.05 → **+ mini-batch TTT 12.35** → + learnable $\eta$ 11.99 → + Mamba backbone 11.09. Mini-batching is the single biggest win.

**Results worth carrying around.** At 1.3B on Books at 32K context: TTT-Linear 8.82 ppl, TTT-MLP 8.81, **Mamba 8.93 — and Mamba's perplexity bottoms out at 16K then gets *worse*, while TTT keeps improving with more context.** Decode latency: TTT-Linear ≈ Mamba (≈0.035 ms/token, context-independent); TTT-MLP ≈ 6.4× slower due to memory I/O — its honest open problem.

---

## 8. LaCT: test-time training done right

**Paper:** Tianyuan Zhang, Sai Bi, …, Songlin Yang, …, Hao Tan, *Test-Time Training Done Right* ([arXiv:2505.23884](https://arxiv.org/abs/2505.23884)).

LaCT's critique of §7 is a systems critique: inner mini-batches of 16–64 tokens make the update kernel memory-bound — **often below 5% FLOPs utilization** — and require heroic custom kernels, which in turn make big nonlinear states and fancy inner optimizers impractical. Prior TTT states topped out at ~0.1–5% of model parameters.

**The fix: make the chunk *enormous*.** Large-Chunk TTT splits the sequence into chunks of **2K up to 1M tokens**, computes one fast-weight update from the *whole chunk*, applies it, moves on:

$$g = \nabla_W \sum_{i=1}^{b} \eta_i\, \mathcal{L}(f_W(k_i), v_i), \qquad W \leftarrow \mathrm{L2\text{-}Normalize}\big(W - \mathrm{Muon}(g)\big), \qquad o_i = f_W(q_i),$$

with a SwiGLU-MLP fast-weight network $f_W(x) = W_2[\mathrm{SiLU}(W_1 x) \circ (W_3 x)]$, a negative-dot-product inner loss $\mathcal{L} = -f_W(k_i)^\top v_i$, per-token learned learning rates $\eta_i = \mathrm{softplus}(w_\eta^\top x_i + b_\eta)$, and optionally **Muon** (Newton–Schulz orthogonalized momentum) as the inner optimizer. Everything is a handful of big matmuls: pure PyTorch, ~70% GPU utilization, and fast-weight states up to **~40% of model parameters**.

**The accepted trade-off, and the hybrid.** Within a chunk, tokens are an unordered set — they don't see each other through the fast weights (LaCT's LM variant is *apply-then-update*: each chunk is processed with only previous chunks' weights, preserving causality). Local, ordered, fine-grained structure is delegated to a **window attention layer** that shares QKV with the TTT layer. This division of labor — *fast weights for global context, attention for local precision* — is the architecture pattern that both TTT-E2E and our In-Place TTT inherit.

**Results.** Novel view synthesis up to 128 views ≈ 1M tokens — at 48 input views (512²), prefill is 1.4s vs 16.1s for full attention; AR video diffusion on 14B Wan with 56K-token sequences matching full attention; LMs at 760M/3B on 32K context beating GLA+SWA and DeltaNet+SWA per-position loss, with Muon > momentum > plain GD throughout.

> Note the contrast with §5: DeltaNet's chunkwise form is an *exact* reformulation of a per-token recurrence; LaCT's large chunk *changes the model* — coarser update granularity is the price of hardware saturation. In-Place TTT (§10) gets large chunks a third way: by keeping attention around so the fast weights never have to do fine-grained token mixing at all.

---

## 9. TTT-E2E: the inner loss is the actual LM loss

**Paper:** Tandon, Dalal, Li, Koceja, Rød, …, Yu Sun, *End-to-End Test-Time Training for Long Context* ([arXiv:2512.23675](https://arxiv.org/abs/2512.23675)). (Local copy: `e2e-ttt-paper/`.)

Every method so far trains its memory on a **proxy loss** — reconstruct $\theta_V x$ from $\theta_K x$, bind keys to values. TTT-E2E asks: why not train it on the thing we actually care about? Reframe long context as **continual learning, not architecture design**: keep a standard Transformer (with sliding-window attention), and at test time keep training it — by ordinary **next-token prediction on the context being read**:

$$\ell_t(W) = \mathrm{CE}\big(f(x_{t-1}; W),\, x_t\big), \qquad W_i = W_{i-1} - \eta\,\frac{1}{b}\!\!\sum_{t=(i-1)b+1}^{ib}\!\!\nabla \ell_t(W_{i-1}).$$

"End-to-end" in two senses:

1. **At test time** — the inner objective is the end-of-network CE loss, not a per-layer reconstruction. In their derivation-from-prior-work ablation, swapping the key-value-binding reconstruction loss for the NTP loss is the step that matters (760M, DCLM: SWA 2.827 → TTT-KVB 2.818 → TTT-E2E 2.805).
2. **At training time** — the outer loop directly optimizes the post-TTT loss via meta-learning (gradients-of-gradients), learning an initialization $W_0$ *meant to be fine-tuned*. The contrast baseline, "TTT-naive," is classic dynamic evaluation: train a static model, bolt on test-time updates afterwards — it barely helps. The meta-learning is the whole game.

**What gets updated, exactly:** only MLP layers, only in the **last 1/4 of blocks** (a storage-vs-backprop-cost trade-off), with a second *frozen* MLP per TTT'd block as "safe storage" against forgetting pre-trained knowledge. Inner mini-batch $b{=}1$K, SWA window $k{=}8$K with $k \ge b$ so attention covers each chunk before TTT updates on it. A detail the paper doesn't state but the released code reveals: the inner optimizer is plain SGD with **η = 1.0** and global-norm clipping at 1.0, no momentum, nothing learned per-token — because $W_0$ is meta-learned *through* the update rule, the learning rate is effectively absorbed into the initialization.

**Results.** At 3B/164B tokens, TTT-E2E **scales with context length the same way full attention does** out to 128K — while SWA, Mamba-2, Gated DeltaNet, and TTT-KVB all degrade — with constant inference latency, 2.7× faster than full attention at 128K. It even improves on *top of* full attention by 0.018 loss at 8K, so the gain is orthogonal to attention span. Honest negatives: NIAH recall collapses (0.06 vs full attention's 0.99 at 128K — compression discards needle-like detail by design), and training is 3.4× slower than full attention at 8K (gradients-of-gradients; no FlashAttention double-backward).

---

## 10. In-Place TTT: our approach

**Paper:** Feng, Luo, Hua, Zhang, He, Huang, Cai, *In-Place Test-Time Training* (**ICLR 2026 Oral**, [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)). Code: [ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT). (Local: `paper/`, `In-Place-TTT/`.)

Every method above adds or substitutes a module: TTT-Linear/MLP *replace attention* with a new fast-weight layer; LaCT *adds* a SwiGLU fast-weight block; even TTT-E2E adds a second MLP per block and demands from-scratch meta-pretraining. For modern LLMs this is the wrong deal — replacing attention is a high-risk modification of exactly the component billions of dollars of pretraining went into, and any new randomly-initialized layer conflicts with billions of trained parameters.

**Our move: don't add the memory — find it.** Transformer MLP blocks are already key-value memories (Geva et al. 2020) — that's where pretraining stores its "slow" knowledge. So let the *same* matrix also store fast, transient, in-context knowledge. In a gated MLP,

$$Z = \phi(H W_{\text{gate}}^\top)\odot(H W_{\text{up}}^\top), \qquad O = Z\,W_{\text{down}}^\top,$$

we freeze $W_{\text{up}}, W_{\text{gate}}$ as slow weights and treat **$W_{\text{down}}$ — a matrix the model already has — as the fast weights**, updated in place at inference time. No new mixing module, no architectural change, attention fully intact: a drop-in enhancement of any pretrained checkpoint.

In fast-weight vocabulary, the mapping is exact:

| Fast-weight concept | In-Place TTT realization |
|---|---|
| memory matrix $S$ | $W_{\text{down}}$ ($d_{\text{model}} \times d_{\text{ff}}$ — *huge state, for free*) |
| key $k_t$ | $z_t$, the MLP's post-gating intermediate activation |
| value $v_t$ | $\hat v_t$, an LM-aligned target (below) |
| read $S q$ | the MLP's ordinary down-projection $W_{\text{down}} z$ |
| write | a gradient step on an inner loss |

### 10.1 The LM-aligned objective

Prior TTT objectives *reconstruct the current token* ($k$ and $v$ are both projections of the same $x_t$). We argue this is misaligned with what an LM is for: it should not memorize what it just saw, it should store what helps **predict what comes next**. So our target injects future-token information:

$$\hat{V} = \mathrm{Conv1D}(X_0)\, W_{\text{target}},$$

where $X_0$ is the token-embedding sequence, the Conv1D is causal-padded depthwise (kernel 5), and $W_{\text{target}}$ is learned — the pure next-token target is the special case (identity projection, kernel that picks position $t{+}1$); in practice the model learns a localized combination of future tokens, in the spirit of multi-token prediction.

The paper formalizes the advantage in the induction-head setting (a pair $(k^*, v^*)$ appears in context; later $k^*$ reappears and the model must predict $v^*$). Under mild assumptions (near-orthogonal embeddings, key-query alignment of the $z$'s), **Theorem 1**: one fast-weight update with the LM-aligned target $\hat v_t = E_{x_{t+1}}$ raises the *correct next-token logit* by at least $\lambda_{\text{lr}} c_{\text{norm}}^2 c_{\text{align}}$ while moving all other logits by at most $O(\epsilon)$ — whereas the reconstruction target $\hat v_t = E_{x_t}$ moves the correct logit by at most $O(\epsilon)$, i.e. **reconstruction writes provide essentially zero predictive benefit** in exactly the regime in-context learning lives in.

### 10.2 The update: apply-then-update over large chunks

Split the sequence into chunks of $C$ tokens (512–1024 is optimal). For chunk $i$:

1. **Apply:** $\;O_{[i]} = Z_{[i]} \big(W_{\text{down}}^{(i)}\big)^\top$ — the current fast weights process the chunk;
2. **Update:** one gradient step with loss $\mathcal{L}(\cdot,\cdot) = -\langle \cdot,\cdot\rangle_F$, which has the closed form

$$\boxed{\;W_{\text{down}}^{(i+1)} = W_{\text{down}}^{(i)} + \eta\, \hat{V}_{[i]}^\top Z_{[i]}\;}$$

— a rank-$C$ outer-product (Hebbian) write, the linear-attention update of §2 transplanted into the backbone. Chunk $i$'s update helps chunks $i{+}1, i{+}2, \dots$, never itself: strictly causal at chunk granularity.

Why can we get away with chunks 30–60× larger than TTT's $b{=}16$? Because **attention is still there doing the fine-grained token mixing**. Standalone TTT layers must replace attention, which forces per-token (or tiny-batch) updates; our fast weights only need to carry slowly-accumulating global context, which is precisely a large-chunk job. This is the same division of labor LaCT discovered (fast weights global / attention local) — implemented with zero new modules.

### 10.3 Why this parallelizes embarrassingly well

Because the inner loss is *linear* in $W$, the gradient $-\hat V^\top Z$ **does not depend on the current weights**. Each chunk's delta $\Delta W_i = \hat V_{[i]}^\top Z_{[i]}$ can be computed independently, and the recurrence collapses to an exclusive **prefix sum**:

1. compute all $\Delta W_i$ in parallel;
2. one scan: $\;\Delta S_i = \sum_{j<i} \Delta W_j$;
3. in parallel: $\;W^{(i-1)}_{\text{down}} = W^{(0)}_{\text{down}} + \eta\, \Delta S_i$, then $O_{[i]} = Z_{[i]}(W^{(i-1)}_{\text{down}})^\top$.

No WY representation, no UT transform, no custom kernel — that machinery exists (§5.3) because state-dependent updates compose nontrivially; a purely additive rule is just a cumsum. The scan is associative, so the algorithm is **context-parallelism-native**: shard the sequence across devices, compute local deltas, exchange one prefix sum of weight deltas. Causal Conv1D padding keeps each delta free of future information; at document boundaries the fast weights **reset to the pretrained $W_{\text{down}}$**. Two practical details: a warm-start initialization (zero-init Conv1D ⇒ $\Delta W \approx 0$, so a pretrained model starts *exactly* at its original behavior and TTT emerges smoothly during continual training), and at inference a Frobenius-norm clip on each delta ($\tau = 10^{-5}$ in our Qwen3-4B runs) for long-horizon stability.

**Outer loop.** Standard NTP cross-entropy backprops through the (differentiable) chunk recurrence; the meta-learned components are the target generator (Conv1D kernel + $W_{\text{target}}$) plus, in continual/from-scratch training, the backbone itself. The model literally **learns how to write into its own MLPs**.

### 10.4 Results

**Drop-in continual training** (Qwen3-4B-Base, ~20B tokens @32K then ~15B @128K, TTT on every 6th layer; identical curriculum for the baseline): RULER 64K **78.7 vs 74.3 (+4.4)**, 128K **77.0 vs 74.8 (+2.2)**, and at 256K — *beyond* the 128K training length — **43.9 vs 41.7**, i.e. the advantage survives extrapolation. Same recipe transfers: LLaMA-3.1-8B +2.1 at 64K, Qwen3-14B-Base +2.7 at 64K.

**From scratch**, two experiments. At 500M/20B and 1.5B/60B tokens (32K sequences), against SWA Transformer, GLA, DeltaNet, and **LaCT** (both LaCT and ours on the same SWA backbone): lowest sliding-window perplexity at every context length up to 32K, monotonically improving with context. Separately, at 4B/120B tokens (8K context), comparing full-attention and SWA backbones with and without In-Place TTT: RULER-16K goes 6.58 → 19.99 on full attention, RULER-8K 9.91 → 26.80 on SWA, with commonsense scores (HellaSwag/ARC/MMLU/PIQA) mostly improved too.

**Ablations:** more TTT layers (state size) monotonically helps; chunk 512–1024 is the sweet spot; both halves of the target generator matter — the Conv1D (future information) is essential at long context, the projection at short. Prefill throughput and memory overhead: negligible at 8K–128K.

---

## 11. Incorporating fancier update rules into the weights

This section is the punchline of the doc: §6's zoo and §10's in-place framing are **orthogonal axes**, and composing them is our research program.

### 11.1 The general recipe

In-Place TTT is best understood not as one model but as a *recipe* for turning any existing weight matrix into a fast-weight memory:

1. **Pick the memory.** A projection the backbone already owns. We use $W_{\text{down}}$ — it's huge ($d_{\text{model}} \times d_{\text{ff}}$), it's already an associative key→value map, and its input $Z$ is a high-dimensional, sparse-ish feature vector that makes a great address space.
2. **Pick the key.** Whatever activation already feeds that matrix — for us, $Z$. No new key projection needed.
3. **Pick the value.** This is a free design choice, and ours is the LM-aligned target $\hat V$ (§10.1).
4. **Pick the update rule.** *Any row of the §6 table now applies.* The paper deliberately uses the simplest one; the framework is explicitly "orthogonal to the specific choice of loss functions and optimizers."

Step 4 is where the zoo plugs in.

### 11.2 Additive (the paper): linear attention in the weights

$$W_{j+1} = W_j + \eta\, \hat V_{[j]}^\top Z_{[j]}$$

This is the Hebbian write of §2 — with all of its virtues (state-independent deltas ⇒ prefix-sum parallelism, CP-native, trivially stable to warm-start) and its known vice: **no interference management**. If similar keys recur with different values over a very long context, the memory accumulates conflicting associations (the inference-time delta clipping is a band-aid for exactly this). Expanded, the read by a later token is

$$\Delta o_q = \eta \sum_{i<q} (z_i^\top z_q)\, \hat v_i$$

— literally unnormalized linear attention over the context, living inside the MLP.

### 11.3 Delta and Gated-Delta In-Place TTT: DeltaNet in the weights

Transplant §5 wholesale. Keep the decomposition $W_j = W_0 + \Delta W_j$ (pretrained slow knowledge + fast context). Decay the fast state, read what the decayed memory predicts, and write only the gated correction:

$$\widetilde{\Delta W}_j = \rho_j\, \Delta W_j, \qquad E_j = \hat V_j - Z_j\,\big(W_0 + \widetilde{\Delta W}_j\big)^\top \qquad \text{(prediction error of the current memory)}$$

$$\boxed{\;\Delta W_{j+1} = \widetilde{\Delta W}_j + \eta_j\, E_j^\top Z_j\;}$$

This is Gated DeltaNet's $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ with the dictionary $S \leftrightarrow \Delta W$, $k \leftrightarrow z$, $v \leftrightarrow \hat v$, $\alpha \leftrightarrow \rho_j$ (retention gate on the *fast* state only — the decay applies to $\Delta W$, never to the pretrained $W_0$, so forgetting context can't erode pretrained knowledge), $\beta \leftrightarrow \eta_j$ (write-strength gate). The one deliberate deviation from GDN: the read goes through $W_0 + \widetilde{\Delta W}$, not $\widetilde{\Delta W}$ alone, so the memory effectively stores *residuals relative to what the pretrained model already predicts* — it only spends capacity on what the context adds. The repo already ships this as the `gated_delta` update rule alongside `inplace`, with learned gates and a token-level update mode.

Why bother? The additive rule answers "what should I store?"; the delta rule also answers "**what does the memory already believe about this key?**" — which is exactly the interference-management mechanism §10's clipping approximates crudely. The induction-head theory of §10.1 says LM-aligned *values* put the right content in memory; the delta correction keeps that content *retrievable* when keys collide over 100K+ tokens.

The cost is the same one DeltaNet paid in §5.3: the error $E_j$ depends on the current state, so chunk deltas are no longer independent and the prefix-sum trick dies. The options, in increasing engineering effort:

| Update rule in the weights | Parallel structure | Cost of causality |
|---|---|---|
| Additive (paper) | exclusive prefix sum over $\Delta W_j$ — fully parallel, CP-native | free |
| Delta / gated-delta | sequential scan over chunks (apply→update→…), parallel within chunk | one $d_{\text{ff}}\times d_{\text{model}}$ matmul chain of length $T/C$; fine at $C = 512$–4096 since $T/C$ is small |
| Delta w/ intra-chunk recurrence | DeltaNet-style WY/UT machinery lifted to $W_{\text{down}}$-sized states | custom kernels; open engineering question at $d_{\text{ff}} \times d_{\text{model}}$ scale |

The middle row is the pragmatic regime — with chunks of 512–1024 the sequential chain is 128–512 steps at 128K–256K context (32–64 steps with the repo's continual-training chunk of 4096), and each step is dense matmuls (this is the same coarse-granularity bet LaCT makes). Note this trade-off is intrinsic, not incidental: *state-independent updates are scannable; state-dependent updates are what make memories self-correcting.* You pay in parallelism for exactly the property you wanted.

### 11.4 The rest of the zoo, in-place

Each remaining axis of §6's grid has a natural in-place analogue, in roughly ascending order of ambition:

- **Retention (Mamba-2/GLA-style):** $\Delta W_{j+1} = \rho_j \Delta W_j + \eta\,\hat V_j^\top Z_j$ — decay *without* the delta correction. Keeps the scan parallel (a prefix "weighted sum" with cumulative products of scalars $\rho$ is still an associative scan), making it the cheapest interference fix available. The natural first ablation between §11.2 and §11.3.
- **Momentum / surprise (Titans-style):** carry a velocity matrix $M_j = \theta_j M_{j-1} + \hat V_j^\top Z_j$, write $\Delta W_{j+1} = \rho_j \Delta W_j + \eta_j M_j$. Two chained scans; still parallelizable.
- **Muon (LaCT/Atlas-style):** orthogonalize the per-chunk delta with a few Newton–Schulz iterations before writing. Large chunks make this affordable (it's a handful of $d \times d$ matmuls per chunk); LaCT's ablations (Muon > momentum > GD) suggest it's worth testing on $W_{\text{down}}$ directly.
- **Windowed objectives (Atlas-style Omega rule):** let each chunk's update also reduce the loss on the previous chunk's tokens — a cheap second gradient pass that approximates "memorize the context, not the chunk."
- **Closed-form / exact (Longhorn, Mesa):** the proximal step has a closed form even at $W_{\text{down}}$ scale (it's still rank-$C$ via Woodbury on the chunk Gram matrix); the full Mesa solve is almost certainly too expensive here, but is the limit point that tells us how much headroom the optimizer axis has.

The point of the framework is that these are now **apples-to-apples comparable**: same memory ($W_{\text{down}}$), same keys ($Z$), same LM-aligned values ($\hat V$), same chunk schedule — only the write rule changes. Our `fair_baselines` harness runs both kinds of comparison under one training recipe (FineWeb-Edu at 32K, chunk 1024 for the scratch ablations): pure sequence-*mixer* baselines (SWA, DeltaNet, Gated DeltaNet, GDN-2, Kimi Linear, E2E-TTT-in-place) with tokenizer/optimizer/data/model shape held fixed, and the In-Place TTT *write-rule* variants (`inplace_ttt`, `gated_delta_ttt`) where memory, keys, values, and chunk schedule are all identical and only the chunkwise update rule is swapped. (Don't confuse the two: `gated_deltanet` the mixer and `inplace_ttt --ttt-update-rule=gated_delta` the MLP fast-weight path are different experiments.)

### 11.5 How this differs from TTT-E2E's way of writing into weights

TTT-E2E (§9) also writes into the backbone's own MLPs — it's the closest cousin, and the contrast is instructive:

| | TTT-E2E | In-Place TTT |
|---|---|---|
| What's updated | full MLPs, last 1/4 of blocks | $W_{\text{down}}$ only, chosen layers |
| Write signal | backprop of the *end-of-network CE loss* through L/4 blocks | local, closed-form outer product $\hat V^\top Z$ per layer |
| Value/target | implicit (whatever CE backprop produces) | explicit, learned $\hat V$ — the memory operation is read/compare/write you can inspect |
| Outer training | requires from-scratch meta-pretraining (grad-of-grad; 3.4× slower at 8K) | warm-starts from any pretrained checkpoint; standard backprop through a scan |
| Parallelism | sequential scan over $T/b$ chunks with inner backprops | prefix sum (additive) or short sequential chain (delta) |
| Trade-off bought | the *true* objective, maximal alignment | locality, parallelism, drop-in deployability, explicit memory semantics |

E2E's CE gradient is the "correct" signal by construction, but it is opaque and expensive — there is no explicit target, so there is nothing like the delta rule's "read what's stored, write the difference" to reason about or improve. Our explicit key/value/update factorization is what makes §11.3–11.4 possible at all: you can't swap the update rule of a method that never exposes one.

---

## 12. The unified view

Read the whole doc backwards through one lens:

$$\text{sequence model} \;=\; \big(\,\text{memory } \mathcal{M},\; \text{inner loss } \ell,\; \text{optimizer step},\; \text{retention},\; \text{granularity},\; \textbf{where } \mathcal{M} \textbf{ lives}\,\big)$$

- **Linear attention** discovered the memory ($\mathcal{M}$ = matrix) and used the crudest optimizer (one GD step on an inner product, no forgetting).
- **RetNet / GLA / Mamba** added retention — scalar, then per-channel, then derived from control theory — and the chunkwise training template.
- **DeltaNet and descendants** upgraded the loss (squared error ⇒ self-correcting memory) and showed how to keep non-diagonal updates trainable (WY/UT); the zoo then explored every remaining cell: vector gates (RWKV-7), momentum + depth (Titans), implicit steps (Longhorn), exact solves (Mesa), multi-step (DeltaProduct), windowed losses + Muon (Atlas), fine-grained decay at scale (KDA).
- **TTT** named what everyone was doing — online learning — made the loss *meta-learned* and the memory an arbitrary network, at the price of tiny inner batches.
- **LaCT** fixed the systems story: huge chunks, huge nonlinear states, real optimizers, plus window attention for what big chunks can't do.
- **TTT-E2E** made the inner loss the true LM loss and the outer loop honest meta-learning — long context as continual learning into the model's own MLPs.
- **In-Place TTT (ours)** answers the *where* question: the memory should be a weight matrix the pretrained model already has, written with an objective aligned to next-token prediction, at chunk sizes attention makes affordable, with a scan that makes it free to parallelize — and §11 is the program of carrying every update rule the field invented into that setting.

The field converged on this from four directions at once — kernelized attention, control theory, fast-weight programming, and meta-learning — which is usually a sign the abstraction is real.

---

## 13. References

**Foundations**
- Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)
- Schmidhuber. *Learning to control fast-weight memories.* Neural Computation, 1992.
- Sun et al. *Retentive Network: A Successor to Transformer.* 2023. [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

**SSMs**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Dao, Gu. *Transformers are SSMs (Mamba-2 / SSD).* ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

**Delta-rule family and the zoo**
- Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet).* NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) — and the blog series: [I](https://sustcsonglin.github.io/blog/2024/deltanet-1/) · [II](https://sustcsonglin.github.io/blog/2024/deltanet-2/) · [III](https://sustcsonglin.github.io/blog/2024/deltanet-3/) · [talks](http://sustcsonglin.github.io/talk/)
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks.* ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Peng et al. *RWKV-7 "Goose" with Expressive Dynamic State Evolution.* 2025. [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)
- Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- Liu et al. *Longhorn: State Space Models are Amortized Online Learners.* 2024. [arXiv:2407.14207](https://arxiv.org/abs/2407.14207)
- von Oswald et al. *MesaNet: Sequence Modeling by Locally Optimal Test-Time Training.* 2025. [arXiv:2506.05233](https://arxiv.org/abs/2506.05233)
- Siems et al. *DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders.* 2025. [arXiv:2502.10297](https://arxiv.org/abs/2502.10297)
- Behrouz et al. *Atlas: Learning to Optimally Memorize the Context at Test Time.* 2025. [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)
- Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture.* 2025. [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Grazzi et al. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025. [arXiv:2411.12537](https://arxiv.org/abs/2411.12537)

**TTT line**
- Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* 2024. [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
- Zhang et al. *Test-Time Training Done Right (LaCT).* 2025. [arXiv:2505.23884](https://arxiv.org/abs/2505.23884)
- Tandon, Dalal, Li, …, Sun. *End-to-End Test-Time Training for Long Context (TTT-E2E).* 2025. [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
- **Feng, Luo, Hua, Zhang, He, Huang, Cai. *In-Place Test-Time Training.* ICLR 2026 (Oral). [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)**
