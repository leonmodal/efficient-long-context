# MemFlow: Multi-Timescale Memory-Flow Linear Attention — Implementation Plan

This is an executable engineering spec. Build it in the phase order given. Each phase
has a single, testable deliverable. **Do not skip the correctness tests** — they are the
difference between "trains and is silently wrong" and "trains and is right."

Scope: a linear-attention layer with multiple matrix-memory "buckets" at different
timescales, where slower buckets are written using *readouts of the faster bucket*
("memory of memory"). Phase 1 uses a **fixed** number of buckets (no growth). Phases 3–4
add **log-linear growth** of the number of buckets. Nothing else (no kNN, no RAG) is in scope.

Target: a small GPT-style LM where this layer replaces self-attention. Dev scale is small
so iteration is fast on one GPU.

---

## 0. Conventions and notation (read first — used everywhere)

- `B` batch, `T` sequence length, `d_model` model width.
- `H` heads per layer; each head has its own independent set of memories. All math below
  is written for **one head**; vectorize over `H` and `B` in code.
- `d_k` key dim, `d_v` value dim (default `d_k = d_v = d_model // H`, call it `d_head`).
- A memory state is a matrix `S ∈ R^{d_v × d_k}`. **Read:** `o = S @ q` with `q ∈ R^{d_k}`, `o ∈ R^{d_v}`.
- Buckets are indexed by level `ℓ = 0, 1, 2, ...`. Level 0 is the fast bucket. `S^(ℓ)` is its state.
- `P_ℓ` = write period of level `ℓ` (in tokens). `P_0 = 1`. Default `[1, 128, 512, 2048]`.
- `C` = compute chunk size for the kernel (default 64). **`C` is independent of every `P_ℓ`.**

### The single update operator (gated delta rule, KDA/Gated-DeltaNet style)

Writing a key/value pair `(k, v)` into state `S`, with channel-wise decay `α ∈ (0,1)^{d_k}`
and scalar write strength `β ∈ (0,1)`:

```
U(S; k, v, α, β) = S · diag(α) · (I − β·k·kᵀ) + β · v · kᵀ
```

Equivalent delta form (use whichever is convenient; they're identical):

```
U(S; k, v, α, β) = S·diag(α) + β · (v − (S·diag(α))·k) · kᵀ
```

Requirements:
- `k` is **L2-normalized** (‖k‖₂ = 1) before the write. `q` is L2-normalized before a read.
- `α = sigmoid(W_α x + b_α)`, init `b_α` high so α starts near 1 (slow forgetting). Channel-wise (a vector).
- `β = sigmoid(W_β x + b_β)`, scalar per token.
- Shapes: `S` (d_v, d_k); `diag(α)` (d_k, d_k); `(I − β k kᵀ)` (d_k, d_k); `v kᵀ` (d_v, d_k). Output (d_v, d_k). ✓

This single operator is used for **every** bucket write. The buckets differ only in
(a) what `(k, v)` they get fed and (b) how often.

### Memory flow (the core idea)

- **Fast bucket (level 0):** standard linear attention. `k,v,q` are projections of the token.
  Updated **every token**. `S^(0)_t = U(S^(0)_{t-1}; k^(0)_t, v^(0)_t, α^(0)_t, β^(0)_t)`.
- **Slower bucket (level ℓ>0):** the **value written is a readout of level ℓ−1**, not the token.
  - probe `p_t = W_p^(ℓ) x_t` (lives in level ℓ−1's key space; see §6 for tying option)
  - readout (the value) `r_t = S^(ℓ-1)_t · p_t`   ← this is the "memory of memory"
  - key `k^(ℓ)_t = W_k^(ℓ) x_t` (token-derived, L2-normalized after aggregation)
  - written once per `P_ℓ` tokens via Option C aggregation (below).

### Option C aggregation (read every token, write once per period)

For level `ℓ>0`, over each window of `P_ℓ` tokens, maintain a running softmax-weighted average
of `(probe-key, readout-value)` so you never buffer the window:

```
salience  s_t = w_sal^(ℓ) · x_t           # scalar score per token
running:  m  = running max of s_t          # numerical stability
          den   += exp(s_t − m)            # rescale prior accumulators when m updates
          numK  += exp(s_t − m) · k^(ℓ)_t  # (d_k,)
          numR  += exp(s_t − m) · r_t      # (d_v,)
at window boundary:
          k̄ = l2norm(numK / den)          # aggregated key, renormalized
          r̄ = numR / den                  # aggregated value (NOT normalized)
          S^(ℓ) = U(S^(ℓ); k̄, r̄, α^(ℓ), β^(ℓ))   # the one write
          reset m, den, numK, numR
```

Note: when `m` updates to `m_new`, multiply `den, numK, numR` by `exp(m_old − m_new)` (FlashAttention-style).
For the **reference** impl you may instead gather the `P_ℓ` per-token values and do a batch
softmax-weighted sum — identical result, simpler to verify. Optimize to the online form later.

### Readout (layer output, every token)

Query every **live** bucket, gate, sum, project:

```
o^(ℓ)_t = S^(ℓ) · q^(ℓ)_t          # q^(ℓ)_t = l2norm(W_q^(ℓ) x_t); per-level read query
g^(ℓ)_t = sigmoid(W_g^(ℓ) x_t)      # per-level output gate (d_v,)
y_t = Norm( Σ_ℓ live(ℓ,t) · g^(ℓ)_t ⊙ o^(ℓ)_t ) · W_o
```

`live(ℓ,t) ∈ {0,1}` masks a bucket until its first write (see §Masking). A masked bucket
contributes exactly nothing. (Reading an unwritten zero state also returns zero, but explicit
masking avoids feeding spurious zeros into Norm.)

### Masking empty buckets

Level `ℓ` is **not live** until token `t ≥ P_ℓ` (its first write happens at the first boundary,
i.e. `t = P_ℓ`). Before that, drop it from the readout sum. This is the entire answer to
"the bucket is empty at the start" — you don't read it until it has content.

---

## 1. Repo layout

```
memflow/
  __init__.py
  config.py            # MemFlowConfig dataclass (all hyperparams)
  delta_rule.py        # gated delta op: recurrent ref + chunkwise + read-with-probe
  bucket.py            # MemoryBucket: holds S, exposes update() and read()
  consolidation.py     # Option C: probe, read-lower, online aggregate, boundary write
  memflow_layer.py     # MemFlowLayer: orchestrates buckets, consolidation, readout
  growing.py           # Phase 3: Fenwick scheduler + bucketed levels + merge
  model.py             # GPT-style decoder LM that uses MemFlowLayer
  data.py              # loaders: FineWeb shard, enwik8 (char), synthetic MQAR, NIAH
  train.py             # training loop + configs
  eval.py              # ppl, MQAR acc, NIAH/RULER, throughput/memory
tests/
  test_delta_equiv.py  # chunkwise == recurrent (allclose)
  test_consolidation.py# writes only at boundaries; online agg == batch agg
  test_masking.py      # empty bucket contributes exactly 0
  test_growing.py      # #levels == O(log T); distant-fact coverage
configs/
  dev_1bucket.yaml
  dev_2bucket.yaml
  dev_Nbucket.yaml
  dev_growing.yaml
```

Dependencies: `torch`, `einops`, `numpy`, optionally `flash-linear-attention` (FLA) for the
fast-bucket kernel. Start **without** FLA (pure-PyTorch reference), add it in Phase 1.5 for speed.

---

## 2. MemFlowConfig (config.py)

```python
@dataclass
class MemFlowConfig:
    # model
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 64          # d_k = d_v = head_dim
    vocab_size: int = 50304
    max_seq_len: int = 8192
    # memory flow
    periods: tuple = (1, 128, 512, 2048)   # P_ℓ per level; periods[0] must be 1
    chunk_size: int = 64                    # compute chunk C (independent of periods)
    tie_probe_to_lower_key: bool = False    # W_p^(ℓ) = W_k^(ℓ-1) if True
    value_source: str = "memory"            # "memory" (ours) | "token" (baseline ablation)
    use_output_gate: bool = True
    seed_on_create: bool = False            # Phase 3: seed a new level from level below
    # growth (Phase 3+)
    grow: bool = False                      # False = fixed buckets (Phase 1/2)
    growth_base: int = 2                    # new level every time chunk-count crosses base^ℓ
    # training stability
    detach_value_steps: int = 2000          # stop-grad on r_t -> S^(ℓ-1) for first N steps
    alpha_bias_init: float = 4.0            # high => α≈1 at init (slow forgetting)
    beta_bias_init: float = -4.0            # low  => small writes at init
```

---

## 3. Phase 0 — Scaffolding + single-bucket baseline

**Goal:** a correct, trainable single-bucket linear-attention LM. This is the foundation; if the
delta rule is wrong here, everything downstream is wrong.

### 3.1 delta_rule.py
Implement three functions, single-head, batched over `(B, H)`:

```python
def gated_delta_recurrent(q, k, v, alpha, beta, S0=None):
    """Reference. Loops over T. Returns (o, S_final).
       q,k,v: (B,H,T,d); alpha: (B,H,T,d_k); beta: (B,H,T,1).
       k,q already L2-normalized by caller. o: (B,H,T,d_v)."""
    # S = S0 or zeros (B,H,d_v,d_k)
    # for t: S = S·diag(alpha_t)·(I − beta_t k_t k_tᵀ) + beta_t v_t k_tᵀ ; o_t = S q_t
    # return o, S

def gated_delta_chunkwise(q, k, v, alpha, beta, S0=None, chunk_size=64):
    """Parallel form. Same I/O as recurrent. Use FLA's chunk_gated_delta_rule if available,
       else implement standard chunk recurrence. MUST match recurrent to atol=1e-4."""

def read_state_with_query(S0, q_probe, k, v, alpha, beta, chunk_size=64):
    """Reads the EVOLVING fast state with an alternate query stream q_probe (the probes P),
       sharing k,v,alpha,beta. Returns r_t = S_t · q_probe_t for all t. (B,H,T,d_v).
       Reference: identical to recurrent but emit S_t·q_probe_t instead of S_t·q_t."""
```

### 3.2 model.py
Standard pre-norm decoder: token+pos embedding → N×(MemFlowLayer + MLP, each with RMSNorm and
residual) → final RMSNorm → tied LM head. For Phase 0 the MemFlowLayer is just the fast bucket.

### 3.3 Tests / acceptance
- `test_delta_equiv.py`: random inputs, assert `allclose(recurrent, chunkwise, atol=1e-4)` and
  `allclose(recurrent_final_S, chunkwise_final_S, atol=1e-4)`. **Must pass before Phase 1.**
- Train on enwik8 (char-level) or a 100M-token FineWeb-Edu shard for ~2k steps; assert loss
  drops clearly (e.g. char bpc < 1.6, or LM loss visibly decreasing and not NaN).

**Deliverable:** single-bucket LM trains; equivalence test green.

---

## 4. Phase 1 — Two-bucket fixed memory flow (THE CORE)

**Goal:** add slow bucket (level 1) that consolidates from fast (level 0) via Option C, fixed
`periods=(1,128)`, `grow=False`. This is the central contribution. Get it correct, then ablate.

### 4.1 consolidation.py
```python
class Consolidator(nn.Module):
    """Builds level ℓ from level ℓ-1 (Option C)."""
    def __init__(self, cfg, level):  # holds W_p, W_k, w_sal, alpha/beta heads for this level
    def forward(self, x_chunk, S_lower_evolving_read_fn, S_upper, acc_state):
        # 1. P = W_p(x_chunk); KK = W_k(x_chunk); sal = w_sal(x_chunk)        # per-token
        # 2. if value_source=="memory": R = S_lower_evolving_read_fn(P)        # r_t = S^(ℓ-1)_t p_t
        #    else (ablation "token"):   R = W_v_token(x_chunk)                 # baseline
        # 3. online-accumulate (KK, R) weighted by exp(sal) into acc_state
        # 4. if boundary in this chunk: kbar=l2norm(numK/den); rbar=numR/den
        #       S_upper = U(S_upper, kbar, rbar, alpha_ℓ, beta_ℓ); reset acc
        # return S_upper, acc_state
```

Use `read_state_with_query` (from delta_rule.py) as `S_lower_evolving_read_fn`, passing the
probes `P` as the query stream against level ℓ−1's (k,v,α,β). For the recurrent reference,
just compute `r_t = S^(ℓ-1)_t @ p_t` inside the token loop.

### 4.2 memflow_layer.py (forward, recurrent reference first)
```
forward(X):                       # X: (B,T,d_model) -> reshape per head
  init S0=zeros, S1=zeros, acc1=empty
  outputs = []
  for t in range(T):
     # fast (level 0) update + immediate readout
     S0 = U(S0; k0_t, v0_t, α0_t, β0_t)
     o0 = S0 @ q0_t
     # consolidation level 1 (Option C), value from S0
     r_t = S0 @ p1_t                            # memory-of-memory read
     acc1 = online_accumulate(acc1, k1_t, r_t, sal1_t)
     if (t+1) % P_1 == 0:
        k̄,r̄ = finalize(acc1); S1 = U(S1; l2norm(k̄), r̄, α1_t, β1_t); acc1 = reset()
     # readout (mask level 1 until first write at t=P_1-? -> live when t+1 >= P_1)
     live1 = (t+1) >= P_1
     o1 = (S1 @ q1_t) if live1 else 0
     y = RMSNorm(g0_t⊙o0 + (g1_t⊙o1 if live1 else 0)) @ W_o
     outputs.append(y)
  return stack(outputs)
```
Then write the **chunked** version: fast bucket via `gated_delta_chunkwise`; `R` via
`read_state_with_query`; slow writes via a loop over the (few) boundaries. Verify it matches
the recurrent version (atol 1e-4) on random input.

Apply `detach_value_steps`: for the first N training steps, `r_t = (S0.detach()) @ p1_t` so the
slow objective doesn't yank the fast bucket early; anneal to no-detach after N steps.

### 4.3 Tests / acceptance
- `test_consolidation.py`:
  - (a) **boundary-only writes:** run with periods=(1,4); assert `S1` is byte-identical between
    boundaries and only changes at t=3,7,11,... .
  - (b) **online == batch:** assert online-accumulated `(k̄,r̄)` equals a direct batch
    softmax-weighted sum over the window (atol 1e-5).
- `test_masking.py`: with periods=(1,128), assert at t<128 the output equals the fast-only output
  exactly (level 1 contributes 0).
- Gradient sanity: one backward pass, assert no NaN/Inf and that `W_p, W_k^(1), w_sal^(1)` all
  receive nonzero grad after the first boundary.
- **Ablation parity run:** train two configs to equal steps on a synthetic MQAR task (below):
  - ours: `value_source="memory"`
  - baseline: `value_source="token"` (multi-rate token reading)
  - and a single-bucket baseline with `head_dim` doubled (matched total state).
  Record MQAR accuracy for all three.

**Deliverable:** two-bucket model trains; on MQAR, `value_source="memory"` should ≥ the
`"token"` ablation and ≥ matched-state single-bucket. If it doesn't, the memory flow isn't
earning its keep — investigate before adding levels.

### 4.4 Synthetic MQAR (multi-query associative recall) — put in data.py
Generate sequences of `(key, value)` pairs followed by queries that re-ask earlier keys; the
model must emit the matching value. Standard recall probe; vary #pairs and gap length. This is
the fastest signal that memory flow helps; it does **not** require big training.

---

## 5. Phase 2 — N fixed buckets (generalize the cascade)

**Goal:** generalize to `periods=(1,128,512,2048)`, still `grow=False`. Level `ℓ>0` consolidates
from level `ℓ-1` at period `P_ℓ`.

Changes:
- `memflow_layer.py`: hold a list of states `S[0..L-1]` and a `Consolidator` per level `ℓ>0`.
  In the loop, after updating `S[ℓ-1]`, run consolidator `ℓ`. Order matters: update level 0,
  then 1, then 2, ... within the same step so each reads the freshest lower state.
- Readout sums over all live levels with per-level gates.
- `seed_on_create` flag: when level `ℓ` does its **first** write, optionally first do one
  consolidation that reads the *entire current* state of level `ℓ-1` into level `ℓ` (so it isn't
  born empty). Off by default in Phase 2.

Tests:
- N-level equivalence (recurrent vs chunked) at periods=(1,4,16).
- Per-level masking: level ℓ contributes 0 until t ≥ P_ℓ.
- No NaN at level switch-on.

**Deliverable:** configurable N-bucket model; sanity train at N=3,4. Diagnostics logged:
per-level effective decay (mean α) and per-level mean gate over training.

---

## 6. Projection sharing (decide here, applies to all phases)

Default: **separate** `W_q, W_k, W_v, W_α, W_β, W_p, w_sal, W_g` per level. Cheap and expressive.

Two options to expose via config:
- `tie_probe_to_lower_key=True`: set `W_p^(ℓ) = W_k^(ℓ-1)`. Rationale: the read `r = S^(ℓ-1) p`
  is governed by `(k^(ℓ-1) · p)`, so the probe must live in level ℓ−1's key space; tying makes
  the read a content-addressable lookup and saves params. Ablate this; it often helps.
- (Optional, only if param-constrained) shared low-level trunk + small per-level heads.

Do **not** share the read query `q^(ℓ)` across levels — each level wants its own read geometry.

---

## 7. Phase 3 — Growing log-linear (Fenwick) state

**Goal:** the **number** of levels grows with sequence length, giving O(log T) state that covers
all of history (coarsely). This is the payoff for very long context. `grow=True`.

### 7.1 Why bucketed, not running
A purely decaying (running) cascade loses old spans: by the time a high level reads a low level,
the low level has already faded its early content. The fix is **bucketed levels with timed merges**:
each level seals its span and is merged upward *while still fresh*. Use `growing.py` for this.

### 7.2 Mechanics
- Allocate level `ℓ` lazily when chunk-count first crosses `growth_base^ℓ` (zeros). Mask until first write.
- Schedule via the **least-significant-set-bit** of the chunk index (standard Fenwick/binary-indexed
  tree). At chunk index `c`, the levels that "complete" are determined by the trailing set bits of `c`.
- **Merge = your consolidator applied upward:** when level `ℓ`'s current bucket completes, consolidate
  it into level `ℓ+1` (probe → aggregate → delta-write), then reset level `ℓ`'s bucket for the next span.
  This is `seed_on_create`/merge done at every doubling, not just at birth.
- Readout: query all live levels, combine with **learned per-level weights** `λ_ℓ(t)` (softmax over
  levels or per-level sigmoids) + output gate. Recent → low levels (sharp), distant → high levels (coarse).

### 7.3 Implementation order
1. Reference Fenwick scheduler in pure Python: given `T` and `growth_base`, produce, per chunk,
   the list of (which levels complete, which merge into which). Unit-test the schedule against a
   hand-computed small case (e.g., T/C = 8).
2. Wire merges through the existing `Consolidator`. Reference (recurrent) first.
3. (Optional, for speed) adapt the log-linear-attention chunkwise kernel: it decomposes the
   inter-chunk term into per-level contributions — your buckets slot into those levels. Verify
   against the reference (atol 1e-4). Don't write Triton until the reference is correct and trains.

### 7.4 Tests / acceptance
- `test_growing.py`:
  - **state size:** for T = 1024, 4096, 16384, assert number of allocated levels ≈ `log_base(T/C)+const`.
  - **coverage:** construct a sequence with a unique fact at position ~5% in; at the end of a long
    sequence, a query for it retrieves it well above chance (it should survive coarsely in a high level).
    Compare against the fixed-bucket model, which should fade it to chance.
  - schedule correctness (the hand-computed case).
- Equivalence reference-vs-kernel if the kernel is added.

**Deliverable:** growing-state model; on NIAH/RULER at increasing lengths, recall **holds** where
the fixed-bucket and single-state models fade. Log #levels and per-level λ over length.

---

## 8. Training harness (train.py) and configs

- Optimizer: AdamW, `lr=3e-4`, betas (0.9, 0.95), weight_decay 0.1, grad clip 1.0.
- Schedule: linear warmup 5% then cosine to 10% of peak.
- Precision: bf16 autocast; keep memory states in fp32 (they accumulate — fp16 will drift).
- Batch: pack sequences to `max_seq_len`; gradient accumulation to taste.
- Dev model: `d_model=512, n_layers=6, n_heads=8, head_dim=64` (~60M params). Train 10–30B tokens
  of FineWeb-Edu for a real signal; use enwik8 / synthetics for fast loops.
- Log every step: loss; per-level mean α (effective timescale); per-level mean gate / λ; grad norm.
- Checkpoint + resume. Deterministic seed for the equivalence tests.

---

## 9. Evaluation (eval.py) and baselines

Metrics:
- **LM perplexity / bpc** on held-out FineWeb-Edu / enwik8.
- **MQAR accuracy** (Phase 1+): sweep #pairs and gap.
- **NIAH / RULER** at lengths {1k, 4k, 16k, 64k} (Phase 3 payoff).
- **Throughput** (tokens/s) and **peak memory**, train and decode.

Baselines (run all at matched params/FLOPs — this is the whole point):
1. **Single-bucket** gated-delta with `head_dim` scaled so total state matches the multi-bucket model.
2. **Multi-rate token reading** = same architecture with `value_source="token"` (isolates whether
   "memory of memory" beats "just read tokens at a slower rate").
3. (Optional) FLA Gated DeltaNet / Mamba-2 of equal params, as external sanity.
4. (Phase 3) the fixed-bucket model, to show growth holds recall where fixed fades.

**Success criterion for the project:** at matched params/FLOPs, the memory-flow model
(`value_source="memory"`) beats both (1) and (2) on MQAR and long-context recall, at comparable
throughput. The growing variant additionally holds recall vs the fixed variant as length grows.
If it merely ties, the contribution is efficiency/interpretability + the parallelism/growth result.

---

## 10. Correctness & debugging checklist (do not deviate)

- [ ] L2-normalize `k` (incl. aggregated `k̄`) and `q` before every write/read.
- [ ] Keep memory states in **fp32**, even under bf16 autocast.
- [ ] Compute chunk `C` is **decoupled** from every period `P_ℓ` (e.g., C=64, P=128/512/2048).
- [ ] Online softmax uses the running-max rescale (or clamp salience to a safe range).
- [ ] Mask each level out of the readout until `t ≥ P_ℓ` (its first write).
- [ ] Update levels **in order** (0,1,2,...) each step so each reads the freshest lower state.
- [ ] `detach` the `r_t → S^(ℓ-1)` path for the first `detach_value_steps` steps, then anneal in.
- [ ] Init `α` near 1 (high `alpha_bias_init`), `β` small (low `beta_bias_init`) — especially for slow levels.
- [ ] Chunkwise output == recurrent output (atol 1e-4) at every phase before training.

---

## 11. Known risks and what to watch

- **Slow level collapses to mirror the fast one.** Watch per-level effective decay; if slow α ≈ fast α
  and slow gate stays high, it's not specializing. Mitigate: stronger α-init toward 1, small slowness
  penalty (penalize α far from 1 on high levels), or a hard write-period floor.
- **Summaries-of-summaries information loss** (deep cascades / high Fenwick levels): monitor recall at
  depth; the bucketed+timed-merge structure mitigates but cannot eliminate — fixed state cannot hold
  unbounded history losslessly. This is expected, not a bug.
- **Training instability from reading a fast state that changes every step:** handled by the detach
  warmup; if still unstable, lengthen `detach_value_steps` or lower the slow `β`.
- **fp16 state drift:** if you see slow divergence, confirm states are fp32.
- **Premature kernels:** do not write Triton until the pure-PyTorch reference is correct and trains.
  Use FLA for the fast bucket; keep slow writes as a simple boundary loop (they're sparse).

---

## 12. Build order (TL;DR for the agent)

1. Phase 0: delta rule (recurrent + chunkwise) + equivalence test + single-bucket LM that learns.
2. Phase 1: two-bucket memory flow (recurrent ref → chunked), Option C, masking; pass consolidation
   + masking tests; run the 3-way ablation on MQAR. **This is the make-or-break milestone.**
3. Phase 2: generalize to N fixed buckets; diagnostics.
4. (Decide projection tying via §6 ablation.)
5. Phase 3: Fenwick growth (scheduler → merges via consolidator → readout over live levels); pass
   state-size + coverage tests; NIAH/RULER vs fixed-bucket.
6. Throughout: keep every equivalence test green; log per-level α and gates.

Start at Phase 0. Do not start Phase 1 until the equivalence test passes.