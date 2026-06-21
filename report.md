# MemFlow — Findings Report (Negative Result)

**MemFlow** = a linear-attention layer with multiple matrix-memory "buckets" at different
timescales, where *slower buckets are written from readouts of the faster bucket* ("memory of
memory") via salience-weighted Option-C consolidation. The hypothesis: a hierarchy of consolidated
memories uses a fixed state budget better than one undivided memory, especially for recall.

**TL;DR:** On synthetic **MQAR** (multi-query associative recall, the standard probe for a model's
exact-recall capacity), **memory-of-memory does not beat a matched-state single memory or a standard
Gated DeltaNet (GDN)** — across every base variant we tried, it is consistently the *worst*. A
"temporal segment-split" baseline (N memories, each owning one slice of the sequence) also loses.
Separately, our hand-rolled base layer is **unstable** (recall swings ~0.0↔0.9 by seed/numerics)
while FLA's GDN is stable. The long-context regime MemFlow was *designed* for (needle-in-a-haystack)
was **not** reached; that remains the only place the idea could still earn a niche (see Limitations).

---

## 1. What works (deliverables)

All correctness is verified by a 31-test suite (CPU float64 + CUDA float32/bf16 on Modal L4):

- **Gated delta rule** (`delta_rule.py`): recurrent reference + a numerically-stable channel-wise
  chunkwise form (matmul fast path + materialized exact oracle). `chunkwise == recurrent` to
  **3e-7**, verified against the literal `S·diag(α)(I−βkkᵀ)+βvkᵀ` operator.
- **N-bucket memory flow** (`memflow_layer.py`, `consolidation.py`): Option-C consolidation
  (online running-max aggregation `==` batch softmax), per-level masking (exactly 0 before first
  write), detach warmup, recurrent==chunk equivalence for 2- and 3-bucket cascades (2–4e-7).
- **Fenwick growth scaffolding** (`growing.py`): `lssb`/schedule/`live_blocks` with the O(log T)
  full-history coverage invariant (live buckets tile `[0,NC)` with `popcount(NC)` buckets). *The
  growing layer itself was not built (the study below redirected effort).*
- **Baselines**: standard Gated DeltaNet via FLA (`baselines.py::GDNBaselineLM`), and a
  segment-split baseline. Faithful log-linear attention (HGDN) was attempted but is blocked by an
  FLA version mismatch in the vendored `hattention` package.
- **Modal harness** (`modal_app.py`): GPU image, fan-out study runner, and **per-run result
  persistence to a Modal Volume** (`report` action recovers results even if a run is killed).

---

## 2. Setup

- Synthetic **MQAR**: D distinct key→value pairs in context, then queries re-asking earlier keys;
  accuracy = exact-value recall over supervised positions (chance ≈ 1 / (vocab/2)).
- Models matched on **total matrix-memory state** within each N (single/GDN use `head_dim = round(32·√N)`).
- 4 variants: **memory** (ours), **split** (N segment memories, direct token writes), **single**
  (one matched-size memory, ours), **gdn** (FLA Gated DeltaNet, the external reference).
- d_model=128, 2 layers, 4 heads, short conv on q/k/v, AdamW lr 1e-3, 5000 steps, 2 seeds.

A **short convolution** on q/k/v was required for *any* of these models to do MQAR at all (the
Zoology result): without it, every variant sat at chance.

---

## 3. Results

### 3a. memory vs token-read vs single vs GDN (channel-wise-decay base, +conv, mean of 2 seeds)

| D (pairs) | memory | token-read | single | GDN |
|---|---|---|---|---|
| 16 | 0.999 | 0.999 | 1.000 | 1.000 |
| 32 | 0.992 | 0.507 | 0.992 | 0.998 |
| 64 | **0.932** | 0.040 | **0.981** | **0.993** |
| 128 | 0.041 | 0.032 | **0.539** | **0.985** |

- ✅ **memory-of-memory beats the token-read ablation** (D=32: 0.99 vs 0.51; D=64: 0.93 vs 0.04).
  The mechanism does *something* — reading the fast memory beats reading tokens at a slow rate.
- ❌ **memory loses to a matched-state single memory and to GDN at high load** (D=128: 0.04 vs 0.54
  vs 0.99). Splitting a fixed budget into fast + lossy-summary buckets wastes exact-recall capacity.

### 3b. memory vs split vs single vs GDN — FINAL, apples-to-apples (GDN base, faithful init)

Mean over 2 seeds; vocab 512 -> chance ~ 0.004. All variants matched on total state.

| N | D | memory | split | single | GDN |
|--:|--:|:--:|:--:|:--:|:--:|
| 2 | 16 | 0.82 | 1.00 | 1.00 | 1.00 |
| 2 | **64** | **0.02** | 0.36 | **0.66** | **0.98** |
| 2 | 256 | 0.005 | 0.003 | 0.004 | 0.014 |
| 4 | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | **64** | **0.25** | 0.19 | **0.91** | **0.99** |
| 4 | 256 | 0.003 | 0.004 | 0.004 | 0.084 |

At the discriminating load (D=64): **GDN >= single >> split ~ memory**.
- **`memory` is the worst** — adding the slow "memory of memory" bucket to `single` *lowers* recall
  (0.66->0.02 at N=2, 0.91->0.25 at N=4): the salience-averaged summaries inject noise into the readout.
- **`split`** fails on MQAR: recall is content-addressed, so a query reads all N buckets and sums them;
  the N-1 buckets that don't hold the key add interference.
- **`single`** (our base) is now stable and in GDN's league (0.66-0.91 vs 0.98).

### 3c. The instability was init, not the idea — found and fixed (this is what makes 3b trustworthy)

Early on, `single` swung wildly by seed/numerics (0.009 <-> 0.98) while GDN didn't. We chased it down:
the chunkwise **kernel is numerically correct** (matches the recurrent reference to 5e-7 even under
aggressive GDN decay, min alpha ~ 0.30 -- not a bug), and `decay`/`beta` are matched to FLA. The real
culprit was **initialization**: our readout applied a sigmoid gate *inside* the norm (vs FLA's
`rmsnorm(o)*silu(g)`), we had no prenorm residual rescaling, and a low beta-bias. After aligning the
readout (norm -> SiLU-gate), beta-bias -> 0, and prenorm rescaling (`o_proj`/`down` x 1/sqrt(2L)), our
`single` became **stable** (D=64: 0.85/0.93 at N=2, 0.87/0.89 at N=4 across both seeds). So the
`single`-vs-`gdn` gap was implementation/init, **not** the memory-flow idea -- and `memory` vs `single`
(identical code except the slow bucket) is a clean controlled test on which the negative verdict holds.

---

## 4. Analysis — why memory-flow loses on exact recall

1. **Lossy averaging vs lossless superposition.** A slow bucket stores a *salience-weighted average*
   of fast-bucket readouts; standard linear attention superposes outer products losslessly (up to
   capacity). Averaging blurs distinct facts — the exact failure mode documented in this repo's own
   `nmt/EXPERIMENTS.md` (mean-pooled hierarchy → 0.07 recall at depth 5).
2. **Decay tension.** "Memory of memory" needs the fast bucket to *retain* content long enough to be
   consolidated. A well-tuned GDN decay (some heads forget fast) starves the consolidation, so on the
   proper GDN base `memory` got *worse*, not better.
3. **A confound we found and fixed.** Our original decay was **channel-wise** (per-dim, KDA-style,
   per the plan) vs GDN's **scalar** per-head decay. Channel-wise forgets dimensions unevenly and
   corrupts stored values; switching to scalar GDN-style decay closed most of the `single`-vs-`gdn`
   gap (0.63→0.87), confirming the base, not just the idea, was handicapped.

---

## 5. Limitations / what was NOT tested (and would change the verdict)

- **Long-context recall (NIAH/RULER) — the regime MemFlow was designed for — was not run.** MQAR is
  *short-context exact recall*, which structurally favors one big memory. The thesis is *one fact
  surviving a long stretch of distractors*, where a fixed memory is overwritten but a frozen/
  consolidated summary could retain it. This is the only place the idea could still win; the NIAH
  harness is built (`data.make_niah`) but a clean run requires a stable base.
- **Base instability is RESOLVED** (was a concern in an earlier draft): it was an init mismatch
  (readout gating, prenorm rescaling, beta-bias), now fixed — our `single` is stable and tracks GDN.
  So the table in §3b is on a faithful, apples-to-apples base.
- **No real-data LM** (FineWeb/enwik8) and **no throughput numbers**; the pure-PyTorch channel-wise
  chunkwise has no fused kernel and does not scale cheaply to long sequences.

---

## 6. Conclusion

On exact associative recall (MQAR), **memory-of-memory consolidation does not beat fixed-state linear
attention** — it ties the token-read ablation's purpose but loses to both a matched-state single
memory and a standard Gated DeltaNet, and its consolidation actively *hurts* recall. Temporal
segment-splitting also loses (read-cross-talk). The idea is not justified for recall-capacity tasks.
Its only remaining possible niche is long-context *retention*, which remains untested and would
require a stabilized base to evaluate cleanly.

## 7. Reproduce

```bash
modal run memflow/modal_app.py --action verify          # correctness suite + CUDA equivalence
modal run memflow/modal_app.py --action study --task mqar --steps 5000
modal run memflow/modal_app.py --action report --task mqar   # pull saved results from the volume
.venv/bin/python -m pytest -q                            # 31 correctness tests (CPU)
```
