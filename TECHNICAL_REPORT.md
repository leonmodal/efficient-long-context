# Efficient Long-Context Modeling — Technical Report

Softmax attention is near-lossless at recall but pays a **growing KV cache**: O(T) memory and O(T)
work per token, O(T²) prefill — the wall at 128K–1M context. The alternative is to replace the growing
cache with a **fixed-size memory updated online** as the sequence streams by — equivalently, a weight
matrix trained online (linear attention, Mamba, DeltaNet, Gated DeltaNet are all instances; the delta
rule is one SGD step on `½‖Sk−v‖²`, GDN adds decay).

This report covers three separate projects, each attacking efficient long-context modeling from a
different angle:

1. **Gated TTT** — put the memory in the backbone's own weights, written with a self-correcting rule.
2. **Neural Memory Tree** — scale associative memory with a routable tree of memories.
3. **Memory Flow** — a multi-timescale cascade where slow memories summarize faster ones.

---

# Project 1 — Gated TTT

**Goal.** Long-context as *continual learning* rather than a new architecture: keep attention for local
precision, and carry global context in fast weights that adapt at test time — without adding any module.

**Method.** Build on **In-Place TTT** (ICLR 2026 Oral, arXiv:2604.06169): a Transformer's gated-MLP is
already a key→value store, so treat its down-projection **`W_down` as fast weights**, updated in place
at inference (slow weights `W_up, W_gate` frozen, attention untouched). The key is `z` (the post-gating
MLP activation); the value `v̂` is an **LM-aligned target** (causal Conv1D over future-token embeddings
— store "what predicts next," not "reconstruct current"); the read is the ordinary `W_down z`. Updates
happen once per **large chunk** (512–1024 tokens), because attention does the fine-grained token mixing
and the fast weights only need to carry slowly-accumulating context.

**The Gated-TTT contribution (the write rule):**
- **Baseline (additive, from In-Place TTT):** `W ← W + η·V̂ᵀZ` — a Hebbian / linear-attention write.
  State-independent ⇒ collapses to a **prefix-sum scan**, so it's context-parallel-native and free to
  parallelize. Weakness: no interference management — similar keys with different values accumulate.
- **Gated Delta (ours):** transplant Gated DeltaNet into the weights —
  `ΔW ← ρ·ΔW + η·(V̂ − Z(W₀+ρΔW)ᵀ)ᵀZ`. Decay applies to the *fast* state `ΔW` only (never `W₀`, so
  context can't erode pretrained knowledge); read what memory predicts, write the *correction*. This
  buys self-correction at the cost of parallelism: the update is state-dependent, so the scan becomes a
  short sequential chain over `T/C` chunks (cheap at C=512–4096). A `fair_baselines` harness compares
  the rules apples-to-apples — same memory/keys/values/chunks, only the write rule changes.

**Results (In-Place TTT base, published):**

| Setup | Metric | Baseline | + In-Place TTT |
|---|---|--:|--:|
| Qwen3-4B (continual) | RULER 64K | 74.3 | **78.7** (+4.4) |
| Qwen3-4B | RULER 128K | 74.8 | 77.0 (+2.2) |
| Qwen3-4B | RULER 256K (> train len) | 41.7 | 43.9 (+2.2) |
| LLaMA-3.1-8B | RULER 64K | — | +2.1 |
| Qwen3-14B | RULER 64K | — | +2.7 |
| from scratch 0.5–1.5B, 32K | sliding-window ppl | SWA/GLA/DeltaNet/LaCT | **lowest at every length** |

Ablations: more TTT layers (state) → monotonically better; chunk 512–1024 optimal; the future-info
Conv1D target is essential at long context. The **additive vs Gated-Delta write-rule** head-to-head is
the open experiment (`fair_baselines`, all else held fixed).

**Status.** The additive In-Place TTT is the strong, validated result. The **Gated-Delta** write rule
is the active contribution (derivation in `TTT/report.md`, framework in `TTT/weights-as-memory.md §11`,
code as the `gated_delta` rule + `fair_baselines`); its head-to-head numbers are the open experiment.

---

# Project 2 — Neural Memory Tree (NMT)

**Goal.** Beat the capacity limit of a single fixed memory (only ~`d` facts before retrieval
interference) by scaling to **many** memories and routing to the right one in ~log time — the way
billion-scale vector search does.

**Method.** A tree of leaves, each a DeltaNet matrix-memory holding a span's facts, with small sets of
routing keys per node; a query beam-searches top-down to the relevant leaf. All experiments are
**synthetic, pure-torch, mostly no-training** (H200) — they probe memory *geometry and capacity*, not
learned behavior.

**Findings — positive.** *No forget gate for leaves* — DeltaNet retains aged in-leaf facts while gated
delta (GDA) erodes them (oldest-fact cosine / magnitude):

| gate | oldest | middle | newest |
|---|--:|--:|--:|
| DeltaNet (α=1) | 0.96 / **0.93** | 0.98 / 0.97 | 1.0 / 1.0 |
| GDA (α=0.9) | 0.77 / **0.19** | 0.96 / 0.46 | 1.0 / 1.0 |

*The lever is #keys, not pooling* (buried-fact retrieval, fixed beam): single-key **0.62** → 8 keys
**1.00**; `multi_mean == multi_kmeans` (k-means adds nothing). Beam-tree retrieval hit@4 = **0.94** (flat
exact 1.00).

*MQAR reproduces "memory caching"* — one fixed memory saturates past capacity (~d=64); caching restores
it; sparse == dense (accuracy):

| reader | N=64 | N=128 | N=256 |
|---|--:|--:|--:|
| fixed 1-memory | 1.00 | 0.91 | **0.49** |
| + caching (dense, GRM) | 1.00 | 1.00 | **1.00** |
| + sparse top-2 (SSC) | 1.00 | 1.00 | **1.00** |

*Routing geometry (HOPE)* — as segments densify, single-mean-pool routing collapses while multi-key
holds (hit@1):

| router | P=16 | P=32 | P=64 | P=128 |
|---|--:|--:|--:|--:|
| Memory-Caching (mean-pool) | 0.84 | 0.61 | 0.35 | 0.26 |
| **NMT (multi-key)** | **1.00** | **1.00** | **0.94** | **0.59** |

**Finding — negative (the make-or-break): the naive hierarchy does NOT scale** (routing hit@4 vs #leaves):

| #leaves L | 512 | 4096 | 32768 |
|---|--:|--:|--:|
| flat-exact (8 keys, scan all) | 1.00 | 1.00 | 1.00 |
| **NMT beam tree** (~log L) | 0.998 | **0.47** | **0.073** |

A temporal (arrival-order) tree with mean-pooled parent keys collapses to **0.073 at depth 5** — worse
than flat — despite scoring 72× fewer nodes: mean-pooling *unrelated* leaves blurs the needle ~1/√branch
**per level**, so the correct branch is pruned at the top. **Fix (untested): a content-organized tree**
(recursive k-means / ANN-style) so each subtree is a content cluster and routing follows meaning.

**Status.** Multi-key routing and memory-caching wins are solid (synthetic). The hierarchy currently
fails; content-clustering is the proposed fix. No real LM / NIAH yet.

---

# Project 3 — Memory Flow (MemFlow)

**Goal.** Cover long context with O(log T)-style **multi-timescale** memory, where slower buckets hold
coarser/older information — and write those slow buckets from **readouts of the faster bucket**
("memory of memory") so each level summarizes processed memory rather than raw tokens.

**Method.** N gated-delta buckets at periods `(1, P₁, P₂, …)`. The fast bucket (period 1) is ordinary
linear attention. Each slow bucket, every `Pℓ` tokens, does **Option-C consolidation**: probe the
faster bucket, aggregate `(token-key, fast-bucket-readout)` with a salience-weighted running softmax,
and do **one** gated-delta write. Readout queries every live bucket, gates, sums, norms. (Phase 3 adds
Fenwick-scheduled growth for genuinely O(log T) levels — scheduler + coverage invariant built; growing
layer left as future work.)

**Built & verified** (31-test suite, CPU f64 + CUDA f32/bf16 on Modal): gated-delta rule (recurrent +
numerically-stable channel-wise chunkwise, equal to **3e-7**), N-bucket Option-C (online running-max ==
batch softmax), masking, detach warmup, Fenwick scheduler + O(log T) coverage; FLA Gated DeltaNet
baseline; segment-split baseline; Modal harness with per-run result persistence.

**Result — MQAR, apples-to-apples, GDN base, matched state (mean of 2 seeds, chance ≈ 0.004):**

| N | metric @ D=64 | memory | split | single (ours) | GDN |
|--:|:--|:--:|:--:|:--:|:--:|
| 2 | accuracy | **0.02** | 0.36 | 0.66 | 0.98 |
| 4 | accuracy | **0.25** | 0.19 | 0.91 | 0.99 |

- **Memory-of-memory loses decisively.** Adding the slow bucket to `single` *lowers* recall
  (0.66→0.02, 0.91→0.25): the salience-averaged summaries inject noise. `memory` < `single` in **every
  seed**, and the two are identical code except the slow bucket — a clean controlled test.
- It does beat the weaker "read tokens at a slow rate" ablation, so the mechanism does *something* —
  just not enough to justify the state it consumes.

**Why the verdict is trustworthy (the instability detour).** `single` first looked unstable (0.0↔0.9 by
seed) vs GDN. We ruled out the kernel (correct to 5e-7 even under aggressive decay) and traced it to
**initialization**: a sigmoid gate *inside* the output norm (vs FLA's `rmsnorm(o)·silu(g)`), no prenorm
residual rescaling, and a low β-bias — plus an earlier confound, channel-wise (KDA-style) decay vs GDN's
scalar, which corrupts stored values. After matching all of these, `single` became **stable and tracked
GDN**, confirming the gap was implementation, not the idea — so the negative `memory` result sits on a
faithful, apples-to-apples base.

**Status.** Negative on exact recall (`memflow/report.md`). **Not tested:** long-context **NIAH/RULER**
(the regime it was designed for — harness built), real-data LM, throughput. Since the failure mode is
lossy averaging — exactly what long-context retention must avoid — NIAH is the only door left, and the
prior is not encouraging.

---

# Reflection

Going in, the shared bet across all three projects was that a cleverer way of *organizing* a fixed-size
memory — self-correcting writes, a routing tree, a timescale cascade — could narrow the long-context gap
with attention. What we actually learned is more sobering, and more useful, than that bet.

The lesson that recurred, and that I'd weight most heavily next time, is that **compression is not free,
and averaging is the wrong kind of compression**. Twice — the Neural Memory Tree's mean-pooled parents
and Memory Flow's salience-averaged consolidation — we built a summary *on top of* memories and watched
recall collapse the same way: the one fact that matters gets averaged into the many that don't. The
delta rule stores associations losslessly up to capacity; a mean does not. We had already written this
down in `nmt/EXPERIMENTS.md` (0.07 recall at depth 5) and should have taken our own warning more
seriously before building Memory Flow on the same primitive. If there's a single transferable rule, it's
*prefer mechanisms that add state and correct (delta/GDN, more memories) over mechanisms that summarize
(pooling, averaging)*.

The second realization is that **capacity is the real currency, and the boring levers move it.** What
worked was simply having more of the resource — more cached segments, more routing keys per node;
k-means-over-mean didn't matter, and splitting a fixed budget into a summary hierarchy actively *spent*
capacity for no recall gain. It's telling that the one validated winner, Gated TTT, doesn't compress
harder — it finds a *bigger* memory that was already sitting in the backbone (the whole `W_down`).

Third, **the baseline was humbling, and the methodology mattered more than any single result.** A
properly-tuned Gated DeltaNet beat everything we hand-built on recall, and a good chunk of our early
"memory-flow is worse" signal turned out to be our own initialization bug, not the idea. Refusing to
believe any number until `single == gdn` — ruling out the kernel, matching decay, readout, and init —
was the most valuable habit of the project. "Ours vs the library" gaps are implementation until proven
otherwise.

Finally, **we kept stopping at the doorstep of the actual question.** Long-context retrieval (NIAH/RULER)
is where every one of these ideas is supposed to pay off, and only Gated TTT actually ran it (and won,
+4.4 RULER@64K). The Neural Memory Tree and Memory Flow both delivered their verdicts on short synthetic
recall — the regime that *least* favors them — and even strong TTT methods show NIAH recall can collapse
under compression, a reminder that long-context *loss* improving is not the same as needle retrieval. The
next dollar of compute belongs on the long-context evals, not on more short-context tuning.

**Where that leaves the three:** Gated TTT is the validated direction. The Neural Memory Tree has a real
routing win wrapped around a failed naive hierarchy — with a concrete, untested fix (content-clustering).
Memory Flow's "memory of memory" does not earn its keep on recall — a clean, trustworthy negative result,
with long-context retention the one door we never opened.

---

# Deliverables & reproduction

- **Gated TTT** (`TTT/`) — `report.md` (gated-delta-in-place derivation), `weights-as-memory.md` (field
  survey + method + the gated-delta program), `fair_baselines/`, `In-Place-TTT/`, `e2e-ttt-paper/`.
- **Neural Memory Tree** (`nmt/`) — `EXPERIMENTS.md` (full log); `modal run nmt/nmt_{modal,mqar_modal,
  hope_modal,hierarchy_modal}.py` (H200, synthetic).
- **Memory Flow** (`memflow/`) — full package + `report.md`; `modal run memflow/modal_app.py
  --action verify` (correctness + CUDA equivalence), `--action study --task mqar` (the comparison),
  `--action report` (pull saved results from the Modal volume); `.venv/bin/python -m pytest -q` (31 tests).
