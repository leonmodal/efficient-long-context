# Efficient Long-Context Modeling Report

Softmax attention recalls almost losslessly, but it pays a growing KV cache: O(T) memory and O(T) work
per token, O(T squared) prefill. That is the wall at 128K to 1M context. The alternative is to replace
the growing cache with a fixed-size memory that gets updated online as the sequence streams by. That
fixed-size memory is, in effect, a weight matrix trained online: linear attention, Mamba, DeltaNet, and
Gated DeltaNet are all instances of this, where the delta rule is one SGD step on the memory loss
`½‖Sk - v‖²` and Gated DeltaNet adds decay.

This report covers three separate projects, each going after efficient long context from a different
angle:

1. Gated TTT: put the memory in the backbone's own weights, written with a self-correcting rule.
2. Neural Memory Tree: scale associative memory with a routable tree of memories.
3. Memory Flow: a multi-timescale cascade where slow memories summarize faster ones.

---

# Project 1: Gated TTT

**Goal.** Treat long context as continual learning rather than a new architecture. Keep attention for
local precision, and carry global context in fast weights that adapt at test time, without adding any
new module.

**Method.** This builds on In-Place TTT (ICLR 2026 Oral, arXiv:2604.06169). A Transformer's gated MLP is
already a key/value store, so it treats the down-projection `W_down` as fast weights and updates it in
place at inference, with the up and gate projections frozen and attention left untouched. The key is `z`
(the post-gating MLP activation). The value `v̂` is an LM-aligned target: a causal Conv1D over
future-token embeddings, so the write stores "what predicts next" rather than "reconstruct current." The
read is the ordinary `W_down z`. Updates happen once per large chunk (512 to 1024 tokens), because
attention handles the fine-grained token mixing and the fast weights only need to carry slowly
accumulating context.

**The Gated TTT direction (the write rule).** There are two write rules to compare:

- Additive (the In-Place TTT baseline): `W ← W + η·V̂ᵀZ`, a Hebbian linear-attention write. The delta is
  state-independent, so it collapses to a prefix-sum scan, which is context-parallel-native and free to
  parallelize. Its weakness is no interference management: similar keys with different values accumulate.
- Gated Delta (the proposed direction): transplant Gated DeltaNet into the weights,
  `ΔW ← ρ·ΔW + η·(V̂ - Z(W₀+ρΔW)ᵀ)ᵀZ`. Decay applies to the fast state `ΔW` only, never to `W₀`, so
  forgetting context cannot erode pretrained knowledge. It reads what the memory predicts and writes
  only the correction. This buys self-correction at the cost of parallelism: the update is
  state-dependent, so the scan turns into a short sequential chain over `T/C` chunks (cheap when C is
  512 to 4096). A `fair_baselines` harness compares the two rules with everything else held fixed (same
  memory, keys, values, chunk schedule), so only the write rule changes.

**Results (In-Place TTT base, published numbers):**

| Setup | Metric | Baseline | With In-Place TTT |
|---|---|--:|--:|
| Qwen3-4B (continual) | RULER 64K | 74.3 | 78.7 (+4.4) |
| Qwen3-4B | RULER 128K | 74.8 | 77.0 (+2.2) |
| Qwen3-4B | RULER 256K (past train length) | 41.7 | 43.9 (+2.2) |
| LLaMA-3.1-8B | RULER 64K | n/a | +2.1 |
| Qwen3-14B | RULER 64K | n/a | +2.7 |
| from scratch 0.5B to 1.5B, 32K | sliding-window ppl | SWA, GLA, DeltaNet, LaCT | lowest at every length |

Ablations: more TTT layers (more state) helps monotonically; chunk 512 to 1024 is the sweet spot; the
future-info Conv1D target matters at long context. The additive versus Gated-Delta write-rule head to
head is the open experiment.

**Status.** The additive In-Place TTT is the strong, published result. The Gated-Delta write rule is the
active direction: the derivation is in `TTT/report.md`, the framework in `TTT/weights-as-memory.md`
section 11, and the code ships it as the `gated_delta` rule plus `fair_baselines`. Its head-to-head
numbers are not in yet.

---

# Project 2: Neural Memory Tree (NMT)

**Goal.** Get past the capacity limit of a single fixed memory (only about `d` facts before retrieval
interference) by scaling to many memories and routing to the right one in roughly log time, the way
billion-scale vector search does.

**Method.** A tree of leaves, each a DeltaNet matrix-memory holding a span's facts, with a small set of
routing keys per node. A query beam-searches top down to the relevant leaf. The runs are synthetic,
pure torch, and mostly without training (H200), so they probe memory geometry and capacity, not learned
behavior.

**Positive findings.** No forget gate for leaves: DeltaNet retains aged in-leaf facts, while gated delta
(GDA) erodes them (oldest-fact cosine and magnitude):

| gate | oldest | middle | newest |
|---|--:|--:|--:|
| DeltaNet (α=1) | 0.96 / 0.93 | 0.98 / 0.97 | 1.0 / 1.0 |
| GDA (α=0.9) | 0.77 / 0.19 | 0.96 / 0.46 | 1.0 / 1.0 |

The lever is the number of keys, not the pooling method (buried-fact retrieval, fixed beam): single key
0.62, eight keys 1.00, and `multi_mean == multi_kmeans` (k-means adds nothing). Beam-tree retrieval
hit@4 is 0.94 (flat exact is 1.00).

MQAR reproduces the "memory caching" result. One fixed memory saturates past its capacity (about d=64),
caching restores it, and sparse matches dense (accuracy):

| reader | N=64 | N=128 | N=256 |
|---|--:|--:|--:|
| fixed single memory | 1.00 | 0.91 | 0.49 |
| with caching (dense, GRM) | 1.00 | 1.00 | 1.00 |
| with sparse top-2 (SSC) | 1.00 | 1.00 | 1.00 |

Routing geometry (HOPE): as segments get dense, single-mean-pool routing collapses while multi-key
routing holds (hit@1):

| router | P=16 | P=32 | P=64 | P=128 |
|---|--:|--:|--:|--:|
| Memory-Caching (mean-pool) | 0.84 | 0.61 | 0.35 | 0.26 |
| NMT (multi-key) | 1.00 | 1.00 | 0.94 | 0.59 |

**Negative finding (the make-or-break): the naive hierarchy does not scale** (routing hit@4 versus the
number of leaves):

| number of leaves | 512 | 4096 | 32768 |
|---|--:|--:|--:|
| flat exact (8 keys, scan all) | 1.00 | 1.00 | 1.00 |
| NMT beam tree (about log L) | 0.998 | 0.47 | 0.073 |

A temporal (arrival-order) tree with mean-pooled parent keys collapses to 0.073 at depth 5, worse than
flat, even though it scores 72 times fewer nodes. The cause is that mean-pooling unrelated leaves blurs
the needle by about 1/sqrt(branch) per level, so the correct branch gets pruned at the top. The proposed
fix, which is untested, is a content-organized tree (recursive k-means or ANN-style) so each subtree is
a content cluster and routing follows meaning.

**Status.** The multi-key routing and memory-caching wins are solid on synthetic tasks. The hierarchy
currently fails, and content-clustering is the proposed fix. No real LM or NIAH yet.

---

# Project 3: Memory Flow (MemFlow)

This is the project I built and evaluated in this session.

**Goal.** Cover long context with multi-timescale memory, where slower buckets hold coarser, older
information. The idea I wanted to test is writing those slow buckets from readouts of the faster bucket
("memory of memory"), so each level summarizes processed memory rather than raw tokens.

**Method.** N gated-delta buckets at periods (1, P1, P2, and so on). The fast bucket (period 1) is
ordinary linear attention. Every P_l tokens, each slow bucket does an Option-C consolidation: probe the
faster bucket, aggregate (token-key, fast-bucket-readout) with a salience-weighted running softmax, and
do one gated-delta write. The readout queries every live bucket, gates, sums, and norms. Phase 3 adds a
Fenwick-scheduled growth so the number of levels grows like log T. I built and tested that scheduler and
its coverage invariant, but I left the growing layer itself as future work.

**What I built and verified** (31-test suite, CPU float64 plus CUDA float32/bf16 on Modal): the
gated-delta rule (a recurrent reference and a numerically-stable channel-wise chunkwise form, equal to
3e-7), the N-bucket Option-C consolidation (online running-max equals batch softmax), masking, the
detach warmup, and the Fenwick scheduler with its O(log T) coverage invariant. I also wired an FLA Gated
DeltaNet baseline, a segment-split baseline, and a Modal harness that persists every run's result so
nothing is lost on a failure.

**Result on MQAR, apples to apples on a Gated DeltaNet base, matched state (mean of 2 seeds, chance about
0.004):**

| N | metric at D=64 | memory | split | single (ours) | GDN |
|--:|:--|:--:|:--:|:--:|:--:|
| 2 | accuracy | 0.02 | 0.36 | 0.66 | 0.98 |
| 4 | accuracy | 0.25 | 0.19 | 0.91 | 0.99 |

Memory-of-memory loses clearly. Adding the slow bucket to `single` lowers recall (0.66 down to 0.02 at
N=2, 0.91 down to 0.25 at N=4): the salience-averaged summaries inject noise into the readout. `memory`
is below `single` in every seed, and the two are the same code apart from the slow bucket, so this is a
clean controlled test. It does beat the weaker "read tokens at a slow rate" ablation, so the mechanism
does something, just not enough to justify the state it uses.

**Why I trust the verdict (the instability I had to chase down).** `single` first looked unstable (it
swung between roughly 0.0 and 0.9 across seeds) while GDN did not. I ruled out the kernel first: I
checked that the chunkwise matches the recurrent reference to 3e-7 even under aggressive decay, so it
was not a correctness bug. I traced the instability to initialization: I had a sigmoid gate inside the
output norm (FLA uses rmsnorm(o) times silu(g)), no prenorm residual rescaling, and a low beta bias. I
had also started with channel-wise (KDA-style) decay instead of GDN's scalar decay, which corrupts
stored values. After I matched all of these (the readout, beta bias set to 0, prenorm rescaling on
`o_proj` and the MLP down by 1/sqrt(2L)), `single` became stable and tracked GDN. That confirmed the gap
was my implementation, not the idea, and it put the negative `memory` result on a faithful base.

**Status.** Negative on exact recall (full writeup in `memflow/report.md`). I did not test long-context
NIAH or RULER (the regime it was designed for, although I built the harness), real-data LM, or
throughput. The failure mode is lossy averaging, which is exactly what long-context retention also has
to avoid, so NIAH is the one door I never opened and I would not expect a surprise there.

---

# Reflection

Going in, the bet across all three projects was that a smarter way of organizing a fixed memory (a
self-correcting write, a routing tree, a timescale cascade) could narrow the long-context gap with
attention. What I actually learned is more sobering, and more useful, than that bet.

The lesson that came up twice, and the one I would weight most heavily next time, is that compression is
not free, and averaging is the wrong kind of compression. The Neural Memory Tree's mean-pooled parents
and Memory Flow's salience-averaged consolidation both built a summary on top of memories and watched
recall collapse the same way: the one fact that matters gets averaged into the many that do not. The
delta rule stores associations losslessly up to capacity; a mean does not. The warning was already
written down in `nmt/EXPERIMENTS.md` (0.07 recall at depth 5), and I should have taken it more seriously
before building Memory Flow on the same primitive. If there is one rule worth carrying forward, it is to
prefer mechanisms that add state and correct (the delta rule, more memories) over mechanisms that
summarize (pooling, averaging).

The second thing is that capacity is the real currency, and the boring levers move it. What worked was
simply having more of the resource: more cached segments, more routing keys per node. K-means over mean
did not matter, and splitting a fixed budget into a summary hierarchy actually spent capacity for no
recall gain. It is worth noting that the one validated winner, Gated TTT, does not compress harder. It
finds a bigger memory that was already sitting in the backbone (the whole `W_down`).

The third thing is that the baseline was humbling, and the method mattered more than any single number.
A properly tuned Gated DeltaNet beat everything I hand-built on recall, and a good chunk of my early
"memory flow is worse" signal turned out to be my own init bug, not the idea. Refusing to believe any
number until `single` matched GDN (ruling out the kernel, matching the decay, the readout, and the init)
was the most valuable habit in the project. An "ours versus the library" gap is implementation until
proven otherwise.

Finally, I kept stopping at the doorstep of the actual question. Long-context retrieval (NIAH or RULER)
is where every one of these ideas is supposed to pay off, and only Gated TTT actually ran it and won
(+4.4 RULER at 64K). The Neural Memory Tree and Memory Flow both gave their verdicts on short synthetic
recall, the regime that least favors them. Even strong TTT methods show NIAH recall can collapse under
compression, which is a reminder that a better long-context loss is not the same as needle retrieval.
The next round of compute belongs on the long-context evals, not on more short-context tuning.

Where that leaves the three: Gated TTT is the validated direction. The Neural Memory Tree has a real
routing win around a failed naive hierarchy, with a concrete and untested fix (content-clustering).
Memory Flow's "memory of memory" does not earn its keep on recall. That is a clean, trustworthy negative
result, with long-context retention the one door I never opened.

---

# Deliverables and reproduction

- Gated TTT (`TTT/`): `report.md` (the gated-delta in-place derivation), `weights-as-memory.md` (the
  field survey, the method, and the gated-delta program), plus `fair_baselines/`, `In-Place-TTT/`, and
  `e2e-ttt-paper/`.
- Neural Memory Tree (`nmt/`): `EXPERIMENTS.md` (the full log), with `modal run nmt/nmt_modal.py` and the
  `nmt_mqar_modal.py`, `nmt_hope_modal.py`, `nmt_hierarchy_modal.py` variants (H200, synthetic).
- Memory Flow (`memflow/`): the full package plus `report.md`. Run `modal run memflow/modal_app.py
  --action verify` for the correctness suite and CUDA equivalence, `--action study --task mqar` for the
  comparison, and `--action report` to pull saved results from the Modal volume. Run
  `.venv/bin/python -m pytest -q` for the 31 tests.
