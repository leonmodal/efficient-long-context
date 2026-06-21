# NMT Prototype — Experiment Log

Code: `nmt/nmt_modal.py` · Run: `modal run nmt/nmt_modal.py` · GPU: H200 (synthetic, pure torch, **no training**).

## Memory matrix sizes (reference)

Memory matrix = recurrent state per head `[d_k, d_v]` × `num_heads`. Repo "mid" config: `head_k=192, head_v=384, num_heads=6, layers=21, hidden=1536`.

| | per-head state | levels | total / layer | bf16 |
|---|---|---|---|---|
| **DeltaNet** (what we use) | `[192, 384]` | 1 | 6×192×384 = 442K | **0.84 MB** |
| **Log-Linear** | `[192, 384]` | L≈log₂T (≤15) | 6.6M | ~13 MB |

Log-Linear = DeltaNet × L Fenwick levels. Papers: DeltaNet [2406.06484] (d_k=d_v=d_model, 340M/1.3B); Log-Linear [2506.04761] (Gated DeltaNet + O(log T) states).

## Run @ real per-head dims (d_k=192, d_v=384, 512 leaves, beam=8) — overparametrized

**[1] Retrieval** — beam tree vs flat exact:
- tree hit@4 = **0.94**, flat recall = 1.00, 136/512 nodes scored. (At toy d=128 this was 0.81 → real dims already close most of the gap.)
- beam ablation: 16→1.00, so remaining misses are recoverable by mild beam widening.

**[2] Read sanity** — needle as most-recent write: read cosine **1.000** (DeltaNet and GDA). Delta rule recovers a recent write exactly.

**[3] Retention** — DeltaNet vs GDA as an in-leaf fact ages (16 writes/leaf, measuring recovered value's cosine, magnitude):

| gate | oldest | middle | newest |
|---|---|---|---|
| 1.0 (DeltaNet) | (0.96, **0.93**) | (0.98, 0.97) | (1.0, 1.0) |
| 0.98 (GDA) | (0.94, **0.68**) | (0.98, 0.84) | (1.0, 1.0) |
| 0.9 (GDA) | (0.77, **0.19**) | (0.96, 0.46) | (1.0, 1.0) |

→ DeltaNet retains old in-leaf content; GDA erodes it with age. **Confirms: no forget gate for leaves.**

**[4] Routing-key comparison** @ fixed beam=8 (key quality, not widened search):

| scenario | single_key | multi_mean | multi_kmeans |
|---|---|---|---|
| needle (buried fact) | **0.62** | **1.00** | **1.00** |
| theme (chunk gist) | 1.00 | 1.00 | 1.00 |

(all at 136 nodes scored — same cost.)

## Takeaways

1. **Overparametrized works** end-to-end at real per-head dims: self-test 1.0, retrieval 0.94, read 1.0, retention strong.
2. **The lever is #keys, not the pooling method.** single→multi (1→8 keys) fixes buried-fact retrieval (0.62→1.00). **multi_mean and multi_kmeans are identical (both 1.00)** — k-means gives **no** benefit here. → Use 8 keys + plain mean-by-group; **drop k-means** unless recall degrades at much larger scale.
3. **A single key is only lossy for buried distinctive facts.** For coherent/gist retrieval, 1 key is perfect (1.00). (This is why "mean doesn't work" was too strong — it only fails for buried items.)
4. Caveat: at d=192 the synthetic task saturates at 1.0, so it can't separate mean vs k-means; their tie may not hold at million-leaf depth. Revisit only if needed.

## Training run 1 — associative recall, end-to-end (`nmt_train_modal.py`, H200)

Learned: `W_k` (routing/payload keys), `val_embed`, an **untied** readout head, routing temperature `tau`. Trained **dense (GRM** over all 16 leaves) on FRESH random (key→value-class) sets each step — so the weights learn the *mechanism* (write/route/read), not facts. 256 classes, 256 pairs/example, 16 leaves.

Learning curve (chance = 0.0039 = 1/256):

| step | loss | mem_acc | no-mem control |
|---|---|---|---|
| 0 | 5.545 (=ln 256) | 0.006 (chance) | 0.002 |
| 150 | 0.043 | **1.000** | 0.000 |
| 3000 | 0.000 | 1.000 | 0.004 |

Final eval (fresh data):

| readout | acc |
|---|---|
| dense (all 16 leaves, GRM) | **1.000** |
| sparse top-4 (SSC) | **1.000** |
| sparse top-1 | **1.000** |
| no-memory control | 0.004 (chance) |

Takeaways:
1. **NMT memory is trainable end-to-end** — genuine curve from chance → perfect recall (~150 steps).
2. **No-memory control stays at chance** → the memory is doing the work (the query key carries zero info about its value; only the per-example memory has it).
3. **Train-dense / infer-sparse confirmed**: trained with dense GRM, but sparse top-1/top-4 at inference = dense = 1.000.
4. *Note (first version had a tied readout → solved at init, no learning; untying the readout exposed the real curve.)*

Caveat: **easy regime** — converges by ~150 steps, exact-key queries, top-1 already perfect (routing is trivial when the query equals a stored key). It proves trainability + necessity-of-memory + dense→sparse, but doesn't yet stress routing or the hierarchy.

## Training run 2 — MQAR, like the Memory Caching paper (`nmt_mqar_modal.py`, H200)

Multi-Query Associative Recall (Zoology), the paper's synthetic recall benchmark. Trained from scratch, fixed small dim **d=64**, **8 cached segments**, sweeping #key-value pairs N. Three readers: baseline (one fixed memory, capacity ~d), GRM (dense over 8 segments), SSC (top-2 sparse = our NMT read).

| reader | N=16 | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|---|
| baseline (1 fixed mem) | 1.00 | 1.00 | 0.998 | 0.911 | **0.489** |
| + GRM (dense) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| + SSC (top-2, our NMT read) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

Reproduces the paper's core finding:
1. **Fixed-state RNN saturates** once #pairs > capacity (~d=64): 0.91 @ N=128, **0.49 @ N=256** (rank-d interference).
2. **Memory caching restores recall** — GRM holds 1.00 to N=256 (8 segments → ~8× effective capacity).
3. **SSC (sparse top-2 = our NMT read) == GRM (dense)** — train-dense/infer-sparse holds, and sparse routing finds the right segment.

Caveat: MC has its own ceiling (~8×d here); push N past ~512 and even MC saturates → that's where *more segments / the hierarchical tree* matter. This is MQAR only, not the full LM/NIAH/LongBench suite (that needs 760M–1.3B on 30–100B FineWeb tokens — days of multi-GPU).

## HOPE experiment — NMT routing vs the Memory Caching baseline (`nmt_hope_modal.py`, H200)

The MC paper routes with ONE mean-pooled key per segment (`r_i=<u, MeanPooling(S_i)>`). NMT keeps a small set of keys per node (8) and max-scores. Task: 16 segments each packed with P distinct (key→value) pairs; query a buried key; sweep P (segment density). No training — pure routing geometry, Monte-Carlo over 256 sequences.

| metric | router | P=8 | P=16 | P=32 | P=64 | P=128 |
|---|---|---|---|---|---|---|
| hit@1 | MC | 0.99 | 0.84 | 0.61 | 0.35 | 0.26 |
| hit@1 | **NMT** | 1.00 | **1.00** | **1.00** | **0.94** | **0.59** |
| recall@1 | MC | 1.00 | 0.93 | 0.75 | 0.50 | 0.38 |
| recall@1 | **NMT** | 1.00 | **1.00** | **1.00** | **0.98** | **0.74** |

**Result:** sparse segments (P=8) → tie. As segments get DENSE, MC's mean-pool routing collapses (hit@1 0.84→0.26) while NMT holds (1.0 until P=64). At P=32–64, NMT is **~2–3× MC** on recall. This is the regime that matters: the paper's real segments are 256 tokens (P large → many facts each), so MC's single-key routing should blur buried facts exactly as shown → predicts NMT > MC on real long-context recall.

Mechanism: it's the **multi-key vs single-mean-pool routing key** (the tree enables multi-key at every node). Honest caveats: synthetic, no training, exact-key queries; NMT uses 8× the routing-key storage (tiny vs payload); the *hierarchy* itself (log-time) isn't tested here, only multi-key routing.

## HIERARCHY experiment — NMT beam-tree vs flat at scale (`nmt_hierarchy_modal.py`, H200) — NEGATIVE

Sparse leaves (8 clean keys each), grow #leaves L; branch-8 tree, beam 16. Routing hit@4 + nodes scored:

| L (leaves) | 64 | 512 | 4096 | 32768 | (depth) |
|---|---|---|---|---|---|
| flat-MC (1 mean key, scan all) | 0.99 | 0.96 | 0.87 | 0.67 | — |
| flat-exact (8 keys, scan all) | 1.00 | 1.00 | 1.00 | 1.00 | — |
| **NMT tree** (beam, ~log L) | 1.00 | 0.998 | **0.47** | **0.073** | 2/3/4/5 |
| tree nodes scored | 72 | 200 | 328 | 456 | (vs L) |

**The naive tree does NOT scale.** Cost win is huge (456 vs 32768 nodes, 72×) but recall **collapses to 7%** at depth 5 — *worse than flat-MC*. Cause: mean-by-group parent keys blur the needle by ~1/√branch **per level**, so by depth 5 the signal at the root is ~`0.35⁵≈0.005` → the true branch is pruned at the TOP level → never reached. This is a **temporal (arrival-order) tree**: each node summarizes *unrelated* leaves, so its pooled key is uninformative. Confirms the day-one "temporal tree vs content routing" risk.

**Fix (the real next experiment):** a **content-organized** hierarchy — cluster the leaf keys *into* the tree (recursive k-means / ANN-style, à la HNSW/IVF/ScaNN) so each subtree is a content cluster and routing follows meaning. That's how billion-scale vector search routes at log cost. The temporal tree was the wrong structure.

## Next
- **Content-clustered tree** (recursive k-means over leaves) re-tested at scale — make-or-break for the hierarchy claim.
- **Translate the dense-segment win to a TRAINED comparison** (MC-routing vs multi-key NMT-routing on MQAR) — that win is solid and worth banking.
- **S-NIAH passkey** (the paper's long-context retrieval eval) — synthetic, feasible at small scale.
- **Streaming/causal** (grow tree, past-only reads) — designed (Fenwick), not yet in code.
- **Scaled LM** (FineWeb, small backbone + memory + next-token loss) — bigger lift; possible on Modal multi-GPU if we want the full eval suite.
