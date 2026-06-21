Yes. In the notation we’ve been using, **(Z) comes from the normal Transformer MLP block**. It is not a new attention key, not a new module, and not a special learned memory vector.

For a LLaMA/Qwen-style gated MLP, you usually have something like:

[
Z = \mathrm{SiLU}(XW_{\text{gate}}^\top)\odot (XW_{\text{up}}^\top)
]

[
O = ZW_{\text{down}}^\top
]

where:

* (X\in\mathbb{R}^{L\times d_{\text{model}}}) is the hidden state entering the MLP;
* (W_{\text{up}}, W_{\text{gate}}) map from (d_{\text{model}}) to (d_{\text{ff}});
* (Z\in\mathbb{R}^{L\times d_{\text{ff}}}) is the **intermediate MLP activation**;
* (W_{\text{down}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}) maps back to the model dimension;
* (O\in\mathbb{R}^{L\times d_{\text{model}}}) is the MLP output.

So (Z) is simply the vector right before the MLP down-projection. In-Place TTT treats the **final MLP projection** (W_{\text{down}}) as the fast weight, while the input/up/gate projections remain slow weights. The paper describes this as repurposing the existing MLP block instead of adding a new TTT layer, and specifically says the final projection matrix of the MLP block is used as adaptable fast weights. ([arXiv][1])

---

# 1. Standard MLP block

A standard Transformer block has attention and MLP. Ignore attention for a second. The MLP part is usually:

[
X \rightarrow Z \rightarrow O
]

For a gated MLP:

[
U = XW_{\text{up}}^\top
]

[
G = XW_{\text{gate}}^\top
]

[
Z = \phi(G)\odot U
]

[
O = ZW_{\text{down}}^\top
]

where (\phi) is usually SiLU/SwiGLU-style activation.

So (Z) is the MLP’s expanded hidden feature. If (d_{\text{model}}=4096) and (d_{\text{ff}}=11008), then:

[
x_t\in\mathbb{R}^{4096}
]

[
z_t\in\mathbb{R}^{11008}
]

[
W_{\text{down}}\in\mathbb{R}^{4096\times11008}
]

The usual MLP output is:

[
o_t = W_{\text{down}}z_t
]

or in row-vector notation:

[
o_t = z_tW_{\text{down}}^\top
]

That is the object In-Place TTT turns into a fast-weight memory.

---

# 2. In-Place TTT: what it does

The core In-Place TTT move is:

> The MLP down-projection is already a big matrix that maps feature activations (Z) into residual-stream updates. Let it adapt while reading the context.

Normally:

[
W_{\text{down}} = W_0
]

is fixed after training.

In In-Place TTT:

[
W_{\text{down}}^{(j)}
=====================

W_0 + \Delta W_j
]

where (j) indexes chunks.

The model processes a long sequence in chunks:

[
C_1,C_2,\dots,C_J
]

For chunk (j), it computes the normal MLP intermediate activations:

[
Z_j = \mathrm{MLP}_{\text{up/gate}}(X_j)
]

Then applies the current fast weight:

[
O_j = Z_j(W_0+\Delta W_j)^\top
]

Then it computes a target value (\hat V_j), and updates the fast weight for future chunks.

The paper describes this as an **apply-then-update** cycle: for each chunk, the current fast weights process the intermediate activations, then the weights are updated using those activations and target values. ([arXiv][1])

---

# 3. Why (Z) acts like a key

In TTT language, you usually have:

[
\text{key } k_t,\quad \text{value } v_t,\quad \text{fast weights } W_t
]

The fast weights learn an association:

[
k_t \mapsto v_t
]

In In-Place TTT, the role of the key is played by:

[
z_t
]

the MLP intermediate activation.

So:

[
z_t = \text{key}
]

[
\hat v_t = \text{value/target}
]

[
W_{\text{down}} = \text{fast-weight memory}
]

The memory read is:

[
\hat o_t = W_{\text{down}}z_t
]

or row-wise:

[
\hat o_t = z_tW_{\text{down}}^\top
]

So (Z) comes from the existing MLP, but once you view (W_{\text{down}}) as memory, (Z) becomes the **address** used to read/write that memory.

That is why (Z) is important.

---

# 4. Original In-Place TTT update

A simplified additive version is:

[
\Delta W_{j+1}
==============

\Delta W_j
+
\eta \hat V_j^\top Z_j
]

or:

[
W_{j+1}
=======

W_j
+
\eta \hat V_j^\top Z_j
]

where:

[
Z_j\in\mathbb{R}^{C\times d_{\text{ff}}}
]

[
\hat V_j\in\mathbb{R}^{C\times d_{\text{model}}}
]

[
\hat V_j^\top Z_j\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}
]

This is an outer-product memory write.

For a single token:

[
\Delta W_t = \eta \hat v_t z_t^\top
]

So after many previous tokens:

[
W_t = W_0+\eta\sum_{i<t}\hat v_i z_i^\top
]

Now a future token with feature (z_q) reads:

[
\Delta o_q
==========

\Delta W_t z_q
]

# [

\eta\sum_{i<t}\hat v_i(z_i^\top z_q)
]

This is basically **linear attention / associative memory**:

[
\text{output} = \sum_i \text{similarity}(z_q,z_i)\cdot \hat v_i
]

So In-Place TTT makes the MLP down-projection act like a big associative memory, where (Z) is the key space.

---

# 5. Where does (\hat V) come from?

This is the other important piece.

In classic TTT, the target value often reconstructs the current input. In-Place TTT argues that this is misaligned with language modeling. A language model does not ultimately care about reconstructing the current hidden state; it cares about predicting future tokens.

So the paper contrasts:

[
\text{reconstruction target: } \hat v_t = E(x_t)
]

with:

[
\text{LM-aligned target: } \hat v_t = E(x_{t+1})
]

where (E(\cdot)) is a token embedding. The paper’s theoretical section explicitly compares current-token reconstruction targets with next-token LM-aligned targets. ([arXiv][1])

In practice, it is usually not just the raw next-token embedding. The paper says they generate the value with a causal 1D convolution and use causal padding so that each chunk’s update delta contains no future information. ([arXiv][1])

So conceptually:

[
Z_j = \text{MLP feature/key}
]

[
\hat V_j = \text{LM-aligned value to write}
]

[
W_{\text{down}} = \text{fast memory}
]

---

# 6. Why chunking matters

If you updated after every token, it would be expensive.

If you updated once after the whole sequence, it would not help during the sequence and could leak future information.

So In-Place TTT chunks the sequence:

```text
chunk 1: use W0, then update W
chunk 2: use updated W, then update W
chunk 3: use updated W, then update W
...
```

Mathematically:

[
W^{(0)}=W_0
]

[
O_j = Z_j(W^{(j)})^\top
]

[
W^{(j+1)}=W^{(j)}+\eta\hat V_j^\top Z_j
]

Chunk (j)’s update helps chunk (j+1) and later chunks, not itself. This preserves causality.

The paper emphasizes that because In-Place TTT keeps attention intact and adapts only MLP blocks, it can use relatively large chunks, with ablations indicating chunk sizes around 512–1024 work well. ([arXiv][1])

---

# 7. Why the original update is efficient

The additive update:

[
\Delta W_j = \eta\hat V_j^\top Z_j
]

does not depend on the current (W_j).

So you can compute all chunk deltas:

[
\Delta W_1,\Delta W_2,\dots,\Delta W_J
]

then do an exclusive prefix sum:

[
W^{(j)}=W_0+\sum_{i<j}\Delta W_i
]

This gives each chunk the correct causal fast weight, but the computation can be parallelized.

That is the reason In-Place TTT is hardware-friendly. The paper says the implementation uses a parallel scan to process chunks while preserving strict causal semantics, and that the design is compatible with context parallelism. ([arXiv][1])

---

# 8. The weakness of additive In-Place TTT

The additive write is:

[
W_{j+1}=W_j+\eta\hat V_j^\top Z_j
]

It writes the value into memory, but it does not ask:

[
\text{What does the current memory already predict for this key?}
]

So if similar keys appear many times with different values, the memory can accumulate conflicting associations.

Example:

```text
A -> red
...
A -> blue
...
A -> green
```

Additive memory tends to mix:

[
\text{red}+\text{blue}+\text{green}
]

Delta-style memory tries to correct:

[
\text{current prediction for A} \rightarrow \text{new value for A}
]

That is where DeltaNet comes in.

---

# 9. Linear attention before DeltaNet

DeltaNet starts from linear attention.

Standard softmax attention is:

[
o_t=\sum_{i\le t}\mathrm{softmax}(q_t^\top k_i)v_i
]

Linear attention replaces the softmax kernel with a linear dot-product-like feature interaction. In simple form:

[
S_t = S_{t-1}+v_tk_t^\top
]

[
o_t=S_tq_t
]

Here:

* (S_t\in\mathbb{R}^{d_v\times d_k}) is a memory matrix;
* (k_t) is the write key;
* (v_t) is the value;
* (q_t) is the read query.

Expanded:

[
S_t=\sum_{i\le t}v_ik_i^\top
]

[
o_t=\sum_{i\le t}v_i(k_i^\top q_t)
]

So again it is associative memory:

[
\text{similar key/query} \Rightarrow \text{retrieve value}
]

The DeltaNet paper explains that linear attention can be formulated as a linear RNN with matrix-valued hidden states, eliminating the KV cache and enabling constant-memory inference. ([ar5iv][2])

---

# 10. The problem with additive linear attention

The linear attention update:

[
S_t=S_{t-1}+v_tk_t^\top
]

is like original additive In-Place TTT.

It just keeps adding associations.

If two writes have similar keys but different values, they collide.

Suppose:

[
k_1 \approx k_2
]

but:

[
v_1\ne v_2
]

Then querying with that key retrieves a mixture:

[
S q \approx v_1+v_2
]

That is bad for exact retrieval and overwriting.

The DeltaNet paper says purely additive updates make it hard to deallocate old key-value associations and lead to key collisions when sequence length exceeds memory capacity. ([ar5iv][2])

---

# 11. DeltaNet: memory correction

DeltaNet changes the update.

Instead of:

[
S_t=S_{t-1}+v_tk_t^\top
]

it first reads the current memory at key (k_t):

[
\hat v_t=S_{t-1}k_t
]

Then computes the error:

[
e_t=v_t-\hat v_t
]

Then writes the correction:

[
S_t=S_{t-1}+\beta_t e_tk_t^\top
]

So:

[
\boxed{
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top
}
]

This is the core delta rule.

Expanded:

[
S_t
===

S_{t-1}
+
\beta_t v_tk_t^\top
-------------------

\beta_t S_{t-1}k_tk_t^\top
]

[
S_t
===

S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top
]

So it has two effects:

[
+\beta_t v_tk_t^\top
]

writes the new value, while:

[
-\beta_t S_{t-1}k_tk_t^\top
]

removes the old value currently associated with that key direction.

That is why DeltaNet is better at overwriting.

The DeltaNet paper describes this as replacing the additive update of linear transformers with the delta rule, improving associative recall, and then contributes a hardware-efficient algorithm to scale it to language modeling. ([ar5iv][2])

---

# 12. DeltaNet as SGD on a memory

DeltaNet is also exactly one gradient step on a local regression loss.

Define:

[
L(S)=\frac12|Sk_t-v_t|^2
]

Gradient:

[
\nabla_S L=(Sk_t-v_t)k_t^\top
]

Gradient descent:

[
S_t=S_{t-1}-\beta_t\nabla_S L
]

[
S_t
===

## S_{t-1}

\beta_t(S_{t-1}k_t-v_t)k_t^\top
]

[
S_t
===

S_{t-1}
+
\beta_t(v_t-S_{t-1}k_t)k_t^\top
]

That is DeltaNet.

So DeltaNet is not just “linear attention but different.” It is:

[
\boxed{
\text{linear associative memory + online error-corrective learning}
}
]

---

# 13. Gated DeltaNet

Gated DeltaNet adds a decay/forget gate.

A simple version:

[
\tilde S_{t-1}=\alpha_t S_{t-1}
]

[
e_t=v_t-\tilde S_{t-1}k_t
]

[
S_t=\tilde S_{t-1}+\beta_t e_tk_t^\top
]

Equivalently:

[
\boxed{
S_t
===

\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)
+
\beta_t v_tk_t^\top
}
]

where:

* (\alpha_t\in[0,1]) controls forgetting/retention;
* (\beta_t\in[0,1]) controls write strength;
* the delta term controls targeted correction.

Gated DeltaNet’s paper explicitly frames gating and the delta rule as complementary: gating enables adaptive memory erasure, while the delta rule enables targeted updates. ([arXiv][3])

So:

```text
Linear attention:
    add new association.

DeltaNet:
    correct the association for this key.

Gated DeltaNet:
    correct the association and decide how much old memory to retain.
```

---

# 14. Connecting In-Place TTT and DeltaNet

Now the important connection.

In-Place TTT additive update:

[
W_{j+1}=W_j+\eta\hat V_j^\top Z_j
]

Linear attention additive update:

[
S_t=S_{t-1}+v_tk_t^\top
]

These are the same kind of memory write.

Mapping:

[
S \leftrightarrow W_{\text{down}}
]

[
k_t \leftrightarrow z_t
]

[
v_t \leftrightarrow \hat v_t
]

[
q_t \leftrightarrow z_{\text{future}}
]

So In-Place TTT is already very close to a **linear associative memory inside the MLP**.

That is why your idea is natural.

---

# 15. Gated Delta In-Place TTT

Original In-Place TTT:

[
\Delta W_{j+1}
==============

\Delta W_j+\eta\hat V_j^\top Z_j
]

Your Delta version:

[
W_j=W_0+\Delta W_j
]

[
\hat O_j=Z_jW_j^\top
]

[
E_j=\hat V_j-\hat O_j
]

[
\Delta W_{j+1}
==============

\Delta W_j+\eta E_j^\top Z_j
]

Your Gated Delta version:

[
\boxed{
\Delta W_{j+1}
==============

\rho_j\Delta W_j
+
\eta_j
\left[
\hat V_j-Z_j(W_0+\Delta W_j)^\top
\right]^\top Z_j
}
]

This is literally the DeltaNet idea transplanted into In-Place TTT.

Mapping:

| DeltaNet                               | Gated Delta In-Place TTT               |
| -------------------------------------- | -------------------------------------- |
| (S_t) memory matrix                    | (\Delta W_j) or (W_0+\Delta W_j)       |
| (k_t) key                              | (z_t), the MLP intermediate activation |
| (v_t) value                            | (\hat v_t), the LM-aligned target      |
| (S_{t-1}k_t) current memory prediction | (z_t(W_0+\Delta W_j)^\top)             |
| (v_t-S_{t-1}k_t) correction            | (\hat v_t-z_t(W_0+\Delta W_j)^\top)    |
| (\alpha_t) forget gate                 | (\rho_j) decay/retention gate          |
| (\beta_t) write gate                   | (\eta_j) learning-rate/write gate      |

This is why I think In-Place TTT is the right base for your project.

---

# 16. Why this is cleaner than E2E TTT

E2E TTT updates weights by CE gradient:

[
W_{j+1}=W_j-\eta\nabla_W \mathrm{CE}_j
]

That can be useful, but it does not naturally expose:

[
\text{target} - \text{current memory prediction}
]

The error signal is whatever comes out of full backprop through the network. For a linear projection it still has an outer-product form, but the “target” is implicit.

In Gated Delta In-Place TTT, the memory operation is explicit:

[
\text{read: } Z_jW_j^\top
]

[
\text{compare: } \hat V_j-Z_jW_j^\top
]

[
\text{write correction: } E_j^\top Z_j
]

That is a much cleaner memory story.

---

# 17. One subtlety: parallelism

Original In-Place TTT is easy to parallelize because:

[
\Delta W_j=\eta\hat V_j^\top Z_j
]

does not depend on (W_j).

But Gated Delta In-Place TTT uses:

[
E_j=\hat V_j-Z_j(W_0+\Delta W_j)^\top
]

which depends on the current fast state.

So the update is more sequential:

[
\Delta W_j \rightarrow \Delta W_{j+1}
]

This is exactly the same challenge DeltaNet had. The DeltaNet and Gated DeltaNet papers spend a lot of effort deriving chunkwise/parallel training algorithms so these recurrent memory updates can run efficiently on hardware. ([ar5iv][2])

So your project’s tradeoff is:

[
\text{Original In-Place TTT}
============================

\text{fast, additive, prefix-scannable}
]

[
\text{Gated Delta In-Place TTT}
===============================

\text{more correct memory update, but harder to parallelize}
]

That is the core research tradeoff.

---

# 18. Clean mental model

**In-Place TTT:**

> Use the MLP hidden feature (Z) as a key. Write an LM-aligned value (\hat V) into the MLP down-projection (W_{\text{down}}). Later similar (Z)’s retrieve that value.

**DeltaNet:**

> Use (k) as a key. Read the current memory prediction (Sk). Write only the error (v-Sk), so the memory is corrected rather than just accumulated.

**Gated DeltaNet:**

> Do DeltaNet’s correction, but also learn how much old memory to forget.

**Your Gated Delta In-Place TTT:**

> Use In-Place TTT’s (Z) and (W_{\text{down}}), but replace blind additive writes with Gated DeltaNet-style corrective writes.

In one equation:

[
\boxed{
\Delta W_{j+1}
==============

\rho_j\Delta W_j
+
\eta_j
\left[
\hat V_j-Z_j(W_0+\Delta W_j)^\top
\right]^\top Z_j
}
]

That is the cleanest form of the idea.

[1]: https://arxiv.org/html/2604.06169v1 "In-Place Test-Time Training"
[2]: https://ar5iv.org/html/2406.06484v6 "[2406.06484] Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
[3]: https://arxiv.org/abs/2412.06464 "[2412.06464] Gated Delta Networks: Improving Mamba2 with Delta Rule"
