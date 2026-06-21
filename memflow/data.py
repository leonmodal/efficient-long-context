"""Data: synthetic MQAR (multi-query associative recall) and helpers (plan §4.4, §9).

MQAR (Arora et al., "Zoology"): the model sees key/value pairs, then query keys re-asking earlier
keys, and must emit the matching value. This is the fastest signal that memory helps, and the
substrate for the Phase 1 ablation (memory-of-memory vs token-read vs matched single bucket).

Format (LM-style, next-token): keys are drawn from [0, V/2), values from [V/2, V). The sequence is
  context:  k0 v0 k1 v1 ... k_{D-1} v_{D-1}        (D distinct pairs)
  queries:  kq0 vq0 kq1 vq1 ...                    (Q repeated pairs)
We supervise next-token prediction ONLY at query-key positions (predicting the value), so a correct
answer requires recalling the association formed in the context. Everything else is ignore (-100).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

IGNORE = -100


def make_mqar(batch: int, num_pairs: int, num_queries: int, vocab: int,
              generator: torch.Generator = None, device="cpu") -> Tuple[Tensor, Tensor]:
    """Return (inputs, targets), both (batch, L-1) with L = 2*num_pairs + 2*num_queries.

    `targets` is IGNORE everywhere except the positions that predict a queried value.
    """
    assert vocab % 2 == 0, "vocab split into equal key/value halves"
    half = vocab // 2
    g = generator
    D, Q = num_pairs, num_queries

    # distinct keys per row from [0, half); values from [half, vocab)
    keys = torch.empty(batch, D, dtype=torch.long, device=device)
    for b in range(batch):
        perm = torch.randperm(half, generator=g, device=device)[:D]
        keys[b] = perm
    vals = torch.randint(half, vocab, (batch, D), generator=g, device=device)

    # context: interleave keys/values -> (B, 2D)
    context = torch.stack([keys, vals], dim=-1).reshape(batch, 2 * D)

    # queries: pick Q of the D pairs (with replacement) per row
    qsel = torch.randint(0, D, (batch, Q), generator=g, device=device)
    qkeys = torch.gather(keys, 1, qsel)
    qvals = torch.gather(vals, 1, qsel)
    queries = torch.stack([qkeys, qvals], dim=-1).reshape(batch, 2 * Q)

    seq = torch.cat([context, queries], dim=1)              # (B, 2D+2Q)
    inputs = seq[:, :-1].contiguous()
    nexttok = seq[:, 1:].contiguous()                       # next-token targets
    targets = torch.full_like(nexttok, IGNORE)

    # supervise the value of each query: query q lives at seq positions [2D+2q, 2D+2q+1];
    # predicting its value happens at target index (2D+2q+1)-1 = 2D+2q.
    qpos = 2 * D + 2 * torch.arange(Q, device=device)       # (Q,)
    targets[:, qpos] = nexttok[:, qpos]
    return inputs, targets


def make_niah(batch: int, seq_len: int, vocab: int, depth: float = 0.1,
              generator: torch.Generator = None, device="cpu") -> Tuple[Tensor, Tensor]:
    """Single-needle-in-a-haystack (RULER-style), token/LM form. Returns (inputs, targets) each
    (batch, seq_len). One needle (nk->nv) is buried at `depth` among distractor key/value pairs;
    the final token re-asks nk and we supervise predicting nv there (everything else IGNORE).

    Tests long-range recall: the needle must survive `~(1-depth)*seq_len` tokens of distractors.
    A fixed state of capacity ~d overwrites it once #distractors >> d; multi-timescale/growing
    state should retain it coarsely. Keys [0,V/2), values [V/2,V).
    """
    assert vocab % 2 == 0 and seq_len % 2 == 1, "need even vocab and odd seq_len (pairs + query)"
    half = vocab // 2
    g = generator
    n_pairs = (seq_len - 1) // 2                      # KV pairs before the final query token
    assert n_pairs <= half, f"need vocab/2 >= n_pairs ({half} < {n_pairs}); raise vocab for NIAH"

    inputs = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch, seq_len), IGNORE, dtype=torch.long, device=device)
    early = max(1, int(depth * n_pairs))             # needle lands in the early region (long gap)
    for b in range(batch):
        keys = torch.randperm(half, generator=g, device=device)[:n_pairs]          # distinct keys
        vals = torch.randint(half, vocab, (n_pairs,), generator=g, device=device)
        nslot = int(torch.randint(0, early, (1,), generator=g, device=device))     # random early slot
        nk, nv = keys[nslot].item(), vals[nslot].item()
        seq = torch.stack([keys, vals], dim=-1).reshape(-1)                        # (2*n_pairs,)
        inputs[b, :2 * n_pairs] = seq
        inputs[b, -1] = nk                                                          # query the needle
        targets[b, -1] = nv                                                         # supervise here
    return inputs, targets


def mqar_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Token accuracy over supervised (non-IGNORE) positions."""
    mask = targets != IGNORE
    if mask.sum() == 0:
        return float("nan")
    pred = logits.argmax(-1)
    return (pred[mask] == targets[mask]).float().mean().item()
