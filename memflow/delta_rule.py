"""Gated delta rule: the single update operator used for *every* bucket write.

Plan §0 (the single update operator):

    U(S; k, v, alpha, beta) = S . diag(alpha) . (I - beta k kᵀ) + beta v kᵀ

Equivalent delta form (identical):

    U(S; k, v, alpha, beta) = S.diag(alpha) + beta (v - (S.diag(alpha)) k) kᵀ

Shapes (one head): S (d_v, d_k); alpha (d_k,) channel-wise on the KEY dim; beta scalar;
k (d_k,) L2-normalized; v (d_v,). Read: o = S @ q with q (d_k,) L2-normalized -> o (d_v,).

This module provides three functions, all batched over (B, H):

  * gated_delta_recurrent     -- the reference; loops over T. Source of truth.
  * gated_delta_chunkwise     -- parallel chunk form; MUST match recurrent to atol 1e-4.
  * read_state_with_query     -- reads the *evolving* state with an alternate query stream
                                 (the probes P), sharing k,v,alpha,beta. r_t = S_t @ p_t.

States are kept in fp32 (plan §10) regardless of autocast dtype of the inputs.

NOTE on timing convention (plan §4.2): the read happens *after* the write at step t, i.e.
    S_t = U(S_{t-1}; k_t, v_t, ...);  o_t = S_t @ q_t.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GDADecay(nn.Module):
    """Gated DeltaNet (Mamba-style) per-head scalar decay, replicated faithfully from FLA:

        alpha = exp( -exp(A_log) * softplus(a_proj(x) + dt_bias) )   in (0,1), per head per token.

    Returns (B,T,H). Use this (default) so each bucket is a true GDA atom; the alternative is the
    channel-wise (per-dim) decay, which is more expressive but weaker for exact recall.
    """

    def __init__(self, d_model: int, n_heads: int, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_floor: float = 1e-4):
        super().__init__()
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        A = torch.empty(n_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))                       # A ~ U(0,16)
        dt = torch.exp(torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, x: Tensor) -> Tensor:
        g = self.a_proj(x).float() + self.dt_bias.float()
        return torch.exp(-torch.exp(self.A_log.float()) * F.softplus(g))


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2-normalize along `dim`. Used on k and q before every write/read (plan §0)."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# fast-path chunk decay clamp: e^{-logg} is capped at e^{_DECAY_CLAMP} to avoid fp32 overflow.
# Inactive whenever the per-chunk cumulative decay stays above e^{-_DECAY_CLAMP} (alpha near 1).
_DECAY_CLAMP = 85.0


# --------------------------------------------------------------------------- #
# recurrent reference (source of truth)
# --------------------------------------------------------------------------- #
def _recurrent_core(
    read_q: Tensor,        # (B,H,T,d_k) query stream used to read the state after each write
    k: Tensor,             # (B,H,T,d_k) L2-normalized keys
    v: Tensor,             # (B,H,T,d_v) values
    alpha: Tensor,         # (B,H,T,d_k) channel-wise decay in (0,1)
    beta: Tensor,          # (B,H,T,1)   scalar write strength in (0,1)
    S0: Optional[Tensor],  # (B,H,d_v,d_k) initial state or None -> zeros
) -> Tuple[Tensor, Tensor]:
    """Shared loop. Evolves S via the gated delta op and reads with `read_q` after each write.

    Returns (o, S_final) with o (B,H,T,d_v) and S_final (B,H,d_v,d_k), both fp32.
    """
    B, H, T, d_k = k.shape
    d_v = v.shape[-1]
    compute_dtype = torch.float32  # states accumulate -> keep fp32 (plan §10)

    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    af = alpha.to(compute_dtype)
    bf = beta.to(compute_dtype)
    qf = read_q.to(compute_dtype)

    if S0 is None:
        S = torch.zeros(B, H, d_v, d_k, dtype=compute_dtype, device=k.device)
    else:
        S = S0.to(compute_dtype).clone()

    outs = []
    for t in range(T):
        a_t = af[:, :, t, :]                       # (B,H,d_k)
        k_t = kf[:, :, t, :]                       # (B,H,d_k)
        v_t = vf[:, :, t, :]                       # (B,H,d_v)
        b_t = bf[:, :, t, :]                       # (B,H,1)
        q_t = qf[:, :, t, :]                       # (B,H,d_k)

        Sa = S * a_t.unsqueeze(-2)                 # S.diag(alpha): scale columns -> (B,H,d_v,d_k)
        Sa_k = (Sa * k_t.unsqueeze(-2)).sum(-1)    # (S.diag(alpha)) k -> (B,H,d_v)
        delta = b_t * (v_t - Sa_k)                 # beta (v - (S diag a) k) -> (B,H,d_v)
        S = Sa + delta.unsqueeze(-1) * k_t.unsqueeze(-2)   # + outer(delta, k) -> (B,H,d_v,d_k)

        o_t = (S * q_t.unsqueeze(-2)).sum(-1)      # S @ q_t -> (B,H,d_v)
        outs.append(o_t)

    o = torch.stack(outs, dim=2)                   # (B,H,T,d_v)
    return o, S


def gated_delta_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    S0: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference recurrent gated delta rule. Loops over T. Returns (o, S_final).

    q,k,v: (B,H,T,d); alpha: (B,H,T,d_k); beta: (B,H,T,1).
    k,q are assumed already L2-normalized by the caller. o: (B,H,T,d_v).
    """
    return _recurrent_core(q, k, v, alpha, beta, S0)


def delta_update(S: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """One gated delta write: S' = U(S; k, v, alpha, beta) = S diag(a)(I - b k kᵀ) + b v kᵀ.

    Batched over leading dims. S (...,d_v,d_k); k,alpha (...,d_k); v (...,d_v); beta (...,1).
    `k` must be L2-normalized by the caller. Used for every bucket write (fast + consolidation).
    """
    Sa = S * alpha.unsqueeze(-2)                       # S diag(alpha): scale key-columns
    Sa_k = (Sa * k.unsqueeze(-2)).sum(-1)              # (S diag a) k -> (...,d_v)
    delta = beta * (v - Sa_k)                          # b (v - (S diag a) k)
    return Sa + delta.unsqueeze(-1) * k.unsqueeze(-2)  # + outer(delta, k)


# --------------------------------------------------------------------------- #
# chunkwise parallel form (pure PyTorch, numerically stable for channel-wise alpha)
# --------------------------------------------------------------------------- #
#
# Derivation (one head, one chunk of length C starting from state S0; alpha_t in R^{d_k}):
#   Let u_t = beta_t (v_t - S_{t-1} diag(alpha_t) k_t)  (pseudo-value), so the recurrence
#   S_t = S_{t-1} diag(alpha_t) + u_t k_tᵀ  unrolls to  S_t = (S0 + sum_{i<=t} u_i (k_i/g_i)ᵀ) diag(g_t)
#   with cumulative decay g_t = prod_{i<=t} alpha_i (channel-wise, g_0 = 1).
#   Then, writing the pairwise decay g_t/g_i = exp(logg_t - logg_i) (<= 1 for i<=t, never the
#   overflow-prone 1/g_i), one gets, with logg = cumsum(log alpha):
#     A[t,i] = sum_c k[i,c] k[t,c] exp(logg[t,c]-logg[i,c])      (i<t)   "decayed key-key"
#     B[t,i] = sum_c q[t,c] k[i,c] exp(logg[t,c]-logg[i,c])      (i<=t)  "decayed query-key"
#     Vbar0[t] = S0 (g_t ⊙ k_t);  RHS[t] = beta_t (v_t - Vbar0[t])
#     (I + L) U = RHS,  L[t,i] = beta_t A[t,i]  (strictly lower)  -> solve for U (pseudo-values)
#     o_t = S0 (g_t ⊙ q_t) + sum_{i<=t} B[t,i] u_i               (read AFTER write at t)
#     S_C = S0 diag(g_C) + sum_i u_i k~_iᵀ,  k~_i = k_i ⊙ exp(logg_C - logg_i)   (state carry)
#   All exponentials have argument <= 0 after the causal mask, so the form is overflow-free.
#   The recurrent loop is replaced by one triangular solve per chunk; chunks are sequential.
def _chunk_step(
    q_c: Tensor,       # (B,H,L,d_k)
    k_c: Tensor,       # (B,H,L,d_k)
    v_c: Tensor,       # (B,H,L,d_v)
    a_c: Tensor,       # (B,H,L,d_k) channel-wise alpha in (0,1)
    b_c: Tensor,       # (B,H,L,1)   beta in (0,1)
    S: Tensor,         # (B,H,d_v,d_k) state at chunk start
    read_q: Tensor,    # (B,H,L,d_k) query stream used for the read (q or a probe)
    exact: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """One chunk of the gated delta rule. Returns (o_chunk, U pseudo-values, S_new).

    `exact=True` materializes the (B,H,L,L,d_k) pairwise-decay tensor: overflow-free for ANY alpha
    (the test oracle). `exact=False` builds the decay-attention via two matmuls A=(k·e^{logg})(k·e^{-logg})ᵀ
    — far faster and lower-memory, exact when the per-chunk decay stays within e^{±_DECAY_CLAMP}
    (always true once alpha is near 1, which alpha_bias_init guarantees in training).
    """
    Lc = k_c.shape[-2]
    logg = torch.cumsum(torch.log(a_c), dim=-2)            # (B,H,L,d_k) cumulative log-decay
    idx = torch.arange(Lc, device=k_c.device)
    causal = (idx.unsqueeze(-1) >= idx.unsqueeze(-2)).view(1, 1, Lc, Lc)  # t>=i
    strict = (idx.unsqueeze(-1) > idx.unsqueeze(-2)).view(1, 1, Lc, Lc)   # t>i
    expg = torch.exp(logg)                                 # g_t (<=1)
    if exact:
        D = logg.unsqueeze(-2) - logg.unsqueeze(-3)        # (B,H,L,L,d_k); dim -3=t, -2=i
        D = D.masked_fill(~causal.unsqueeze(-1), float("-inf"))
        decay = torch.exp(D)                               # <=1 on causal, 0 above
        A = torch.einsum("bhtc,bhic,bhtic->bhti", k_c, k_c, decay)         # decayed key-key
        Bm = torch.einsum("bhtc,bhic,bhtic->bhti", read_q, k_c, decay)     # decayed query-key
    else:
        expmg = torch.exp((-logg).clamp(max=_DECAY_CLAMP))  # e^{-logg}, clamped against overflow
        Qg = k_c * expmg                                    # (B,H,L,d_k)
        A = torch.matmul(k_c * expg, Qg.transpose(-1, -2)) * causal        # (B,H,L,L)
        Bm = torch.matmul(read_q * expg, Qg.transpose(-1, -2)) * causal
    kdec = k_c * expg                                      # g_t ⊙ k_t
    qdec = read_q * expg                                   # g_t ⊙ q_t
    Vbar0 = torch.einsum("bhvk,bhtk->bhtv", S, kdec)       # S0 (g_t ⊙ k_t)
    rhs = b_c * (v_c - Vbar0)                              # (B,H,L,d_v)
    L_mat = b_c * (A * strict)                             # strictly-lower, scaled by beta_t
    eye = torch.eye(Lc, device=k_c.device, dtype=k_c.dtype)
    M = eye + L_mat                                        # unit lower-triangular
    U = torch.linalg.solve_triangular(M, rhs, upper=False, unitriangular=False)  # (B,H,L,d_v)
    o = torch.einsum("bhvk,bhtk->bhtv", S, qdec) + torch.einsum("bhti,bhiv->bhtv", Bm, U)
    # state carry
    gC = expg[:, :, -1, :]                                 # (B,H,d_k) cumulative decay over chunk
    ktil = k_c * torch.exp(logg[:, :, -1:, :] - logg)      # k_i ⊙ (g_C/g_i), <=1
    S_new = S * gC.unsqueeze(-2) + torch.einsum("bhiv,bhik->bhvk", U, ktil)
    return o, U, S_new


def _chunkwise_scan(
    read_q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    S0: Optional[Tensor],
    chunk_size: int,
    exact: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Evolve S via the chunk form (from k,v,alpha,beta) and read it with `read_q`. fp32.

    Returns (o, S_final) with o_t = S_t @ read_q_t. `read_q` may be the host query (-> identical
    to the recurrent output) or a probe stream (-> the memory-of-memory read r_t)."""
    B, H, T, d_k = k.shape
    d_v = v.shape[-1]
    f32 = torch.float32
    rq, kf, vf = read_q.to(f32), k.to(f32), v.to(f32)
    af, bf = alpha.to(f32), beta.to(f32)
    S = torch.zeros(B, H, d_v, d_k, dtype=f32, device=k.device) if S0 is None else S0.to(f32).clone()
    outs = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        o_c, _, S = _chunk_step(rq[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e],
                                af[:, :, s:e], bf[:, :, s:e], S, rq[:, :, s:e], exact=exact)
        outs.append(o_c)
    return torch.cat(outs, dim=2), S


def gated_delta_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    S0: Optional[Tensor] = None,
    chunk_size: int = 64,
    exact: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Parallel chunk form of the gated delta rule. Same I/O as `gated_delta_recurrent`.

    `exact=True` (default, the test oracle) matches the recurrent reference to atol 1e-4 for any
    alpha. `exact=False` uses the fast matmul intra-chunk form (used by the layer/training).
    """
    return _chunkwise_scan(q, k, v, alpha, beta, S0, chunk_size, exact=exact)


def read_state_with_query(
    q_probe: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    S0: Optional[Tensor] = None,
    chunk_size: int = 64,
    mode: str = "recurrent",
    exact: bool = True,
) -> Tensor:
    """Read the EVOLVING state with an alternate query stream `q_probe` (the probes P),
    sharing k,v,alpha,beta with the host stream. Returns r_t = S_t @ q_probe_t for all t.

    This is the "memory-of-memory" read: q_probe lives in the host stream's key space.
    `mode="recurrent"` is the reference; `mode="chunk"` uses the parallel scan (they agree to atol).
    Returns r: (B,H,T,d_v).
    """
    if mode == "chunk":
        r, _ = _chunkwise_scan(q_probe, k, v, alpha, beta, S0, chunk_size, exact=exact)
    else:
        r, _ = _recurrent_core(q_probe, k, v, alpha, beta, S0)
    return r
