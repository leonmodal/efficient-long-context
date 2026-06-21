"""Option-C consolidation: build level ℓ from level ℓ-1 (plan §4.1).

For a slower bucket (level ℓ>0), over each window of P_ℓ tokens we keep a salience-weighted
running average of (token-derived key kᵘ_t, memory-readout value r_t), then do ONE gated-delta
write at the window boundary with the aggregated (k̄, r̄). The value r_t is the "memory of memory":
r_t = S^(ℓ-1)_t · p_t, a readout of the *lower* bucket's evolving state with probe p_t.

This module owns the per-level WRITE-path projections (probe, key, salience, alpha, beta, and an
optional token value for the baseline ablation) and the aggregation primitives. The orchestration
(which lower state to read, the sequential boundary writes, the readout) lives in memflow_layer.py.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import MemFlowConfig
from .delta_rule import GDADecay, l2norm


class ConsolProj(NamedTuple):
    probe: Tensor   # (B,H,T,d_k) L2-normalized probe into the lower bucket's key space
    key: Tensor     # (B,H,T,d_k) raw upper key (normalized AFTER aggregation)
    sal: Tensor     # (B,H,T)     salience score per token
    alpha: Tensor   # (B,H,T,d_k) channel-wise decay for the upper write
    beta: Tensor    # (B,H,T,1)   write strength for the upper write
    vtok: Optional[Tensor]  # (B,H,T,d_v) token value (only for value_source="token")


class Consolidator(nn.Module):
    """Write-path projections + Option-C aggregation for one level ℓ (built from ℓ-1)."""

    def __init__(self, cfg: MemFlowConfig, level: int, lower_key_proj: Optional[nn.Linear] = None):
        super().__init__()
        self.cfg = cfg
        self.level = level
        self.H, self.dk, self.dv = cfg.n_heads, cfg.d_k, cfg.d_v
        inner_k, inner_v = self.H * self.dk, self.H * self.dv

        # probe lives in the LOWER level's key space; optionally tie to the lower key projection
        self.tie_probe = cfg.tie_probe_to_lower_key and lower_key_proj is not None
        self.lower_key_proj = lower_key_proj if self.tie_probe else None
        if not self.tie_probe:
            self.p_proj = nn.Linear(cfg.d_model, inner_k, bias=False)

        self.k_proj = nn.Linear(cfg.d_model, inner_k, bias=False)     # upper key (token-derived)
        self.sal_proj = nn.Linear(cfg.d_model, self.H, bias=False)    # salience scalar per head
        self.scalar_decay = cfg.decay_mode == "scalar"
        if self.scalar_decay:
            self.decay = GDADecay(cfg.d_model, self.H)               # GDA per-head decay
        else:
            self.a_proj = nn.Linear(cfg.d_model, inner_k, bias=True)  # channel-wise alpha
            nn.init.constant_(self.a_proj.bias, cfg.alpha_bias_init)
        self.b_proj = nn.Linear(cfg.d_model, self.H, bias=True)       # upper scalar beta
        if cfg.value_source == "token":
            self.v_proj = nn.Linear(cfg.d_model, inner_v, bias=False)  # baseline ablation value
        nn.init.constant_(self.b_proj.bias, cfg.beta_bias_init)

    def project(self, x: Tensor) -> ConsolProj:
        """x: (B,T,d_model) -> per-token write-path tensors in (B,H,T,*) layout."""
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv

        def heads_k(t):  # (B,T,H*dk) -> (B,H,T,dk)
            return t.view(B, T, H, dk).transpose(1, 2)

        probe_w = self.lower_key_proj if self.tie_probe else self.p_proj
        probe = l2norm(heads_k(probe_w(x)))                    # read query into lower state
        key = heads_k(self.k_proj(x))                          # raw; normalized after aggregation
        sal = self.sal_proj(x).transpose(1, 2)                 # (B,H,T)
        if self.scalar_decay:
            alpha = self.decay(x).unsqueeze(-1).expand(B, T, H, dk).transpose(1, 2)  # (B,H,T,dk)
        else:
            alpha = torch.sigmoid(heads_k(self.a_proj(x)))     # (B,H,T,dk)
        beta = torch.sigmoid(self.b_proj(x)).view(B, T, H, 1).transpose(1, 2)  # (B,H,T,1)
        vtok = None
        if self.cfg.value_source == "token":
            vtok = self.v_proj(x).view(B, T, H, dv).transpose(1, 2)
        return ConsolProj(probe, key, sal, alpha, beta, vtok)


# --------------------------------------------------------------------------- #
# aggregation primitives over complete windows of length `period`
# --------------------------------------------------------------------------- #
def _num_windows(T: int, period: int) -> int:
    return T // period


def aggregate_batch(key: Tensor, value: Tensor, sal: Tensor, period: int):
    """Batch softmax-weighted average over each complete window of `period` tokens.

    key (B,H,T,d_k), value (B,H,T,d_v), sal (B,H,T). Returns:
      kbar (B,H,W,d_k)  -- aggregated key, NOT yet L2-normalized (caller normalizes),
      rbar (B,H,W,d_v)  -- aggregated value,
      boundary_idx (W,) -- token index of each window boundary ((w+1)*period - 1).
    W = T // period (incomplete trailing window is dropped: it never reaches a boundary).
    """
    B, H, T, dk = key.shape
    dv = value.shape[-1]
    W = _num_windows(T, period)
    if W == 0:
        empty = key.new_zeros(B, H, 0, dk)
        return empty, value.new_zeros(B, H, 0, dv), torch.zeros(0, dtype=torch.long, device=key.device)
    k = key[:, :, : W * period].reshape(B, H, W, period, dk)
    v = value[:, :, : W * period].reshape(B, H, W, period, dv)
    s = sal[:, :, : W * period].reshape(B, H, W, period)
    w = torch.softmax(s, dim=-1)                               # (B,H,W,period)
    kbar = (w.unsqueeze(-1) * k).sum(3)                        # (B,H,W,dk)
    rbar = (w.unsqueeze(-1) * v).sum(3)                        # (B,H,W,dv)
    boundary_idx = torch.arange(1, W + 1, device=key.device) * period - 1
    return kbar, rbar, boundary_idx


def aggregate_online(key: Tensor, value: Tensor, sal: Tensor, period: int):
    """Streaming Option-C with FlashAttention-style running-max rescale (plan §4.1).

    Identical result to `aggregate_batch` (tested), but never buffers the window -- this is the
    reference for the eventual online kernel. Same return signature.
    """
    B, H, T, dk = key.shape
    dv = value.shape[-1]
    W = _num_windows(T, period)
    device = key.device
    if W == 0:
        return (key.new_zeros(B, H, 0, dk), value.new_zeros(B, H, 0, dv),
                torch.zeros(0, dtype=torch.long, device=device))
    kbars, rbars = [], []
    for w in range(W):
        m = key.new_full((B, H), float("-inf"))
        den = key.new_zeros(B, H)
        numK = key.new_zeros(B, H, dk)
        numR = value.new_zeros(B, H, dv)
        for j in range(period):
            t = w * period + j
            s_t = sal[:, :, t]                                 # (B,H)
            m_new = torch.maximum(m, s_t)
            rescale = torch.exp(m - m_new)                     # exp(m_old - m_new); 0 on the first step
            rescale = torch.where(torch.isinf(m), torch.zeros_like(rescale), rescale)
            w_t = torch.exp(s_t - m_new)                       # (B,H)
            den = den * rescale + w_t
            numK = numK * rescale.unsqueeze(-1) + w_t.unsqueeze(-1) * key[:, :, t]
            numR = numR * rescale.unsqueeze(-1) + w_t.unsqueeze(-1) * value[:, :, t]
            m = m_new
        kbars.append(numK / den.unsqueeze(-1))
        rbars.append(numR / den.unsqueeze(-1))
    kbar = torch.stack(kbars, dim=2)                           # (B,H,W,dk)
    rbar = torch.stack(rbars, dim=2)
    boundary_idx = torch.arange(1, W + 1, device=device) * period - 1
    return kbar, rbar, boundary_idx
