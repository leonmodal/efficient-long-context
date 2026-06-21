"""External baselines for matched comparison (plan §9, baseline 3): standard Gated DeltaNet.

We reuse the SAME GPT shell as MemFlowLM (embedding, RMSNorm, SwiGLU MLP, tied head, masked
cross-entropy) and swap ONLY the token mixer, so any accuracy gap is attributable to the mixer,
not the surrounding architecture. The GDN mixer is FLA's `GatedDeltaNet` (Triton; CUDA only).

`flash-linear-attention` (the `fla` package) is imported lazily so this module imports fine on a
box without it; the Gated DeltaNet path only runs on the GPU image that provides `fla`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import MLP
from .memflow_layer import RMSNorm


class GDNBaselineLM(nn.Module):
    """GPT-style LM whose token mixer is FLA Gated DeltaNet (scalar decay + short conv + gate)."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, head_dim: int, vocab_size: int,
                 expand_v: float = 1.0, use_short_conv: bool = True, mlp_ratio: float = 4.0,
                 norm_eps: float = 1e-5, tie_embeddings: bool = True):
        super().__init__()
        from fla.layers import GatedDeltaNet  # lazy: requires CUDA + triton
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn_norms = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(n_layers):
            self.attn_norms.append(RMSNorm(d_model, eps=norm_eps))
            self.mixers.append(GatedDeltaNet(
                hidden_size=d_model, head_dim=head_dim, num_heads=n_heads, expand_v=expand_v,
                use_gate=True, use_short_conv=use_short_conv, mode="chunk", layer_idx=i,
                norm_eps=norm_eps))
            self.mlp_norms.append(RMSNorm(d_model, eps=norm_eps))
            self.mlps.append(MLP(d_model, mlp_ratio))
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        # NOTE: deliberately do NOT re-init / rescale the FLA mixer — it carries FLA's tuned init,
        # and touching it destabilizes GDN. The baseline keeps FLA's native initialization.

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None, mode: str = "chunk",
                detach_value: bool = False):
        x = self.embed(idx)
        for an, mix, mn, mlp in zip(self.attn_norms, self.mixers, self.mlp_norms, self.mlps):
            x = x + mix(an(x))[0]            # GatedDeltaNet returns (o, None, cache)
            x = x + mlp(mn(x))
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n


class HGDNBaselineLM(nn.Module):
    """Log-linear attention (gated-delta variant): the PRIMARY same-state-class baseline.

    Wraps the `hattention` package's `HGatedDeltaNetForCausalLM` (Fenwick hierarchy of O(log T)
    gated-delta states with learned per-level lambdas). We bypass its fused loss and compute the
    masked MQAR cross-entropy ourselves, matching the rest of the comparison exactly.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, head_dim: int, vocab_size: int,
                 expand_v: float = 1.0, use_short_conv: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        from hattention.configuration_h_gated_deltanet import HGatedDeltaNetConfig
        from hattention.modeling_h_gated_deltanet import HGatedDeltaNetForCausalLM
        cfg = HGatedDeltaNetConfig(
            hidden_size=d_model, num_hidden_layers=n_layers, num_heads=n_heads, head_dim=head_dim,
            expand_v=expand_v, use_gate=True, use_short_conv=use_short_conv, vocab_size=vocab_size,
            norm_eps=norm_eps, fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=False,
            fuse_linear_cross_entropy=False, use_l2warp=False, tie_word_embeddings=True,
            max_position_embeddings=65536)
        self.model = HGatedDeltaNetForCausalLM(cfg)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None, mode: str = "chunk",
                detach_value: bool = False):
        logits = self.model(input_ids=idx).logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        return sum(p.numel() for p in self.parameters())
