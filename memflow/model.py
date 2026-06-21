"""GPT-style decoder LM that uses MemFlowLayer for token mixing (plan §3.2).

Pre-norm decoder: token embedding -> N x (RMSNorm -> MemFlowLayer -> residual,
RMSNorm -> MLP -> residual) -> final RMSNorm -> (optionally tied) LM head.

No positional embedding by default: linear attention is order-aware via its recurrence, and
absolute positions would cap sequence length (counter to the Phase 3 long-context goal). A learned
absolute embedding can be enabled with use_pos_emb=True for short-context experiments.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MemFlowConfig
from .memflow_layer import MemFlowLayer, RMSNorm


class MLP(nn.Module):
    """SwiGLU feed-forward (gated GELU-style), bias-free."""

    def __init__(self, d_model: int, mlp_ratio: float):
        super().__init__()
        hidden = int(mlp_ratio * d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: MemFlowConfig, layer_idx: int, mixer_factory=MemFlowLayer):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = mixer_factory(cfg, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: Tensor, mode: str = "chunk", detach_value: bool = False) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mode=mode, detach_value=detach_value)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MemFlowLM(nn.Module):
    def __init__(self, cfg: MemFlowConfig, use_pos_emb: bool = False, tie_embeddings: bool = True,
                 mixer_factory=MemFlowLayer):
        super().__init__()
        self.cfg = cfg
        self.use_pos_emb = use_pos_emb
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i, mixer_factory) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # prenorm residual rescaling (GPT-2 / FLA style): scale the projections that write into the
        # residual stream by 1/sqrt(2*n_layers) so the residual variance doesn't grow with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith(".down.weight"):
                p.data.mul_(scale)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None and not getattr(m, "_keep_bias_init", False):
                # leave a_proj/b_proj biases (set in MemFlowLayer) untouched
                pass
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None, mode: str = "chunk",
                detach_value: bool = False):
        B, T = idx.shape
        x = self.embed(idx)
        if self.use_pos_emb:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x, mode=mode, detach_value=detach_value)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not (self.lm_head.weight is self.embed.weight):
                n -= self.lm_head.weight.numel()
        return n
