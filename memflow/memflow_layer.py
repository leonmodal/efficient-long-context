"""MemFlowLayer — orchestrates the memory buckets, consolidation, and readout.

A single layer holds L = len(cfg.periods) buckets. Level 0 is the fast gated-delta linear
attention (write every token). Level ℓ>0 is consolidated from level ℓ-1 via Option C
(consolidation.py): every P_ℓ tokens, a salience-weighted aggregate of (token key, memory readout)
is written into S^(ℓ) with one gated-delta write. The readout queries every *live* bucket, gates,
sums, norms, and projects (plan §0).

Two execution modes that must agree (tested):
  * "recurrent" — the per-token reference (source of truth),
  * "chunk"     — the parallel form: fast bucket + probe reads via the chunkwise scan, slow writes
                  as a short loop over the (few) window boundaries.

This generalizes to any number of fixed buckets, so it serves Phase 1 (2 buckets) and Phase 2 (N).
`detach_value=True` stops gradient from the memory-readout value back into the lower bucket
(the detach warmup, plan §4.2): it changes gradients only, not forward values.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MemFlowConfig
from .consolidation import Consolidator, aggregate_batch
from .delta_rule import (
    GDADecay,
    delta_update,
    gated_delta_chunkwise,
    gated_delta_recurrent,
    l2norm,
    read_state_with_query,
)


class ShortConv(nn.Module):
    """Causal depthwise 1D conv + SiLU on a projection stream (standard for AR recall, e.g. GDN).

    Input/output (B, T, dim). Left-padded so position t sees only positions <= t (no future leak).
    """

    def __init__(self, dim: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        xt = x.transpose(1, 2)                              # (B, dim, T)
        xt = F.pad(xt, (self.kernel - 1, 0))               # causal left pad
        return F.silu(self.conv(xt).transpose(1, 2))       # (B, T, dim)


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class SegmentSplitLayer(nn.Module):
    """Baseline mixer (equal state to an N-bucket MemFlow): N independent gated-delta buckets, each
    written DIRECTLY from tokens but only during its contiguous sequence segment, then FROZEN.

    Bucket b writes only while the token is in segment b (alpha=1, beta=0 outside -> frozen, so its
    early content is never overwritten), and is read at every token. This is the "use 2x state by
    partitioning the sequence" control for the memory-of-memory idea: same #buckets, same head_dim,
    direct token writes instead of reading the faster bucket. N = len(cfg.periods)."""

    def __init__(self, cfg: MemFlowConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.N = len(cfg.periods)
        self.H, self.dk, self.dv = cfg.n_heads, cfg.d_k, cfg.d_v
        self.chunk_size = cfg.chunk_size
        ik, iv = self.H * self.dk, self.H * self.dv
        mk = lambda d: nn.ModuleList([nn.Linear(cfg.d_model, d, bias=False) for _ in range(self.N)])
        self.q_proj, self.k_proj, self.v_proj = mk(ik), mk(ik), mk(iv)
        self.scalar_decay = cfg.decay_mode == "scalar"
        if self.scalar_decay:
            self.decay = nn.ModuleList([GDADecay(cfg.d_model, self.H) for _ in range(self.N)])
        else:
            self.a_proj = nn.ModuleList([nn.Linear(cfg.d_model, ik, bias=True) for _ in range(self.N)])
        self.b_proj = nn.ModuleList([nn.Linear(cfg.d_model, self.H, bias=True) for _ in range(self.N)])
        self.g_proj = mk(iv) if cfg.use_output_gate else None
        if cfg.use_short_conv:
            self.q_conv = nn.ModuleList([ShortConv(ik, cfg.conv_size) for _ in range(self.N)])
            self.k_conv = nn.ModuleList([ShortConv(ik, cfg.conv_size) for _ in range(self.N)])
            self.v_conv = nn.ModuleList([ShortConv(iv, cfg.conv_size) for _ in range(self.N)])
        self.o_norm = RMSNorm(self.dv, eps=cfg.norm_eps)
        self.o_proj = nn.Linear(iv, cfg.d_model, bias=False)
        for b in range(self.N):
            if not self.scalar_decay:
                nn.init.constant_(self.a_proj[b].bias, cfg.alpha_bias_init)
            nn.init.constant_(self.b_proj[b].bias, cfg.beta_bias_init)

    def forward(self, x: Tensor, mode: str = "chunk", detach_value: bool = False) -> Tensor:
        B, T, _ = x.shape
        H, dk, dv, N = self.H, self.dk, self.dv, self.N
        seg_id = (torch.arange(T, device=x.device) * N) // T          # (T,) segment per position
        outs = []
        for b in range(N):
            qp, kp, vp = self.q_proj[b](x), self.k_proj[b](x), self.v_proj[b](x)
            if self.cfg.use_short_conv:
                qp, kp, vp = self.q_conv[b](qp), self.k_conv[b](kp), self.v_conv[b](vp)
            q = l2norm(qp.view(B, T, H, dk).transpose(1, 2))
            k = l2norm(kp.view(B, T, H, dk).transpose(1, 2))
            v = vp.view(B, T, H, dv).transpose(1, 2)
            if self.scalar_decay:
                alpha = self.decay[b](x).unsqueeze(-1).expand(B, T, H, dk).transpose(1, 2)
            else:
                alpha = torch.sigmoid(self.a_proj[b](x)).view(B, T, H, dk).transpose(1, 2)
            beta = torch.sigmoid(self.b_proj[b](x)).view(B, T, H, 1).transpose(1, 2)
            in_seg = (seg_id == b).view(1, 1, T, 1).to(x.dtype)        # 1 inside segment b
            alpha = torch.where(in_seg.bool(), alpha, torch.ones_like(alpha))  # freeze outside
            beta = beta * in_seg                                       # no writes outside
            if mode == "recurrent":
                o_b, _ = gated_delta_recurrent(q, k, v, alpha, beta)
            else:
                o_b, _ = gated_delta_chunkwise(q, k, v, alpha, beta, chunk_size=self.chunk_size, exact=False)
            normed = self.o_norm(o_b)                                  # FLA-style: norm then SiLU-gate
            if self.g_proj is not None:
                g = F.silu(self.g_proj[b](x)).view(B, T, H, dv).transpose(1, 2)
                normed = g * normed
            outs.append(normed)
        summed = outs[0]
        for o in outs[1:]:
            summed = summed + o
        y = summed.transpose(1, 2).reshape(B, T, H * dv)
        return self.o_proj(y)


class MemFlowLayer(nn.Module):
    def __init__(self, cfg: MemFlowConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.H, self.dk, self.dv = cfg.n_heads, cfg.d_k, cfg.d_v
        self.periods = list(cfg.periods)
        self.L = len(self.periods)
        self.chunk_size = cfg.chunk_size
        inner_k, inner_v = self.H * self.dk, self.H * self.dv

        # ---- fast bucket (level 0) ----
        self.q_proj = nn.Linear(cfg.d_model, inner_k, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner_k, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner_v, bias=False)
        self.scalar_decay = cfg.decay_mode == "scalar"
        if self.scalar_decay:
            self.fast_decay = GDADecay(cfg.d_model, self.H)
        else:
            self.a_proj = nn.Linear(cfg.d_model, inner_k, bias=True)
        self.b_proj = nn.Linear(cfg.d_model, self.H, bias=True)
        if cfg.use_short_conv:
            self.q_conv = ShortConv(inner_k, cfg.conv_size)
            self.k_conv = ShortConv(inner_k, cfg.conv_size)
            self.v_conv = ShortConv(inner_v, cfg.conv_size)

        # ---- consolidators + read heads for levels 1..L-1 ----
        consolidators, read_q, read_g = [], [], []
        for lvl in range(1, self.L):
            lower_key = self.k_proj if lvl == 1 else consolidators[lvl - 2].k_proj
            consolidators.append(Consolidator(cfg, level=lvl, lower_key_proj=lower_key))
            read_q.append(nn.Linear(cfg.d_model, inner_k, bias=False))
            read_g.append(nn.Linear(cfg.d_model, inner_v, bias=False))
        self.consolidators = nn.ModuleList(consolidators)
        self.read_q = nn.ModuleList(read_q)        # read query for levels 1..L-1
        self.read_g = nn.ModuleList(read_g)        # output gate for levels 1..L-1

        # ---- level-0 output gate + shared readout ----
        if cfg.use_output_gate:
            self.g_proj = nn.Linear(cfg.d_model, inner_v, bias=False)
        self.o_norm = RMSNorm(self.dv, eps=cfg.norm_eps)
        self.o_proj = nn.Linear(inner_v, cfg.d_model, bias=False)

        self._init_bias()

    def _init_bias(self):
        if not self.scalar_decay:
            nn.init.constant_(self.a_proj.bias, self.cfg.alpha_bias_init)
        nn.init.constant_(self.b_proj.bias, self.cfg.beta_bias_init)

    def _alpha(self, x: Tensor) -> Tensor:
        """Fast-bucket decay as (B,H,T,d_k): per-head GDA scalar (expanded) or channel-wise."""
        B, T, _ = x.shape
        if self.scalar_decay:
            return self.fast_decay(x).unsqueeze(-1).expand(B, T, self.H, self.dk).transpose(1, 2)
        return torch.sigmoid(self._heads_k(self.a_proj(x), B, T))

    # ---------------------------------------------------------------- helpers
    def _heads_k(self, t, B, T):
        return t.view(B, T, self.H, self.dk).transpose(1, 2)

    def _fast_qkv(self, x: Tensor):
        B, T, _ = x.shape
        qp, kp, vp = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.cfg.use_short_conv:
            qp, kp, vp = self.q_conv(qp), self.k_conv(kp), self.v_conv(vp)
        q = l2norm(self._heads_k(qp, B, T))
        k = l2norm(self._heads_k(kp, B, T))
        v = vp.view(B, T, self.H, self.dv).transpose(1, 2)
        alpha = self._alpha(x)
        beta = torch.sigmoid(self.b_proj(x)).view(B, T, self.H, 1).transpose(1, 2)
        return q, k, v, alpha, beta

    def _level_read_query(self, x: Tensor, lvl: int) -> Tensor:
        """L2-normalized read query for level lvl (lvl>=1)."""
        B, T, _ = x.shape
        return l2norm(self._heads_k(self.read_q[lvl - 1](x), B, T))

    def _readout(self, x: Tensor, out_per_level: List[Tensor]) -> Tensor:
        """FLA-style gated readout per level: sum_ℓ silu(g^ℓ) ⊙ RMSNorm(o^ℓ), then W_o.

        Gate is applied AFTER the norm with SiLU (FusedRMSNormGated form) — the proven-stable
        readout — rather than a sigmoid gate inside the norm. Masked levels have o=0 -> contribute 0.
        """
        B, T, _ = x.shape
        summed = None
        for lvl, o in enumerate(out_per_level):
            normed = self.o_norm(o)                                    # RMSNorm over d_v
            if self.cfg.use_output_gate:
                proj = self.g_proj if lvl == 0 else self.read_g[lvl - 1]
                g = F.silu(proj(x)).view(B, T, self.H, self.dv).transpose(1, 2)
                normed = g * normed
            summed = normed if summed is None else summed + normed
        y = summed.transpose(1, 2).reshape(B, T, self.H * self.dv)
        return self.o_proj(y)

    def _piecewise_state_read(self, states: List[Tensor], period: int, query: Tensor, T: int) -> Tensor:
        """Read a piecewise-constant bucket: out_t = states[idx(t)] @ query_t, 0 before first write.

        `states[w]` is the bucket state AFTER window w's boundary write. Token t reads the state of
        the most recent boundary <= t, i.e. window w covers tokens [(w+1)P-1, (w+2)P-1).
        """
        B, H = query.shape[:2]
        out = query.new_zeros(B, H, T, self.dv)
        for w in range(len(states)):
            start = (w + 1) * period - 1
            if start >= T:
                break
            end = min((w + 2) * period - 1, T)
            out[:, :, start:end] = torch.einsum("bhvk,bhtk->bhtv", states[w], query[:, :, start:end])
        return out

    @torch.no_grad()
    def diagnostics(self, x: Tensor) -> dict:
        """Per-level effective decay (mean alpha) and mean output gate (plan §5/§8 logging)."""
        B, T, _ = x.shape
        out = {}
        out["alpha/level0"] = self._alpha(x).mean().item()
        if self.cfg.use_output_gate:
            out["gate/level0"] = torch.sigmoid(self.g_proj(x)).mean().item()
        for lvl in range(1, self.L):
            p = self.consolidators[lvl - 1].project(x)
            out[f"alpha/level{lvl}"] = p.alpha.mean().item()
            if self.cfg.use_output_gate:
                out[f"gate/level{lvl}"] = torch.sigmoid(self.read_g[lvl - 1](x)).mean().item()
        return out

    def recurrent_state_history(self, x: Tensor):
        """Diagnostic: per-level state snapshots at every token (recurrent path).

        Returns a list of length L; entry lvl is a tensor (T,B,H,d_v,d_k) of S^(lvl) after each
        token. Used to verify boundary-only writes and to inspect bucket dynamics. Small inputs only.
        """
        return self._forward_recurrent(x, detach_value=False, record_states=True)[1]

    # ---------------------------------------------------------------- recurrent reference
    def _forward_recurrent(self, x: Tensor, detach_value: bool, record_states: bool = False):
        B, T, _ = x.shape
        H, dk, dv, L = self.H, self.dk, self.dv, self.L
        dev = x.device
        q0, k0, v0, a0, b0 = self._fast_qkv(x)
        projs = [c.project(x) for c in self.consolidators]          # index lvl-1
        read_qs = [None] + [self._level_read_query(x, lvl) for lvl in range(1, L)]

        S = [torch.zeros(B, H, dv, dk, device=dev) for _ in range(L)]
        # online accumulators for levels 1..L-1
        m = [None] + [x.new_full((B, H), float("-inf")) for _ in range(1, L)]
        den = [None] + [x.new_zeros(B, H) for _ in range(1, L)]
        numK = [None] + [x.new_zeros(B, H, dk) for _ in range(1, L)]
        numR = [None] + [x.new_zeros(B, H, dv) for _ in range(1, L)]

        out_per_level = [x.new_zeros(B, H, T, dv) for _ in range(L)]
        history: List[List[Tensor]] = [[] for _ in range(L)]
        for t in range(T):
            # level 0 fast write + read
            S[0] = delta_update(S[0], k0[:, :, t], v0[:, :, t], a0[:, :, t], b0[:, :, t])
            out_per_level[0][:, :, t] = torch.einsum("bhvk,bhk->bhv", S[0], q0[:, :, t])
            # levels 1..L-1, in order (each reads the freshest lower state)
            for lvl in range(1, L):
                p = projs[lvl - 1]
                P = self.periods[lvl]
                lower = S[lvl - 1].detach() if detach_value else S[lvl - 1]
                if p.vtok is not None:                               # token ablation
                    r_t = p.vtok[:, :, t]
                else:                                                # memory-of-memory readout
                    r_t = torch.einsum("bhvk,bhk->bhv", lower, p.probe[:, :, t])
                # online salience accumulation (running-max rescale)
                s_t = p.sal[:, :, t]
                m_new = torch.maximum(m[lvl], s_t)
                rescale = torch.where(torch.isinf(m[lvl]), torch.zeros_like(s_t), torch.exp(m[lvl] - m_new))
                w_t = torch.exp(s_t - m_new)
                den[lvl] = den[lvl] * rescale + w_t
                numK[lvl] = numK[lvl] * rescale.unsqueeze(-1) + w_t.unsqueeze(-1) * p.key[:, :, t]
                numR[lvl] = numR[lvl] * rescale.unsqueeze(-1) + w_t.unsqueeze(-1) * r_t
                m[lvl] = m_new
                if (t + 1) % P == 0:                                 # window boundary -> one write
                    kbar = l2norm(numK[lvl] / den[lvl].unsqueeze(-1))
                    rbar = numR[lvl] / den[lvl].unsqueeze(-1)
                    S[lvl] = delta_update(S[lvl], kbar, rbar, p.alpha[:, :, t], p.beta[:, :, t])
                    m[lvl] = x.new_full((B, H), float("-inf"))
                    den[lvl] = x.new_zeros(B, H)
                    numK[lvl] = x.new_zeros(B, H, dk)
                    numR[lvl] = x.new_zeros(B, H, dv)
                if (t + 1) >= P:                                     # live mask
                    out_per_level[lvl][:, :, t] = torch.einsum("bhvk,bhk->bhv", S[lvl], read_qs[lvl][:, :, t])
            if record_states:
                for lvl in range(L):
                    history[lvl].append(S[lvl].detach().clone())

        y = self._readout(x, out_per_level)
        if record_states:
            return y, [torch.stack(h, dim=0) for h in history]   # each (T,B,H,d_v,d_k)
        return y

    # ---------------------------------------------------------------- chunked
    def _forward_chunk(self, x: Tensor, detach_value: bool) -> Tensor:
        B, T, _ = x.shape
        L = self.L
        q0, k0, v0, a0, b0 = self._fast_qkv(x)
        o0, _ = gated_delta_chunkwise(q0, k0, v0, a0, b0, chunk_size=self.chunk_size, exact=False)
        out_per_level = [o0]
        lower_states: Optional[List[Tensor]] = None   # post-window states of the level below

        for lvl in range(1, L):
            p = self.consolidators[lvl - 1].project(x)
            P = self.periods[lvl]
            # value stream r_t = readout of the lower bucket with probe p_t
            if p.vtok is not None:
                r = p.vtok
            elif lvl == 1:
                # lower is the fast (dense-evolving) bucket -> chunkwise probe read
                if detach_value:
                    r = read_state_with_query(p.probe, k0.detach(), v0.detach(), a0.detach(),
                                              b0.detach(), chunk_size=self.chunk_size,
                                              mode="chunk", exact=False)
                else:
                    r = read_state_with_query(p.probe, k0, v0, a0, b0, chunk_size=self.chunk_size,
                                              mode="chunk", exact=False)
            else:
                # lower is a piecewise-constant consolidated bucket
                st = [s.detach() for s in lower_states] if detach_value else lower_states
                r = self._piecewise_state_read(st, self.periods[lvl - 1], p.probe, T)
            # Option C aggregation over complete windows
            kbar, rbar, bidx = aggregate_batch(p.key, r, p.sal, P)
            W = kbar.shape[2]
            # sequential boundary writes
            states: List[Tensor] = []
            S = x.new_zeros(B, self.H, self.dv, self.dk)
            for w in range(W):
                a_w = p.alpha[:, :, bidx[w]]
                b_w = p.beta[:, :, bidx[w]]
                S = delta_update(S, l2norm(kbar[:, :, w]), rbar[:, :, w], a_w, b_w)
                states.append(S)
            # readout for this level (piecewise; 0 before first boundary handles the live mask)
            rq = self._level_read_query(x, lvl)
            o_lvl = self._piecewise_state_read(states, P, rq, T)
            out_per_level.append(o_lvl)
            lower_states = states

        return self._readout(x, out_per_level)

    # ---------------------------------------------------------------- public
    def forward(self, x: Tensor, mode: str = "chunk", detach_value: bool = False) -> Tensor:
        if mode == "recurrent":
            return self._forward_recurrent(x, detach_value)
        return self._forward_chunk(x, detach_value)
