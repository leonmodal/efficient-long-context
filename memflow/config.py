"""MemFlowConfig — all hyperparameters for the multi-timescale memory-flow layer.

See plan.md §2. Conventions (plan §0):
  - One memory state per head: S in R^{d_v x d_k}. Read: o = S @ q.
  - d_k = d_v = head_dim by default.
  - Buckets indexed by level l = 0,1,2,...; level 0 is the fast bucket (period 1).
  - C (chunk_size) is the *compute* chunk and is independent of every period P_l.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MemFlowConfig:
    # ---- model ----
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 64           # d_k = d_v = head_dim
    vocab_size: int = 50304
    max_seq_len: int = 8192
    mlp_ratio: float = 4.0       # feed-forward expansion in the decoder block
    norm_eps: float = 1e-5

    # ---- memory flow ----
    periods: Tuple[int, ...] = (1, 128, 512, 2048)   # P_l per level; periods[0] must be 1
    chunk_size: int = 64                              # compute chunk C (independent of periods)
    tie_probe_to_lower_key: bool = False              # W_p^(l) = W_k^(l-1) if True
    value_source: str = "memory"                      # "memory" (ours) | "token" (baseline ablation)
    use_output_gate: bool = True
    use_short_conv: bool = True                       # causal depthwise conv on fast q/k/v (standard for AR recall)
    conv_size: int = 4                                # short-conv kernel width
    decay_mode: str = "scalar"                        # "scalar" = Gated-DeltaNet per-head decay (default);
    #                                                   "channelwise" = per-dim decay (KDA-style, weaker on recall)
    seed_on_create: bool = False                      # Phase 3: seed a new level from level below

    # ---- growth (Phase 3+) ----
    grow: bool = False                                # False = fixed buckets (Phase 1/2)
    growth_base: int = 2                              # new level every time chunk-count crosses base^l

    # ---- training stability ----
    detach_value_steps: int = 2000                    # stop-grad on r_t -> S^(l-1) for first N steps
    alpha_bias_init: float = 4.0                      # high => alpha~1 at init (channel-wise mode only)
    beta_bias_init: float = 0.0                       # 0 => beta starts ~0.5, matching FLA Gated DeltaNet

    def __post_init__(self) -> None:
        if self.periods[0] != 1:
            raise ValueError(f"periods[0] must be 1 (fast bucket every token), got {self.periods[0]}")
        if any(p <= 0 for p in self.periods):
            raise ValueError(f"all periods must be positive, got {self.periods}")
        if list(self.periods) != sorted(self.periods):
            raise ValueError(f"periods must be non-decreasing, got {self.periods}")
        if self.value_source not in ("memory", "token"):
            raise ValueError(f"value_source must be 'memory' or 'token', got {self.value_source!r}")
        if self.d_model % self.n_heads != 0 and self.head_dim * self.n_heads != self.d_model:
            # head_dim may be set independently of d_model//n_heads; only warn via attribute.
            pass

    # ---- derived ----
    @property
    def d_k(self) -> int:
        return self.head_dim

    @property
    def d_v(self) -> int:
        return self.head_dim

    @property
    def n_levels(self) -> int:
        """Number of fixed levels (Phase 1/2). For grow=True this is the *cap* / initial size."""
        return len(self.periods)

    @property
    def inner_dim(self) -> int:
        """Total width across heads for one projection stream (q/k/v)."""
        return self.n_heads * self.head_dim

    @classmethod
    def from_yaml(cls, path: str) -> "MemFlowConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "periods" in data and isinstance(data["periods"], list):
            data["periods"] = tuple(data["periods"])
        fields = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)
