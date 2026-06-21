"""Phase 2 acceptance (plan §5): per-level masking and stability at level switch-on.

A bucket must contribute exactly nothing before its first write (t < P_ℓ), and its state must be
zero until then and change only at its own boundaries. No NaN/Inf as levels turn on.
"""
import torch

from memflow.config import MemFlowConfig
from memflow.memflow_layer import MemFlowLayer


def _cfg(periods, C):
    return MemFlowConfig(d_model=48, n_heads=2, head_dim=12, vocab_size=16,
                         periods=periods, chunk_size=C, max_seq_len=512)


def test_per_level_state_zero_until_first_write():
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        periods = (1, 4, 16)
        layer = MemFlowLayer(_cfg(periods, 8))
        x = torch.randn(1, 48, 48)
        hist = layer.recurrent_state_history(x)         # list[L] of (T, B,H,dv,dk)
        for lvl, P in enumerate(periods):
            S = hist[lvl]
            first_boundary = P - 1                       # 0-indexed token of first write
            # zero strictly before the first boundary
            if first_boundary > 0:
                assert S[:first_boundary].abs().max().item() == 0.0, f"level {lvl} nonzero pre-write"
            # changes occur only at this level's boundaries
            changed = [t for t in range(1, S.shape[0]) if (S[t] - S[t - 1]).abs().max().item() > 0]
            expected = [t for t in range(1, S.shape[0]) if (t + 1) % P == 0]
            assert changed == expected, f"level {lvl}: changed {changed} != boundaries {expected}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_no_nan_through_switch_on():
    torch.manual_seed(1)
    layer = MemFlowLayer(_cfg((1, 4, 16), 8))
    x = torch.randn(2, 40, 48, requires_grad=True)      # spans both switch-ons (t=3 and t=15)
    for mode in ("chunk", "recurrent"):
        y = layer(x, mode=mode)
        assert torch.isfinite(y).all(), f"non-finite output in {mode}"
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_diagnostics_ranges():
    torch.manual_seed(2)
    layer = MemFlowLayer(_cfg((1, 4, 16), 8))
    d = layer.diagnostics(torch.randn(2, 32, 48))
    assert set(d) >= {"alpha/level0", "alpha/level1", "alpha/level2"}
    for k, v in d.items():
        assert 0.0 < v < 1.0, f"{k}={v} out of (0,1)"
