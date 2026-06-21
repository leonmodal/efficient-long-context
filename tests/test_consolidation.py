"""Phase 1/2 acceptance (plan §4.3, §5):
  - Option-C online aggregation == batch aggregation,
  - the chunked multi-bucket layer == the recurrent reference (atol 1e-4),
  - slow buckets are written ONLY at window boundaries (state constant between them),
  - consolidation projections receive gradient after the first boundary.
"""
import torch

from memflow.config import MemFlowConfig
from memflow.consolidation import aggregate_batch, aggregate_online
from memflow.memflow_layer import MemFlowLayer


def _cfg(periods, C, **kw):
    return MemFlowConfig(d_model=48, n_heads=2, head_dim=12, vocab_size=16,
                         periods=periods, chunk_size=C, max_seq_len=512, **kw)


def test_online_equals_batch():
    torch.manual_seed(0)
    B, H, T, dk, dv, P = 2, 3, 48, 12, 16, 8
    key = torch.randn(B, H, T, dk, dtype=torch.float64)
    val = torch.randn(B, H, T, dv, dtype=torch.float64)
    sal = torch.randn(B, H, T, dtype=torch.float64) * 3.0   # large scores stress the running max
    kb, rb, bi = aggregate_batch(key, val, sal, P)
    ko, ro, bo = aggregate_online(key, val, sal, P)
    assert torch.equal(bi, bo)
    assert (kb - ko).abs().max().item() < 1e-9
    assert (rb - ro).abs().max().item() < 1e-9


def _equiv(periods, T, C, seed=0):
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        layer = MemFlowLayer(_cfg(periods, C))
        x = torch.randn(2, T, 48)
        with torch.no_grad():
            yr = layer(x, mode="recurrent")
            yc = layer(x, mode="chunk")
        return (yr - yc).abs().max().item()
    finally:
        torch.set_default_dtype(torch.float32)


def test_layer_recurrent_equals_chunk_two_bucket():
    assert _equiv((1, 4), 24, 8) < 1e-4
    assert _equiv((1, 8), 40, 16) < 1e-4
    assert _equiv((1, 12), 48, 16) < 1e-4    # period not a multiple of the compute chunk


def test_layer_recurrent_equals_chunk_n_bucket():
    assert _equiv((1, 4, 16), 64, 8) < 1e-4
    assert _equiv((1, 8, 32), 96, 16) < 1e-4


def test_boundary_only_writes():
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(2)
        layer = MemFlowLayer(_cfg((1, 4), 4))
        x = torch.randn(1, 16, 48)
        hist = layer.recurrent_state_history(x)        # list[L] of (T,B,H,dv,dk)
        S1 = hist[1]
        changed = [t for t in range(1, 16) if (S1[t] - S1[t - 1]).abs().max().item() > 0]
        assert changed == [3, 7, 11, 15], changed
        # level 0 (fast) changes every token
        S0 = hist[0]
        changed0 = [t for t in range(1, 16) if (S0[t] - S0[t - 1]).abs().max().item() > 0]
        assert changed0 == list(range(1, 16))
    finally:
        torch.set_default_dtype(torch.float32)


def test_masking_before_first_write():
    """periods=(1,128), T<128: the 2-bucket output equals the fast-only output exactly."""
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(1)
        layer = MemFlowLayer(_cfg((1, 128), 16))
        fast = MemFlowLayer(_cfg((1,), 16))
        for n in ["q_proj", "k_proj", "v_proj", "fast_decay", "b_proj", "g_proj", "o_norm", "o_proj",
                  "q_conv", "k_conv", "v_conv"]:
            getattr(fast, n).load_state_dict(getattr(layer, n).state_dict())
        x = torch.randn(2, 100, 48)
        for mode in ("chunk", "recurrent"):
            assert (layer(x, mode=mode) - fast(x, mode=mode)).abs().max().item() == 0.0
    finally:
        torch.set_default_dtype(torch.float32)


def test_detach_value_changes_grad_not_forward():
    """detach_value stops gradient through the memory-readout into the fast bucket, but the
    forward values are unchanged; the (non-detached) probe still receives gradient."""
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(4)
        layer = MemFlowLayer(_cfg((1, 4), 4))
        x = torch.randn(2, 16, 48)
        with torch.no_grad():
            y_off = layer(x, mode="chunk", detach_value=False)
            y_on = layer(x, mode="chunk", detach_value=True)
        assert (y_off - y_on).abs().max().item() == 0.0      # forward identical

        def grads(detach):
            layer.zero_grad(set_to_none=True)
            layer(x, mode="chunk", detach_value=detach).sum().backward()
            return (layer.v_proj.weight.grad.clone(),
                    layer.consolidators[0].p_proj.weight.grad.clone())

        gv_off, gp_off = grads(False)
        gv_on, gp_on = grads(True)
        # fast-bucket value proj: gradient differs because the memory-readout path is cut
        assert (gv_off - gv_on).abs().max().item() > 1e-9
        # probe is not detached -> still learns under the warmup
        assert gp_on.abs().sum().item() > 0
    finally:
        torch.set_default_dtype(torch.float32)


def test_consolidation_params_get_gradient():
    torch.manual_seed(3)
    layer = MemFlowLayer(_cfg((1, 4), 4))
    x = torch.randn(2, 16, 48, requires_grad=True)
    y = layer(x, mode="chunk")
    y.sum().backward()
    c = layer.consolidators[0]
    for name, p in [("p_proj", c.p_proj.weight), ("k_proj", c.k_proj.weight),
                    ("sal_proj", c.sal_proj.weight), ("decay.a_proj", c.decay.a_proj.weight),
                    ("b_proj", c.b_proj.weight), ("read_q", layer.read_q[0].weight)]:
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"no grad for {name}"
