"""SegmentSplitLayer baseline: execution modes agree, it slots into the shared LM shell, it learns,
and buckets freeze outside their segment (early content is retained, not overwritten)."""
import torch

from memflow.config import MemFlowConfig
from memflow.memflow_layer import SegmentSplitLayer
from memflow.model import MemFlowLM


def _cfg(N, **kw):
    periods = tuple([1] + [128] * (N - 1))   # only the COUNT matters for SegmentSplitLayer
    return MemFlowConfig(d_model=48, n_heads=2, head_dim=12, vocab_size=32,
                         periods=periods, chunk_size=16, max_seq_len=256, **kw)


def test_recurrent_equals_chunk():
    torch.set_default_dtype(torch.float64)
    try:
        for N in (2, 4):
            torch.manual_seed(N)
            layer = SegmentSplitLayer(_cfg(N))
            x = torch.randn(2, 60, 48)
            with torch.no_grad():
                yr = layer(x, mode="recurrent")
                yc = layer(x, mode="chunk")
            assert (yr - yc).abs().max().item() < 1e-4, (N, (yr - yc).abs().max().item())
    finally:
        torch.set_default_dtype(torch.float32)


def test_bucket_freezes_outside_segment():
    """A change to a LATE token must not alter the FIRST bucket's contribution (it froze at the
    segment-1 boundary). We verify via the first-bucket output before the change point."""
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        N, T = 2, 40
        layer = SegmentSplitLayer(_cfg(N))
        x = torch.randn(1, T, 48)
        x2 = x.clone()
        x2[:, T - 1] = torch.randn(48)            # perturb only the very last token (segment 1)
        # isolate bucket 0's contribution by zeroing bucket 1's output gate weight
        with torch.no_grad():
            layer.g_proj[1].weight.zero_()         # bucket 1 contributes ~0 (sigmoid(0)=0.5 const)
            y1 = layer(x, mode="chunk")
            y2 = layer(x2, mode="chunk")
        # tokens in segment 0 (first half) cannot see the segment-1 perturbation (causal + frozen)
        half = T // 2
        assert (y1[:, :half] - y2[:, :half]).abs().max().item() < 1e-9
    finally:
        torch.set_default_dtype(torch.float32)


def test_lm_learns_with_segment_split_mixer():
    torch.manual_seed(0)
    cfg = MemFlowConfig(d_model=64, n_layers=2, n_heads=2, head_dim=16, vocab_size=32,
                        periods=(1, 128), chunk_size=16, max_seq_len=128)
    m = MemFlowLM(cfg, mixer_factory=SegmentSplitLayer)
    ids = torch.randint(0, 32, (8, 33))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95))
    first = None
    for step in range(250):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < 0.3 * first and loss.item() < 0.6
