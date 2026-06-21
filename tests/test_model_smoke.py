"""Phase 0 acceptance: single-bucket LM forwards, the two execution modes agree, gradients are
finite, and the model can drive training loss down (it learns)."""
import math

import torch

from memflow.config import MemFlowConfig
from memflow.model import MemFlowLM


def _tiny_cfg():
    return MemFlowConfig(d_model=64, n_layers=2, n_heads=2, head_dim=16, vocab_size=32,
                         periods=(1,), chunk_size=16, max_seq_len=64)


def test_forward_shape_and_mode_equivalence():
    torch.manual_seed(0)
    m = MemFlowLM(_tiny_cfg())
    idx = torch.randint(0, 32, (4, 48))
    with torch.no_grad():
        lo_c, _ = m(idx, mode="chunk")
        lo_r, _ = m(idx, mode="recurrent")
    assert lo_c.shape == (4, 48, 32)
    assert (lo_c - lo_r).abs().max().item() < 1e-4  # chunk == recurrent end-to-end


def test_gradients_finite():
    torch.manual_seed(0)
    m = MemFlowLM(_tiny_cfg())
    idx = torch.randint(0, 32, (4, 32))
    _, loss = m(idx[:, :-1], idx[:, 1:])
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"


def test_model_learns():
    torch.manual_seed(0)
    m = MemFlowLM(_tiny_cfg())
    ids = torch.randint(0, 32, (8, 33))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95))
    first = None
    for step in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
    assert first > math.log(32) - 0.5     # starts near uniform
    assert loss.item() < 0.5              # learns the fixed batch
    assert loss.item() < 0.2 * first      # large relative drop
    assert all(torch.isfinite(p).all() for p in m.parameters())
