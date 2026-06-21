"""Phase 0 acceptance: chunkwise gated delta rule == recurrent reference (atol 1e-4).

Covers the plan §3.3 requirement plus stress cases:
  - output o and final state S match,
  - channel-wise alpha (KDA-style), varying decay strengths,
  - chunk_size that does NOT divide T (ragged last chunk),
  - read_state_with_query reduces to the recurrent output when the probe == q.
"""
import torch

from memflow.delta_rule import (
    gated_delta_chunkwise,
    gated_delta_recurrent,
    l2norm,
    read_state_with_query,
)


def _make_inputs(B, H, T, d_k, d_v, seed, alpha_bias=0.0, beta_bias=0.0):
    g = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(B, H, T, d_k, generator=g, dtype=torch.float64))
    k = l2norm(torch.randn(B, H, T, d_k, generator=g, dtype=torch.float64))
    v = torch.randn(B, H, T, d_v, generator=g, dtype=torch.float64)
    alpha = torch.sigmoid(torch.randn(B, H, T, d_k, generator=g, dtype=torch.float64) + alpha_bias)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, generator=g, dtype=torch.float64) + beta_bias)
    return q, k, v, alpha, beta


def _check(B, H, T, d_k, d_v, chunk_size, seed, alpha_bias=0.0, beta_bias=0.0, atol=1e-4):
    q, k, v, alpha, beta = _make_inputs(B, H, T, d_k, d_v, seed, alpha_bias, beta_bias)
    o_rec, S_rec = gated_delta_recurrent(q, k, v, alpha, beta)
    o_chk, S_chk = gated_delta_chunkwise(q, k, v, alpha, beta, chunk_size=chunk_size)
    o_err = (o_rec - o_chk).abs().max().item()
    s_err = (S_rec - S_chk).abs().max().item()
    assert o_err < atol, f"output mismatch {o_err:.2e} (T={T},C={chunk_size},seed={seed})"
    assert s_err < atol, f"state mismatch {s_err:.2e} (T={T},C={chunk_size},seed={seed})"
    return o_err, s_err


def test_equiv_basic():
    _check(B=2, H=3, T=64, d_k=16, d_v=16, chunk_size=64, seed=0)


def test_equiv_multichunk():
    _check(B=2, H=2, T=192, d_k=16, d_v=24, chunk_size=64, seed=1)


def test_equiv_ragged_chunk():
    # chunk_size does NOT divide T -> exercises the ragged final chunk
    _check(B=2, H=2, T=100, d_k=16, d_v=16, chunk_size=32, seed=2)


def test_equiv_chunk1_equals_recurrent():
    # chunk_size == 1 must reproduce the per-token recurrence exactly
    _check(B=2, H=2, T=40, d_k=12, d_v=12, chunk_size=1, seed=3)


def test_equiv_slow_decay():
    # alpha near 1 (slow forgetting), as for high-level buckets
    _check(B=2, H=2, T=128, d_k=16, d_v=16, chunk_size=64, seed=4, alpha_bias=4.0, beta_bias=-2.0)


def test_equiv_fast_decay():
    # aggressive forgetting (small alpha) -> stress the decay normalization
    _check(B=2, H=2, T=128, d_k=16, d_v=16, chunk_size=64, seed=5, alpha_bias=-2.0)


def test_equiv_nontrivial_initial_state():
    q, k, v, alpha, beta = _make_inputs(2, 2, 96, 16, 16, seed=6)
    S0 = torch.randn(2, 2, 16, 16, dtype=torch.float64)
    o_rec, S_rec = gated_delta_recurrent(q, k, v, alpha, beta, S0=S0)
    o_chk, S_chk = gated_delta_chunkwise(q, k, v, alpha, beta, S0=S0, chunk_size=32)
    assert (o_rec - o_chk).abs().max().item() < 1e-4
    assert (S_rec - S_chk).abs().max().item() < 1e-4


def test_fast_chunkwise_matches_recurrent_alpha_near_one():
    """The fast matmul intra-chunk form (exact=False, used in training) matches the recurrent
    reference whenever alpha is near 1 (alpha_bias_init guarantees this regime)."""
    # alpha_bias=4 -> alpha ~ 0.98; cumulative decay over a chunk stays well within the clamp
    q, k, v, alpha, beta = _make_inputs(2, 3, 192, 24, 24, seed=21, alpha_bias=4.0, beta_bias=-1.0)
    o_rec, S_rec = gated_delta_recurrent(q, k, v, alpha, beta)
    o_fast, S_fast = gated_delta_chunkwise(q, k, v, alpha, beta, chunk_size=64, exact=False)
    assert (o_rec - o_fast).abs().max().item() < 1e-4
    assert (S_rec - S_fast).abs().max().item() < 1e-4


def test_recurrent_matches_literal_operator():
    """Verify the recurrent reference faithfully implements the plan's U operator
    U(S;k,v,a,b) = S diag(a) (I - b k kᵀ) + b v kᵀ, built here with EXPLICIT matrices
    (an independent code path from the delta-form loop). Read o_t = S_t q_t after the write.
    """
    q, k, v, alpha, beta = _make_inputs(2, 3, 37, 12, 14, seed=11)
    B, H, T, d_k = k.shape
    d_v = v.shape[-1]
    o_ref, S_ref = gated_delta_recurrent(q, k, v, alpha, beta)

    S = torch.zeros(B, H, d_v, d_k, dtype=torch.float64)
    eye = torch.eye(d_k, dtype=torch.float64)
    outs = []
    for t in range(T):
        a_t, k_t, v_t = alpha[:, :, t], k[:, :, t], v[:, :, t]
        b_t = beta[:, :, t]                                  # (B,H,1)
        diagA = torch.diag_embed(a_t)                        # (B,H,d_k,d_k)
        P = eye - b_t.unsqueeze(-1) * (k_t.unsqueeze(-1) * k_t.unsqueeze(-2))  # I - b k kᵀ
        vkt = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)          # v kᵀ -> (B,H,d_v,d_k)
        S = S @ diagA @ P + b_t.unsqueeze(-1) * vkt
        outs.append((S @ q[:, :, t].unsqueeze(-1)).squeeze(-1))
    o_lit = torch.stack(outs, dim=2)
    # float64 rounding accumulates over T steps; the two paths are algebraically identical.
    assert (o_ref - o_lit).abs().max().item() < 1e-6
    assert (S_ref - S).abs().max().item() < 1e-6


def test_read_with_query_matches_recurrent_when_probe_is_q():
    q, k, v, alpha, beta = _make_inputs(2, 2, 50, 16, 16, seed=7)
    o_rec, _ = gated_delta_recurrent(q, k, v, alpha, beta)
    r = read_state_with_query(q, k, v, alpha, beta)
    assert (o_rec - r).abs().max().item() < 1e-6


if __name__ == "__main__":
    import sys
    errs = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                errs.append(name)
    sys.exit(1 if errs else 0)
