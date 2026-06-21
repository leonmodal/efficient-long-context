"""Training harness (plan §8): optimizer, schedule, and task trainers.

AdamW (lr 3e-4, betas (0.9,0.95), wd 0.1), grad clip 1.0, linear warmup (5%) then cosine to 10%
of peak. bf16 autocast on CUDA; memory states stay fp32 inside the delta rule. The MQAR trainer
is used by the Phase 1 ablation (value_source memory vs token vs matched single bucket).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import MemFlowConfig
from .data import make_mqar, mqar_accuracy
from .model import MemFlowLM


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.05, floor_frac: float = 0.10):
    warmup = max(1, int(warmup_frac * total))
    if step < warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return peak * (floor_frac + (1 - floor_frac) * cos)


def make_optimizer(model, lr: float, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def matched_single_bucket(cfg: MemFlowConfig) -> MemFlowConfig:
    """Single-bucket config whose state size matches the multi-bucket model's total state.

    Total matrix-memory state across L buckets is L * head_dim^2; a single bucket matches it with
    head_dim' = round(head_dim * sqrt(L)) (plan §9 baseline 1)."""
    L = len(cfg.periods)
    new_head = int(round(cfg.head_dim * math.sqrt(L)))
    return replace(cfg, periods=(1,), head_dim=new_head)


def train_model_recall(model, data_fn, steps: int = 2000, lr: float = 3e-4, batch: int = 64,
                       device: str = "cpu", seed: int = 0, detach_steps: int = 0,
                       mode: str = "chunk", eval_batches: int = 8, log_every: int = 0) -> Dict:
    """Train any LM (forward(idx, targets, mode, detach_value)->(logits, loss)) on a recall task.

    `data_fn(batch, generator, device) -> (inputs, targets)` produces the task (MQAR or NIAH). The
    optimizer/schedule/loss/eval are identical across all models in the comparison; only the model
    and data_fn differ. Accuracy is over the supervised (non-IGNORE) positions."""
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    model = model.to(device)
    opt = make_optimizer(model, lr)
    use_amp = device == "cuda"
    curve = []
    model.train()
    for step in range(steps):
        x, y = data_fn(batch, g, device)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, steps, lr)
        detach = step < detach_steps
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y, mode=mode, detach_value=detach)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            curve.append((step, float(loss.item())))
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = data_fn(batch, g, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x, mode=mode)
            accs.append(mqar_accuracy(logits.float(), y))
    return {"accuracy": float(sum(accs) / len(accs)),
            "final_loss": curve[-1][1] if curve else None, "curve": curve,
            "params": model.num_params(), "eval_batches": eval_batches}


def mqar_data_fn(num_pairs: int, num_queries: int, vocab: int):
    from .data import make_mqar
    return lambda b, g, dev: make_mqar(b, num_pairs, num_queries, vocab, generator=g, device=dev)


def niah_data_fn(seq_len: int, vocab: int, depth: float = 0.1):
    from .data import make_niah
    sl = seq_len if seq_len % 2 == 1 else seq_len + 1     # make_niah needs odd length
    return lambda b, g, dev: make_niah(b, sl, vocab, depth=depth, generator=g, device=dev)


def train_model_mqar(model, vocab: int, num_pairs: int, num_queries: int, **kw) -> Dict:
    return train_model_recall(model, mqar_data_fn(num_pairs, num_queries, vocab), **kw)


def train_mqar(cfg: MemFlowConfig, num_pairs: int, num_queries: int, **kw) -> Dict:
    """Train a MemFlow config on MQAR (builds the model, then runs the shared loop)."""
    model = MemFlowLM(cfg)
    res = train_model_mqar(model, cfg.vocab_size, num_pairs, num_queries, **kw)
    res.update(head_dim=cfg.head_dim, periods=list(cfg.periods), value_source=cfg.value_source,
               total_state=len(cfg.periods) * cfg.head_dim ** 2 * cfg.n_heads)
    return res
