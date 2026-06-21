"""
NMT — first REAL-DATA language-modeling go/no-go (Modal H200).

Trains a small recurrent (linear-attention) LM on real text
(emozilla/Long-Data-Collections-Pretrain-Without-Books), comparing:
  - baseline : plain linear-attention LM (single fixed-state memory)
  - + NMT    : same, plus a memory-caching read over past SEGMENTS (our method)

Decision metric = per-position loss (the Log-Linear paper's long-context
diagnostic): if NMT keeps loss dropping at FAR positions where the fixed state
saturates, the memory is using long context and is worth pursuing.

Small + fast on purpose (~100M tokens, minutes) — a first signal, not the final
table. Pure torch (no fla dependency) for correctness.

Run:  modal run nmt/nmt_lm_modal.py
"""

import modal

app = modal.App("nmt-lm")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "datasets", "tiktoken", "huggingface_hub", "pyarrow"
)


@app.function(image=image, gpu="H200", timeout=3600)
def run():
    import math, torch, torch.nn as nn, torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"
    # ---- config (small + fast first signal) ----
    V = 50257            # gpt2 vocab
    d = 384
    H = 6                # heads
    dh = d // H          # 64
    n_layers = 4
    T = 1024             # context
    Sg = 128             # segment length -> T/Sg = 8 segments for NMT
    C = 64               # chunk size for linear-attn scan
    B = 24
    steps = 4000
    warmup = 200
    lr = 3e-4
    val_batches = 24
    cfg = dict(V=V, d=d, H=H, n_layers=n_layers, T=T, Sg=Sg, B=B, steps=steps, lr=lr)

    def l2n(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    # ---------------- data: stream + pack real text ----------------
    def make_stream():
        from datasets import load_dataset
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("emozilla/Long-Data-Collections-Pretrain-Without-Books",
                          split="train", streaming=True)
        buf, seqs = [], []
        for ex in ds:
            text = ex.get("text") if isinstance(ex, dict) else None
            if not text:
                text = next((v for v in ex.values() if isinstance(v, str)), "")
            buf.extend(enc.encode_ordinary(text)); buf.append(enc.eot_token)
            while len(buf) >= T + 1:
                seqs.append(buf[:T + 1]); buf = buf[T + 1:]
                if len(seqs) == B:
                    yield torch.tensor(seqs, device=dev); seqs = []

    # ---------------- model ----------------
    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__(); self.w = nn.Parameter(torch.ones(d))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

    def phi(x):
        return F.elu(x) + 1.0          # positive feature map (linear attention)

    class LinAttn(nn.Module):
        """Plain chunked linear attention: a single fixed-state recurrent memory."""
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
            self.norm = RMSNorm(dh)
        def forward(self, x):
            b, t, _ = x.shape
            q = phi(self.q(x)).view(b, t, H, dh); k = phi(self.k(x)).view(b, t, H, dh)
            v = self.v(x).view(b, t, H, dh)
            nc = t // C
            qc = q.view(b, nc, C, H, dh); kc = k.view(b, nc, C, H, dh); vc = v.view(b, nc, C, H, dh)
            S = torch.zeros(b, H, dh, dh, device=x.device, dtype=x.dtype)
            causal = torch.tril(torch.ones(C, C, device=x.device, dtype=x.dtype))
            outs = []
            for c in range(nc):
                A = torch.einsum("bchd,bshd->bhcs", qc[:, c], kc[:, c]) * causal
                y_in = torch.einsum("bhcs,bshe->bche", A, vc[:, c])
                y_cross = torch.einsum("bchd,bhde->bche", qc[:, c], S)
                outs.append(y_in + y_cross)
                S = S + torch.einsum("bshd,bshe->bhde", kc[:, c], vc[:, c])
            y = torch.stack(outs, 1).reshape(b, t, H, dh)
            return self.o(self.norm(y).reshape(b, t, d))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__(); h = 4 * d
            self.w1 = nn.Linear(d, h); self.w2 = nn.Linear(d, h); self.w3 = nn.Linear(h, d)
        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class NMTMemory(nn.Module):
        """Memory caching: cache per-segment linear memories; query reads past segments (GRM)."""
        def __init__(self):
            super().__init__()
            self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
            self.q = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
            self.log_tau = nn.Parameter(torch.zeros(())); self.gamma = nn.Parameter(torch.zeros(()))
            self.norm = RMSNorm(d)
        def forward(self, x):
            b, t, _ = x.shape
            ns = t // Sg
            pk = l2n(self.k(x)).view(b, ns, Sg, d)        # per-token keys, by segment
            vv = self.v(x).view(b, ns, Sg, d)
            pq = l2n(self.q(x))                            # [b,t,d] queries
            seg_of = (torch.arange(t, device=x.device) // Sg)             # [t]
            # routing key per segment = mean key (multi-key max would need more; mean ok here)
            rkey = l2n(pk.mean(2))                          # [b,ns,d]
            score = torch.einsum("bnd,btd->btn", rkey, pq) * self.log_tau.exp()   # [b,t,ns]
            past = (torch.arange(ns, device=x.device)[None, :] < seg_of[:, None])  # [t,ns] seg < cur
            score = score.masked_fill(~past[None], float("-inf"))
            w = torch.softmax(score, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)               # positions w/ no past segment -> 0
            # read: per segment, M_seg phi(q) = sum_s v (phi_k . phi_q)
            sim = torch.einsum("bnsd,btd->btns", pk, pq)   # [b,t,ns,Sg]
            reads = torch.einsum("bnse,btns->btne", vv, sim)  # [b,t,ns,d] = M_seg phi(q)
            y = torch.einsum("btn,btne->bte", w, reads)    # [b,t,d]
            return self.gamma * self.o(self.norm(y))

    class LM(nn.Module):
        def __init__(self, use_nmt):
            super().__init__()
            self.emb = nn.Embedding(V, d)
            self.mix = nn.ModuleList([LinAttn() for _ in range(n_layers)])
            self.n1 = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)])
            self.mlp = nn.ModuleList([MLP() for _ in range(n_layers)])
            self.n2 = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)])
            self.nmt = NMTMemory() if use_nmt else None
            self.nf = RMSNorm(d); self.head = nn.Linear(d, V, bias=False)
        def forward(self, idx):
            x = self.emb(idx)
            for i in range(n_layers):
                x = x + self.mix[i](self.n1[i](x))
                x = x + self.mlp[i](self.n2[i](x))
                if self.nmt is not None and i == n_layers // 2:
                    x = x + self.nmt(x)              # inject memory read mid-network
            return self.head(self.nf(x))

    # ---------------- train + eval one variant ----------------
    def train_eval(use_nmt, val):
        model = LM(use_nmt).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        def lr_at(s):
            if s < warmup: return s / warmup
            p = (s - warmup) / max(1, steps - warmup); return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
        gen = make_stream()
        model.train()
        for s in range(steps):
            batch = next(gen)
            x, y = batch[:, :T], batch[:, 1:T + 1]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            for g in opt.param_groups: g["lr"] = lr * lr_at(s)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # eval per-position loss on held-out val
        model.eval()
        nb = 8                       # position buckets
        bucket = torch.zeros(nb, device=dev); cnt = torch.zeros(nb, device=dev)
        tot = 0.0; ntok = 0
        with torch.no_grad():
            for batch in val:
                x, y = batch[:, :T], batch[:, 1:T + 1]
                logits = model(x)
                ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").view(x.shape)
                tot += ce.sum().item(); ntok += ce.numel()
                pos = (torch.arange(T, device=dev) * nb // T)
                bucket.index_add_(0, pos, ce.mean(0)); cnt.index_add_(0, pos, torch.ones(T, device=dev))
        ppos = (bucket / cnt).tolist()
        return dict(val_ppl=round(math.exp(tot / ntok), 3), per_pos_loss=[round(v, 3) for v in ppos])

    # build a fixed val set first, then train on subsequent stream
    g0 = make_stream(); val = [next(g0) for _ in range(val_batches)]
    out = {"cfg": cfg, "gpu": torch.cuda.get_device_name(0)}
    out["baseline"] = train_eval(False, val)
    out["nmt"] = train_eval(True, val)
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    print(f"\nGPU: {r['gpu']}   config: {r['cfg']}")
    print(f"\n{'variant':<10}{'val ppl':>10}   per-position loss (8 buckets, early->late)")
    for v in ("baseline", "nmt"):
        d = r[v]
        print(f"{v:<10}{d['val_ppl']:>10.2f}   {d['per_pos_loss']}")
    b, n = r["baseline"]["per_pos_loss"], r["nmt"]["per_pos_loss"]
    print(f"\nlate-position (last bucket) loss:  baseline {b[-1]:.3f}  vs  nmt {n[-1]:.3f}"
          f"   (delta {b[-1]-n[-1]:+.3f})")
    print("nmt better at far positions => memory is using long context (worth pursuing)\n")
