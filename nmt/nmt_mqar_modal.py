"""
MQAR — train + eval like the Memory Caching paper (Modal H200).

Multi-Query Associative Recall (Arora et al., Zoology), the synthetic recall
benchmark used in the Memory Caching paper. A sequence presents N (key->value)
token pairs; the model must recall the value for each of Q queried keys.

The paper's thesis: a FIXED-size RNN memory saturates once #pairs exceeds its
capacity (~dim), and Memory Caching restores recall by caching multiple segment
memories. We reproduce that at a fixed small model dim, sweeping #pairs, with
three readers:
  - baseline : ONE linear-attention memory over all pairs            (capacity ~d)
  - GRM      : dense aggregation over L cached segment memories       (paper's MC)
  - SSC      : top-k sparse over segment memories                     (= our NMT read)

Train from scratch (CE on query positions), report accuracy vs #pairs.

Run:  modal run nmt/nmt_mqar_modal.py
"""

import modal

app = modal.App("nmt-mqar")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(image=image, gpu="H200", timeout=2400)
def run():
    import torch, torch.nn as nn, torch.nn.functional as F

    dev = "cuda"
    V = 8192               # vocab
    d = 64                 # model dim (SMALL on purpose: baseline saturates ~d pairs)
    L = 8                  # segments (for GRM/SSC)
    Q = 32                 # queries per sequence (multi-query)
    B = 256
    steps = 1500
    lr = 1e-3
    topk = 2
    N_sweep = [16, 32, 64, 128, 256]   # #key-value pairs
    seeds = 2

    def l2norm(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def gen(batch, N):
        # distinct keys per sequence; arbitrary value tokens
        keys = torch.argsort(torch.rand(batch, V, device=dev), dim=-1)[:, :N]  # [B,N] distinct
        vals = torch.randint(0, V, (batch, N), device=dev)                     # [B,N]
        qidx = torch.randint(0, N, (batch, Q), device=dev)                     # which pairs to query
        b = torch.arange(batch, device=dev)[:, None]
        tgt = vals[b, qidx]                                                    # [B,Q] target value tokens
        return keys, vals, qidx, tgt

    def make_model():
        E = nn.Embedding(V, d).to(dev)        # token embedding
        W_k = nn.Linear(d, d, bias=False).to(dev)
        readout = nn.Linear(d, V).to(dev)     # untied head -> chance at init
        log_tau = nn.Parameter(torch.zeros((), device=dev))
        return nn.ModuleDict(dict(E=E, W_k=W_k, readout=readout)), log_tau

    def forward(model, log_tau, keys, vals, qidx, N, variant):
        bsz = keys.shape[0]
        E, W_k, readout = model["E"], model["W_k"], model["readout"]
        phi_k = l2norm(W_k(E(keys)))                       # [B,N,d]
        v_emb = E(vals)                                    # [B,N,d]
        b = torch.arange(bsz, device=dev)[:, None]
        phi_q = l2norm(W_k(E(keys[b, qidx])))              # [B,Q,d]  query = the pair's key

        if variant == "baseline":
            sim = torch.einsum("bnd,bqd->bqn", phi_k, phi_q)     # [B,Q,N]
            y = torch.einsum("bnv,bqn->bqv", v_emb, sim)         # one memory over ALL pairs
        else:
            seg = N // L
            phi_k_l = phi_k.view(bsz, L, seg, d)
            v_l = v_emb.view(bsz, L, seg, d)
            sim = torch.einsum("blsd,bqd->bqls", phi_k_l, phi_q)  # [B,Q,L,seg]
            reads = torch.einsum("blsv,bqls->bqlv", v_l, sim)     # [B,Q,L,d] = M_l phi(q)
            score = sim.max(-1).values * log_tau.exp()            # [B,Q,L] route over segments
            if variant == "ssc":
                thr = score.topk(topk, dim=-1).values[..., -1:]
                score = score.masked_fill(score < thr, float("-inf"))
            w = F.softmax(score, dim=-1)                          # GRM (dense) or SSC (top-k)
            y = torch.einsum("bql,bqlv->bqv", w, reads)           # [B,Q,d]
        return readout(y)                                         # [B,Q,V]

    def train_one(N, variant, seed):
        torch.manual_seed(seed)
        model, log_tau = make_model()
        params = list(model.parameters()) + [log_tau]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            keys, vals, qidx, tgt = gen(B, N)
            logits = forward(model, log_tau, keys, vals, qidx, N, variant)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cor = tot = 0
            for _ in range(20):
                keys, vals, qidx, tgt = gen(512, N)
                logits = forward(model, log_tau, keys, vals, qidx, N, variant)
                cor += (logits.argmax(-1) == tgt).sum().item(); tot += tgt.numel()
        return cor / tot

    results = {}
    for variant in ("baseline", "grm", "ssc"):
        results[variant] = {}
        for N in N_sweep:
            accs = [train_one(N, variant, s) for s in range(seeds)]
            results[variant][N] = round(sum(accs) / len(accs), 4)
    return dict(cfg=dict(V=V, d=d, L=L, Q=Q, topk=topk, steps=steps, seeds=seeds),
                N_sweep=N_sweep, results=results, gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = run.remote()
    c = r["cfg"]
    print(f"\nGPU: {r['gpu']}   MQAR  (dim d={c['d']}, {c['L']} segments, {c['Q']} queries, vocab {c['V']})")
    print(f"baseline = single fixed memory (capacity ~d={c['d']});  grm/ssc = {c['L']} cached segment memories\n")
    Ns = r["N_sweep"]
    print(f"{'reader':<10}" + "".join(f"{('N=' + str(n)):>9}" for n in Ns))
    print("-" * (10 + 9 * len(Ns)))
    for variant in ("baseline", "grm", "ssc"):
        print(f"{variant:<10}" + "".join(f"{r['results'][variant][n]:>9.3f}" for n in Ns))
    print("\n(accuracy; baseline should collapse once N exceeds ~d; grm/ssc should hold)\n")
