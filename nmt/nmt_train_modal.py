"""
NMT — first TRAINING experiment (Modal H200).

Associative recall, trained end-to-end. Each example is a FRESH random set of
(key -> value-class) pairs, split into leaves. The model must:
  - write each leaf's pairs into a linear-attention payload  M_l = sum_p v_p phi(W_k k_p)^T
  - route a query key to the right leaf via LEARNED keys (W_k) + a learned temperature
  - read the value back and classify it.

Because the pairs are random PER EXAMPLE, the weights cannot memorize facts —
they can only learn the *mechanism* (write / route / read). This is the real test
of whether the NMT memory is trainable.

Trained DENSE (GRM: weight ALL leaves by softmax of routing scores) — fully
differentiable, no top-k. At eval we also try SPARSE top-k (SSC) to confirm the
train-dense / infer-sparse story, plus a NO-MEMORY control (predict from the query
key alone -> must be at chance, since the key carries no info about its value).

Run:  modal run nmt/nmt_train_modal.py
"""

import modal

app = modal.App("nmt-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(image=image, gpu="H200", timeout=1800)
def train():
    import torch, torch.nn as nn, torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"
    d, d_v, C = 192, 384, 256          # key dim, value dim, num value classes
    S, ppl = 256, 16                    # pairs per example, pairs per leaf
    L = S // ppl                        # 16 leaves
    B, steps, lr = 512, 3000, 2e-3
    cfg = dict(d=d, d_v=d_v, C=C, S=S, pairs_per_leaf=ppl, n_leaves=L, batch=B, steps=steps)

    def l2norm(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    # ---- learnable NMT memory params ----
    W_k = nn.Linear(d, d, bias=False).to(dev)          # routing + payload key projection
    val_embed = nn.Embedding(C, d_v).to(dev)           # value-class embeddings (stored in memory)
    readout = nn.Linear(d_v, C).to(dev)                # SEPARATE head: random at init -> chance at init
    log_tau = nn.Parameter(torch.zeros((), device=dev))  # routing sharpness (learned)
    mem_params = (list(W_k.parameters()) + list(val_embed.parameters())
                  + list(readout.parameters()) + [log_tau])
    opt = torch.optim.Adam(mem_params, lr=lr)

    # ---- no-memory control: predict class from the query key alone ----
    base = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, C)).to(dev)
    opt_b = torch.optim.Adam(base.parameters(), lr=lr)

    def gen(batch):
        K = l2norm(torch.randn(batch, S, d, device=dev))      # fresh random keys
        cls = torch.randint(0, C, (batch, S), device=dev)     # random value classes
        qi = torch.randint(0, S, (batch,), device=dev)        # which pair to query
        return K, cls, qi

    def forward_mem(K, cls, qi, topk=None):
        bsz = K.shape[0]
        phi = l2norm(W_k(K))                                   # [B,S,d]
        V = val_embed(cls)                                     # [B,S,d_v]
        phi_l = phi.view(bsz, L, ppl, d)
        V_l = V.view(bsz, L, ppl, d_v)
        b = torch.arange(bsz, device=dev)
        phi_q = l2norm(W_k(K[b, qi]))                          # [B,d]
        # per-pair similarity to the query (this expands M_l @ phi(q) without building M)
        sim = torch.einsum("blpd,bd->blp", phi_l, phi_q)       # [B,L,ppl]
        reads = torch.einsum("blpv,blp->blv", V_l, sim)        # [B,L,d_v]  = M_l phi(q)
        s = sim.max(-1).values * log_tau.exp()                 # [B,L] routing score (max over leaf keys)
        if topk is not None:                                   # SPARSE (SSC): keep only top-k leaves
            thr = s.topk(topk, dim=-1).values[:, -1:]
            s = s.masked_fill(s < thr, float("-inf"))
        w = F.softmax(s, dim=-1)                               # GRM weights (dense if topk=None)
        y = torch.einsum("bl,blv->bv", w, reads)               # [B,d_v]
        logits = readout(y)                                    # [B,C]  (untied -> must be learned)
        return logits, cls[b, qi]

    def forward_base(K, cls, qi):
        b = torch.arange(K.shape[0], device=dev)
        return base(K[b, qi]), cls[b, qi]

    # ---- train ----
    curve = []
    for step in range(steps):
        K, cls, qi = gen(B)
        logits, tgt = forward_mem(K, cls, qi)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

        lb, tb = forward_base(K, cls, qi)
        loss_b = F.cross_entropy(lb, tb)
        opt_b.zero_grad(); loss_b.backward(); opt_b.step()

        if step % 150 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt).float().mean().item()
                accb = (lb.argmax(-1) == tb).float().mean().item()
            curve.append(dict(step=step, loss=round(loss.item(), 3),
                              mem_acc=round(acc, 3), base_acc=round(accb, 3),
                              tau=round(log_tau.exp().item(), 1)))

    # ---- eval: dense vs sparse top-k vs no-memory (fresh data) ----
    with torch.no_grad():
        accs = {}
        for name, tk in (("dense_all_16", None), ("sparse_top4", 4), ("sparse_top1", 1)):
            cor = tot = 0
            for _ in range(20):
                K, cls, qi = gen(512)
                logits, tgt = forward_mem(K, cls, qi, topk=tk)
                cor += (logits.argmax(-1) == tgt).sum().item(); tot += tgt.numel()
            accs[name] = round(cor / tot, 4)
        cor = tot = 0
        for _ in range(20):
            K, cls, qi = gen(512)
            lb, tb = forward_base(K, cls, qi)
            cor += (lb.argmax(-1) == tb).sum().item(); tot += tb.numel()
        accs["no_memory"] = round(cor / tot, 4)

    return dict(cfg=cfg, chance=round(1.0 / C, 4), curve=curve,
                eval=accs, final_tau=round(log_tau.exp().item(), 1),
                gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = train.remote()
    print(f"\nGPU: {r['gpu']}   config: {r['cfg']}")
    print(f"chance accuracy = {r['chance']:.4f}  (1/{r['cfg']['C']} classes)\n")
    print(f"{'step':>6}{'loss':>9}{'mem_acc':>10}{'base_acc':>10}{'tau':>8}")
    for c in r["curve"]:
        print(f"{c['step']:>6}{c['loss']:>9.3f}{c['mem_acc']:>10.3f}{c['base_acc']:>10.3f}{c['tau']:>8.1f}")
    print(f"\nFinal eval (learned routing temperature = {r['final_tau']}):")
    for k, v in r["eval"].items():
        print(f"  {k:<14} acc = {v:.4f}")
    print()
