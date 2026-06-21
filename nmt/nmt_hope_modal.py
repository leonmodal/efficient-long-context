"""
HOPE experiment — does NMT routing beat the Memory Caching baseline? (Modal H200)

The Memory Caching paper routes with ONE mean-pooled key per segment
(r_i = <u, MeanPooling(S_i)>). That blurs a distinctive fact buried in a dense
segment. NMT keeps a small set of keys per node (max-scored) — so a buried key
survives as its own prototype.

Task: L segments, each packed with P distinct (key->value) pairs; query a buried
key. Sweep P (segment density). No training — pure routing/read geometry.

Metrics per router:
  hit@1 / hit@2 : is the queried key's segment the top-1 / in the top-2 selected?
  recall@1      : after a GRM read over the selected segments, does the recalled
                  value rank #1 among all N stored values?

Run:  modal run nmt/nmt_hope_modal.py
"""

import modal

app = modal.App("nmt-hope")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(image=image, gpu="H200", timeout=600)
def run():
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    d = 128
    L = 16            # segments
    topk = 2
    n_keys = 8        # NMT keys per segment (vs MC's 1 mean key) — fixed small budget
    B = 256           # sequences (Monte-Carlo)
    P_sweep = [8, 16, 32, 64, 128]

    def l2n(x, dim=-1):
        return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

    def eval_router(score, k, v, qk, qseg, N):
        b = torch.arange(qk.shape[0], device=dev)
        top = score.topk(topk, dim=-1).indices                       # [B,topk]
        hit1 = (score.argmax(-1) == qseg).float().mean().item()
        hit2 = (top == qseg[:, None]).any(-1).float().mean().item()
        # GRM read over the selected top-k segments
        masked = torch.full_like(score, float("-inf"))
        masked.scatter_(1, top, score.gather(1, top))
        w = torch.softmax(masked, dim=-1)                            # [B,L]
        sim = torch.einsum("blpd,bd->blp", k, qk)                    # [B,L,P]
        reads = torch.einsum("blpe,blp->ble", v, sim)                # [B,L,d] = M_seg phi(q)
        y = torch.einsum("bl,ble->be", w, reads)                     # [B,d]
        allv = v.reshape(qk.shape[0], N, d)
        cos = torch.einsum("bnd,bd->bn", l2n(allv), l2n(y))          # rank true value among all N
        true_idx = qseg * k.shape[2] + qpos_global[b]
        recall1 = (cos.argmax(-1) == true_idx).float().mean().item()
        return dict(hit1=round(hit1, 3), hit2=round(hit2, 3), recall1=round(recall1, 3))

    global qpos_global
    results = {}
    for P in P_sweep:
        N = L * P
        k = l2n(torch.randn(B, L, P, d, device=dev))
        v = l2n(torch.randn(B, L, P, d, device=dev))
        qseg = torch.randint(0, L, (B,), device=dev)
        qpos = torch.randint(0, P, (B,), device=dev)
        qpos_global = qpos
        b = torch.arange(B, device=dev)
        qk = k[b, qseg, qpos]                                        # exact buried key

        # --- Memory Caching router: ONE mean key per segment ---
        mc_key = l2n(k.mean(2))                                      # [B,L,d]
        mc_score = torch.einsum("bld,bd->bl", mc_key, qk)

        # --- NMT router: n_keys per segment (sub-pooled), max-scored ---
        g = max(1, P // n_keys); nk = min(n_keys, P)
        nmt_keys = l2n(k[:, :, :nk * g].reshape(B, L, nk, g, d).mean(3))   # [B,L,nk,d]
        nmt_score = torch.einsum("blkd,bd->blk", nmt_keys, qk).max(-1).values

        results[P] = dict(
            N=N,
            mc=eval_router(mc_score, k, v, qk, qseg, N),
            nmt=eval_router(nmt_score, k, v, qk, qseg, N),
        )
    return dict(cfg=dict(d=d, L=L, topk=topk, n_keys_nmt=n_keys, B=B),
                results=results, gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = run.remote()
    c = r["cfg"]
    print(f"\nGPU: {r['gpu']}   L={c['L']} segments, top-{c['topk']} selected,"
          f" MC=1 mean key/seg vs NMT={c['n_keys_nmt']} keys/seg\n")
    Ps = list(r["results"].keys())
    for metric in ("hit1", "hit2", "recall1"):
        print(f"== {metric} ==")
        print(f"{'pairs/seg':<12}" + "".join(f"{('P=' + str(P)):>9}" for P in Ps))
        for router in ("mc", "nmt"):
            print(f"{router:<12}" + "".join(f"{r['results'][P][router][metric]:>9.3f}" for P in Ps))
        print()
    print("(N pairs = 16 x P; MC routes by a blurred mean key, NMT by multi-key max.")
    print(" widening MC->NMT gap as P grows = the tree/multi-key advantage over flat MC)\n")
