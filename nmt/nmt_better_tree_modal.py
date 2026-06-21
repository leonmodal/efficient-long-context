"""
Better tree — content-organized + parallelizable, in the SIMILARITY regime (Modal H200).

Lesson from the failed run: a bounded-summary log-tree CANNOT do exact unique-key
recall (info-theoretically impossible — compressing N random keys loses the needle).
Trees are for SIMILARITY search: retrieve pages RELEVANT to a query (a neighborhood),
which is the real NMT use case and HAS content structure to exploit.

We compare, on clustered data with similarity queries, at scale:
  flat       : scan all L, exact top-k                              (O(L), reference)
  temporal   : balanced tree, leaves in ARRIVAL order               (content-blind)
  content    : balanced tree, leaves ordered by a PARALLEL LSH sort  (content-organized)

Everything is parallel: LSH sort = matmul + sort; tree build = batched mean-pool;
beam search = log_branch(L) batched steps. No sequential per-item clustering.

Metric: hit@k = is the true nearest leaf in the method's top-k?  + nodes scored.

Run:  modal run nmt/nmt_better_tree_modal.py
"""

import modal

app = modal.App("nmt-better-tree")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(image=image, gpu="H200", timeout=1200)
def run():
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    d = 64; G = 1024; sigma = 0.15; branch = 8; beam_size = 16; top_k = 8; nbits = 24
    L_batch = {512: 256, 4096: 128, 32768: 48}

    def l2n(x, dim=-1):
        return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

    def build_levels(lk):                       # lk [B,L,d] -> list of [B, L/b^l, d]
        lv = [lk]; cur = lk
        while cur.shape[1] > 1:
            B_, n, d_ = cur.shape
            lv.append(l2n(cur.view(B_, n // branch, branch, d_).mean(2))); cur = lv[-1]
        return lv

    def gather(level, idx):                     # level [B,n,d], idx [B,M] -> [B,M,d]
        B_, M = idx.shape
        return torch.gather(level, 1, idx.view(B_, M, 1).expand(B_, M, level.shape[2]))

    def tree_retrieve(levels, q):
        B_ = q.shape[0]; beam = torch.zeros(B_, 1, dtype=torch.long, device=dev); scored = 0
        for l in range(len(levels) - 1, 0, -1):
            cand = (beam.unsqueeze(-1) * branch + torch.arange(branch, device=dev)).reshape(B_, -1)
            s = torch.einsum("bmd,bd->bm", gather(levels[l - 1], cand), q); scored += cand.shape[1]
            beam = torch.gather(cand, 1, s.topk(min(beam_size, cand.shape[1]), -1).indices)
        ls = torch.einsum("bmd,bd->bm", gather(levels[0], beam), q)
        sel = torch.gather(beam, 1, ls.topk(min(top_k, beam.shape[1]), -1).indices)
        return sel, scored                      # positions in the (possibly reordered) leaf array

    res = {}
    for L, B in L_batch.items():
        cent = l2n(torch.randn(B, G, d, device=dev))
        assign = torch.randint(0, G, (B, L), device=dev)
        b1 = torch.arange(B, device=dev)[:, None]
        lk = l2n(cent[b1, assign] + sigma * torch.randn(B, L, d, device=dev))     # leaf keys (clustered)
        qc = torch.randint(0, G, (B,), device=dev)
        q = l2n(cent[torch.arange(B, device=dev), qc] + sigma * torch.randn(B, d, device=dev))  # similarity query
        true_nn = torch.einsum("bld,bd->bl", lk, q).argmax(-1)                    # exact nearest leaf

        # parallel LSH content ordering: sign-bits of random projections -> sort
        proj = torch.randn(d, nbits, device=dev)
        codes = ((lk @ proj) > 0).to(torch.float32) @ (2.0 ** torch.arange(nbits, device=dev))
        order = codes.argsort(dim=1)                                             # [B,L] content order

        def tree_hit(perm):
            lk_o = torch.gather(lk, 1, perm.unsqueeze(-1).expand(B, L, d))
            sel, scored = tree_retrieve(build_levels(lk_o), q)
            sel_orig = torch.gather(perm, 1, sel)                                # map back to original ids
            return (sel_orig == true_nn[:, None]).any(-1).float().mean().item(), scored

        ident = torch.arange(L, device=dev).expand(B, L)
        temp_hit, temp_nodes = tree_hit(ident)
        cont_hit, cont_nodes = tree_hit(order)
        flat_hit = 1.0  # flat exact top-k always contains the true NN by definition

        res[L] = dict(tokens=L * 128,
                      flat=dict(hit=flat_hit, nodes=L),
                      temporal=dict(hit=round(temp_hit, 3), nodes=temp_nodes),
                      content=dict(hit=round(cont_hit, 3), nodes=cont_nodes))
    return dict(cfg=dict(d=d, G=G, sigma=sigma, branch=branch, beam=beam_size, top_k=top_k),
                res=res, gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = run.remote()
    c = r["cfg"]
    print(f"\nGPU: {r['gpu']}   similarity retrieval, {c['G']} clusters, top-{c['top_k']}, leaf=128 tok\n")
    Ls = list(r["res"].keys())
    toks = [r["res"][L]["tokens"] for L in Ls]
    hdr = "".join(f"{(str(t//1000)+'K' if t<1_000_000 else str(t//1_000_000)+'M'):>11}" for t in toks)
    print(f"== hit@{c['top_k']} (true nearest leaf in top-k) ==")
    print(f"{'method':<22}{hdr}")
    print(f"{'  (#leaves)':<22}" + "".join(f"{L:>11}" for L in Ls))
    for m in ("flat", "temporal", "content"):
        print(f"{m:<22}" + "".join(f"{r['res'][L][m]['hit']:>11.3f}" for L in Ls))
    print(f"\n== nodes scored ==")
    print(f"{'method':<22}{hdr}")
    for m in ("flat", "temporal", "content"):
        print(f"{m:<22}" + "".join(f"{r['res'][L][m]['nodes']:>11}" for L in Ls))
    print("\ncontent tree (parallel LSH order) should track flat at ~log-L nodes;")
    print("temporal tree (content-blind) should collapse at scale.\n")
