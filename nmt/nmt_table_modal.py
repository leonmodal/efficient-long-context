"""
Method comparison table — scale (tokens) x performance x search cost (Modal H200).

Methods (all retrieve a buried key->value from a long memory; no training):
  rnn          : single fixed state over the WHOLE stream (no caching)        -> O(1) read
  mc           : flat Memory Caching, 1 mean key / segment, scan all          -> O(L)
  mc_multikey  : flat, K keys / segment (max-route), scan all                 -> O(L)
  tree         : OUR branch-8 beam tree, K keys / node                        -> ~O(log L)

Leaf = 128-token block (so tokens = L * 128). Metrics: recall@1 (rank true value
among ALL stored values), nodes scored (compute proxy), wall-time per query.

Run:  modal run nmt/nmt_table_modal.py
"""

import modal

app = modal.App("nmt-table")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(image=image, gpu="H200", timeout=1200)
def run():
    import torch, time

    torch.manual_seed(0)
    dev = "cuda"
    d = 128; K = 8; branch = 8; beam_size = 16; top_k = 4
    tok_per_leaf = 128
    L_batch = {64: 512, 512: 256, 4096: 128, 32768: 48}

    def l2n(x, dim=-1):
        return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

    def sc(keys, q):                                   # keys[B,n,K,d], q[B,d] -> [B,n]
        return torch.einsum("bnkd,bd->bnk", keys, q).max(-1).values

    def gather(level, idx):                            # level[B,n,K,d], idx[B,M] -> [B,M,K,d]
        B_, M = idx.shape
        return torch.gather(level, 1, idx.view(B_, M, 1, 1).expand(B_, M, level.shape[2], level.shape[3]))

    def build_levels(lk):
        lv = [lk]; cur = lk
        while cur.shape[1] > 1:
            B_, n, k_, d_ = cur.shape
            lv.append(l2n(cur.view(B_, n // branch, branch, k_, d_).mean(3))); cur = lv[-1]
        return lv

    def tree_route(levels, q):
        B_ = q.shape[0]
        beam = torch.zeros(B_, 1, dtype=torch.long, device=dev); scored = 0
        for l in range(len(levels) - 1, 0, -1):
            cand = (beam.unsqueeze(-1) * branch + torch.arange(branch, device=dev)).reshape(B_, -1)
            s = sc(gather(levels[l - 1], cand), q); scored += cand.shape[1]
            beam = torch.gather(cand, 1, s.topk(min(beam_size, cand.shape[1]), -1).indices)
        ls = sc(gather(levels[0], beam), q)
        top = ls.topk(min(top_k, beam.shape[1]), -1)
        return torch.gather(beam, 1, top.indices), top.values, scored

    def read_rank(sel, sel_s, lk, lv, q, true, N):
        selk, selv = gather(lk, sel), gather(lv, sel)
        sim = torch.einsum("btkd,bd->btk", selk, q)
        reads = torch.einsum("btke,btk->bte", selv, sim)
        y = torch.einsum("bt,bte->be", torch.softmax(sel_s, -1), reads)
        cos = torch.einsum("bnd,bd->bn", l2n(lv.reshape(y.shape[0], N, d)), l2n(y))
        return (cos.argmax(-1) == true).float().mean().item()

    def timed(fn, reps=8):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); return (time.time() - t0) / reps

    res = {}
    for L, B in L_batch.items():
        N = L * K
        lk = l2n(torch.randn(B, L, K, d, device=dev))
        lv = l2n(torch.randn(B, L, K, d, device=dev))
        qseg = torch.randint(0, L, (B,), device=dev); qpos = torch.randint(0, K, (B,), device=dev)
        b = torch.arange(B, device=dev); q = lk[b, qseg, qpos]; true = qseg * K + qpos
        levels = build_levels(lk)
        mc_key = l2n(lk.mean(2))

        def do_rnn():
            sim = torch.einsum("blkd,bd->blk", lk, q)
            y = torch.einsum("blke,blk->be", lv, sim)
            cos = torch.einsum("bnd,bd->bn", l2n(lv.reshape(B, N, d)), l2n(y))
            return (cos.argmax(-1) == true).float().mean().item()

        def do_flat(score):
            top = score.topk(top_k, -1)
            return read_rank(top.indices, top.values, lk, lv, q, true, N)

        rnn_r = do_rnn()
        mc_r = do_flat(torch.einsum("bld,bd->bl", mc_key, q))
        mk_r = do_flat(sc(lk, q))
        sel, sel_s, scored = tree_route(levels, q)
        tree_r = read_rank(sel, sel_s, lk, lv, q, true, N)

        t_rnn = timed(do_rnn)
        t_mc = timed(lambda: do_flat(torch.einsum("bld,bd->bl", mc_key, q)))
        t_mk = timed(lambda: do_flat(sc(lk, q)))
        t_tree = timed(lambda: read_rank(*tree_route(levels, q)[:2], lk, lv, q, true, N))

        res[L] = dict(
            tokens=L * tok_per_leaf,
            recall=dict(rnn=round(rnn_r, 3), mc=round(mc_r, 3), mc_multikey=round(mk_r, 3), tree=round(tree_r, 3)),
            nodes=dict(rnn=1, mc=L, mc_multikey=L, tree=scored),
            us_per_q=dict(rnn=round(t_rnn / B * 1e6, 1), mc=round(t_mc / B * 1e6, 1),
                          mc_multikey=round(t_mk / B * 1e6, 1), tree=round(t_tree / B * 1e6, 1)),
        )
    return dict(cfg=dict(d=d, K=K, branch=branch, beam=beam_size, top_k=top_k), res=res,
                gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = run.remote()
    Ls = list(r["res"].keys())
    toks = [r["res"][L]["tokens"] for L in Ls]
    methods = [("rnn", "RNN (no cache)"), ("mc", "MC (1 key)"),
               ("mc_multikey", "MC + multi-key"), ("tree", "Tree + multi-key (ours)")]
    print(f"\nGPU: {r['gpu']}   (leaf = 128 tokens; recall@1 = recover buried value)\n")

    def tbl(title, key, fmt):
        print(f"== {title} ==")
        print(f"{'method':<26}" + "".join(f"{(str(t//1000)+'K' if t<1_000_000 else str(t//1_000_000)+'M'):>10}" for t in toks))
        print(f"{'  (#leaves)':<26}" + "".join(f"{L:>10}" for L in Ls))
        for mk, name in methods:
            print(f"{name:<26}" + "".join(fmt(r["res"][L][key][mk]) for L in Ls))
        print()

    tbl("recall@1 (performance)", "recall", lambda v: f"{v:>10.3f}")
    tbl("nodes scored (search compute)", "nodes", lambda v: f"{v:>10}")
    tbl("wall-time us/query (search)", "us_per_q", lambda v: f"{v:>10.1f}")
    print("tokens = #leaves x 128.  RNN=O(1) but saturates; flat MC=O(L); ours=O(log L) nodes.")
    print("note: GPU wall-time favors flat (one big matmul) vs the tree's sequential levels at these L.\n")
