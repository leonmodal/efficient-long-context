"""
HIERARCHY advantage — NMT beam-tree vs flat Memory Caching at scale (Modal H200).

Sparse segments (P=8 pairs each, so per-segment keys are clean) but MANY of them
(L up to 32768 = "bigger length"). Query a buried key. Compare routing:
  - flat-MC    : 1 mean key / leaf, scan ALL L leaves, top-k          (paper's MC, O(L))
  - flat-exact : 8 keys / leaf, scan ALL L leaves (max), top-k        (good-key upper bound, O(L))
  - NMT-tree   : 8 keys / node, branch-8 beam tree search             (ours, ~O(log L))

Reports hit@4 (recall) and nodes-scored (cost). The hierarchy win = NMT-tree keeps
recall near flat-exact while scoring O(log L) nodes instead of O(L), and beats
flat-MC on recall too.  No training — pure routing geometry.

Run:  modal run nmt/nmt_hierarchy_modal.py
"""

import modal

app = modal.App("nmt-hierarchy")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(image=image, gpu="H200", timeout=900)
def run():
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    d = 128
    K = 8                 # keys per leaf (= P pairs)
    branch = 8
    beam_size = 16
    top_k = 4
    # L (leaves) = powers of 8 -> clean balanced tree;  batch shrinks as L grows
    L_batch = {64: 1024, 512: 512, 4096: 256, 32768: 96}

    def l2n(x, dim=-1):
        return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

    def score(keys, q):                                   # keys [B,n,K,d], q [B,d] -> [B,n]
        return torch.einsum("bnkd,bd->bnk", keys, q).max(-1).values

    def build_levels(leaf_keys):
        levels = [leaf_keys]; cur = leaf_keys
        while cur.shape[1] > 1:
            B_, n, k_, d_ = cur.shape
            grp = cur.view(B_, n // branch, branch, k_, d_)
            levels.append(l2n(grp.mean(dim=3)))           # parent = mean of each child's keys (branch keys)
            cur = levels[-1]
        return levels                                      # levels[0]=leaves ... levels[-1]=root

    def gather_keys(level, idx):                           # level [B,n,K,d], idx [B,M] -> [B,M,K,d]
        B_, M = idx.shape
        return torch.gather(level, 1, idx.view(B_, M, 1, 1).expand(B_, M, level.shape[2], level.shape[3]))

    def tree_retrieve(levels, q):
        B_ = q.shape[0]
        beam = torch.zeros(B_, 1, dtype=torch.long, device=dev)   # root index
        scored = 0
        for l in range(len(levels) - 1, 0, -1):
            cand = (beam.unsqueeze(-1) * branch + torch.arange(branch, device=dev)).reshape(B_, -1)
            s = score(gather_keys(levels[l - 1], cand), q)
            scored += cand.shape[1]
            top = s.topk(min(beam_size, cand.shape[1]), dim=-1).indices
            beam = torch.gather(cand, 1, top)               # indices into level l-1
        leaf_s = score(gather_keys(levels[0], beam), q)     # final leaves
        sel = torch.gather(beam, 1, leaf_s.topk(min(top_k, beam.shape[1]), dim=-1).indices)
        return sel, scored                                  # [B,top_k] leaf indices, nodes scored

    results = {}
    for L, B in L_batch.items():
        leaf_keys = l2n(torch.randn(B, L, K, d, device=dev))       # K clean keys per leaf
        qseg = torch.randint(0, L, (B,), device=dev)
        qpos = torch.randint(0, K, (B,), device=dev)
        b = torch.arange(B, device=dev)
        qk = leaf_keys[b, qseg, qpos]                               # exact buried key

        # flat-MC: 1 mean key / leaf
        mc_score = torch.einsum("bld,bd->bl", l2n(leaf_keys.mean(2)), qk)
        mc_hit = (mc_score.topk(top_k, -1).indices == qseg[:, None]).any(-1).float().mean().item()
        # flat-exact: K keys / leaf, scan all
        ex_score = score(leaf_keys, qk)
        ex_hit = (ex_score.topk(top_k, -1).indices == qseg[:, None]).any(-1).float().mean().item()
        # NMT tree
        levels = build_levels(leaf_keys)
        sel, scored = tree_retrieve(levels, qk)
        tree_hit = (sel == qseg[:, None]).any(-1).float().mean().item()

        results[L] = dict(
            depth=len(levels) - 1, batch=B,
            mc=dict(hit=round(mc_hit, 3), nodes=L),
            flat_exact=dict(hit=round(ex_hit, 3), nodes=L),
            tree=dict(hit=round(tree_hit, 3), nodes=scored),
        )
    return dict(cfg=dict(d=d, K=K, branch=branch, beam=beam_size, top_k=top_k),
                results=results, gpu=torch.cuda.get_device_name(0))


@app.local_entrypoint()
def main():
    r = run.remote()
    c = r["cfg"]
    print(f"\nGPU: {r['gpu']}   branch={c['branch']}, beam={c['beam']}, top-{c['top_k']},"
          f" {c['K']} keys/node\n")
    Ls = list(r["results"].keys())
    print(f"{'L (leaves)':<12}" + "".join(f"{L:>12}" for L in Ls))
    print(f"{'depth':<12}" + "".join(f"{r['results'][L]['depth']:>12}" for L in Ls))
    print("-- recall hit@4 --")
    for name in ("mc", "flat_exact", "tree"):
        print(f"{name:<12}" + "".join(f"{r['results'][L][name]['hit']:>12.3f}" for L in Ls))
    print("-- nodes scored (cost) --")
    for name in ("mc", "flat_exact", "tree"):
        print(f"{name:<12}" + "".join(f"{r['results'][L][name]['nodes']:>12}" for L in Ls))
    print("\nhierarchy win = tree hit ~ flat_exact, at nodes-scored << L (and tree hit > mc)\n")
