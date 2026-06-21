"""
Neural Memory Tree -- minimal GPU prototype (Modal).

Goal: verify the NMT mechanism end-to-end with *real* neural payloads:
    leaf payload  = a (gated) DeltaNet memory matrix written from a token block
    routing keys  = mean-pooled key vectors (leaf sub-blocks; parent = mean of child keys)
    retrieval     = beam tree search over routing keys (max_k q.k)
    read          = softmax-weighted sum of  M_i phi(q)

We compare two payload variants on a synthetic associative-recall stream:
    - "deltanet": plain delta rule, forget gate a_t = 1   (no forgetting)
    - "gda":      gated delta rule, forget gate a_t < 1   (forgetting)

and run a needle-position sweep to test the hypothesis that forgetting is
*not* wanted inside a leaf (it erases content written early in the block).

Run:  modal run nmt/nmt_modal.py
"""

import modal

app = modal.App("nmt-prototype")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


# --------------------------------------------------------------------------- #
# Core math (torch imported inside each fn so the module loads without torch). #
# --------------------------------------------------------------------------- #
def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def delta_rule_state(k, v, beta, gate):
    """Final (gated) DeltaNet memory matrix per leaf, sequential over tokens.

    k:    [L, n, d_k]   (L2-normalized inside -- this is phi(k))
    v:    [L, n, d_v]
    beta: [L, n]        write strength in (0, 1]
    gate: [L, n]        forget gate a_t in (0, 1];  a_t == 1 -> plain DeltaNet
    returns S: [L, d_v, d_k]

    Recurrence:  S_t = a_t * S_{t-1} + beta_t * (v_t - a_t * (S_{t-1} @ k_t)) @ k_t^T
    """
    import torch

    L, n, d_k = k.shape
    d_v = v.shape[-1]
    kn = l2norm(k, dim=-1)
    S = torch.zeros(L, d_v, d_k, device=k.device, dtype=k.dtype)
    for t in range(n):
        kt = kn[:, t, :]                                    # [L, d_k]
        vt = v[:, t, :]                                     # [L, d_v]
        at = gate[:, t].view(L, 1)                          # [L, 1]
        bt = beta[:, t].view(L, 1)                          # [L, 1]
        pred = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)   # [L, d_v]  = S @ kt
        delta = bt * (vt - at * pred)                       # [L, d_v]
        S = at.unsqueeze(-1) * S + torch.bmm(delta.unsqueeze(-1), kt.unsqueeze(1))
    return S                                                # [L, d_v, d_k]


def read_payload(S, q):
    """y(q) = S @ phi(q).  S: [..., d_v, d_k], q: [..., d_k] -> [..., d_v]."""
    import torch
    qn = l2norm(q, dim=-1)
    return torch.matmul(S, qn.unsqueeze(-1)).squeeze(-1)


def mean_pool_keys(keys, num_keys):
    """Mean-pool [L, m, d] into [L, num_keys, d] over contiguous groups, then L2-norm."""
    L, m, d = keys.shape
    g = m // num_keys
    pooled = l2norm(keys, dim=-1)[:, : num_keys * g, :].reshape(L, num_keys, g, d).mean(2)
    return l2norm(pooled, dim=-1)


# --------------------------------------------------------------------------- #
# Tree (forest) over leaves: branch-factor B, parent key = mean of child keys. #
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("level", "keys", "children", "leaf_index", "span")

    def __init__(self, level, keys, children=None, leaf_index=None, span=None):
        self.level = level
        self.keys = keys                     # [num_keys, d_k] (L2-normed)
        self.children = children or []
        self.leaf_index = leaf_index
        self.span = span

    @property
    def is_leaf(self):
        return self.leaf_index is not None


def kmeans(x, k, iters=12):
    """Tiny k-means: cluster [N,d] into k centroids (L2-normed). Used for content parent keys."""
    import torch
    N = x.shape[0]
    if N <= k:
        return l2norm(x, dim=-1)
    C = x[torch.randperm(N, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = x[m].mean(0)
    return l2norm(C, dim=-1)


def build_parent(children, method="mean_group", num_keys=8):
    """Summarize children's keys into parent keys.
    'mean_all'   -> 1 key  = mean of ALL child keys          (the single-key baseline)
    'mean_group' -> 1 key per child = that child's key-mean   (current default)
    'kmeans'     -> num_keys content centroids over all child keys
    """
    import torch
    child_keys = torch.cat([c.keys for c in children], 0)            # [sum_keys, d]
    if method == "mean_all":
        keys = child_keys.mean(0, keepdim=True)
    elif method == "kmeans":
        keys = kmeans(child_keys, num_keys)
    else:  # mean_group
        keys = torch.stack([c.keys.mean(0) for c in children], 0)
    return Node(children[0].level + 1, l2norm(keys, dim=-1),
                children=list(children),
                span=(children[0].span[0], children[-1].span[1]))


def build_tree(leaf_keys, branch_factor, parent_method="mean_group", parent_num_keys=8):
    """leaf_keys: [L, num_keys, d_k]. Bottom-up binary-counter build -> forest roots."""
    levels = []

    def insert(node):
        level = node.level
        while True:
            while len(levels) <= level:
                levels.append([])
            levels[level].append(node)
            if len(levels[level]) < branch_factor:
                break
            group = levels[level]
            levels[level] = []
            node = build_parent(group, parent_method, parent_num_keys)
            level += 1

    leaves = [Node(0, leaf_keys[i], leaf_index=i, span=(i, i + 1)) for i in range(leaf_keys.shape[0])]
    for lf in leaves:
        insert(lf)
    roots = [n for lvl in levels for n in lvl]
    return roots, leaves


def beam_retrieve(roots, q, beam_size, top_k):
    """Beam tree search. Returns (top_k leaf Nodes, scores, num_nodes_scored)."""
    qn = l2norm(q, dim=-1)

    def score(node):
        return float((node.keys @ qn).max())

    beam = list(roots)
    scored = 0
    while not all(n.is_leaf for n in beam):
        cand = []
        for n in beam:
            cand.append(n) if n.is_leaf else cand.extend(n.children)
        scored += len(cand)
        cand.sort(key=score, reverse=True)
        beam = cand[:beam_size]
    beam.sort(key=score, reverse=True)
    leaves = beam[:top_k]
    return leaves, [score(n) for n in leaves], scored


def flat_topk(leaf_keys, q, top_k):
    """Exact flat retrieval: max over each leaf's keys of q.k, take top_k leaves."""
    import torch
    qn = l2norm(q, dim=-1)
    s = torch.einsum("lkd,d->lk", leaf_keys, qn).max(dim=1).values   # [L]
    idx = torch.topk(s, top_k).indices.tolist()
    return idx, s


# --------------------------------------------------------------------------- #
# Routing-key configs + scenarios (for the single vs mean vs k-means comparison) #
# --------------------------------------------------------------------------- #
CONFIGS = {
    "single_key":   dict(num_keys=1, parent_method="mean_all"),     # 1 key/node, mean of all below
    "multi_mean":   dict(num_keys=8, parent_method="mean_group"),   # 8 keys/node, mean-by-child (current)
    "multi_kmeans": dict(num_keys=8, parent_method="kmeans"),       # 8 keys/node, content centroids
}


def build_needle(L, W, d, n_q, dev):
    """Distinctive buried fact: random keys; plant a unique needle key as ONE of a leaf's W keys."""
    import torch
    base_k = torch.randn(L, W, d, device=dev)
    targets = torch.randperm(L, device=dev)[:n_q]
    qk = l2norm(torch.randn(n_q, d, device=dev))
    for j, lf in enumerate(targets.tolist()):
        base_k[lf, 0] = qk[j]
    return base_k, targets, qk


def build_theme(L, W, d, n_q, dev):
    """Coherent chunk: each leaf's keys cluster around a theme; query = a leaf's noisy theme."""
    import torch
    themes = l2norm(torch.randn(L, d, device=dev))
    noise = l2norm(torch.randn(L, W, d, device=dev), dim=-1)
    base_k = themes.unsqueeze(1) + 0.4 * noise                       # coherent keys (mean_pool re-norms)
    targets = torch.randperm(L, device=dev)[:n_q]
    qk = l2norm(themes[targets] + 0.4 * l2norm(torch.randn(n_q, d, device=dev), dim=-1))
    return base_k, targets, qk


def eval_config(base_k, targets, qk, kc, beam, top_k, branch):
    """Build the tree under routing-key config kc, query each target, report hit@k / recall / nodes."""
    leaf_keys = mean_pool_keys(base_k, kc["num_keys"])
    roots, _ = build_tree(leaf_keys, branch, kc["parent_method"], kc["num_keys"])
    hit = recall = scored = 0
    for j, tgt in enumerate(targets.tolist()):
        lvs, _, sc = beam_retrieve(roots, qk[j], beam, top_k)
        scored += sc
        hit += int(tgt in [n.leaf_index for n in lvs])
        recall += int(tgt in flat_topk(leaf_keys, qk[j], top_k)[0])
    N = len(targets)
    return dict(hit=round(hit / N, 3), flat=round(recall / N, 3), nodes=round(scored / N, 1))


# --------------------------------------------------------------------------- #
# Experiment                                                                    #
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu="H200", timeout=600)
def run_experiment():
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    cfg = dict(
        d_k=192, d_v=384, leaf_tokens=128, writes_per_leaf=16, num_keys=8,   # real per-head DeltaNet dims (mid config)
        n_leaves=512, branch_factor=8, beam_size=8, top_k=4, n_needles=16,
    )
    d_k, d_v = cfg["d_k"], cfg["d_v"]
    n, W = cfg["leaf_tokens"], cfg["writes_per_leaf"]
    L, B = cfg["n_leaves"], cfg["branch_factor"]
    out = {"cfg": cfg, "gpu": torch.cuda.get_device_name(0)}

    # ---- self-test: delta rule must recover orthogonal planted associations ----
    kt = l2norm(torch.eye(4, d_k, device=dev).unsqueeze(0))          # 4 orthonormal keys
    vt = l2norm(torch.randn(1, 4, d_v, device=dev))
    St = delta_rule_state(kt, vt, torch.ones(1, 4, device=dev), torch.ones(1, 4, device=dev))
    rec = read_payload(St[0], kt[0])                                  # [4, d_v]
    cos = torch.nn.functional.cosine_similarity(rec, vt[0], dim=-1)
    out["selftest_recall_cos"] = round(cos.mean().item(), 4)

    # ---------------------- build the synthetic stream -------------------------
    # Per leaf: W associations written at random positions in an n-token block
    # (beta=1 there, 0 elsewhere). Routing keys = the written keys, mean-pooled.
    base_k = torch.randn(L, n, d_k, device=dev)
    base_v = l2norm(torch.randn(L, n, d_v, device=dev))
    beta = torch.zeros(L, n, device=dev)
    write_pos = torch.stack([torch.randperm(n, device=dev)[:W] for _ in range(L)])  # [L, W]
    beta.scatter_(1, write_pos, 1.0)

    # plant needles: distinct leaves, needle written at the LAST token (pos n-1)
    needle_leaves = torch.randperm(L, device=dev)[: cfg["n_needles"]]
    needle_k = l2norm(torch.randn(cfg["n_needles"], d_k, device=dev))
    needle_v = l2norm(torch.randn(cfg["n_needles"], d_v, device=dev))
    for j, lf in enumerate(needle_leaves.tolist()):
        base_k[lf, n - 1] = needle_k[j]
        base_v[lf, n - 1] = needle_v[j]
        beta[lf, n - 1] = 1.0
        write_pos[lf, -1] = n - 1                                    # ensure it's a routing key

    # routing keys: gather written keys -> [L, W, d_k] -> mean-pool to num_keys
    wk = torch.gather(base_k, 1, write_pos.unsqueeze(-1).expand(-1, -1, d_k))
    leaf_keys = mean_pool_keys(wk, cfg["num_keys"])                  # [L, num_keys, d_k]
    roots, leaves = build_tree(leaf_keys, B)
    out["n_roots"] = len(roots)
    out["max_level"] = max(n_.level for n_ in roots)

    # ---- [1] retrieval (gate-independent: routing keys don't depend on gate) --
    def retrieval_stats(beam_size, top_k):
        hit = recall = scored_tot = 0
        for j, lf in enumerate(needle_leaves.tolist()):
            q = needle_k[j]
            leaves_r, _, scored = beam_retrieve(roots, q, beam_size, top_k)
            scored_tot += scored
            hit += int(lf in [nd.leaf_index for nd in leaves_r])
            recall += int(lf in flat_topk(leaf_keys, q, top_k)[0])
        N = cfg["n_needles"]
        return dict(beam=beam_size, top_k=top_k, hit_at_k=round(hit / N, 3),
                    flat_recall=round(recall / N, 3),
                    tree_nodes=round(scored_tot / N, 1), flat_nodes=L)

    out["retrieval"] = retrieval_stats(cfg["beam_size"], cfg["top_k"])
    out["retrieval_ablation"] = [retrieval_stats(b, k) for (b, k) in
                                 ((8, 4), (16, 8), (32, 16), (64, 32))]

    # ---- [2] read sanity: needle as MOST-RECENT write -> delta rule exact ------
    def read_recent(gate_val):
        S = delta_rule_state(base_k, base_v, beta, torch.full((L, n), gate_val, device=dev))
        cos = [torch.nn.functional.cosine_similarity(
                   read_payload(S[lf], needle_k[j]), needle_v[j], dim=0).item()
               for j, lf in enumerate(needle_leaves.tolist())]
        return round(sum(cos) / len(cos), 4)

    out["read_recent_cos"] = {"deltanet": read_recent(1.0), "gda_0.98": read_recent(0.98)}

    # ---- [3] retention: DeltaNet vs GDA as the in-leaf needle AGES -------------
    # nw associations per leaf, needle at slot s (0=oldest), nw-1-s competing writes
    # after it; gate applied every step. cosine = interference, magnitude = decay.
    def retention(gate_val, slot, nw):
        pk = l2norm(torch.randn(L, d_k, device=dev))
        pv = l2norm(torch.randn(L, d_v, device=dev))
        K = l2norm(torch.randn(L, nw, d_k, device=dev)); K[:, slot] = pk
        V = l2norm(torch.randn(L, nw, d_v, device=dev)); V[:, slot] = pv
        S = delta_rule_state(K, V, torch.ones(L, nw, device=dev),
                             torch.full((L, nw), gate_val, device=dev))
        y = read_payload(S, pk)
        cos = torch.nn.functional.cosine_similarity(y, pv, dim=-1).mean().item()
        mag = (y * pv).sum(-1).mean().item()                       # signed projection onto v*
        return [round(cos, 3), round(mag, 3)]

    nw = cfg["writes_per_leaf"]
    slots = {"oldest": 0, "middle": nw // 2, "newest": nw - 1}
    out["retention"] = {f"gate_{g}": {name: retention(g, s, nw) for name, s in slots.items()}
                        for g in (1.0, 0.98, 0.9)}

    # ---- [4] routing-key comparison: single vs multi(mean) vs multi(kmeans) @ fixed beam ----
    out["key_comparison"] = {}
    for scen, builder in (("needle", build_needle), ("theme", build_theme)):
        bk, targets, qk = builder(L, 8, d_k, cfg["n_needles"], dev)
        out["key_comparison"][scen] = {name: eval_config(bk, targets, qk, kc, cfg["beam_size"], cfg["top_k"], B)
                                       for name, kc in CONFIGS.items()}
    return out


@app.local_entrypoint()
def main():
    r = run_experiment.remote()
    cfg = r["cfg"]
    print(f"\nGPU: {r['gpu']}")
    print(f"config: {cfg}")
    print(f"delta-rule self-test recall cosine: {r['selftest_recall_cos']} (want ~1.0)")
    print(f"tree: {r['n_roots']} root(s), max level {r['max_level']}")

    print("\n[1] Retrieval (gate-independent) -- can beam tree search find the needle leaf?")
    rt = r["retrieval"]
    print(f"    tree hit@{cfg['top_k']} = {rt['hit_at_k']:.2f}   flat exact recall@{cfg['top_k']} = {rt['flat_recall']:.2f}"
          f"   nodes scored {rt['tree_nodes']}/{rt['flat_nodes']} (tree/flat)")
    print("    beam/top_k ablation (wider search recovers recall lost to mean-pool routing):")
    print(f"      {'beam':>6}{'top_k':>7}{'tree hit':>10}{'tree nodes':>12}")
    for a in r["retrieval_ablation"]:
        print(f"      {a['beam']:>6}{a['top_k']:>7}{a['hit_at_k']:>10.2f}{a['tree_nodes']:>12.1f}")

    print("\n[2] Read sanity -- needle as most-recent write (delta rule -> exact recovery):")
    for k_, v_ in r["read_recent_cos"].items():
        print(f"      {k_:<10} read cosine = {v_:.3f}")

    print("\n[3] Retention -- DeltaNet vs GDA as the in-leaf needle ages "
          f"({cfg['writes_per_leaf']} writes/leaf, with competition).")
    print("    (cosine = interference-robustness, magnitude = how much survives decay)\n")
    ret = r["retention"]
    cols = list(next(iter(ret.values())).keys())
    print(f"    {'gate':<10}" + "".join(f"{c + ' (cos,mag)':>20}" for c in cols))
    for g, row in ret.items():
        print(f"    {g:<10}" + "".join(f"{str(tuple(row[c])):>20}" for c in cols))

    print("\n[4] Routing-key comparison @ fixed beam=8 (key quality, NOT widened search):")
    for scen, configs in r["key_comparison"].items():
        tag = "distinctive buried fact" if scen == "needle" else "coherent chunk gist"
        print(f"  scenario = {scen}  ({tag}):")
        print(f"    {'config':<14}{'tree hit@4':>12}{'flat recall':>13}{'tree nodes':>12}")
        for name, m in configs.items():
            print(f"    {name:<14}{m['hit']:>12.2f}{m['flat']:>13.2f}{m['nodes']:>12.1f}")
    print()
