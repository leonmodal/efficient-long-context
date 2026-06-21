"""Modal app for MemFlow: run the correctness suite and experiments on real GPUs.

Usage (from the repo root):
    modal run memflow/modal_app.py                      # default: verify on L4
    modal run memflow/modal_app.py --action verify
    MEMFLOW_GPU=H100 modal run memflow/modal_app.py --action gpu-equiv

The `memflow` package and `tests/` are baked into the image. Heavy imports (torch, memflow)
live inside the remote functions so the local entrypoint only needs the `modal` SDK.
"""
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (has memflow/, tests/)
GPU = os.environ.get("MEMFLOW_GPU", "L4")  # override with e.g. MEMFLOW_GPU=H100 modal run ...

app = modal.App("memflow")

# Persistent results: every run writes its own JSON here the instant it finishes, so a mid-run
# failure (timeout, quota, disabled app) never loses completed work. `report`/`study` read it back.
results_vol = modal.Volume.from_name("memflow-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "einops", "pytest")
    .add_local_dir(os.path.join(REPO, "memflow"), remote_path="/root/memflow",
                   ignore=["__pycache__", "*.pyc"], copy=True)
    .add_local_dir(os.path.join(REPO, "tests"), remote_path="/root/tests",
                   ignore=["__pycache__", "*.pyc"], copy=True)
    .add_local_file(os.path.join(REPO, "pyproject.toml"), remote_path="/root/pyproject.toml", copy=True)
)

# Image for external baselines: bakes the vendored flash-linear-attention (`fla`) package so
# `import fla` works without pip resolution against our torch build (Triton ships with torch).
FLA_SRC = os.path.join(REPO, "TTT", "baselines", "flash-linear-attention", "fla")
image_gdn = (
    image.pip_install("transformers", "ninja")
    .add_local_dir(FLA_SRC, remote_path="/root/fla", ignore=["__pycache__", "*.pyc"], copy=True)
)

# Image for the log-linear (HGDN) baseline: adds the `hattention` package (which imports `fla`).
HATT_SRC = os.path.join(REPO, "log-linear-attention", "hattention")
image_hgdn = (
    image_gdn.pip_install("jaxtyping", "einops")
    .add_local_dir(HATT_SRC, remote_path="/root/hattention", ignore=["__pycache__", "*.pyc"], copy=True)
)


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def run_pytest() -> int:
    """Run the full correctness suite inside the GPU image."""
    import subprocess
    print("=== torch / device info ===", flush=True)
    import torch
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    r = subprocess.run(["python", "-m", "pytest", "-q"], cwd="/root")
    return r.returncode


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def gpu_equiv() -> dict:
    """CUDA equivalence in float32 (real training dtype) + bf16 autocast forward/backward.

    The CPU pytest uses float64; this confirms the chunkwise==recurrent property survives on
    GPU in the precisions we actually train with.
    """
    import torch
    from memflow.delta_rule import gated_delta_chunkwise, gated_delta_recurrent, l2norm
    from memflow.config import MemFlowConfig
    from memflow.model import MemFlowLM

    dev = "cuda"
    out = {}

    # --- float32 equivalence on CUDA ---
    g = torch.Generator(device="cpu").manual_seed(0)
    B, H, T, dk, dv, C = 4, 4, 256, 64, 64, 64
    q = l2norm(torch.randn(B, H, T, dk, generator=g)).to(dev)
    k = l2norm(torch.randn(B, H, T, dk, generator=g)).to(dev)
    v = torch.randn(B, H, T, dv, generator=g).to(dev)
    alpha = torch.sigmoid(torch.randn(B, H, T, dk, generator=g) + 3.0).to(dev)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, generator=g)).to(dev)
    o_rec, S_rec = gated_delta_recurrent(q, k, v, alpha, beta)
    o_chk, S_chk = gated_delta_chunkwise(q, k, v, alpha, beta, chunk_size=C)
    out["f32_o_err"] = (o_rec - o_chk).abs().max().item()
    out["f32_S_err"] = (S_rec - S_chk).abs().max().item()

    # --- model fwd/bwd under bf16 autocast on CUDA (states stay fp32 inside delta rule) ---
    cfg = MemFlowConfig(d_model=256, n_layers=4, n_heads=4, head_dim=64, vocab_size=512,
                        periods=(1,), chunk_size=64, max_seq_len=1024)
    m = MemFlowLM(cfg).to(dev)
    idx = torch.randint(0, 512, (4, 512), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(idx[:, :-1], idx[:, 1:])
    loss.backward()
    out["bf16_loss"] = loss.item()
    out["bf16_grad_finite"] = all(torch.isfinite(p.grad).all().item()
                                  for p in m.parameters() if p.grad is not None)
    out["device"] = torch.cuda.get_device_name(0)
    return out


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def train_sanity(steps: int = 400) -> dict:
    """Overfit a fixed batch on GPU to confirm the layer learns (loss drops, no NaN)."""
    import math
    import torch
    from memflow.config import MemFlowConfig
    from memflow.model import MemFlowLM

    dev = "cuda"
    torch.manual_seed(0)
    cfg = MemFlowConfig(d_model=128, n_layers=3, n_heads=4, head_dim=32, vocab_size=64,
                        periods=(1,), chunk_size=32, max_seq_len=128)
    m = MemFlowLM(cfg).to(dev)
    ids = torch.randint(0, 64, (16, 65), device=dev)
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95))
    first = None
    for step in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
    return {"first_loss": first, "final_loss": loss.item(), "ln_V": math.log(64),
            "params": m.num_params(), "device": torch.cuda.get_device_name(0)}


@app.function(image=image, gpu=GPU, timeout=30 * 60)
def mqar_run(spec: dict) -> dict:
    """Train one MQAR config (one point in the ablation) and return its accuracy + metadata."""
    import torch
    from memflow.config import MemFlowConfig
    from memflow.train import train_mqar, matched_single_bucket

    base = MemFlowConfig(
        d_model=spec["d_model"], n_layers=spec["n_layers"], n_heads=spec["n_heads"],
        head_dim=spec["head_dim"], vocab_size=spec["vocab"], periods=(1, spec["P"]),
        chunk_size=spec["chunk_size"], max_seq_len=4096, value_source="memory",
    )
    variant = spec["variant"]
    if variant == "memory":
        cfg = base
        detach = spec["steps"] // 5
    elif variant == "token":
        cfg = MemFlowConfig(**{**base.__dict__, "value_source": "token"})
        detach = 0
    elif variant == "single":            # matched-total-state single bucket
        cfg = matched_single_bucket(base)
        detach = 0
    else:
        raise ValueError(variant)

    res = train_mqar(cfg, num_pairs=spec["D"], num_queries=spec["Q"], steps=spec["steps"],
                     lr=spec["lr"], batch=spec["batch"], device="cuda", seed=spec["seed"],
                     detach_steps=detach, mode="chunk", eval_batches=8)
    res.update(variant=variant, D=spec["D"], P=spec["P"], seed=spec["seed"])
    return res


def _mqar_specs(steps: int, Ds=(16, 32, 64, 128), seeds=(0, 1), variants=("memory", "token", "single")) -> list:
    base = dict(d_model=128, n_layers=2, n_heads=4, head_dim=32, vocab=256, P=64,
                chunk_size=64, Q=32, lr=1e-3, batch=32, steps=steps)
    specs = []
    for seed in seeds:
        for D in Ds:
            for variant in variants:
                specs.append({**base, "variant": variant, "D": D, "seed": seed})
    return specs


def _periods_for(N: int, task: str, size: int) -> list:
    """Geometric MemFlow periods spanning the sequence (so all N buckets activate)."""
    if task == "mqar":
        return {2: [1, 64], 4: [1, 16, 64, 256]}[N]
    L = size  # niah length
    return {2: [1, max(2, L // 2)],
            4: [1, max(2, L // 8), max(4, L // 4), max(8, L // 2)]}[N]


def _study_specs(task: str, steps: int, seeds=(0, 1), Ns=(2, 4)):
    """Build (memflow_family_specs, gdn_specs) for the segment-split study on one task."""
    h = 32
    vocab = {"mqar": 512, "niah": 4096}[task]    # vocab/2 >= max #distinct keys (D / n_pairs)
    batch = {"mqar": 24, "niah": 12}[task]       # smaller batch for the longer NIAH sequences
    steps_t = steps if task == "mqar" else min(steps, 3000)   # NIAH-long is the slow pole
    base = dict(d_model=128, n_layers=2, n_heads=4, vocab=vocab, chunk_size=32, lr=1e-3,
                steps=steps_t, batch=batch, max_seq_len=8192)
    sizes = {"mqar": [16, 64, 256], "niah": [512, 1024, 2048]}[task]
    mf, gd = [], []
    for seed in seeds:
        for N in Ns:
            hs = int(round(h * (N ** 0.5)))            # matched-state single/gdn head_dim
            for size in sizes:
                common = {**base, "N": N, "task": task, "seed": seed}
                if task == "mqar":
                    common.update(D=size, Q=16)
                else:
                    common.update(length=size, depth=0.1)
                mf.append({**common, "variant": "memory", "head_dim": h, "periods": _periods_for(N, task, size)})
                mf.append({**common, "variant": "split", "head_dim": h, "periods": [1] * N})
                mf.append({**common, "variant": "single", "head_dim": hs, "periods": [1]})
                gd.append({**common, "variant": "gdn", "head_dim": hs})
    return mf, gd


def _print_study(results: list, task: str):
    from collections import defaultdict
    agg = defaultdict(list)                            # (N, size, variant) -> [acc]
    for r in results:
        agg[(r["N"], r["size"], r["variant"])].append(r["accuracy"])
    Ns = sorted({r["N"] for r in results})
    sizes = sorted({r["size"] for r in results})
    variants = ("memory", "split", "single", "gdn")
    label = "D" if task == "mqar" else "len"
    chance = 1.0 / (256 if task == "mqar" else 2048)   # 1 / value-space size
    for N in Ns:
        print(f"\n=== {task.upper()}  N={N} buckets (mean over seeds; chance≈{chance:.4f}) ===")
        print(f"{label:>6} | " + " ".join(f"{v:>9}" for v in variants))
        print("-" * (9 + 10 * len(variants)))
        for s in sizes:
            cells = []
            for v in variants:
                vals = agg[(N, s, v)]
                cells.append(f"{(sum(vals)/len(vals)) if vals else float('nan'):>9.3f}")
            print(f"{s:>6} | " + " ".join(cells))


def _hgdn_specs(steps: int, Ds=(16, 32, 64, 128), seeds=(0, 1)) -> list:
    # head_dim 64 keeps the log-linear triton kernel happy; matched across the comparison.
    base = dict(d_model=128, n_layers=2, n_heads=4, head_dim=64, vocab=256,
                Q=32, lr=1e-3, batch=32, steps=steps)
    return [{**base, "D": D, "seed": seed} for seed in seeds for D in Ds]


def _gdn_specs(steps: int, Ds=(16, 32, 64, 128), seeds=(0, 1)) -> list:
    # GDN head_dim set so its single-bucket state matches the memory model's TOTAL state
    # (2 buckets * 32^2 == 1 bucket * 45^2, per head): the matched-state standard-GDN baseline.
    base = dict(d_model=128, n_layers=2, n_heads=4, gdn_head_dim=45, vocab=256,
                Q=32, lr=1e-3, batch=32, steps=steps)
    return [{**base, "D": D, "seed": seed} for seed in seeds for D in Ds]


def _build_data_fn(spec):
    from memflow.train import mqar_data_fn, niah_data_fn
    if spec["task"] == "mqar":
        return mqar_data_fn(spec["D"], spec["Q"], spec["vocab"])
    return niah_data_fn(spec["length"], spec["vocab"], spec["depth"])


def _save_result(res: dict):
    """Persist one run's result to the volume immediately (survives any later failure)."""
    import json
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    size = res.get("size")
    fn = f"{res['task']}_{res['variant']}_N{res['N']}_sz{size}_seed{res['seed']}.json"
    with open(os.path.join(RESULTS_DIR, fn), "w") as f:
        json.dump(res, f)
    results_vol.commit()


@app.function(image=image, volumes={RESULTS_DIR: results_vol}, timeout=5 * 60)
def collect() -> list:
    """Read back every saved result from the volume (for report/aggregation)."""
    import json
    import glob
    import os
    results_vol.reload()
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out


@app.function(image=image, gpu=GPU, timeout=60 * 60, max_containers=8,
              volumes={RESULTS_DIR: results_vol})
def recall_run(spec: dict) -> dict:
    """Train a MemFlow-family variant (memory / split / single) on MQAR or NIAH."""
    from memflow.config import MemFlowConfig
    from memflow.model import MemFlowLM
    from memflow.memflow_layer import SegmentSplitLayer
    from memflow.train import train_model_recall

    cfg = MemFlowConfig(d_model=spec["d_model"], n_layers=spec["n_layers"], n_heads=spec["n_heads"],
                        head_dim=spec["head_dim"], vocab_size=spec["vocab"],
                        periods=tuple(spec["periods"]), chunk_size=spec["chunk_size"],
                        max_seq_len=spec["max_seq_len"], value_source="memory")
    factory = SegmentSplitLayer if spec["variant"] == "split" else None
    model = MemFlowLM(cfg, mixer_factory=factory) if factory else MemFlowLM(cfg)
    detach = spec["steps"] // 5 if spec["variant"] == "memory" else 0
    res = train_model_recall(model, _build_data_fn(spec), steps=spec["steps"], lr=spec["lr"],
                             batch=spec["batch"], device="cuda", seed=spec["seed"],
                             detach_steps=detach, mode="chunk", eval_batches=8)
    res.update(variant=spec["variant"], N=spec["N"], task=spec["task"], seed=spec["seed"],
               size=spec.get("D", spec.get("length")),
               total_state=spec["n_heads"] * spec["head_dim"] ** 2 * len(spec["periods"]))
    _save_result(res)
    return res


@app.function(image=image_gdn, gpu=GPU, timeout=60 * 60, max_containers=8,
              volumes={RESULTS_DIR: results_vol})
def gdn_recall_run(spec: dict) -> dict:
    """Train the standard Gated DeltaNet (fixed-state reference) on MQAR or NIAH."""
    from memflow.baselines import GDNBaselineLM
    from memflow.train import train_model_recall
    model = GDNBaselineLM(d_model=spec["d_model"], n_layers=spec["n_layers"],
                          n_heads=spec["n_heads"], head_dim=spec["head_dim"],
                          vocab_size=spec["vocab"], expand_v=1.0, use_short_conv=True)
    res = train_model_recall(model, _build_data_fn(spec), steps=spec["steps"], lr=spec["lr"],
                             batch=spec["batch"], device="cuda", seed=spec["seed"],
                             mode="chunk", eval_batches=8)
    res.update(variant="gdn", N=spec["N"], task=spec["task"], seed=spec["seed"],
               size=spec.get("D", spec.get("length")),
               total_state=spec["n_heads"] * spec["head_dim"] ** 2)
    _save_result(res)
    return res


@app.function(image=image_hgdn, gpu=GPU, timeout=30 * 60)
def hgdn_run(spec: dict) -> dict:
    """Train the log-linear (HGDN) baseline on the SAME MQAR + loop. Primary same-state-class peer."""
    from memflow.baselines import HGDNBaselineLM
    from memflow.train import train_model_mqar
    model = HGDNBaselineLM(d_model=spec["d_model"], n_layers=spec["n_layers"],
                           n_heads=spec["n_heads"], head_dim=spec["head_dim"],
                           vocab_size=spec["vocab"], expand_v=1.0, use_short_conv=True)
    res = train_model_mqar(model, spec["vocab"], spec["D"], spec["Q"], steps=spec["steps"],
                           lr=spec["lr"], batch=spec["batch"], device="cuda", seed=spec["seed"],
                           mode="chunk", eval_batches=8)
    res.update(variant="loglinear", D=spec["D"], seed=spec["seed"])
    return res


@app.function(image=image_gdn, gpu=GPU, timeout=30 * 60)
def gdn_run(spec: dict) -> dict:
    """Train a standard Gated DeltaNet baseline (FLA) on the SAME MQAR + shell + loop."""
    from memflow.baselines import GDNBaselineLM
    from memflow.train import train_model_mqar
    model = GDNBaselineLM(d_model=spec["d_model"], n_layers=spec["n_layers"],
                          n_heads=spec["n_heads"], head_dim=spec["gdn_head_dim"],
                          vocab_size=spec["vocab"], expand_v=1.0, use_short_conv=True)
    res = train_model_mqar(model, spec["vocab"], spec["D"], spec["Q"], steps=spec["steps"],
                           lr=spec["lr"], batch=spec["batch"], device="cuda", seed=spec["seed"],
                           mode="chunk", eval_batches=8)
    res.update(variant="gdn", D=spec["D"], seed=spec["seed"],
               total_state=spec["n_heads"] * spec["gdn_head_dim"] ** 2)
    return res


def _print_mqar_table(results: list, variants=("memory", "token", "single", "loglinear", "gdn")):
    from collections import defaultdict
    agg = defaultdict(list)               # (variant, D) -> [acc over seeds]
    for r in results:
        agg[(r["variant"], r["D"])].append(r["accuracy"])
    present = [v for v in variants if any(k[0] == v for k in agg)]
    Ds = sorted({r["D"] for r in results})
    print("\nMQAR accuracy (mean over seeds) — higher is better")
    print(f"{'D':>5} | " + " ".join(f"{v:>10}" for v in present))
    print("-" * (8 + 11 * len(present)))
    for D in Ds:
        cells = []
        for v in present:
            vals = agg[(v, D)]
            cells.append(f"{(sum(vals) / len(vals)) if vals else float('nan'):>10.3f}")
        print(f"{D:>5} | " + " ".join(cells))
    by_v = {r["variant"]: r for r in results}
    print("\ntotal matrix-state per layer per head-group:",
          {v: by_v[v].get("total_state") for v in present if v in by_v})
    print("params:", {v: by_v[v].get("params") for v in present if v in by_v})


@app.local_entrypoint()
def main(action: str = "verify", steps: int = 4000, task: str = "both"):
    if action in ("verify", "pytest"):
        rc = run_pytest.remote()
        print(f"pytest exit code: {rc}")
        if action == "verify":
            print("gpu_equiv:", gpu_equiv.remote())
            print("train_sanity:", train_sanity.remote())
    elif action in ("study", "report"):
        import json
        tasks = ("mqar", "niah") if task == "both" else (task,)
        if action == "study":                         # fire runs; each self-saves to the volume
            for t in tasks:
                mf, gd = _study_specs(t, steps)
                print(f"\n[{t}] launching {len(mf)} MemFlow-family + {len(gd)} GDN runs...")
                done = sum(1 for _ in recall_run.map(mf, return_exceptions=True))
                done += sum(1 for _ in gdn_recall_run.map(gd, return_exceptions=True))
                print(f"[{t}] {done} runs returned (results persisted per-run to the volume)")
        saved = collect.remote()                      # source of truth: whatever made it to disk
        print(f"\n{len(saved)} results on the volume")
        for t in tasks:
            sub = [r for r in saved if r.get("task") == t]
            if sub:
                _print_study(sub, t)
        keys = ("variant", "N", "task", "size", "seed", "accuracy", "params", "total_state")
        rel = [r for r in saved if r.get("task") in tasks]
        print("\nRAW_RESULTS_JSON:", json.dumps([{k: r.get(k) for k in keys} for r in rel]))
    elif action == "hgdn":
        hspecs = _hgdn_specs(steps)
        print(f"running log-linear (HGDN) baseline: {len(hspecs)} runs ({steps} steps each)...")
        results = list(hgdn_run.map(hspecs))
        _print_mqar_table(results, variants=("loglinear",))
        import json
        keys = ("variant", "D", "seed", "accuracy", "params", "total_state")
        print("RAW_RESULTS_JSON:", json.dumps([{k: r.get(k) for k in keys} for r in results]))
    elif action == "gdn":
        gspecs = _gdn_specs(steps)
        print(f"running GDN baseline: {len(gspecs)} runs ({steps} steps each)...")
        results = list(gdn_run.map(gspecs))
        _print_mqar_table(results, variants=("gdn",))
        import json
        keys = ("variant", "D", "seed", "accuracy", "params", "total_state")
        print("RAW_RESULTS_JSON:", json.dumps([{k: r.get(k) for k in keys} for r in results]))
    elif action == "gpu-equiv":
        print("gpu_equiv:", gpu_equiv.remote())
    elif action == "train-sanity":
        print("train_sanity:", train_sanity.remote())
    elif action in ("mqar", "compare"):
        specs = _mqar_specs(steps)
        print(f"running MQAR ablation: {len(specs)} MemFlow runs ({steps} steps each)...")
        results = list(mqar_run.map(specs))
        if action == "compare":
            gspecs = _gdn_specs(steps)
            print(f"running GDN baseline: {len(gspecs)} runs...")
            results += list(gdn_run.map(gspecs))
        _print_mqar_table(results)
        import json
        keys = ("variant", "D", "seed", "accuracy", "params", "total_state")
        print("RAW_RESULTS_JSON:", json.dumps([{k: r.get(k) for k in keys} for r in results]))
    else:
        raise SystemExit(f"unknown action {action!r}")
