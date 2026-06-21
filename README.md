# efficient-long-context

Research on efficient long-context modeling: replacing attention's growing KV cache with a fixed-size
memory updated online. Three projects.

1. Gated TTT (`TTT/`): memory in the backbone's own MLP `W_down`, with additive versus Gated-Delta
   self-correcting write rules. Validated long-context wins (+4.4 RULER at 64K).
2. Neural Memory Tree (`nmt/`): a routable tree of delta-rule memories. Multi-key routing wins, but the
   naive hierarchy fails (content-clustering is the fix).
3. Memory Flow (`memflow/`): multi-timescale "memory of memory" consolidation. Clean negative on exact
   recall; long-context untested.

Start with [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the consolidated findings, results tables,
and a reflection across all three.

## Layout
- `memflow/`: Memory Flow package, correctness suite, and Modal harness (`report.md` is its writeup).
- `nmt/`: Neural Memory Tree experiments (`EXPERIMENTS.md` is the full log).
- `TTT/`: Gated TTT writeups (`report.md`, `weights-as-memory.md`).
- `tests/`, `configs/`: MemFlow tests and configs.

Reproduce MemFlow with `modal run memflow/modal_app.py --action verify` and `pytest -q` (31 tests).

The vendored third-party clones (flash-linear-attention, In-Place-TTT, log-linear-attention, arXiv
sources) and the local `.venv` are intentionally excluded.
