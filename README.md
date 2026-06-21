# efficient-long-context

Research on **efficient long-context modeling** — replacing attention's growing KV cache with a
fixed-size memory updated online. Three projects:

1. **Gated TTT** (`TTT/`) — memory in the backbone's own MLP `W_down`; additive vs Gated-Delta
   self-correcting write rules. *Validated long-context wins (+4.4 RULER@64K).*
2. **Neural Memory Tree** (`nmt/`) — a routable tree of delta-rule memories. *Multi-key routing win;
   naive hierarchy fails (content-clustering is the fix).*
3. **Memory Flow** (`memflow/`) — multi-timescale "memory of memory" consolidation. *Clean negative on
   exact recall; long-context untested.*

👉 **Start with [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)** — consolidated findings, results tables,
and a reflection across all three.

## Layout
- `memflow/` — Memory Flow package + correctness suite + Modal harness (`report.md` = its writeup)
- `nmt/` — Neural Memory Tree experiments (`EXPERIMENTS.md` = full log)
- `TTT/` — Gated TTT writeups (`report.md`, `weights-as-memory.md`)
- `tests/`, `configs/` — MemFlow tests and configs

Reproduce MemFlow: `modal run memflow/modal_app.py --action verify` · `pytest -q` (31 tests).

*Vendored third-party clones (flash-linear-attention, In-Place-TTT, log-linear-attention, arXiv
sources) and the local `.venv` are intentionally excluded.*
