"""Phase 3 scaffolding: the Fenwick (binary-indexed) growth schedule (plan §7, paper §methods).

The number of levels grows as O(log T): at chunk index c (1-indexed), let j = lssb(c) be the
position of the least-significant set bit of c (the largest ℓ with base^ℓ | c). The paper's state
recurrence promotes the lower levels into a freshly-born level j+1:

    S_c^(0)        = (new chunk content)
    S_c^(ℓ)        = 0                         for 0 < ℓ <= j         (reset; merged upward)
    S_c^(j+1)      = merge( S_{c-1}^(0..j) )                          (born from the lower levels)
    S_c^(ℓ)        = S_{c-1}^(ℓ)               for ℓ > j+1            (untouched)

In MemFlow the merge is the learned Option-C consolidation (consolidation.py) applied upward, not a
raw sum — but the *schedule* (which levels merge into which, and when) is exactly this lssb structure.
This module is pure Python (no torch) so the schedule is unit-testable on its own.
"""
from __future__ import annotations

from typing import List, NamedTuple


def lssb(c: int, base: int = 2) -> int:
    """Least-significant-set-bit level: the largest ℓ such that base**ℓ divides c (c >= 1).

    For base 2 this is the number of trailing zero bits: lssb(c) = (c & -c).bit_length() - 1.
    """
    if c < 1:
        raise ValueError(f"chunk index must be >= 1, got {c}")
    if base == 2:
        return (c & -c).bit_length() - 1
    level = 0
    while c % base == 0:
        c //= base
        level += 1
    return level


class MergeEvent(NamedTuple):
    chunk: int            # 1-indexed chunk
    lssb: int             # j = lssb(chunk)
    born: int             # level j+1 that receives the merge
    merged: List[int]     # lower levels 0..j consolidated upward into `born`, then reset


def fenwick_schedule(num_chunks: int, base: int = 2) -> List[MergeEvent]:
    """Per-chunk merge events for chunks 1..num_chunks (the binary-indexed promotion schedule)."""
    events = []
    for c in range(1, num_chunks + 1):
        j = lssb(c, base)
        events.append(MergeEvent(chunk=c, lssb=j, born=j + 1, merged=list(range(0, j + 1))))
    return events


def num_levels(num_chunks: int, base: int = 2) -> int:
    """Number of storage levels needed for `num_chunks` chunks: 2 + max_c lssb(c) = O(log_base T/C).

    Levels are 0 (per-chunk) .. (1 + max born). For base 2 this is floor(log2(num_chunks)) + 2.
    """
    if num_chunks < 1:
        return 1
    return 2 + max(lssb(c, base) for c in range(1, num_chunks + 1))


def live_blocks(num_chunks: int, base: int = 2):
    """The live buckets after `num_chunks` chunks under the binary-counter merge, as the spans of
    history they summarize. Returns a list of (level, start_chunk, end_chunk), oldest-span-first.

    This is the Phase 3 coverage guarantee: the blocks TILE [0, num_chunks) disjointly (every past
    chunk lives in exactly one bucket) using O(log T) buckets (for base 2, popcount(num_chunks) <=
    floor(log2 num_chunks)+1). Old history survives coarsely in a large high-level block; recent
    history is sharp in small low-level blocks. A fixed single bucket has no such guarantee.
    """
    powers = []
    p, lvl = 1, 0
    while p <= num_chunks:
        powers.append((lvl, p))
        p *= base
        lvl += 1
    blocks, cursor, rem = [], 0, num_chunks
    for lvl, p in reversed(powers):
        cnt = rem // p
        for _ in range(cnt):
            blocks.append((lvl, cursor, cursor + p))
            cursor += p
        rem -= cnt * p
    return blocks


def level_birth_chunk(level: int, base: int = 2) -> int:
    """First chunk index at which `level` is born (allocated). Level 0 exists from chunk 1.

    Level ℓ>=1 is first born at chunk base**(ℓ-1) (when lssb first reaches ℓ-1)."""
    if level <= 0:
        return 1
    return base ** (level - 1)
