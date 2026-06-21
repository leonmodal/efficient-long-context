"""Phase 3 acceptance (plan §7.4): Fenwick schedule correctness + O(log T) state growth.

The coverage test (distant-fact recall holds where fixed buckets fade) needs the full growing model
and is added once the merge/readout wiring lands. These tests pin the schedule down first.
"""
import math

from memflow.growing import fenwick_schedule, level_birth_chunk, live_blocks, lssb, num_levels


def test_lssb_matches_bittrick_and_general():
    for c in range(1, 200):
        assert lssb(c, 2) == (c & -c).bit_length() - 1
    # base-3 general path: 9 = 3^2 -> level 2, 6 = 3*2 -> level 1, 5 -> 0
    assert lssb(9, 3) == 2
    assert lssb(6, 3) == 1
    assert lssb(5, 3) == 0


def test_schedule_hand_computed_T_over_C_8():
    # plan §7.4 hand case: 8 chunks, base 2
    events = fenwick_schedule(8, base=2)
    got = [(e.chunk, e.lssb, e.born, tuple(e.merged)) for e in events]
    expected = [
        (1, 0, 1, (0,)),
        (2, 1, 2, (0, 1)),
        (3, 0, 1, (0,)),
        (4, 2, 3, (0, 1, 2)),
        (5, 0, 1, (0,)),
        (6, 1, 2, (0, 1)),
        (7, 0, 1, (0,)),
        (8, 3, 4, (0, 1, 2, 3)),
    ]
    assert got == expected


def test_num_levels_logarithmic():
    # T in {1024, 4096, 16384}, C = 64 -> NC in {16, 64, 256}
    for T in (1024, 4096, 16384):
        nc = T // 64
        L = num_levels(nc, base=2)
        assert L == int(math.log2(nc)) + 2          # floor(log2 NC) + 2
        assert abs(L - (math.log2(nc) + 2)) < 1.0    # O(log T/C) + const


def test_live_blocks_tile_history_with_log_buckets():
    """Coverage guarantee: live buckets tile [0,NC) disjointly using O(log NC) buckets."""
    for nc in (1, 2, 7, 8, 11, 16, 100, 256, 257):
        blocks = live_blocks(nc, base=2)
        # tile [0, nc) with no gaps/overlaps, oldest-first
        cursor = 0
        for lvl, s, e in blocks:
            assert s == cursor, f"gap/overlap at nc={nc}: {blocks}"
            assert e - s == 2 ** lvl
            cursor = e
        assert cursor == nc
        # O(log) buckets: popcount for base 2
        assert len(blocks) == bin(nc).count("1")
        assert len(blocks) <= math.floor(math.log2(nc)) + 1
    # a distant early fact (chunk 0) is always inside the OLDEST (largest) live block, never dropped
    blocks = live_blocks(200, base=2)
    oldest = blocks[0]
    assert oldest[1] == 0 and oldest[2] == 128   # chunk 0 covered by the size-128 block


def test_level_birth_chunk():
    assert level_birth_chunk(0) == 1
    assert [level_birth_chunk(l) for l in range(1, 5)] == [1, 2, 4, 8]
    # a level is never used before it is born
    events = fenwick_schedule(64)
    for e in events:
        assert e.chunk >= level_birth_chunk(e.born)
