"""Unit tests for fastwam.server.chunk_ringbuffer."""
from __future__ import annotations

import threading
from typing import List

import numpy as np
import pytest

from fastwam.server.chunk_ringbuffer import ChunkEntry, ChunkRingBuffer


def _make_entry(chunk_id: int, value: float = 0.0) -> ChunkEntry:
    return ChunkEntry(
        action_abs=np.full((32, 7), value, dtype=np.float32),
        base_capture_ts_ns=1_000_000_000 + chunk_id * 400_000_000,
        step_dt_ns=50_000_000,
        chunk_id=chunk_id,
    )


def test_invalid_capacity_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkRingBuffer(capacity=0)


def test_push_until_full_no_eviction() -> None:
    rb = ChunkRingBuffer(capacity=2)
    assert rb.occupancy() == 0
    assert rb.latest() is None
    assert rb.prev() is None

    e1 = _make_entry(1)
    assert rb.push(e1) is None  # not full -> no eviction
    assert rb.occupancy() == 1
    assert rb.latest() is e1
    assert rb.prev() is None  # only one element

    e2 = _make_entry(2)
    assert rb.push(e2) is None  # exactly fills capacity
    assert rb.occupancy() == 2
    assert rb.latest() is e2
    assert rb.prev() is e1
    assert rb.max_occupancy_seen() == 2


def test_push_full_evicts_oldest() -> None:
    rb = ChunkRingBuffer(capacity=2)
    e1, e2, e3 = _make_entry(1), _make_entry(2), _make_entry(3)
    rb.push(e1)
    rb.push(e2)
    evicted = rb.push(e3)
    assert evicted is e1, "oldest entry must be returned on full-buffer push"
    assert rb.occupancy() == 2
    assert rb.latest() is e3
    assert rb.prev() is e2

    evicted2 = rb.push(_make_entry(4))
    assert evicted2 is e2


def test_clear_resets_state_but_keeps_history() -> None:
    rb = ChunkRingBuffer(capacity=2)
    rb.push(_make_entry(1))
    rb.push(_make_entry(2))
    assert rb.max_occupancy_seen() == 2
    rb.clear()
    assert rb.occupancy() == 0
    assert rb.latest() is None
    assert rb.prev() is None
    # history of max occupancy persists (status() reports lifetime max).
    assert rb.max_occupancy_seen() == 2


def test_capacity_property() -> None:
    rb = ChunkRingBuffer(capacity=5)
    assert rb.capacity == 5


def test_concurrent_push_read_no_crash() -> None:
    rb = ChunkRingBuffer(capacity=2)
    stop = threading.Event()
    errors: List[BaseException] = []

    def writer() -> None:
        try:
            i = 0
            while not stop.is_set():
                rb.push(_make_entry(i))
                i += 1
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                rb.latest()
                rb.prev()
                rb.occupancy()
                rb.max_occupancy_seen()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, daemon=True),
        threading.Thread(target=writer, daemon=True),
        threading.Thread(target=reader, daemon=True),
        threading.Thread(target=reader, daemon=True),
    ]
    for t in threads: t.start()
    threading.Event().wait(0.3)
    stop.set()
    for t in threads: t.join(timeout=2.0)
    assert not errors, f"thread errors: {errors!r}"
    # Buffer must still be in a valid state and bounded by capacity.
    assert rb.occupancy() <= rb.capacity
    assert rb.max_occupancy_seen() <= rb.capacity
