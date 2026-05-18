"""Thread-safe FIFO ring buffer for action chunks.

PR5 scope: holds the latest ``capacity`` :class:`ChunkEntry` produced by
:class:`fastwam.server.closed_loop.ClosedLoopRunner`. On ``push`` the buffer
returns the entry that was evicted (or ``None``) so callers can log eviction
events. ``occupancy`` / ``max_occupancy_seen`` feed the runner ``status()``
report (see design doc risk #12).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkEntry:
    """One snapshot of a 7-DoF cartesian-absolute action chunk."""

    action_abs: np.ndarray         # (chunk_len, 7) -- xyz, rpy, gripper
    base_capture_ts_ns: int        # = head_left.stamp_ns at snapshot
    step_dt_ns: int                # default 50_000_000 (20 Hz)
    chunk_id: int


class ChunkRingBuffer:
    """Bounded FIFO with eviction-on-full semantics."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = int(capacity)
        self._buf: deque[ChunkEntry] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()
        self._max_occupancy_seen = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, entry: ChunkEntry) -> Optional[ChunkEntry]:
        """Append ``entry``. Returns the evicted oldest entry if buffer was full."""
        with self._lock:
            evicted: Optional[ChunkEntry] = None
            if len(self._buf) == self._capacity:
                # deque(maxlen=N) silently drops the head, so capture it ourselves.
                evicted = self._buf[0]
            self._buf.append(entry)
            if len(self._buf) > self._max_occupancy_seen:
                self._max_occupancy_seen = len(self._buf)
            return evicted

    def latest(self) -> Optional[ChunkEntry]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def prev(self) -> Optional[ChunkEntry]:
        """Return the entry just before ``latest()`` (used for blending)."""
        with self._lock:
            if len(self._buf) < 2:
                return None
            return self._buf[-2]

    def occupancy(self) -> int:
        with self._lock:
            return len(self._buf)

    def max_occupancy_seen(self) -> int:
        with self._lock:
            return self._max_occupancy_seen

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


__all__ = ["ChunkEntry", "ChunkRingBuffer"]
