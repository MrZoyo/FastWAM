"""Watchdog — 100 Hz safety supervisor (design §7.6.3, §8, risks 10/11/13)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

_NS_PER_MS = 1_000_000


@dataclass
class _WatchdogMetrics:
    hold_reason: str = ""
    last_emergency_reason: str = ""
    last_ws_age_ms: Optional[float] = None
    last_arm_age_ms: Optional[float] = None
    infer_heartbeat_age_ms: Optional[float] = None
    chunk_stale_count: int = 0
    ws_warn_count: int = 0
    ws_hold_count: int = 0
    ws_estop_count: int = 0
    arm_red_streak_ms: float = 0.0
    infer_warn_count: int = 0
    tick_count: int = 0


class Watchdog:
    """10 ms safety supervisor: stale chunk / WS / ARM / infer heartbeat."""

    def __init__(
        self,
        ringbuffer: Any,
        arm_client: Any,
        ws_ingester: Any,
        dispatcher: Any,
        watchdog_period_ms: int = 10,
        chunk_max_stale_ms: int = 2000,
        ws_warn_stale_ms: int = 200,
        ws_hold_stale_ms: int = 500,
        ws_estop_stale_ms: int = 1500,
        infer_period_ms: int = 400,
        infer_timeout_factor: float = 5.0,
        infer_hold_factor: float = 5.0,
        infer_warn_factor: float = 2.0,
        arm_red_grace_ms: int = 200,
        arm_health_max_age_ms: int = 200,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if watchdog_period_ms <= 0:
            raise ValueError("watchdog_period_ms must be > 0")

        self._ringbuffer = ringbuffer
        self._arm = arm_client
        self._ws = ws_ingester
        self._dispatcher = dispatcher
        self._period_s = watchdog_period_ms / 1000.0
        self._chunk_max_stale_ns = int(chunk_max_stale_ms) * _NS_PER_MS
        self._ws_warn_ns = int(ws_warn_stale_ms) * _NS_PER_MS
        self._ws_hold_ns = int(ws_hold_stale_ms) * _NS_PER_MS
        self._ws_estop_ns = int(ws_estop_stale_ms) * _NS_PER_MS
        self._infer_period_ns = int(infer_period_ms) * _NS_PER_MS
        self._infer_warn_factor = float(infer_warn_factor)
        self._infer_hold_factor = float(infer_hold_factor)
        # accept old name (PR6 contract) for back-compat
        if infer_timeout_factor and infer_timeout_factor != infer_hold_factor:
            self._infer_hold_factor = float(infer_timeout_factor)
        self._arm_red_grace_ms = float(arm_red_grace_ms)
        self._arm_health_max_age_ms = float(arm_health_max_age_ms)
        self._logger = logger or logging.getLogger(__name__)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._metrics = _WatchdogMetrics()
        self._infer_heartbeat_ns: Optional[int] = None
        self._last_seen_chunk_id: Optional[int] = None
        self._last_seen_chunk_base_ns: Optional[int] = None
        self._last_chunk_seen_mono_ns: Optional[int] = None
        self._arm_red_start_mono_ns: Optional[int] = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Watchdog.start called twice")
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="Watchdog", daemon=True
        )
        self._thread.start()
        self._logger.info(
            "[watchdog] start period=%.0f ms infer_period=%d ms",
            self._period_s * 1000.0, self._infer_period_ns // _NS_PER_MS,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    # ---------------------------------------------------------------- public
    def infer_heartbeat(self) -> None:
        """Called by InferLoop at the end of each inference cycle."""
        self._infer_heartbeat_ns = time.monotonic_ns()

    def status(self) -> dict:
        with self._state_lock:
            m = self._metrics
            return {
                "hold_reason": m.hold_reason,
                "last_emergency_reason": m.last_emergency_reason,
                "last_ws_age_ms": m.last_ws_age_ms,
                "last_arm_age_ms": m.last_arm_age_ms,
                "infer_heartbeat_age_ms": m.infer_heartbeat_age_ms,
                "chunk_stale_count": m.chunk_stale_count,
                "ws_warn_count": m.ws_warn_count,
                "ws_hold_count": m.ws_hold_count,
                "ws_estop_count": m.ws_estop_count,
                "infer_warn_count": m.infer_warn_count,
                "arm_red_streak_ms": m.arm_red_streak_ms,
                "tick_count": m.tick_count,
                "running": self._thread is not None and self._thread.is_alive(),
            }

    # ---------------------------------------------------------------- main loop
    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            self._tick()
            next_t += self._period_s
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                if self._stop_evt.wait(timeout=sleep_s):
                    return
            else:
                next_t = time.monotonic()

    def _tick(self) -> None:
        with self._state_lock:
            self._metrics.tick_count += 1
        now_mono_ns = time.monotonic_ns()
        for check in (
            self._check_chunk_stale,
            self._check_ws_stale_wrap,
            self._check_arm_health,
            self._check_infer_heartbeat,
        ):
            check(now_mono_ns)
            if self._stop_evt.is_set():
                return

    def _check_ws_stale_wrap(self, _now: int) -> None: self._check_ws_stale()

    # ---------------------------------------------------------------- checks
    def _check_chunk_stale(self, now_mono_ns: int) -> None:
        chunk = self._ringbuffer.latest()
        if chunk is None:
            if self._last_chunk_seen_mono_ns is not None:
                # we used to have chunks; if it's been a while → stale
                gap_ns = now_mono_ns - self._last_chunk_seen_mono_ns
                if gap_ns > self._chunk_max_stale_ns:
                    self._notify_hold("chunk_missing")
            return

        # track newest seen chunk id / base
        if (
            self._last_seen_chunk_id is None
            or chunk.chunk_id != self._last_seen_chunk_id
        ):
            self._last_seen_chunk_id = chunk.chunk_id
            self._last_seen_chunk_base_ns = chunk.base_capture_ts_ns
            self._last_chunk_seen_mono_ns = now_mono_ns

        chunk_len = int(chunk.action_abs.shape[0])
        chunk_end_ns = chunk.base_capture_ts_ns + chunk_len * chunk.step_dt_ns
        chunk_age_ns = time.time_ns() - chunk_end_ns
        if chunk_age_ns > self._chunk_max_stale_ns:
            # chunk used up, no fresher one seen recently
            if self._last_chunk_seen_mono_ns is not None and \
                    (now_mono_ns - self._last_chunk_seen_mono_ns) > self._chunk_max_stale_ns:
                self._notify_hold("chunk_stale")
                with self._state_lock:
                    self._metrics.chunk_stale_count += 1

    def _check_ws_stale(self) -> None:
        try:
            h = self._ws.health()
        except Exception:  # pragma: no cover
            self._logger.exception("[watchdog] ws.health raised")
            return
        ages = h.get("last_frame_age_per_channel_ms") or {}
        if not ages:
            return
        worst_ms = max(float(v) for v in ages.values())
        worst_ns = int(worst_ms * _NS_PER_MS)
        with self._state_lock:
            self._metrics.last_ws_age_ms = worst_ms

        if worst_ns >= self._ws_estop_ns:
            with self._state_lock:
                self._metrics.ws_estop_count += 1
            self._logger.error("[watchdog] WS estop: worst age %.1f ms", worst_ms)
            self._trigger_emergency(f"ws_stale_estop:{worst_ms:.0f}ms")
        elif worst_ns >= self._ws_hold_ns:
            with self._state_lock:
                self._metrics.ws_hold_count += 1
            self._logger.warning("[watchdog] WS hold: worst age %.1f ms", worst_ms)
            self._notify_hold(f"ws_stale:{worst_ms:.0f}ms")
        elif worst_ns >= self._ws_warn_ns:
            with self._state_lock:
                self._metrics.ws_warn_count += 1
            self._logger.warning("[watchdog] WS warn: worst age %.1f ms", worst_ms)

    def _check_arm_health(self, now_mono_ns: int) -> None:
        try:
            h = self._arm.health()
        except Exception:  # pragma: no cover
            self._logger.exception("[watchdog] arm.health raised")
            return
        age_ms = h.get("last_poll_age_ms")
        lease_alive = bool(h.get("lease_alive", False))
        running = bool(h.get("running", False))
        with self._state_lock:
            self._metrics.last_arm_age_ms = age_ms

        red = False
        if age_ms is None and running:
            # no poll yet → not RED unless poller died
            red = False
        elif age_ms is not None and age_ms > self._arm_health_max_age_ms:
            red = True
        elif running and not lease_alive and h.get("consecutive_fail", 0) >= 5:
            red = True

        if red:
            if self._arm_red_start_mono_ns is None:
                self._arm_red_start_mono_ns = now_mono_ns
            streak_ms = (now_mono_ns - self._arm_red_start_mono_ns) / 1e6
            with self._state_lock:
                self._metrics.arm_red_streak_ms = streak_ms
            if streak_ms > self._arm_red_grace_ms:
                self._logger.error(
                    "[watchdog] ARM RED streak %.1f ms -> emergency", streak_ms
                )
                self._trigger_emergency(f"arm_red:{streak_ms:.0f}ms")
        else:
            self._arm_red_start_mono_ns = None
            with self._state_lock:
                self._metrics.arm_red_streak_ms = 0.0

    def _check_infer_heartbeat(self, now_mono_ns: int) -> None:
        hb = self._infer_heartbeat_ns
        if hb is None:
            with self._state_lock:
                self._metrics.infer_heartbeat_age_ms = None
            return
        age_ns = now_mono_ns - hb
        age_ms = age_ns / 1e6
        with self._state_lock:
            self._metrics.infer_heartbeat_age_ms = age_ms

        hold_ns = int(self._infer_period_ns * self._infer_hold_factor)
        warn_ns = int(self._infer_period_ns * self._infer_warn_factor)
        if age_ns > hold_ns:
            self._logger.error(
                "[watchdog] InferLoop heartbeat stale %.0f ms -> hold", age_ms
            )
            self._notify_hold(f"infer_heartbeat:{age_ms:.0f}ms")
        elif age_ns > warn_ns:
            with self._state_lock:
                self._metrics.infer_warn_count += 1
            self._logger.warning(
                "[watchdog] InferLoop heartbeat warn %.0f ms", age_ms
            )

    # ---------------------------------------------------------------- actions
    def _notify_hold(self, reason: str) -> None:
        with self._state_lock:
            self._metrics.hold_reason = reason
        try:
            self._dispatcher.set_hold(True, reason=reason)
        except Exception:  # pragma: no cover
            self._logger.exception("[watchdog] dispatcher.set_hold raised")

    def _trigger_emergency(self, reason: str) -> None:
        with self._state_lock:
            self._metrics.last_emergency_reason = reason
        try:
            self._arm.emergency_stop(True)
        except Exception:  # pragma: no cover
            self._logger.exception("[watchdog] arm.emergency_stop raised")
        try:
            self._dispatcher.stop()
        except Exception:  # pragma: no cover
            self._logger.exception("[watchdog] dispatcher.stop raised")
        self._stop_evt.set()
