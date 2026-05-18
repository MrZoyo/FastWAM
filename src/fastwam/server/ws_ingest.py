"""WebSocket frame ingester for rgbd_ws_bridge V4 encoded_sync_pair stream.

Decodes paired H.264 frames in a background thread and exposes the latest
FrameSnapshot per channel via a thread-safe accessor. See design doc
``docs/fastwam_http_server_self_fetch_design.md`` §4.1 (protocol) and §7.4
(component contract).
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
import urllib.parse
from collections import deque
from typing import Any, NamedTuple

import av
import numpy as np
import websocket  # websocket-client

PACKET_MAGIC_V4 = b"RGBDWS4\x00"
PACKET_HEADER_V4_FORMAT = "<8sIII"
PACKET_HEADER_V4_SIZE = struct.calcsize(PACKET_HEADER_V4_FORMAT)
EXPECTED_PAYLOAD_FORMAT = "h264_annexb_pair_v1"
EXPECTED_TYPE = "encoded_sync_pair"
WARMUP_MAX_EMPTY_PACKETS = 2  # PyAV may emit empty frames for first ~2 packets


class _WarmupFailure(RuntimeError):
    """Raised internally when PyAV warmup exceeds the empty-packet budget.

    Caught by :meth:`WSFrameIngester._on_message` to reset the offending
    channel's decoder instead of propagating the exception to the
    websocket-client callback (which would kill the ingester thread and
    prevent future reconnects).
    """

    def __init__(self, channel: str, message: str) -> None:
        super().__init__(message)
        self.channel = channel


class FrameSnapshot(NamedTuple):
    bgr: np.ndarray  # (H, W, 3) uint8
    capture_ts_ns: int  # channel.meta['stamp_ns']
    decode_ts_ns: int
    pair_seq: int


def _ensure_no_proxy(ws_url: str) -> str | None:
    """Add ws_url host to NO_PROXY env so LAN bypasses any global proxy."""
    host = urllib.parse.urlparse(ws_url).hostname
    if not host:
        return None
    for key in ("NO_PROXY", "no_proxy"):
        entries = [s.strip() for s in os.environ.get(key, "").split(",") if s.strip()]
        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)
    return host


def parse_packet(message: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    """Parse a single V4 packet → (meta_dict, channel_a_bytes, channel_b_bytes).

    Raises RuntimeError on any structural problem.
    """
    if isinstance(message, str):
        raise RuntimeError("expected binary V4 packet, got text message")
    if len(message) < PACKET_HEADER_V4_SIZE:
        raise RuntimeError("packet header too short")
    if not message.startswith(PACKET_MAGIC_V4):
        raise RuntimeError("packet magic mismatch")
    magic, header_size, size_a, size_b = struct.unpack_from(PACKET_HEADER_V4_FORMAT, message, 0)
    if magic != PACKET_MAGIC_V4:
        raise RuntimeError("packet magic mismatch (post unpack)")
    hb = PACKET_HEADER_V4_SIZE
    he = hb + header_size
    ae = he + size_a
    be = ae + size_b
    if len(message) != be:
        raise RuntimeError(f"packet size mismatch: expect={be} actual={len(message)}")
    meta = json.loads(message[hb:he].decode("utf-8"))
    return meta, bytes(message[he:ae]), bytes(message[ae:be])


def validate_startup_meta(meta: dict[str, Any], expected_channels: set[str]) -> None:
    """Startup self-check. Raise RuntimeError on any mismatch."""
    if meta.get("payload_format") != EXPECTED_PAYLOAD_FORMAT:
        raise RuntimeError(f"payload_format mismatch: got {meta.get('payload_format')!r}")
    if meta.get("type") != EXPECTED_TYPE:
        raise RuntimeError(f"type mismatch: got {meta.get('type')!r}")
    channels = meta.get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        raise RuntimeError(f"channels must be a list of length 2, got {channels!r}")
    got = {str(ch.get("name")) for ch in channels}
    if got != expected_channels:
        raise RuntimeError(
            f"channel name mismatch: got {sorted(got)!r}, expected {sorted(expected_channels)!r}"
        )


class WSFrameIngester:
    """Background thread that decodes WS V4 packets and caches latest frames."""

    def __init__(
        self,
        ws_url: str,
        expected_channels: set[str],
        frame_max_age_ms: int = 250,
        reconnect_backoff_ms_list: list[int] | None = None,
        startup_timeout_ms: int = 30_000,
        logger: logging.Logger | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._expected_channels = set(expected_channels)
        self._frame_max_age_ms = int(frame_max_age_ms)
        self._backoff_ms = list(reconnect_backoff_ms_list or [500, 1000, 2000, 5000, 10000])
        if not self._backoff_ms:
            self._backoff_ms = [1000]
        self._startup_timeout_ms = int(startup_timeout_ms)
        self._logger = logger or logging.getLogger("fastwam.ws_ingest")

        self._latest: dict[str, FrameSnapshot] = {}
        self._latest_lock = threading.Lock()
        self._decoders: dict[str, av.codec.context.CodecContext] = {}
        self._empty_packet_count: dict[str, int] = {}

        # state / stats
        self._startup_event = threading.Event()
        self._startup_error: str | None = None
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocketApp | None = None

        self._recv_times_ns: deque[int] = deque(maxlen=256)
        self._decode_fail_count = 0
        self._reconnect_count = 0
        self._pair_seq_gaps = 0
        self._last_pair_seq: int | None = None
        self._startup_validated = False
        self._decoder_reset_count = 0

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        _ensure_no_proxy(self._ws_url)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("WSFrameIngester already started")
        self._stop_requested.clear()
        self._startup_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_loop, name="ws_ingest", daemon=True
        )
        self._thread.start()
        if not self._startup_event.wait(self._startup_timeout_ms / 1000.0):
            self.stop()
            raise RuntimeError(
                f"WS startup timeout after {self._startup_timeout_ms} ms (no first frame)"
            )
        if self._startup_error is not None:
            err = self._startup_error
            self.stop()
            raise RuntimeError(f"WS startup self-check failed: {err}")

    def stop(self) -> None:
        self._stop_requested.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        thr = self._thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=2.0)
        self._thread = None
        self._ws = None

    # -- consumer api ---------------------------------------------------

    def latest(self, image_key: str) -> FrameSnapshot | None:
        with self._latest_lock:
            snap = self._latest.get(image_key)
        if snap is None:
            return None
        age_ms = (time.time_ns() - snap.capture_ts_ns) / 1e6
        if age_ms > self._frame_max_age_ms:
            return None
        return snap

    def health(self) -> dict[str, Any]:
        now_ns = time.time_ns()
        ages: dict[str, float] = {}
        with self._latest_lock:
            for k, snap in self._latest.items():
                ages[k] = (now_ns - snap.capture_ts_ns) / 1e6
        # fps over last 5s
        cutoff = now_ns - 5_000_000_000
        recent = [t for t in self._recv_times_ns if t >= cutoff]
        fps_5s = (len(recent) / 5.0) if recent else 0.0
        return {
            "last_frame_age_per_channel_ms": ages,
            "fps_5s": fps_5s,
            "decode_fail_count": self._decode_fail_count,
            "reconnect_count": self._reconnect_count,
            "pair_seq_gaps": self._pair_seq_gaps,
            "decoder_reset_count": self._decoder_reset_count,
        }

    # -- internals ------------------------------------------------------

    def _run_loop(self) -> None:
        """Outer loop: connect → run_forever → backoff on disconnect."""
        attempt = 0
        host = urllib.parse.urlparse(self._ws_url).hostname
        run_kwargs: dict[str, Any] = {"ping_interval": 0}
        if host:
            run_kwargs["http_no_proxy"] = [host]

        while not self._stop_requested.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(**run_kwargs)
            except Exception as exc:  # pragma: no cover - runtime
                self._logger.warning("ws run_forever raised: %s", exc)

            if self._stop_requested.is_set():
                break
            # backoff
            attempt += 1
            backoff_idx = min(attempt - 1, len(self._backoff_ms) - 1)
            delay_ms = self._backoff_ms[backoff_idx]
            self._reconnect_count += 1
            self._logger.warning(
                "ws disconnected; reconnect attempt=%d delay_ms=%d", attempt, delay_ms
            )
            if self._stop_requested.wait(delay_ms / 1000.0):
                break

        # if we never validated startup, signal failure so .start() unblocks
        if not self._startup_validated:
            if self._startup_error is None:
                self._startup_error = "WS thread exited before first frame"
            self._startup_event.set()

    def _on_open(self, _ws: Any) -> None:
        self._logger.info("ws connected url=%s", self._ws_url)

    def _on_error(self, _ws: Any, error: Any) -> None:
        self._logger.warning("ws error: %s", error)

    def _on_close(self, _ws: Any, status_code: Any, msg: Any) -> None:
        self._logger.info("ws closed code=%s msg=%s", status_code, msg)

    def _on_message(self, ws: Any, message: Any) -> None:
        recv_ns = time.time_ns()
        try:
            meta, data_a, data_b = parse_packet(message)
        except Exception as exc:
            self._decode_fail_count += 1
            self._logger.warning("packet parse failed: %s", exc)
            return

        if not self._startup_validated:
            try:
                validate_startup_meta(meta, self._expected_channels)
            except RuntimeError as exc:
                self._startup_error = str(exc)
                self._startup_event.set()
                try:
                    ws.close()
                except Exception:
                    pass
                return
            self._startup_validated = True

        # pair_seq monotonicity
        pair_seq = int(meta.get("pair_seq", 0))
        if self._last_pair_seq is not None and pair_seq - self._last_pair_seq > 1:
            self._pair_seq_gaps += 1
        self._last_pair_seq = pair_seq
        self._recv_times_ns.append(recv_ns)

        channels = meta.get("channels", [])
        payloads = (data_a, data_b)
        for ch_meta, raw in zip(channels, payloads):
            name = str(ch_meta.get("name", ""))
            if not name:
                continue
            stamp_ns = int(ch_meta.get("stamp_ns", 0))
            try:
                self._decode_channel(name, raw, stamp_ns, pair_seq)
            except _WarmupFailure as exc:
                # PyAV warmup budget exhausted for this channel. Reset its
                # decoder so future packets get a fresh shot. Do NOT re-raise:
                # propagating would trip websocket-client's on_error path,
                # close the socket and kill the run_forever loop — preventing
                # any further reconnect (see PR13).
                self._reset_channel_decoder(exc.channel, reason=str(exc))

        # signal startup once any decoder produced a frame
        if not self._startup_event.is_set():
            with self._latest_lock:
                got_any = bool(self._latest)
            if got_any:
                self._startup_event.set()

    def _reset_channel_decoder(self, name: str, *, reason: str) -> None:
        """Drop a broken decoder + warmup counter so the channel can recover.

        Called from :meth:`_on_message` when :meth:`_decode_channel` raises
        :class:`_WarmupFailure`. Increments ``_decoder_reset_count`` (visible
        via :meth:`health`) so operators can observe how often warmup
        auto-recovery has kicked in.
        """
        self._decoder_reset_count += 1
        self._decoders.pop(name, None)
        self._empty_packet_count[name] = 0
        self._logger.warning(
            "resetting decoder name=%s reason=%s decoder_reset_count=%d",
            name,
            reason,
            self._decoder_reset_count,
        )

    def _decode_channel(self, name: str, raw: bytes, stamp_ns: int, pair_seq: int) -> None:
        decoder = self._decoders.get(name)
        if decoder is None:
            decoder = av.CodecContext.create("h264", "r")
            self._decoders[name] = decoder
            self._empty_packet_count[name] = 0
        try:
            frames: list[Any] = []
            for pkt in decoder.parse(raw):
                frames.extend(decoder.decode(pkt))
        except Exception as exc:
            self._decode_fail_count += 1
            self._logger.warning("decode failed name=%s err=%s", name, exc)
            return

        if not frames:
            self._empty_packet_count[name] = self._empty_packet_count.get(name, 0) + 1
            if self._empty_packet_count[name] > WARMUP_MAX_EMPTY_PACKETS:
                self._decode_fail_count += 1
                raise _WarmupFailure(
                    name,
                    f"PyAV warmup failed: channel={name} produced "
                    f"{self._empty_packet_count[name]} empty packets in a row "
                    f"(expected first frame by packet 3)",
                )
            return

        # reset warmup counter once we successfully decoded
        self._empty_packet_count[name] = 0
        bgr = frames[-1].to_ndarray(format="bgr24")
        snap = FrameSnapshot(
            bgr=bgr,
            capture_ts_ns=stamp_ns,
            decode_ts_ns=time.time_ns(),
            pair_seq=pair_seq,
        )
        with self._latest_lock:
            self._latest[name] = snap


__all__ = ["FrameSnapshot", "WSFrameIngester", "parse_packet", "validate_startup_meta"]
