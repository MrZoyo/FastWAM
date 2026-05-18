"""Unit tests for ``fastwam.server.ws_ingest``.

We don't spin up a real WebSocketApp here — instead we exercise the parsing
helpers and decoder warmup logic directly via static helpers / private
methods. Network-level reconnect behaviour is left for an integration test.
"""

from __future__ import annotations

import logging
import struct

import pytest

from fastwam.server.ws_ingest import (
    PACKET_MAGIC_V4,
    WSFrameIngester,
    parse_packet,
    validate_startup_meta,
)


EXPECTED_CHANNELS = {"head_left", "right_wrist_left"}


# ---------------------------------------------------------------------------
# parse_packet
# ---------------------------------------------------------------------------


def test_parse_packet_roundtrip(ws_packet_bytes):
    pkt = ws_packet_bytes[0]
    assert pkt.startswith(PACKET_MAGIC_V4), "fixture must start with V4 magic"
    meta, data_a, data_b = parse_packet(pkt)

    assert meta["payload_format"] == "h264_annexb_pair_v1"
    assert meta["type"] == "encoded_sync_pair"
    assert len(meta["channels"]) == 2
    names = {ch["name"] for ch in meta["channels"]}
    assert names == EXPECTED_CHANNELS
    assert meta["channels"][0]["data_size"] == len(data_a)
    assert meta["channels"][1]["data_size"] == len(data_b)
    assert "pair_seq" in meta and isinstance(meta["pair_seq"], int)
    # head/right wrist offset matches what conftest emits
    assert meta["delta_ns"] == 829_502


def test_parse_packet_sequence_monotonic(ws_packet_bytes):
    seqs = []
    for pkt in ws_packet_bytes:
        meta, _, _ = parse_packet(pkt)
        seqs.append(meta["pair_seq"])
    assert seqs == sorted(seqs), f"pair_seq must be monotonic, got {seqs}"
    assert seqs[-1] - seqs[0] == len(seqs) - 1


def test_parse_packet_rejects_text():
    with pytest.raises(RuntimeError, match="text"):
        parse_packet("not bytes")  # type: ignore[arg-type]


def test_parse_packet_rejects_bad_magic():
    bogus = b"BADMAGIC" + b"\x00" * 12
    with pytest.raises(RuntimeError, match="magic"):
        parse_packet(bogus)


def test_parse_packet_rejects_size_mismatch(ws_packet_bytes):
    pkt = ws_packet_bytes[0]
    truncated = pkt[:-16]
    with pytest.raises(RuntimeError, match="size mismatch"):
        parse_packet(truncated)


# ---------------------------------------------------------------------------
# validate_startup_meta
# ---------------------------------------------------------------------------


def test_validate_startup_meta_accepts_fixture(ws_packet_bytes):
    meta, _, _ = parse_packet(ws_packet_bytes[0])
    validate_startup_meta(meta, EXPECTED_CHANNELS)


def test_validate_startup_meta_rejects_bad_payload_format(build_v4_packet):
    pkt = build_v4_packet(b"\x00\x00\x00\x01", b"\x00\x00\x00\x01",
                          payload_format="h264_wrong_v0")
    meta, _, _ = parse_packet(pkt)
    with pytest.raises(RuntimeError, match="payload_format"):
        validate_startup_meta(meta, EXPECTED_CHANNELS)


def test_validate_startup_meta_rejects_bad_type(build_v4_packet):
    pkt = build_v4_packet(b"\x00\x00\x00\x01", b"\x00\x00\x00\x01",
                          type_field="some_other_type")
    meta, _, _ = parse_packet(pkt)
    with pytest.raises(RuntimeError, match="type mismatch"):
        validate_startup_meta(meta, EXPECTED_CHANNELS)


def test_validate_startup_meta_rejects_bad_channel_names(build_v4_packet):
    pkt = build_v4_packet(b"\x00\x00\x00\x01", b"\x00\x00\x00\x01",
                          name_a="left_eye", name_b="right_eye")
    meta, _, _ = parse_packet(pkt)
    with pytest.raises(RuntimeError, match="channel name mismatch"):
        validate_startup_meta(meta, EXPECTED_CHANNELS)


# ---------------------------------------------------------------------------
# PyAV decoder warmup (exercises WSFrameIngester._decode_channel directly)
# ---------------------------------------------------------------------------


def _make_ingester() -> WSFrameIngester:
    return WSFrameIngester(
        ws_url="ws://127.0.0.1:1",  # never actually opened in these tests
        expected_channels=EXPECTED_CHANNELS,
        frame_max_age_ms=10_000,
        reconnect_backoff_ms_list=[100],
        startup_timeout_ms=1000,
        logger=logging.getLogger("test_ws_ingest"),
    )


def test_decode_channel_produces_frame(ws_packet_bytes):
    """Real H.264 chunks must decode to BGR frames (allowing PyAV warmup)."""
    ing = _make_ingester()
    snaps = []
    for i, pkt in enumerate(ws_packet_bytes):
        meta, data_a, _ = parse_packet(pkt)
        ing._decode_channel("head_left", data_a, meta["channels"][0]["stamp_ns"], meta["pair_seq"])
        snap = ing.latest("head_left")
        if snap is not None:
            snaps.append((i, snap))
    assert snaps, "expected at least one decoded frame across the fixture"
    first_i, first_snap = snaps[0]
    assert first_i <= 2, f"PyAV should warm up within 3 packets, got first frame at {first_i}"
    assert first_snap.bgr.ndim == 3
    assert first_snap.bgr.shape[2] == 3
    assert first_snap.bgr.dtype.name == "uint8"
    assert first_snap.pair_seq == 1000 + first_i


def test_decode_channel_rejects_extended_warmup(monkeypatch):
    """If decoder never yields a frame within WARMUP_MAX_EMPTY_PACKETS+1 packets,
    we abort with RuntimeError."""
    ing = _make_ingester()
    # Replace decoder with a stub that always returns no frames.
    class StubDecoder:
        def parse(self, _raw):  # noqa: D401
            return []
        def decode(self, _pkt):  # pragma: no cover
            return []
    ing._decoders["head_left"] = StubDecoder()  # type: ignore[assignment]
    ing._empty_packet_count["head_left"] = 0

    raw = b"\x00\x00\x00\x01\x67dummy"  # never decoded by stub
    for i in range(2):
        ing._decode_channel("head_left", raw, 0, i)
        assert ing.latest("head_left") is None
    # 3rd empty packet must trip the warmup guard
    with pytest.raises(RuntimeError, match="warmup"):
        ing._decode_channel("head_left", raw, 0, 2)


def test_health_starts_empty():
    ing = _make_ingester()
    h = ing.health()
    assert h["fps_5s"] == 0.0
    assert h["last_frame_age_per_channel_ms"] == {}
    assert h["decode_fail_count"] == 0
    assert h["reconnect_count"] == 0
    assert h["pair_seq_gaps"] == 0


def test_latest_returns_none_for_stale_frame(ws_packet_bytes):
    """frame_max_age_ms gating: ancient capture_ts_ns must yield None."""
    ing = WSFrameIngester(
        ws_url="ws://127.0.0.1:1",
        expected_channels=EXPECTED_CHANNELS,
        frame_max_age_ms=1,  # 1 ms — fixture stamps are wall-clock-ish
        reconnect_backoff_ms_list=[100],
        startup_timeout_ms=1000,
        logger=logging.getLogger("test_ws_ingest"),
    )
    import time as _time
    old_stamp = _time.time_ns() - 1_000_000_000  # 1 s old
    meta, data_a, _ = parse_packet(ws_packet_bytes[0])
    ing._decode_channel("head_left", data_a, old_stamp, meta["pair_seq"])
    # might decode or not depending on warmup; force the cache to a stale snap
    from fastwam.server.ws_ingest import FrameSnapshot
    import numpy as np
    ing._latest["head_left"] = FrameSnapshot(
        bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        capture_ts_ns=old_stamp,
        decode_ts_ns=_time.time_ns(),
        pair_seq=meta["pair_seq"],
    )
    assert ing.latest("head_left") is None
