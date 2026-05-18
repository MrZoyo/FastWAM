"""Shared pytest fixtures for fastwam tests.

The shared venv pins ``fastwam`` to the main worktree's ``src/`` via a
``.pth`` entry, so this conftest prepends *this* worktree's ``src/`` to
``sys.path`` to make subpackages added by the current branch importable.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURE_PATH = _THIS_DIR / "fixtures" / "ws_packets_v4.bin"
FIXTURE_INDEX_PATH = _THIS_DIR / "fixtures" / "ws_packets_v4.idx.json"


def _build_v4_packet(
    payload_a: bytes,
    payload_b: bytes,
    name_a: str = "head_left",
    name_b: str = "right_wrist_left",
    pair_seq: int = 0,
    base_stamp_ns: int | None = None,
    payload_format: str = "h264_annexb_pair_v1",
    type_field: str = "encoded_sync_pair",
    extra_channels: int = 0,
) -> bytes:
    """Serialize a synthetic V4 packet matching the rgbd_ws_bridge protocol."""
    base = base_stamp_ns if base_stamp_ns is not None else time.time_ns()
    channels = [
        {
            "data_size": len(payload_a),
            "format": "h264",
            "frame_id": "camera_cam1_frame",
            "name": name_a,
            "original_data_size": len(payload_a),
            "prepended_parameter_sets": True,
            "stamp_ns": base,
            "topic": f"rt/robot/camera/{name_a}/video_encoded",
        },
        {
            "data_size": len(payload_b),
            "format": "h264",
            "frame_id": "camera_cam5_frame",
            "name": name_b,
            "original_data_size": len(payload_b),
            "prepended_parameter_sets": True,
            "stamp_ns": base + 829_502,
            "topic": f"rt/robot/camera/{name_b}/video_encoded",
        },
    ]
    for i in range(extra_channels):
        channels.append({"name": f"extra_{i}", "stamp_ns": base, "format": "h264"})
    meta = {
        "bridge_send_ns": base,
        "channels": channels,
        "delta_ns": 829_502,
        "matched_by": "timestamp",
        "pair_seq": pair_seq,
        "payload_format": payload_format,
        "type": type_field,
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    header = struct.pack(
        "<8sIII",
        b"RGBDWS4\x00",
        len(meta_bytes),
        len(payload_a),
        len(payload_b),
    )
    return header + meta_bytes + payload_a + payload_b


def _encode_h264_packets(num_frames: int = 5, width: int = 320, height: int = 240) -> list[bytes]:
    """Encode ``num_frames`` synthetic frames with libx264 (annex-b)."""
    from fractions import Fraction

    import av

    out: list[bytes] = []
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width = width
    ctx.height = height
    ctx.pix_fmt = "yuv420p"
    ctx.time_base = Fraction(1, 30)
    ctx.framerate = Fraction(30, 1)
    ctx.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "annexb=1"}

    rng = np.random.default_rng(0)
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = (np.arange(width)[None, :] + i * 8) % 256
        img[:, :, 1] = (np.arange(height)[:, None] + i * 4) % 256
        img[:, :, 2] = rng.integers(0, 64, (height, width), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        packets = list(ctx.encode(frame))
        out.append(b"".join(bytes(p) for p in packets) if packets else b"")
    flushed = list(ctx.encode(None))
    if flushed and out:
        out[-1] = out[-1] + b"".join(bytes(p) for p in flushed)
    return out


def _ensure_fixture() -> tuple[bytes, list[tuple[int, int]]]:
    """Force-regenerate V4 packet fixture every session.

    libx264-encoded bytes are not safe to cache across processes: a previously
    generated .bin can fail to decode in a fresh PyAV/h264 decoder context.
    Regenerating per session is cheap (~5 frames, single-digit ms).
    """

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    h264_chunks = _encode_h264_packets(num_frames=5)
    assert all(c for c in h264_chunks), "libx264 produced empty packets"

    blob = bytearray()
    index: list[dict[str, int]] = []
    base = time.time_ns()
    for i, chunk in enumerate(h264_chunks):
        pkt = _build_v4_packet(
            payload_a=chunk,
            payload_b=chunk,
            pair_seq=1000 + i,
            base_stamp_ns=base + i * 33_000_000,
        )
        index.append({"offset": len(blob), "length": len(pkt), "pair_seq": 1000 + i})
        blob.extend(pkt)

    FIXTURE_PATH.write_bytes(bytes(blob))
    FIXTURE_INDEX_PATH.write_text(json.dumps(index, indent=2))
    return bytes(blob), [(int(e["offset"]), int(e["length"])) for e in index]


@pytest.fixture(scope="session")
def ws_packet_bytes() -> list[bytes]:
    blob, idx = _ensure_fixture()
    return [blob[o : o + n] for o, n in idx]


@pytest.fixture(scope="session")
def build_v4_packet():
    return _build_v4_packet


@pytest.fixture(scope="session")
def encode_h264_packets():
    return _encode_h264_packets
