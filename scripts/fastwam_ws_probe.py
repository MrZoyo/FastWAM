#!/usr/bin/env python3
"""WS probe for fastwam — connects to rgbd_ws_bridge V4 stream and dumps
the first packet meta JSON plus per-packet pair_seq / shape / recv timing.

Usage: python scripts/fastwam_ws_probe.py [--url ws://...] [--max-packets N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse

import av
import websocket

# Make the package importable when running this script from repo root.
_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from fastwam.server.ws_ingest import parse_packet  # noqa: E402


def _ensure_no_proxy(url: str) -> str | None:
    host = urllib.parse.urlparse(url).hostname
    if not host:
        return None
    for key in ("NO_PROXY", "no_proxy"):
        entries = [s.strip() for s in os.environ.get(key, "").split(",") if s.strip()]
        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)
    return host


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fastwam WS probe (V4 encoded_sync_pair)")
    p.add_argument("--url", default="ws://192.168.31.66:19095", help="websocket URL")
    p.add_argument("--max-packets", type=int, default=8, help="stop after N packets")
    p.add_argument("--no-decode", action="store_true", help="skip H.264 decode")
    p.add_argument("--connect-timeout-s", type=float, default=10.0, help="overall wallclock budget")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    host = _ensure_no_proxy(args.url)
    print(f"probe url={args.url} host={host} max_packets={args.max_packets}", flush=True)

    decoders: dict[str, object] = {}
    samples: list[dict] = []
    t_start = time.monotonic()

    def on_open(_ws):
        print(f"connected url={args.url}", flush=True)

    def on_err(_ws, err):
        print(f"ws error: {err}", file=sys.stderr, flush=True)

    def on_close(_ws, code, msg):
        print(f"ws closed code={code} msg={msg}", flush=True)

    def on_msg(ws, msg):
        recv_ns = time.time_ns()
        try:
            meta, da, db = parse_packet(msg)
        except Exception as exc:
            print(f"parse error: {exc}", file=sys.stderr, flush=True)
            return
        decoded: dict[str, object] = {}
        if not args.no_decode:
            for ch_meta, raw in zip(meta.get("channels", []), (da, db)):
                name = str(ch_meta.get("name") or ch_meta.get("topic") or "ch?")
                dec = decoders.setdefault(name, av.CodecContext.create("h264", "r"))
                try:
                    frames = []
                    for pkt in dec.parse(raw):
                        frames.extend(dec.decode(pkt))
                    shape = list(frames[-1].to_ndarray(format="bgr24").shape) if frames else None
                    decoded[name] = {"shape": shape, "bytes_in": len(raw)}
                except Exception as exc:
                    decoded[name] = f"decode_error={exc}"
        samples.append({"recv_ns": recv_ns, "pair_seq": meta.get("pair_seq"),
                        "meta": meta, "decoded": decoded})
        if len(samples) >= args.max_packets:
            try:
                ws.close()
            except Exception:
                pass

    ws = websocket.WebSocketApp(
        args.url,
        on_open=on_open,
        on_message=on_msg,
        on_error=on_err,
        on_close=on_close,
    )
    run_kwargs: dict = {"ping_interval": 0}
    if host:
        run_kwargs["http_no_proxy"] = [host]
    try:
        ws.run_forever(**run_kwargs)
    except Exception as exc:
        print(f"run_forever raised: {exc}", file=sys.stderr, flush=True)

    if not samples:
        print("no packets received (upstream WS may be down)", file=sys.stderr)
        return 1

    print("\n========== FIRST PACKET META (full JSON) ==========")
    print(json.dumps(samples[0]["meta"], indent=2, ensure_ascii=False))

    print("\n========== DECODED SHAPES (per packet) ==========")
    for i, s in enumerate(samples):
        print(f"pkt{i} pair_seq={s['pair_seq']} decoded={s['decoded']}")

    print("\n========== TIMING ==========")
    if len(samples) >= 2:
        dts = [(samples[i]["recv_ns"] - samples[i - 1]["recv_ns"]) / 1e6
               for i in range(1, len(samples))]
        print(f"recv intervals (ms): {[round(x, 2) for x in dts]}")
        avg = (len(samples) - 1) * 1000.0 / sum(dts) if sum(dts) > 0 else 0.0
        print(f"avg fps over {len(samples)} packets: {avg:.2f}")
        seqs = [s["pair_seq"] for s in samples]
        print(f"pair_seq: {seqs}  gaps: {[seqs[i]-seqs[i-1] for i in range(1,len(seqs))]}")
        for i, s in enumerate(samples[:3]):
            ch = s["meta"].get("channels", [])
            if len(ch) == 2:
                ts0, ts1 = int(ch[0].get("stamp_ns", 0)), int(ch[1].get("stamp_ns", 0))
                print(f"pkt{i} ch0={ch[0].get('name')} ts={ts0} | "
                      f"ch1={ch[1].get('name')} ts={ts1} | diff_ms={(ts1-ts0)/1e6:.3f}")
    print(f"\ntotal_probe_t={time.monotonic()-t_start:.2f}s n_samples={len(samples)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
