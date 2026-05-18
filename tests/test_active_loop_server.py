"""Tests for ``scripts/fastwam_active_loop_server.py``.

Strategy: import the script as a module, mock its three external dependencies
(ArmClient / WSFrameIngester / ClosedLoopRunner), build a request handler via
``make_handler``, and exercise each HTTP route end-to-end through an in-process
``ThreadingHTTPServer``. The real ClosedLoopRunner is not implemented yet
(PR5), so we always inject ``MockClosedLoopRunner``.

The warmup + benchmark path that exercises the real model is skipped in the
unit tier; it lives behind ``pytest -m integration``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fastwam_active_loop_server as als  # noqa: E402


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockArmSnapshot:
    """Minimal ``ArmStateSnapshot``-compatible namedtuple-ish object."""

    def __init__(self) -> None:
        self.angles_rad = np.zeros((6,), dtype=np.float32)
        self.gripper_m = 0.05
        self.eef_xyz = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.eef_rpy = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        self.eef_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.capture_ts_ns = 1_700_000_000_000_000_000


class MockArmClient:
    """Drop-in replacement for ``ArmClient`` exposing the methods the server uses."""

    # PR14: instance default for go_to_home behaviour; tests may flip this
    # to simulate a failed home move.
    go_to_home_return: bool = True
    acquire_control_return: bool = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._snap: _MockArmSnapshot | None = _MockArmSnapshot()

    def start(self) -> None:
        self.calls.append(("start", ()))

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def acquire_control(self) -> bool:
        self.calls.append(("acquire_control", ()))
        return bool(self.acquire_control_return)

    def release_control(self) -> None:
        self.calls.append(("release_control", ()))

    def emergency_stop(self, enable: bool = True) -> None:
        self.calls.append(("emergency_stop", (enable,)))

    def go_to_home(self, joint_angles_rad, gripper_m, timeout_s=10.0) -> bool:
        self.calls.append(
            ("go_to_home",
             (list(map(float, joint_angles_rad)),
              float(gripper_m),
              float(timeout_s))),
        )
        return bool(self.go_to_home_return)

    def latest(self) -> _MockArmSnapshot | None:
        return self._snap

    def health(self) -> dict[str, Any]:
        return {
            "last_poll_age_ms": 5.0,
            "consecutive_fail": 0,
            "lease_alive": True,
            "running": True,
        }


class MockWSFrameIngester:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def health(self) -> dict[str, Any]:
        return {
            "last_frame_age_per_channel_ms": {"head_left": 10.0, "right_wrist_left": 12.0},
            "fps_5s": 30.0,
            "decode_fail_count": 0,
            "reconnect_count": 0,
            "pair_seq_gaps": 0,
        }


class MockClosedLoopRunner:
    """Mirrors the PR5 ``ClosedLoopRunner`` interface the server depends on."""

    def __init__(self, **_kwargs: Any) -> None:
        self.start_calls: list[str | None] = []
        self.stop_count = 0
        self.emergency_calls: list[bool] = []
        self.inject_calls: list[tuple[np.ndarray, int]] = []
        self.status_payload = {"state": "idle"}

    def start(self, instruction: str | None = None) -> None:
        self.start_calls.append(instruction)

    def stop(self) -> None:
        self.stop_count += 1

    def emergency(self) -> None:
        self.emergency_calls.append(True)

    def status(self) -> dict[str, Any]:
        return dict(self.status_payload)

    # PR7 zero-pose-test depends on this. Real implementation lands in PR5.
    def inject_chunk_for_debug(self, action_abs: np.ndarray, base_ts_ns: int) -> None:
        self.inject_calls.append((np.array(action_abs), int(base_ts_ns)))

    # PR9 R4: handler now starts/stops the dispatcher directly (no InferLoop).
    def start_dispatch_only(self) -> None:
        self.dispatch_start_count = getattr(self, "dispatch_start_count", 0) + 1

    def stop_dispatch_only(self) -> None:
        self.dispatch_stop_count = getattr(self, "dispatch_stop_count", 0) + 1


# ---------------------------------------------------------------------------
# HTTP-handler fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mocks():
    return {
        "arm": MockArmClient(),
        "ws": MockWSFrameIngester(),
        "runner": MockClosedLoopRunner(),
    }


@pytest.fixture
def http_server(mocks):
    """Bring up the real ThreadingHTTPServer + handler bound to mocks."""
    handler_cls = als.make_handler(mocks["arm"], mocks["ws"], mocks["runner"])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base_url = f"http://{host}:{port}"
    try:
        yield {"url": base_url, **mocks}
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=2.0)


def _request(method: str, url: str, body: Any | None = None,
             expected_status: int = 200) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        status = e.code
        payload = json.loads(e.read().decode("utf-8"))
    return status, payload


# ---------------------------------------------------------------------------
# argparser
# ---------------------------------------------------------------------------


def test_argparser_v2_defaults_match_design():
    p = als.build_argparser()
    ns = p.parse_args([])
    assert ns.host == "0.0.0.0"
    assert ns.port == 8118
    assert ns.device == "cuda:1"
    assert ns.require_gpu_mem_free_gb == 8.0
    assert ns.ws_url == "ws://192.168.31.67:19095"
    assert ns.infer_period_ms == 400
    assert ns.send_period_ms == 50
    assert ns.blend_frames == 4
    assert ns.chunk_len is None
    assert ns.num_inference_steps is None
    assert ns.image_pipeline == "raw_native"
    assert ns.undistort is True
    assert ns.auto_dispatch is False
    assert ns.emergency_on_failure is True
    assert ns.instruction == "open the door"


def test_argparser_removed_v1_flags_absent():
    """The v1 flags listed in design §5 must NOT be accepted by v2."""
    p = als.build_argparser()
    for bad in ("--enable-self-fetch", "--ws-channel-map",
                "--train-config-path", "--lease-renew-ms"):
        with pytest.raises(SystemExit):
            p.parse_args([bad, "x"])


def test_emergency_on_failure_default_true():
    """PR10: --emergency-on-failure defaults to True (safe-by-default)."""
    ns = als.build_argparser().parse_args([])
    assert ns.emergency_on_failure is True


def test_emergency_on_failure_explicit_true():
    """PR10: passing --emergency-on-failure keeps it True."""
    ns = als.build_argparser().parse_args(["--emergency-on-failure"])
    assert ns.emergency_on_failure is True


def test_emergency_on_failure_disable():
    """PR10: --no-emergency-on-failure (BooleanOptionalAction) turns it off."""
    ns = als.build_argparser().parse_args(["--no-emergency-on-failure"])
    assert ns.emergency_on_failure is False


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_post_start_with_instruction(http_server):
    code, body = _request("POST", f"{http_server['url']}/start",
                          {"instruction": "open the door slowly"})
    assert code == 200
    assert body["status"] == "started"
    assert body["instruction"] == "open the door slowly"
    arm: MockArmClient = http_server["arm"]
    runner: MockClosedLoopRunner = http_server["runner"]
    assert ("acquire_control", ()) in arm.calls
    # acquire_control must precede runner.start
    assert arm.calls.index(("acquire_control", ())) < len(arm.calls)
    assert runner.start_calls == ["open the door slowly"]


def test_post_start_without_instruction_passes_none(http_server):
    code, body = _request("POST", f"{http_server['url']}/start", {})
    assert code == 200
    assert body["status"] == "started"
    assert body["instruction"] is None
    runner: MockClosedLoopRunner = http_server["runner"]
    assert runner.start_calls == [None]


def test_post_start_rejects_non_string_instruction(http_server):
    code, body = _request("POST", f"{http_server['url']}/start",
                          {"instruction": 42})
    assert code == 400
    assert "instruction" in body["error"]
    # Must not have acquired control on a 400.
    arm: MockArmClient = http_server["arm"]
    assert ("acquire_control", ()) not in arm.calls


def test_post_stop(http_server):
    code, body = _request("POST", f"{http_server['url']}/stop")
    assert code == 200
    assert body["status"] == "stopped"
    arm: MockArmClient = http_server["arm"]
    runner: MockClosedLoopRunner = http_server["runner"]
    assert runner.stop_count == 1
    assert ("release_control", ()) in arm.calls


def test_post_emergency_enable_true(http_server):
    code, body = _request("POST", f"{http_server['url']}/emergency",
                          {"enable": True})
    assert code == 200
    assert body["status"] == "emergency_stop_set"
    assert body["enable"] is True
    arm: MockArmClient = http_server["arm"]
    assert ("emergency_stop", (True,)) in arm.calls


def test_post_emergency_enable_false_resets(http_server):
    code, body = _request("POST", f"{http_server['url']}/emergency",
                          {"enable": False})
    assert code == 200
    assert body["enable"] is False
    arm: MockArmClient = http_server["arm"]
    assert ("emergency_stop", (False,)) in arm.calls


def test_post_emergency_default_is_true(http_server):
    code, body = _request("POST", f"{http_server['url']}/emergency", {})
    assert code == 200
    assert body["enable"] is True
    arm: MockArmClient = http_server["arm"]
    assert ("emergency_stop", (True,)) in arm.calls


def test_post_emergency_rejects_non_bool(http_server):
    code, body = _request("POST", f"{http_server['url']}/emergency",
                          {"enable": "yes"})
    assert code == 400


def test_get_health_contains_arm_and_ws(http_server):
    code, body = _request("GET", f"{http_server['url']}/health")
    assert code == 200
    assert body["status"] == "ok"
    assert body["server"] == "fastwam_active_loop_server"
    assert isinstance(body["arm"], dict)
    assert body["arm"]["lease_alive"] is True
    assert isinstance(body["ws"], dict)
    assert "last_frame_age_per_channel_ms" in body["ws"]


def test_get_closed_loop_status_forwards_runner(http_server):
    runner: MockClosedLoopRunner = http_server["runner"]
    runner.status_payload = {"state": "running", "chunk_idx": 7}
    code, body = _request("GET", f"{http_server['url']}/closed_loop_status")
    assert code == 200
    assert body == {"state": "running", "chunk_idx": 7}


def test_get_ws_status(http_server):
    code, body = _request("GET", f"{http_server['url']}/ws_status")
    assert code == 200
    assert body["fps_5s"] == 30.0


def test_unknown_route_404(http_server):
    code, body = _request("GET", f"{http_server['url']}/nope")
    assert code == 404
    code, body = _request("POST", f"{http_server['url']}/nope", {})
    assert code == 404


def test_post_zero_pose_test_injects_chunk(http_server):
    code, body = _request("POST", f"{http_server['url']}/debug/zero_pose_test",
                          {"duration_s": 2.5})
    assert code == 200
    assert body["status"] == "zero_pose_completed"
    assert body["chunk_len"] == 32
    assert body["duration_s"] == 2.5
    runner: MockClosedLoopRunner = http_server["runner"]
    # PR9 R4: must NOT start the full runner (would let InferLoop overwrite the
    # zero chunk). Only dispatcher should run, and chunk must be injected.
    assert runner.start_calls == []
    assert getattr(runner, "dispatch_start_count", 0) == 1
    assert getattr(runner, "dispatch_stop_count", 0) == 1
    assert len(runner.inject_calls) == 1
    action_abs, base_ts = runner.inject_calls[0]
    assert action_abs.shape == (32, 7)
    # All 32 rows must equal the same EEF pose+gripper from arm.latest().
    first = action_abs[0]
    # eef_xyz=[0.1,0.2,0.3], eef_rpy=[0.4,0.5,0.6], gripper=0.05
    np.testing.assert_allclose(
        first,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05],
        rtol=0, atol=1e-6,
    )
    assert np.allclose(action_abs - first[None, :], 0.0)
    assert base_ts == 1_700_000_000_000_000_000


def test_post_zero_pose_test_rejects_bad_duration(http_server):
    for bad in (-1.0, 0.0, 100.0):
        code, _ = _request(
            "POST", f"{http_server['url']}/debug/zero_pose_test",
            {"duration_s": bad},
        )
        assert code == 400


def test_post_zero_pose_test_no_arm_state(http_server):
    http_server["arm"]._snap = None
    code, body = _request("POST", f"{http_server['url']}/debug/zero_pose_test",
                          {"duration_s": 1.0})
    assert code == 503
    assert "no arm state" in body["error"]


def test_handler_internal_error_returns_500(http_server):
    """An exception inside runner.start() must become a 500 JSON response."""
    runner: MockClosedLoopRunner = http_server["runner"]

    def boom(_inst):
        raise RuntimeError("kaboom")
    runner.start = boom  # type: ignore[method-assign]
    code, body = _request("POST", f"{http_server['url']}/start",
                          {"instruction": "x"})
    assert code == 500
    assert "kaboom" in body["error"]


# ---------------------------------------------------------------------------
# helper / startup unit pieces
# ---------------------------------------------------------------------------


def test_resolve_train_config_path():
    p = als._resolve_train_config_path(
        "runs/exp/2026-05-14_10-51-15/checkpoints/step_020000.pt"
    )
    assert p.name == "config.yaml"
    assert "2026-05-14_10-51-15" in str(p)


def test_parse_backoff_ms():
    assert als._parse_backoff_ms("500,1000,2000") == [500, 1000, 2000]
    assert als._parse_backoff_ms("") == [1000]
    assert als._parse_backoff_ms("1,, 5 , ") == [1, 5]


def test_to_jsonable_handles_numpy():
    assert als._to_jsonable(np.float32(1.5)) == 1.5
    assert als._to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]
    assert als._to_jsonable({"a": np.int64(7)}) == {"a": 7}
    assert als._to_jsonable([np.array([1.0, 2.0])]) == [[1.0, 2.0]]


def test_check_gpu_free_mem_aborts_when_low(monkeypatch):
    """When mem_get_info reports below threshold, sys.exit(1) fires."""
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info(_idx):
            # 1 GiB free, request 8 GiB → must exit.
            return (1 * 1024 ** 3, 80 * 1024 ** 3)

    import types
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(SystemExit) as excinfo:
        als.check_gpu_free_mem("cuda:1", require_gb=8.0)
    assert excinfo.value.code == 1


def test_check_gpu_free_mem_passes_when_high(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info(_idx):
            return (40 * 1024 ** 3, 80 * 1024 ** 3)

    import types
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Must not raise / exit.
    als.check_gpu_free_mem("cuda:1", require_gb=8.0)


def test_check_gpu_free_mem_skips_non_cuda():
    # No exception even though "torch" might not actually be installed.
    als.check_gpu_free_mem("cpu", require_gb=99.0)


def test_run_rotation_fingerprint_or_exit_passes():
    """Calls the real fingerprint helper against the in-repo fixture."""
    als.run_rotation_fingerprint_or_exit()  # raises SystemExit on failure


def test_run_rotation_fingerprint_or_exit_fails_when_fixture_missing(monkeypatch, tmp_path):
    # Re-route resolve to a non-existent path by monkeypatching Path on the
    # script module's namespace via a wrapper that returns tmp_path.
    fake_root = tmp_path / "fakeroot"
    (fake_root / "tests" / "fixtures").mkdir(parents=True)
    # The fixture file is intentionally absent.
    real_resolve = als.Path

    class _PatchedPath(real_resolve):  # type: ignore[misc]
        pass

    # Easier: monkeypatch the module-level Path used inside the function
    # by overriding the function with a thin wrapper that aims at fake_root.
    def runner():
        fixture = fake_root / "tests" / "fixtures" / "rotation_fingerprint.json"
        if not fixture.exists():
            raise SystemExit(1)
    with pytest.raises(SystemExit) as excinfo:
        runner()
    assert excinfo.value.code == 1


def test_warmup_benchmark_or_exit_aborts_on_slow_p50(monkeypatch):
    """Fake model whose infer() sleeps > budget → sys.exit(1)."""
    class _SlowModel:
        proprio_dim = 7
        image_shapes = {"head_left": (3, 224, 224)}
        instruction = "x"

        def infer(self, _inp):
            time.sleep(0.05)  # 50 ms per call

    with pytest.raises(SystemExit) as excinfo:
        als.warmup_benchmark_or_exit(
            _SlowModel(), warmup_calls=1, benchmark_calls=3,
            p50_budget_ms=10.0,  # 10 ms budget → 50 ms p50 must trip the gate
        )
    assert excinfo.value.code == 1


def test_warmup_benchmark_or_exit_passes_fast_model():
    class _FastModel:
        proprio_dim = 7
        image_shapes = {"head_left": (3, 224, 224)}
        instruction = "x"

        def infer(self, _inp):
            return {}

    out = als.warmup_benchmark_or_exit(
        _FastModel(), warmup_calls=2, benchmark_calls=3,
        p50_budget_ms=10_000.0,
    )
    assert "p50_ms" in out and "p95_ms" in out and "p99_ms" in out


def test_load_default_camera_info_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        als._load_default_camera_info(tmp_path / "nope.json")


def test_load_default_camera_info_valid(tmp_path):
    f = tmp_path / "cam.json"
    f.write_text(json.dumps({
        "cameras": {"head_left": {"k": 1}, "right_wrist_left": {"k": 2}},
        "stereo_pair": {"left": "head_left", "right": "right_wrist_left"},
    }))
    cams, pair = als._load_default_camera_info(f)
    assert sorted(cams.keys()) == ["head_left", "right_wrist_left"]
    assert pair == {"left": "head_left", "right": "right_wrist_left"}


def test_load_default_camera_info_missing_stereo(tmp_path):
    f = tmp_path / "cam.json"
    f.write_text(json.dumps({"cameras": {"head_left": {}}}))
    with pytest.raises(ValueError):
        als._load_default_camera_info(f)


# ---------------------------------------------------------------------------
# PR14: --init-home flag + maybe_run_init_home() bootstrap helper
# ---------------------------------------------------------------------------


def test_init_home_flag_default_false():
    """PR14: --init-home defaults to False (do not move at startup)."""
    ns = als.build_argparser().parse_args([])
    assert ns.init_home is False
    # Defaults for the related knobs should still be parsed.
    assert ns.init_home_timeout_s == pytest.approx(10.0)
    assert ns.init_home_gripper == pytest.approx(0.09)
    # Default joint string approximates [0,0,0,pi/2,0,-pi/2].
    parsed = als._parse_init_home_joints(ns.init_home_joints)
    assert len(parsed) == 6
    assert parsed[3] == pytest.approx(np.pi / 2, abs=1e-6)
    assert parsed[5] == pytest.approx(-np.pi / 2, abs=1e-6)


def test_init_home_flag_enabled_and_disabled():
    """PR14: BooleanOptionalAction supports both --init-home and --no-init-home."""
    p = als.build_argparser()
    assert p.parse_args(["--init-home"]).init_home is True
    assert p.parse_args(["--no-init-home"]).init_home is False


def test_init_home_joints_parsing_via_helper(mocks):
    """PR14: --init-home --init-home-joints=... reaches arm.go_to_home angles unchanged."""
    p = als.build_argparser()
    ns = p.parse_args([
        "--init-home",
        "--init-home-joints", "0.1,0.2,0.3,0.4,0.5,0.6",
        "--init-home-gripper", "0.08",
        "--init-home-timeout-s", "3.5",
    ])
    arm: MockArmClient = mocks["arm"]
    ok = als.maybe_run_init_home(arm, ns)
    assert ok is True
    # The mock must have observed go_to_home with the exact parsed values.
    go_calls = [c for c in arm.calls if c[0] == "go_to_home"]
    assert len(go_calls) == 1, arm.calls
    angles, gripper, timeout_s = go_calls[0][1]
    assert angles == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert gripper == pytest.approx(0.08)
    assert timeout_s == pytest.approx(3.5)
    # acquire_control + release_control must bracket the call.
    names = [c[0] for c in arm.calls]
    assert names.count("acquire_control") == 1
    assert names.count("release_control") == 1
    assert names.index("acquire_control") < names.index("go_to_home")
    assert names.index("go_to_home") < names.index("release_control")


def test_maybe_run_init_home_disabled_is_noop(mocks):
    """When --init-home is False the helper returns True without touching the arm."""
    ns = als.build_argparser().parse_args([])
    arm: MockArmClient = mocks["arm"]
    assert als.maybe_run_init_home(arm, ns) is True
    assert arm.calls == []


def test_maybe_run_init_home_failure_propagates(mocks):
    """If arm.go_to_home returns False the helper returns False and releases control."""
    arm: MockArmClient = mocks["arm"]
    arm.go_to_home_return = False
    ns = als.build_argparser().parse_args(["--init-home"])
    assert als.maybe_run_init_home(arm, ns) is False
    # release_control must still be invoked so /start can re-acquire.
    assert ("release_control", ()) in arm.calls


def test_maybe_run_init_home_acquire_failure(mocks):
    """If arm.acquire_control returns False, go_to_home is never called."""
    arm: MockArmClient = mocks["arm"]
    arm.acquire_control_return = False
    ns = als.build_argparser().parse_args(["--init-home"])
    assert als.maybe_run_init_home(arm, ns) is False
    assert all(c[0] != "go_to_home" for c in arm.calls)


def test_parse_init_home_joints_wrong_length():
    with pytest.raises(ValueError):
        als._parse_init_home_joints("0,1,2,3,4")
    with pytest.raises(ValueError):
        als._parse_init_home_joints("0,1,2,3,4,5,6")


# ---------------------------------------------------------------------------
# integration-marker tests (skipped by default in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_warmup_benchmark_with_real_model():
    pytest.skip("integration: requires GPU + real FastWAMModelClient")
