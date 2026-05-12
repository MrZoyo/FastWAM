#!/usr/bin/env python
"""Run minimal real AAO benchmark smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.benchmark import build_argparser, run
from fastwam.closed_loop_eval.episode_recorder import to_jsonable


DEFAULT_PROFILES = ("open_door_airbot_play_gs", "cup_on_coaster_gs_airbot_p7")


def _build_benchmark_args(args: argparse.Namespace, profile: str) -> argparse.Namespace:
    output_dir = Path(args.output_root).expanduser() / profile
    argv = [
        "--profile",
        profile,
        "--model-client",
        "hold",
        "--gpu",
        str(args.gpu),
        "--batch-size",
        str(args.batch_size),
        "--total-episodes",
        str(args.total_episodes),
        "--max-updates",
        str(args.max_updates),
        "--stride",
        str(args.stride),
        "--action-horizon",
        str(args.action_horizon),
        "--sim-loop-frequency",
        "0",
        "--output-dir",
        str(output_dir),
        "--log-level",
        args.log_level,
    ]
    profile_config = args.profile_config.get(profile)
    if profile_config is not None:
        argv.extend(["--profile-config", profile_config])
    for override in args.override:
        argv.extend(["--override", override])
    return build_argparser().parse_args(argv)


def build_argparser_smoke() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", choices=DEFAULT_PROFILES, default=[])
    parser.add_argument(
        "--profile-config",
        action="append",
        default=[],
        help="Optional PROFILE=PATH mapping for testing a custom benchmark profile YAML.",
    )
    parser.add_argument("--output-root", default="/DATA/disk3/tmp/fastwam_aao_benchmark_smoke_tests")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser_smoke().parse_args(argv)
    profile_config: dict[str, str] = {}
    for item in args.profile_config:
        if "=" not in item:
            raise ValueError(f"--profile-config must be PROFILE=PATH, got {item!r}.")
        profile, path = item.split("=", 1)
        if profile not in DEFAULT_PROFILES:
            raise ValueError(f"Unknown smoke profile for --profile-config: {profile!r}.")
        profile_config[profile] = path
    args.profile_config = profile_config
    profiles = tuple(args.profile) if args.profile else DEFAULT_PROFILES
    results = []
    for profile in profiles:
        result = run(_build_benchmark_args(args, profile))
        results.append(
            {
                "profile": profile,
                "output_dir": result["output_dir"],
                "episodes_completed": result["episodes_completed"],
                "success_rate": result["success_rate"],
                "summary_json": result["summary_json"],
                "results_csv": result["results_csv"],
                "results_jsonl": result["results_jsonl"],
            }
        )
    print(json.dumps(to_jsonable({"results": results}), indent=2))


if __name__ == "__main__":
    main()
