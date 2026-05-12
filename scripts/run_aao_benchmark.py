#!/usr/bin/env python
"""Run batch FastWAM AAO benchmark."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.benchmark import main


if __name__ == "__main__":
    main()
