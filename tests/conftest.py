"""Ensure the worktree's src/ is in front of any installed `fastwam` package.

When the user runs `pytest` from this worktree, the venv has `fastwam` installed
from the main checkout (e.g. /data/home/Lyle/Projects/FastWAM/src). We need to
import the worktree's own copy first so tests pick up newly added modules
(`fastwam.server.*`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# If `fastwam` was already imported from somewhere else (e.g. the installed
# venv copy) before this conftest ran, evict the cached modules so subsequent
# imports resolve against the worktree.
for mod_name in list(sys.modules):
    if mod_name == "fastwam" or mod_name.startswith("fastwam."):
        existing = sys.modules[mod_name]
        existing_file = getattr(existing, "__file__", "") or ""
        if not existing_file.startswith(str(SRC)):
            del sys.modules[mod_name]
