"""Ensure this worktree's ``src/`` shadows the venv-shared ``fastwam`` install.

The shared venv adds ``main`` worktree's ``src/`` to ``sys.path`` via a ``.pth``
entry, so without this prepend, ``import fastwam.server`` would resolve against
``main`` (where the subpackage may not yet exist).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
