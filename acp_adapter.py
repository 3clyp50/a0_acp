#!/usr/bin/env python3
"""Console-script launcher: resolve Agent Zero root, then run the ACP adapter."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_agent_zero_root() -> Path:
    env_root = os.getenv("AGENT_ZERO_HOME", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "run_ui.py").is_file() and (candidate / "helpers").is_dir():
            return candidate
    # uvx/PyPI sandboxes: adapter lives in site-packages; probe common fleet paths
    for candidate in (Path("/a0"), Path.home() / "agent-zero" / "PMOVES.AI"):
        if (candidate / "run_ui.py").is_file():
            return candidate
    return here


_root = _find_agent_zero_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from entry import main  # noqa: E402  (A0 root importable before plugin imports)

if __name__ == "__main__":
    main()
