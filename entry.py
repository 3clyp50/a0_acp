from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


def _early_project_root() -> None:
    """Ensure the Agent Zero root is importable before any plugin imports.

    Mirrors _find_agent_zero_root() below, but runs before module-level
    plugin imports so editors launching entry.py from arbitrary working
    directories still resolve the framework (agent, helpers, usr.*).
    """
    env_root = os.getenv("AGENT_ZERO_HOME", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent
        for candidate in (root, *root.parents):
            if (candidate / "run_ui.py").is_file() and (candidate / "helpers").is_dir():
                root = candidate
                break
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_early_project_root()

from usr.plugins.a0_acp import PLUGIN_VERSION


_BENIGN_PROBE_METHODS = {"ping", "health", "healthcheck"}


class _BenignProbeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "Background task failed":
            return True
        exc_info = record.exc_info
        if not exc_info:
            return True
        try:
            from acp.exceptions import RequestError
        except ImportError:
            return True
        exc = exc_info[1]
        if not isinstance(exc, RequestError):
            return True
        if getattr(exc, "code", None) != -32601:
            return True
        data = getattr(exc, "data", None)
        method = data.get("method") if isinstance(data, dict) else None
        return method not in _BENIGN_PROBE_METHODS


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.version:
        print(PLUGIN_VERSION)
        return

    if args.registry:
        registry = Path(__file__).resolve().parent / "acp_registry" / "agent.json"
        print(registry.read_text(encoding="utf-8"))
        return

    if args.check:
        _run_check()
        return

    _setup_logging(debug=args.debug)
    _ensure_project_root()

    acp = _import_acp_or_install()

    from helpers import persist_chat
    from usr.plugins.a0_acp.helpers.server import AgentZeroACPAgent

    persist_chat.load_tmp_chats()
    logger = logging.getLogger(__name__)
    logger.info("Starting Agent Zero ACP adapter")

    try:
        asyncio.run(acp.run_agent(AgentZeroACPAgent(), use_unstable_protocol=True))
    except KeyboardInterrupt:
        logger.info("ACP adapter stopped")
    except Exception:
        logger.exception("ACP adapter crashed")
        sys.exit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-zero-acp",
        description="Run Agent Zero as an Agent Client Protocol stdio server.",
    )
    parser.add_argument("--check", action="store_true", help="Verify ACP imports and exit")
    parser.add_argument("--registry", action="store_true", help="Print ACP registry metadata")
    parser.add_argument("--version", action="store_true", help="Print ACP plugin version")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr")
    return parser.parse_args(argv)


def _setup_logging(*, debug: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_BenignProbeFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _ensure_project_root() -> None:
    project_root = str(_find_agent_zero_root(Path(__file__).resolve().parent))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _find_agent_zero_root(start: Path) -> Path:
    env_root = os.getenv("AGENT_ZERO_HOME", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    for candidate in (start, *start.parents):
        if (candidate / "run_ui.py").is_file() and (candidate / "helpers").is_dir():
            return candidate

    return start


def _run_check() -> None:
    _ensure_project_root()
    _import_acp_or_install()
    from usr.plugins.a0_acp.helpers.server import AgentZeroACPAgent  # noqa: F401

    print("Agent Zero ACP check OK")


def _import_acp_or_install():
    try:
        import acp

        return acp
    except ImportError:
        pass

    from usr.plugins.a0_acp import hooks

    if hooks.ensure_dependencies(raise_on_error=False):
        try:
            import acp

            return acp
        except ImportError:
            pass

    _missing_dependency()


def _missing_dependency() -> None:
    print(
        "Agent Zero ACP requires agent-client-protocol. "
        "The plugin tried to install the ACP SDK into the framework runtime; "
        "manual fallback: pip install agent-client-protocol==0.10.1",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
