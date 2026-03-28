"""Compatibility wrapper for the main application entrypoint in ``core.app``.

Preferred entrypoints:
    uv run -m core.app --practice
    uv run -m core.app --live

This module stays in place so existing ``uv run -m agent`` commands continue
to work.
"""

from core.app import run


if __name__ == "__main__":
    run(prog="agent")
