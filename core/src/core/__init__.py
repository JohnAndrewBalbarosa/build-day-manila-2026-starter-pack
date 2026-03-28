"""Core frame, streaming, and application primitives for Casper agents.

`core` owns the shared `Frame` type, capture/stream utilities, and the
workspace-level app entrypoint in `core.app`. Guess analysis still lives in
`agent`, and live-mode HTTP transport still lives in `api`.
"""

from core.frame import Frame
from core.practice import start_practice
from core.stream import start_stream

__all__ = ["Frame", "start_practice", "start_stream"]
