"""Live mode: subscribe to a LiveKit room and sample video frames.

Phase 2 — not yet implemented. Will use the `livekit` Python SDK
to connect to a LiveKit room, subscribe to the admin's video track,
and yield frames at ~1 FPS.
"""

from __future__ import annotations

from typing import AsyncIterator

from core.frame import Frame


async def start_stream(url: str, token: str) -> AsyncIterator[Frame]:
    """Connect to a LiveKit room and yield video frames.

    Args:
        url: The LiveKit server URL (e.g. wss://project.livekit.cloud).
        token: A subscribe-only JWT token from GET /feed.

    Yields:
        Frame objects with a PIL Image and timestamp.

    Raises:
        NotImplementedError: LiveKit integration is Phase 2.
    """
    raise NotImplementedError(
        "LiveKit streaming is not yet implemented (Phase 2).\n"
        "Use --practice mode for local development."
    )
    # Unreachable — here for reference when implementing Phase 2:
    # pip install livekit
    # from livekit import rtc
    # room = rtc.Room()
    # await room.connect(url, token)
    # ...subscribe to video track, yield frames...
    yield  # type: ignore[misc]  # makes this a valid async generator
