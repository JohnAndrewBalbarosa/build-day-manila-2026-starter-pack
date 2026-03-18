"""Practice mode: capture frames from a local camera via imageio."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import imageio.v3 as iio
from PIL import Image

from core.frame import Frame


async def start_practice(
    camera_index: int = 0,
    fps: int = 1,
) -> AsyncIterator[Frame]:
    """Yield frames from the local camera at the given FPS.

    This is for offline development and prompt tuning.
    No network connection required.

    Args:
        camera_index: Which camera device to use (default 0).
        fps: Frames per second to sample (default 1).

    Yields:
        Frame objects with a PIL Image and timestamp.
    """
    interval = 1.0 / fps

    # imageio uses ffmpeg to capture from the system camera
    # "<video{n}>" is the imageio syntax for camera devices
    device = f"<video{camera_index}>"

    print(f"[practice] Opening camera {camera_index}...")
    print(f"[practice] Sampling at {fps} FPS. Press Ctrl+C to stop.\n")

    while True:
        try:
            # Read a single frame from the camera
            raw = await asyncio.to_thread(iio.imread, device)
            image = Image.fromarray(raw).convert("RGB")

            yield Frame(
                image=image,
                timestamp=datetime.now(timezone.utc),
            )

            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n[practice] Stopped.")
            break
        except Exception as exc:
            print(f"[practice] Error capturing frame: {exc}")
            break
