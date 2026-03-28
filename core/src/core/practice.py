"""Practice mode: capture frames from a local camera via ffmpeg subprocess."""

from __future__ import annotations

import asyncio
import platform
import re
import shutil
from datetime import datetime, timezone
from typing import AsyncIterator

from PIL import Image

from core.frame import Frame


def _detect_ffmpeg() -> str:
    """Find usable ffmpeg binary, preferring system install over imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FileNotFoundError(
        "ffmpeg not found. Install it:\n"
        "  Linux:  sudo apt install ffmpeg\n"
        "  macOS:  brew install ffmpeg\n"
        "  Windows: winget install ffmpeg"
    )


def _build_capture_cmd(ffmpeg: str, camera: int | str) -> list[str]:
    """Build a platform-appropriate ffmpeg command for single-frame capture."""
    system = platform.system()

    if system == "Linux":
        input_fmt = ["-f", "v4l2"]
        device = f"/dev/video{camera}"
    elif system == "Darwin":
        # avfoundation defaults to ~29.97 fps; many Mac cameras only allow 30.0.
        input_fmt = ["-f", "avfoundation", "-framerate", "30"]
        device = str(camera)
    elif system == "Windows":
        input_fmt = ["-f", "dshow"]
        device = f"video={camera}"
    else:
        input_fmt = ["-f", "v4l2"]
        device = f"/dev/video{camera}"

    return [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        *input_fmt,
        "-i", device,
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]


async def _capture_one_frame(cmd: list[str]) -> Image.Image:
    """Run ffmpeg once to grab a single frame, return as PIL Image."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg capture failed (exit {proc.returncode}): {err}")

    if not stdout:
        raise RuntimeError("ffmpeg returned no data")

    raw = stdout
    num_bytes = len(raw)
    for w, h in [(640, 480), (1280, 720), (1920, 1080), (320, 240), (800, 600)]:
        if w * h * 3 == num_bytes:
            return Image.frombytes("RGB", (w, h), raw)

    raise RuntimeError(
        f"Could not determine frame dimensions from {num_bytes} bytes of raw data. "
        "Try specifying resolution with -video_size in the ffmpeg command."
    )


async def _list_windows_video_devices(ffmpeg: str) -> list[str]:
    """Ask ffmpeg/dshow for available Windows video capture devices."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-list_devices",
        "true",
        "-f",
        "dshow",
        "-i",
        "dummy",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

    devices: list[str] = []
    in_video_section = False

    for raw_line in stderr.decode(errors="replace").splitlines():
        line = raw_line.strip()
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            break
        if not in_video_section:
            continue

        match = re.search(r'"([^"]+)"', line)
        if match and "Alternative name" not in line:
            devices.append(match.group(1))

    return devices


async def _resolve_windows_camera_device(ffmpeg: str, camera_index: int) -> str:
    """Map a numeric camera index to the corresponding dshow device name."""
    devices = await _list_windows_video_devices(ffmpeg)
    if not devices:
        raise RuntimeError(
            "No Windows video capture devices were found by ffmpeg/dshow."
        )
    if camera_index < 0 or camera_index >= len(devices):
        available = ", ".join(f"{idx}={name}" for idx, name in enumerate(devices))
        raise RuntimeError(
            f"Camera index {camera_index} is out of range. "
            f"Available Windows cameras: {available}"
        )
    return devices[camera_index]


async def start_practice(
    camera_index: int = 0,
    fps: int = 1,
) -> AsyncIterator[Frame]:
    """Yield frames from the local camera at the given FPS.

    Args:
        camera_index: Which camera device to use (default 0).
        fps: Frames per second to sample (default 1).

    Yields:
        Frame objects with a PIL Image and timestamp.
    """
    interval = 1.0 / fps

    print(f"[practice] Opening camera {camera_index}...")
    print(f"[practice] Sampling at {fps} FPS. Press Ctrl+C to stop.\n")

    try:
        ffmpeg = _detect_ffmpeg()
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return

    camera_label = str(camera_index)
    camera_source: int | str = camera_index

    if platform.system() == "Windows":
        try:
            camera_source = await _resolve_windows_camera_device(ffmpeg, camera_index)
        except Exception as exc:
            print(f"[!] Could not resolve camera {camera_index}: {exc}")
            return
        camera_label = f"{camera_index} ({camera_source})"

    cmd = _build_capture_cmd(ffmpeg, camera_source)

    try:
        test_frame = await _capture_one_frame(cmd)
        print(f"[practice] Camera {camera_label} ready "
              f"({test_frame.size[0]}x{test_frame.size[1]}).\n")
    except Exception as exc:
        print(f"[!] Could not capture from camera {camera_label}: {exc}")
        return

    while True:
        try:
            image = await _capture_one_frame(cmd)

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
