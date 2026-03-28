"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from core import Frame

# ---------------------------------------------------------------------------
# System prompt - tweak this to improve your agent's guessing ability.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are playing a visual charades guessing game.

You will receive 10 image frames from the same short sequence. Most frames are
segmented to keep the acting person and remove background noise. Use the full
set of frames together to infer the charades answer.

Rules:
- Consider all frames before answering.
- Focus on the repeated action, object, or gesture across the sequence.
- Ignore irrelevant background clutter or transparent image regions.
- Give your best guess as a short, specific answer (1-5 words).
- If the sequence is still too ambiguous, respond with exactly "SKIP".
"""

_BATCH_SIZE = 10
_INPUT_DIR = Path(__file__).resolve().parents[3] / "FolderImage" / "input"
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "FolderImage" / "output"
_FRAME_BUFFER: deque["_BufferedFrame"] = deque(maxlen=_BATCH_SIZE)
_FRAME_COUNTER = 0
_BATCH_COUNTER = 0
_LAST_EMITTED_GUESS: str | None = None


@dataclass(frozen=True)
class _BufferedFrame:
    """Saved frame artifacts used for cumulative model inference."""

    sequence_no: int
    timestamp_iso: str
    raw_path: Path
    segmented_path: Path
    model_label: str
    model_bytes: bytes


def _default_model_name() -> str:
    return os.getenv("LLM_MODEL", "openai:gpt-4.1-mini").strip()


def _configure_model_env(model_name: str) -> None:
    """Bridge a generic workspace key to the default OpenAI provider."""
    generic_key = os.getenv("LLM_API_KEY")
    if model_name.startswith("openai:") and generic_key:
        os.environ.setdefault("OPENAI_API_KEY", generic_key)


@lru_cache(maxsize=1)
def _get_agent() -> Agent:
    model_name = _default_model_name()
    _configure_model_env(model_name)
    return Agent(model_name, system_prompt=SYSTEM_PROMPT)


def _ensure_frame_dirs() -> None:
    _INPUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _next_frame_number() -> int:
    global _FRAME_COUNTER
    _FRAME_COUNTER += 1
    return _FRAME_COUNTER


def _next_batch_number() -> int:
    global _BATCH_COUNTER
    _BATCH_COUNTER += 1
    return _BATCH_COUNTER


def _image_to_png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_frame_artifacts(frame: Frame) -> _BufferedFrame:
    sequence_no = _next_frame_number()
    timestamp_iso = frame.timestamp.isoformat()

    _ensure_frame_dirs()

    raw_path = _INPUT_DIR / f"frame_{sequence_no:05d}.png"
    frame.image.save(raw_path)
    print(f"  [input] Saved raw frame -> {raw_path}")

    segmented_image = frame.person_only_image()
    segmented_path = _OUTPUT_DIR / f"frame_{sequence_no:05d}_segmented.png"
    segmented_image.save(segmented_path)
    print(f"  [segment] Saved segmented frame -> {segmented_path}")

    has_subject = False
    if segmented_image.mode == "RGBA":
        has_subject = segmented_image.getchannel("A").getbbox() is not None

    if has_subject:
        model_label = "segmented"
        model_bytes = _image_to_png_bytes(segmented_image)
        print("  [segment] Foreground detected; using segmented frame for the model")
    else:
        model_label = "original"
        model_bytes = _image_to_png_bytes(frame.image)
        print("  [segment] Segmentation empty; falling back to original frame for the model")

    return _BufferedFrame(
        sequence_no=sequence_no,
        timestamp_iso=timestamp_iso,
        raw_path=raw_path,
        segmented_path=segmented_path,
        model_label=model_label,
        model_bytes=model_bytes,
    )


def _build_batch_prompt(frames: list[_BufferedFrame]) -> list[str | BinaryContent]:
    prompt_parts: list[str | BinaryContent] = [
        (
            "These 10 frames are from one charades sequence. Infer the single best "
            'charades answer from the full sequence. Reply with 1-5 words or exactly "SKIP".'
        )
    ]

    for index, buffered_frame in enumerate(frames, start=1):
        prompt_parts.append(
            f"Frame {index} of {len(frames)} at {buffered_frame.timestamp_iso}. "
            f"Preferred image source: {buffered_frame.model_label}."
        )
        prompt_parts.append(
            BinaryContent(
                buffered_frame.model_bytes,
                media_type="image/png",
                identifier=f"frame-{buffered_frame.sequence_no:05d}",
            )
        )

    return prompt_parts


async def _infer_from_batch(frames: list[_BufferedFrame]) -> str | None:
    model_name = _default_model_name()
    _configure_model_env(model_name)

    if model_name.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        print("  [model] Missing OPENAI_API_KEY/LLM_API_KEY; skipping inference")
        return None

    print(f"  [model] Sending {len(frames)} cumulative frames to {model_name}")

    try:
        result = await _get_agent().run(_build_batch_prompt(frames))
    except Exception as exc:
        print(f"  [model] Inference failed: {exc}")
        return None

    answer = result.output.strip()
    print(f"  [model] Raw model response: {answer or '<empty>'}")

    if not answer or answer.upper() == "SKIP":
        return None

    return " ".join(answer.split())


async def analyze(frame: Frame) -> str | None:
    """Analyze a frame and return a cumulative charades guess, or None."""
    global _LAST_EMITTED_GUESS

    print(f"  [stream] Received frame {frame.timestamp.isoformat()}")
    print(f"  [stream] Frame size: {frame.image.size[0]}x{frame.image.size[1]}")

    buffered_frame = _save_frame_artifacts(frame)
    _FRAME_BUFFER.append(buffered_frame)

    print(f"  [buffer] Collected {len(_FRAME_BUFFER)}/{_BATCH_SIZE} frames")
    if len(_FRAME_BUFFER) < _BATCH_SIZE:
        print("  [buffer] Waiting for more frames before calling the model")
        return None

    batch_number = _next_batch_number()
    frames = list(_FRAME_BUFFER)
    _FRAME_BUFFER.clear()

    print(f"  [buffer] Batch #{batch_number} ready")
    guess = await _infer_from_batch(frames)

    if guess is None:
        print(f"  [judge] Batch #{batch_number} result: SKIP")
        return None

    if guess == _LAST_EMITTED_GUESS:
        print(f"  [judge] Duplicate guess suppressed: {guess}")
        return None

    _LAST_EMITTED_GUESS = guess
    print(f"  [judge] Batch #{batch_number} guess: {guess}")
    return guess
