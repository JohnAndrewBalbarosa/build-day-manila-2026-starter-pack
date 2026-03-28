"""Charades AI guessing agent — all phases.

Phases:
  1. Bilingual Foundation — Gemini Flash, preprocessing, bilingual prompt, SKIP
  2. Temporal Intelligence — keyframe detection, multi-frame context, 409 feedback
  3. Confidence Sophistication — rank-vote accumulation, adaptive threshold
  4. Multi-Model Resilience — parallel Gemini + Claude, agreement signals, fallback
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

from PIL import Image
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

from core import Frame

# ============================================================
# STRUCTURED OUTPUT MODEL
# ============================================================


class CandidateGuess(BaseModel):
    guess: str
    confidence: int  # 0-100
    reasoning: str


class FrameAnalysis(BaseModel):
    candidates: list[CandidateGuess]
    is_acting: bool


# ============================================================
# SYSTEM PROMPT — Charades-specific, bilingual
# ============================================================

SYSTEM_PROMPT = """\
You are an expert at playing charades. You are watching a live camera feed where \
a person is ACTING OUT / MIMING a concept. They are NOT holding up objects or signs — \
they are using their body, gestures, and movements to represent something.

CONTEXT:
- This game is played in the Philippines (Manila).
- Answers may be in English OR Tagalog — guess in whichever language fits best.
- Think about Pinoy Henyo categories:
  * Bagay (Object): everyday items, tools, animals
  * Tao (Person): celebrities, professions, historical figures
  * Lugar (Place): landmarks, countries, local spots
  * Pangyayari (Event): activities, holidays, sports
  * Pagkain (Food): dishes, ingredients, drinks
- Filipino references to consider: adobo, sinigang, halo-halo, balut, taho, lechon, \
jeepney, tricycle, tinikling, karaoke, mano po, Boracay, Intramuros, Rizal Park, \
Jollibee, sari-sari store, bahay kubo, sampaguita.

HOW TO ANALYZE:
- Focus ONLY on the person's gestures, body language, and movements.
- Ignore the background, furniture, and other objects unless the person is pointing at them.
- The person may do MULTIPLE sequential actions to describe ONE concept \
(e.g., pointy shape above head + swimming motion = shark/pating).
- If you see multiple frames, they show the SEQUENCE of actions — use ALL of them together.
- Be specific: "golden retriever" is better than "dog", "sinigang" is better than "soup".

Give 3-5 candidate guesses ranked by confidence. Be honest about confidence — \
don't inflate scores. If nothing is being acted out, set is_acting to false.
"""

# ============================================================
# LAZY MODEL AGENTS (created after .env is loaded)
# ============================================================

_primary_agent: Agent[None, FrameAnalysis] | None = None
_secondary_agent: Agent[None, FrameAnalysis] | None = None


def _get_primary() -> Agent[None, FrameAnalysis]:
    global _primary_agent
    if _primary_agent is None:
        _primary_agent = Agent(
            "google-gla:gemini-2.5-flash",
            output_type=FrameAnalysis,
            system_prompt=SYSTEM_PROMPT,
        )
    return _primary_agent


def _get_secondary() -> Agent[None, FrameAnalysis] | None:
    global _secondary_agent
    if _secondary_agent is None:
        try:
            _secondary_agent = Agent(
                "anthropic:claude-sonnet-4-20250514",
                output_type=FrameAnalysis,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception:
            return None
    return _secondary_agent


# ============================================================
# MODULE-LEVEL STATE
# ============================================================

_state: dict[str, Any] = {
    "frame_count": 0,
    "guess_history": [],
    "rejected_guesses": [],
    "last_returned": None,
    "candidates": {},  # guess_text (lowercase) -> cumulative_score
    "keyframe_buffer": [],  # list of jpeg bytes
    "last_keyframe_thumb": None,  # bytes for frame diff
    "round_start": None,
    "guess_count": 0,
}

# ============================================================
# TUNING CONSTANTS
# ============================================================

_MAX_KEYFRAMES = 6
_KEYFRAME_DIFF_THRESHOLD = 12
_RANK_POINTS = [5, 4, 3, 2, 1]
_MIN_FRAMES_BEFORE_GUESS = 2
_MULTI_MODEL_AGREEMENT_BONUS = 1.5


# ============================================================
# PHASE 1: Frame Preprocessing
# ============================================================

def _preprocess(image: Image.Image) -> bytes:
    """Resize to 512px max dimension, compress to JPEG q70."""
    img = image.copy()
    img.thumbnail((512, 512), Image.LANCZOS)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


# ============================================================
# PHASE 2: Keyframe Detection (PIL-only, no numpy needed)
# ============================================================

def _is_keyframe(image: Image.Image) -> bool:
    """Return True if frame is meaningfully different from last keyframe."""
    thumb = image.copy().resize((32, 32)).convert("L")
    thumb_bytes = thumb.tobytes()

    if _state["last_keyframe_thumb"] is None:
        _state["last_keyframe_thumb"] = thumb_bytes
        return True

    old = _state["last_keyframe_thumb"]
    diff = sum(abs(a - b) for a, b in zip(thumb_bytes, old)) / len(thumb_bytes)

    if diff > _KEYFRAME_DIFF_THRESHOLD:
        _state["last_keyframe_thumb"] = thumb_bytes
        return True
    return False


# ============================================================
# PHASE 3: Evidence Accumulation (Rank-Vote)
# ============================================================

def _update_candidates(analysis: FrameAnalysis, bonus: float = 1.0) -> None:
    """Merge new guesses into running scores. Never drops candidates (union-only)."""
    rejected_lower = {r.lower() for r in _state["rejected_guesses"]}

    for i, candidate in enumerate(analysis.candidates[:5]):
        name = candidate.guess.strip().lower()
        if name in rejected_lower or not name:
            continue

        rank_pts = _RANK_POINTS[i] if i < len(_RANK_POINTS) else 1
        conf_bonus = candidate.confidence / 100.0
        score = rank_pts * (1 + conf_bonus) * bonus

        _state["candidates"][name] = _state["candidates"].get(name, 0) + score


# ============================================================
# PHASE 3: Adaptive Guess Threshold
# ============================================================

def _get_threshold() -> float:
    """Dynamic threshold: conservative early, aggressive with time,
    tightens when guesses are scarce."""
    if _state["round_start"] is None:
        return 0.70

    elapsed = time.time() - _state["round_start"]
    remaining = 10 - _state["guess_count"]

    if elapsed < 5:
        base = 0.75
    elif elapsed < 15:
        base = 0.55
    elif elapsed < 30:
        base = 0.40
    else:
        base = 0.30

    if remaining <= 2:
        base = max(base, 0.70)
    elif remaining <= 4:
        base = max(base, 0.50)

    return base


def _should_guess() -> str | None:
    """Return top guess if relative confidence exceeds threshold, else None."""
    if not _state["candidates"]:
        return None

    sorted_cands = sorted(
        _state["candidates"].items(), key=lambda x: x[1], reverse=True
    )
    if not sorted_cands:
        return None

    top_guess, top_score = sorted_cands[0]
    total_score = sum(s for _, s in sorted_cands)

    if total_score == 0:
        return None

    relative_conf = top_score / total_score
    threshold = _get_threshold()

    if _state["frame_count"] < _MIN_FRAMES_BEFORE_GUESS:
        return None

    if relative_conf >= threshold:
        return top_guess

    return None


# ============================================================
# PHASE 4: Multi-Model Inference
# ============================================================

async def _call_model(
    agent: Agent[None, FrameAnalysis], prompt_parts: list[Any]
) -> FrameAnalysis | None:
    """Call a single model, return parsed result or None on failure."""
    try:
        result = await agent.run(prompt_parts)
        return result.output
    except Exception as e:
        print(f"  [agent] Model error: {type(e).__name__}: {e}")
        return None


async def _infer(prompt_parts: list[Any]) -> list[tuple[FrameAnalysis, str]]:
    """Call available models in parallel. Returns (analysis, model_name) tuples."""
    primary = _get_primary()
    secondary = _get_secondary()

    tasks = [_call_model(primary, prompt_parts)]
    names = ["gemini"]

    if secondary is not None:
        tasks.append(_call_model(secondary, prompt_parts))
        names.append("claude")

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[tuple[FrameAnalysis, str]] = []
    for r, name in zip(raw_results, names):
        if isinstance(r, FrameAnalysis):
            results.append((r, name))

    return results


# ============================================================
# BUILD PROMPT WITH MULTI-FRAME CONTEXT
# ============================================================

def _build_prompt() -> list[Any]:
    """Build multi-part prompt: text context + all buffered keyframe images."""
    parts: list[Any] = []

    text = "Analyze these charades frames. A person is acting out something — what is it?"

    if _state["rejected_guesses"]:
        rejected_str = ", ".join(_state["rejected_guesses"])
        text += f"\n\nWRONG GUESSES (do NOT guess these): {rejected_str}"

    if _state["candidates"]:
        top_5 = sorted(
            _state["candidates"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        cands_str = ", ".join(f"{g} (score:{s:.1f})" for g, s in top_5)
        text += f"\n\nRunning candidates from previous frames: {cands_str}"
        text += "\nRefine your confidence based on the latest frame(s)."

    n = len(_state["keyframe_buffer"])
    if n > 1:
        text += f"\n\nYou see {n} frames in chronological order. Frame {n} is most recent."

    parts.append(text)

    for kf_bytes in _state["keyframe_buffer"]:
        parts.append(BinaryContent(data=kf_bytes, media_type="image/jpeg"))

    return parts


# ============================================================
# MAIN ENTRY POINT
# ============================================================

async def analyze(frame: Frame) -> str | None:
    """Analyze a frame and return a guess, or None to skip.

    Called once per captured frame by __main__.py. State persists
    across calls via module-level _state dict.
    """
    _state["frame_count"] += 1

    if _state["round_start"] is None:
        _state["round_start"] = time.time()

    # --- Phase 2: Detect wrong guess (409 feedback) ---
    if _state["last_returned"] is not None:
        rejected = _state["last_returned"]
        if rejected not in _state["rejected_guesses"]:
            _state["rejected_guesses"].append(rejected)
            _state["candidates"].pop(rejected.lower(), None)
            print(f"  [agent] Wrong guess: '{rejected}' — eliminated")
        _state["last_returned"] = None

    # --- Phase 2: Keyframe detection ---
    if not _is_keyframe(frame.image):
        guess = _should_guess()
        if guess:
            _state["last_returned"] = guess
            _state["guess_count"] += 1
            _state["guess_history"].append(guess)
            print(f"  [agent] Guessing from accumulation: '{guess}'")
        return guess

    # --- Phase 1: Preprocess frame ---
    jpeg_bytes = _preprocess(frame.image)

    # --- Phase 2: Buffer keyframes ---
    _state["keyframe_buffer"].append(jpeg_bytes)
    if len(_state["keyframe_buffer"]) > _MAX_KEYFRAMES:
        _state["keyframe_buffer"] = _state["keyframe_buffer"][-_MAX_KEYFRAMES:]

    # --- Phase 4: Multi-model parallel inference ---
    prompt_parts = _build_prompt()
    results = await _infer(prompt_parts)

    if not results:
        print("  [agent] All models failed — skipping frame")
        return None

    # --- Phase 3 + 4: Evidence accumulation + agreement detection ---
    model_top_guesses: list[str] = []
    for analysis, model_name in results:
        if analysis.candidates:
            model_top_guesses.append(analysis.candidates[0].guess.strip().lower())
        _update_candidates(analysis)

    # Multi-model agreement bonus
    if len(model_top_guesses) >= 2 and model_top_guesses[0] == model_top_guesses[1]:
        agreed = model_top_guesses[0]
        if agreed in _state["candidates"]:
            _state["candidates"][agreed] *= _MULTI_MODEL_AGREEMENT_BONUS
            print(f"  [agent] Models agree: '{agreed}' — confidence boosted")

    # --- Phase 3: Guess decision ---
    guess = _should_guess()

    if guess:
        _state["last_returned"] = guess
        _state["guess_count"] += 1
        _state["guess_history"].append(guess)

    # --- Debug output ---
    elapsed = time.time() - (_state["round_start"] or time.time())
    top_3 = sorted(
        _state["candidates"].items(), key=lambda x: x[1], reverse=True
    )[:3]
    threshold = _get_threshold()
    total = sum(s for _, s in _state["candidates"].items()) if _state["candidates"] else 0
    top_conf = (top_3[0][1] / total) if top_3 and total > 0 else 0
    status = (
        f"frame={_state['frame_count']} kf={len(_state['keyframe_buffer'])} "
        f"guesses={_state['guess_count']}/10 t={elapsed:.0f}s "
        f"thr={threshold:.2f} conf={top_conf:.2f}"
    )
    cands = " | ".join(f"{g}({s:.1f})" for g, s in top_3) if top_3 else "(none)"
    print(f"  [agent] {status}")
    print(f"  [agent] Top: {cands}")

    if guess:
        print(f"  [agent] >>> SUBMITTING: '{guess}'")

    return guess
