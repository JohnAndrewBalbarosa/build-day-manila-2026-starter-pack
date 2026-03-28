# Phase 1: Bilingual Foundation - Research

**Researched:** 2026-03-28
**Domain:** Vision LLM integration, bilingual charades prompt engineering, PIL-to-LLM image pipeline
**Confidence:** HIGH

## Summary

Phase 1 delivers the minimum viable competitor: an agent that sends preprocessed frames to Gemini 2.5 Flash, outputs bilingual guesses (English or Tagalog), and skips uncertain frames. All logic lives in a single file (`agent/src/agent/prompt.py`) with a fixed interface: `analyze(frame: Frame) -> str | None`.

The core technical challenge is building a reliable PIL Image -> JPEG bytes -> BinaryContent -> pydantic-ai Agent pipeline that works with `google-gla:gemini-2.5-flash`. The critical environment detail is that pydantic-ai v1.73.0 uses the `GEMINI_API_KEY` environment variable (NOT `GOOGLE_API_KEY` or the starter template's `LLM_API_KEY`). The system prompt must be charades-specific (miming, not scene description) with bilingual Filipino context. Module-level state tracks frame count and guess history across calls.

**Primary recommendation:** Build the simplest working pipeline first (preprocess -> BinaryContent -> Agent.run -> parse SKIP), then layer in the bilingual system prompt and state tracking. This phase is single-frame, single-model only -- multi-frame reasoning and confidence systems come in Phase 2.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use Gemini 2.5 Flash as the primary (and only, for Phase 1) model -- 0.53s TTFT, 215 tok/s, cheapest per frame. Model string: `google-gla:gemini-2.5-flash`
- **D-02:** Use pydantic-ai with BinaryContent to send frames. Pydantic-ai v1.73+ is already installed and supports Gemini natively.
- **D-03:** Set LLM_API_KEY in .env to a Google Gemini API key. Model string: `google-gla:gemini-2.5-flash`
- **D-04:** Resize frames to 512x512 using PIL `Image.thumbnail()` with LANCZOS resampling before sending
- **D-05:** Convert to JPEG bytes (quality 70) for LLM input -- reduces payload without losing gesture detail
- **D-06:** No additional preprocessing (no cropping, no color adjustment, no edge detection)
- **D-07:** Charades-specific prompt: "A person is acting out / miming a concept. Focus on gestures and body language, not background."
- **D-08:** Bilingual instruction: "Answers may be in English or Tagalog. Consider Filipino culture, Pinoy Henyo categories (Bagay/Tao/Lugar/Pangyayari/Pagkain), local food, celebrities, landmarks."
- **D-09:** Output format: "Respond with your best guess as a short phrase (1-5 words). If you're not confident, respond with exactly SKIP."
- **D-10:** Include a small reference list of common Filipino charades categories: food (adobo, sinigang, halo-halo, balut, taho), transport (jeepney, tricycle), cultural (tinikling, karaoke), places (Boracay, Intramuros)
- **D-11:** For Phase 1, rely on the LLM's natural SKIP behavior via prompt instruction -- no separate confidence scoring yet
- **D-12:** The analyze() function returns None when the LLM outputs "SKIP" (case-insensitive)
- **D-13:** Use a module-level dict or class instance in prompt.py for state persistence across analyze() calls
- **D-14:** Track: frame_count (int), guess_history (list of submitted guesses), last_guess (str or None)
- **D-15:** State resets are not needed -- each round is a fresh process invocation
- **D-16:** Guesses are phrases (1-5 words), not single words. "Riding a jeepney" is valid.
- **D-17:** The LLM should output the guess directly, not wrapped in JSON or structured format
- **D-18:** Strip whitespace and quotes from LLM output before submitting

### Claude's Discretion
- Exact wording of the system prompt (within the constraints above)
- PIL image mode conversion details (RGB handling)
- Error message formatting in practice mode output

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | Agent sends frames to a vision LLM and receives text guesses (single-frame baseline) | pydantic-ai BinaryContent API verified (v1.73.0); Agent.run() accepts list of [str, BinaryContent]; GEMINI_API_KEY env var required |
| FOUND-02 | Agent maintains module-level state across analyze() calls (frame history, guess history, candidates) | Python module singleton pattern confirmed; orchestrator imports analyze once and calls repeatedly; no concurrency issues |
| FOUND-03 | Agent returns None (SKIP) when confidence is below threshold, preserving limited guesses | Prompt-based SKIP instruction per D-11; case-insensitive matching per D-12; strip whitespace before comparison |
| FOUND-04 | Frames are preprocessed (resized to 512px, JPEG compressed) before LLM calls to reduce latency | PIL Image.thumbnail(512,512) preserves aspect ratio (1920x1080 -> 512x288); JPEG q70 via BytesIO; must .convert('RGB') defensively before JPEG save |
| FOUND-05 | System prompt is tuned for charades domain: person acting/miming, not scene description | Anti-hallucination prompt pattern from PITFALLS.md: "MIMING, no real objects"; charades-specific per D-07 |
| LANG-01 | Agent outputs guesses as phrases (1-5 words), not just single words | Prompt instruction per D-09; strip quotes/whitespace per D-18 |
| LANG-02 | System prompt includes Filipino cultural context (Pinoy Henyo categories) | Five categories: Bagay/Tao/Lugar/Pangyayari/Pagkain per D-08; reference list per D-10 |
| LANG-03 | Agent can guess in either English or Tagalog based on what fits the action | Prompt instruction per D-08; Gemini 2.5 Flash supports Tagalog output (verified via model capabilities) |
| LANG-04 | Common Filipino charades references included in prompt context | Reference list per D-10: food, transport, cultural, places categories |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Single editable file:** All logic must live in `agent/src/agent/prompt.py`
- **Fixed interface:** `analyze(frame: Frame) -> str | None` -- return guess string or None to skip
- **Frame contract:** `frame.image` is `PIL.Image.Image` (RGB), `frame.timestamp` is `datetime` (UTC)
- **Do not edit:** `__main__.py`, `core/`, `api/` packages
- **Run commands:** `uv run -m agent --practice` (local), `uv run -m agent --live` (competition)
- **Max 10 guesses per round** -- MaxGuessesReached exception on 429
- **Async:** `analyze()` is `async def` -- can use `await` for LLM calls

## Standard Stack

### Core (Already Installed -- No Changes Needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic-ai | 1.73.0 | Agent framework with BinaryContent for vision | Already installed; unified API across providers; async-native |
| Pillow | 12.1.1 | Image preprocessing (thumbnail, JPEG compress) | Already installed via core; no new deps needed |
| numpy | 2.4.3 | Available as transitive dep (not needed for Phase 1) | Available but unused in Phase 1 |
| google-genai | 1.68.0 | Gemini SDK (used internally by pydantic-ai) | Transitive dependency; pydantic-ai wraps it |

### Environment Variables

| Variable | Value | Required By |
|----------|-------|-------------|
| `GEMINI_API_KEY` | Google AI Studio API key | pydantic-ai GoogleGLAProvider -- **CRITICAL: this is the actual env var name** |
| `API_URL` | Dashboard base URL | api package (live mode) |
| `TEAM_TOKEN` | Team authentication key | api package (live mode) |

**IMPORTANT ENV VAR FINDING:** The `.env.example` template uses `LLM_API_KEY`, but pydantic-ai's GoogleGLAProvider reads `GEMINI_API_KEY`. The user's D-03 says "Set LLM_API_KEY in .env" but this will NOT work with pydantic-ai's auto-detection. The `.env` file must set `GEMINI_API_KEY=<key>` for the agent to authenticate with Gemini. Alternatively, the Agent can be created with an explicit provider: `Agent('google-gla:gemini-2.5-flash', provider=GoogleGLAProvider(api_key=os.getenv('LLM_API_KEY')))`.

**Confidence:** HIGH -- verified by reading pydantic-ai v1.73.0 GoogleGLAProvider source code: `api_key = api_key or os.getenv('GEMINI_API_KEY')`.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pydantic-ai BinaryContent | Direct google-genai SDK | Would lose unified API; pydantic-ai already wraps it |
| JPEG q70 | PNG | PNG is 5-10x larger payload; no quality advantage for video frames |
| Image.thumbnail() | Image.resize() | resize() distorts aspect ratio; thumbnail() preserves it |

## Architecture Patterns

### Recommended prompt.py Structure

```
agent/src/agent/prompt.py
    |
    +-- SYSTEM_PROMPT (str constant)
    |
    +-- _state (module-level dict)
    |     frame_count: int
    |     guess_history: list[str]
    |     last_guess: str | None
    |
    +-- _agent (module-level Agent instance)
    |
    +-- _preprocess_frame(image) -> BinaryContent  (private helper)
    |
    +-- analyze(frame) -> str | None  (public entry point)
```

### Pattern 1: Module-Level Agent Instantiation

**What:** Create the pydantic-ai Agent at module level, not inside `analyze()`.
**When to use:** Always. Agent creation has overhead (parsing system prompt, initializing provider).
**Why:** Pitfall 12 from PITFALLS.md -- instantiating per-frame adds cold start penalty.

```python
# Module level -- created once when prompt.py is imported
from pydantic_ai import Agent

SYSTEM_PROMPT = "..."  # charades-specific prompt

_agent = Agent("google-gla:gemini-2.5-flash", system_prompt=SYSTEM_PROMPT)
```

**Confidence:** HIGH -- verified pattern from CLAUDE.md example and PITFALLS.md Pitfall 12.

### Pattern 2: PIL Image to BinaryContent Conversion

**What:** Convert Frame.image to JPEG bytes wrapped in BinaryContent for pydantic-ai.
**When to use:** Every frame before sending to the LLM.

```python
import io
from pydantic_ai import BinaryContent
from PIL import Image

def _preprocess_frame(image: Image.Image) -> BinaryContent:
    """Resize and compress a frame for LLM input."""
    # thumbnail() preserves aspect ratio; modifies image IN PLACE
    img = image.copy()  # don't mutate the original
    img.thumbnail((512, 512), Image.LANCZOS)
    img = img.convert("RGB")  # defensive: JPEG requires RGB mode
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return BinaryContent(data=buf.getvalue(), media_type="image/jpeg")
```

**CRITICAL NOTE on thumbnail():** `Image.thumbnail()` modifies the image **in place** and preserves aspect ratio. A 1920x1080 image becomes 512x288, NOT 512x512. This is correct behavior -- no distortion. The decision D-04 says "512x512" but thumbnail will produce the largest size that fits within that box while preserving aspect ratio.

**Confidence:** HIGH -- verified with Pillow 12.1.1: `Image.new('RGB', (1920, 1080)).thumbnail((512, 512))` produces (512, 288).

### Pattern 3: Sending Image + Text to Agent.run()

**What:** Pass a list of [text_prompt, BinaryContent] to Agent.run() as user_prompt.
**When to use:** Every LLM call.

```python
result = await _agent.run([
    "What is this person acting out?",
    binary_content,  # BinaryContent from _preprocess_frame
])
answer = result.output.strip()
```

**Confidence:** HIGH -- Agent.run() accepts `str | Sequence[UserContent]` where UserContent includes BinaryContent. Verified via source inspection.

### Pattern 4: SKIP Detection

**What:** Return None from analyze() when the LLM says SKIP.

```python
answer = result.output.strip().strip('"').strip("'")
if answer.upper() == "SKIP":
    return None
return answer
```

**Confidence:** HIGH -- straightforward string comparison per D-12.

### Pattern 5: Module-Level State Dict

**What:** Track frame_count, guess_history, and last_guess across calls.

```python
_state: dict = {
    "frame_count": 0,
    "guess_history": [],
    "last_guess": None,
}
```

**Confidence:** HIGH -- Python modules are singletons; the orchestrator imports once and calls repeatedly. No concurrency (sequential per-frame processing).

### Anti-Patterns to Avoid

- **Creating Agent inside analyze():** Cold start penalty on every frame (Pitfall 12). Agent must be module-level.
- **Storing raw PIL Images in state:** Each 1080p image is ~6MB. Store text summaries instead (Pitfall 13). Phase 1 does not need image storage -- just frame_count and guess_history.
- **Not copying image before thumbnail():** `thumbnail()` modifies in place. Must call `image.copy()` first to avoid mutating the Frame's image.
- **Forgetting .convert('RGB'):** JPEG save raises an error on RGBA images. Defensive conversion prevents crashes.
- **Using Image.resize() instead of Image.thumbnail():** resize() distorts aspect ratio. thumbnail() preserves it.
- **Sending PNG instead of JPEG:** 5-10x larger payload, directly impacting API latency.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image-to-LLM pipeline | Custom base64 encoding + raw HTTP calls | pydantic-ai BinaryContent + Agent.run() | Handles provider differences (Gemini vs Claude format), retries, async |
| JPEG compression | Manual pixel manipulation | PIL Image.save(format="JPEG", quality=70) | Battle-tested, handles color profiles, exif stripping |
| Async LLM calls | Raw httpx requests to Gemini API | pydantic-ai Agent (wraps google-genai SDK) | Connection pooling, auth, error handling, structured output |
| Environment variable loading | Manual os.getenv + validation | python-dotenv + pydantic-ai auto-detection | dotenv already loaded by __main__.py; pydantic-ai reads GEMINI_API_KEY automatically |

## Common Pitfalls

### Pitfall 1: Wrong Environment Variable Name
**What goes wrong:** Setting `LLM_API_KEY` or `GOOGLE_API_KEY` in .env but pydantic-ai expects `GEMINI_API_KEY`.
**Why it happens:** The starter template's `.env.example` uses `LLM_API_KEY`. The pydantic-ai GoogleGLAProvider auto-detects from `GEMINI_API_KEY` specifically.
**How to avoid:** Either (a) set `GEMINI_API_KEY` in .env, or (b) explicitly pass the API key when creating the provider. Option (a) is simpler.
**Warning signs:** `UserError: Set the GEMINI_API_KEY environment variable` on first run.

### Pitfall 2: thumbnail() Modifies Image In Place
**What goes wrong:** Calling `frame.image.thumbnail((512, 512))` mutates the Frame's PIL Image object. Since Frame is a frozen dataclass, the reference is immutable, but the Image object itself is mutable. Subsequent code that reads `frame.image` gets the resized version.
**Why it happens:** PIL thumbnail() is an in-place operation, unlike resize() which returns a new image.
**How to avoid:** Always `image.copy()` before calling thumbnail().
**Warning signs:** Image dimensions shrink on every frame; frame logging shows 512xN on second frame onwards.

### Pitfall 3: JPEG Save on Non-RGB Mode
**What goes wrong:** `Image.save(format="JPEG")` raises `OSError: cannot write mode RGBA as JPEG` if the image has an alpha channel.
**Why it happens:** Frame.image is documented as RGB, but defensive coding prevents surprises.
**How to avoid:** Call `.convert("RGB")` before JPEG save.
**Warning signs:** OSError crash on first frame.

### Pitfall 4: Single-Frame Blindness (Phase 1 Acknowledged Limitation)
**What goes wrong:** Charades is temporal -- a single frame is ambiguous. "Arms raised" could be surrender, airplane, tree, or dozens of other concepts.
**Why it happens:** Phase 1 is deliberately single-frame (D-11: no confidence system yet). This is the known tradeoff for getting a working agent fast.
**How to avoid:** Phase 2 adds multi-frame reasoning. For Phase 1, the prompt should instruct: "Consider the FULL BODY pose and ALL visible gestures" to maximize single-frame information extraction.
**Warning signs:** Wildly different guesses on consecutive similar frames.

### Pitfall 5: Hallucinating Real Objects in Charades
**What goes wrong:** LLM describes "person holding a guitar" when they are MIMING guitar-playing with empty hands.
**Why it happens:** LLM language priors fill in objects that should not exist in a charades context.
**How to avoid:** System prompt must explicitly state: "The person is MIMING/acting -- they do NOT have real objects. Describe GESTURES and BODY POSITIONS."
**Warning signs:** Agent outputs specific object descriptions ("red ball", "wooden guitar") for a person with empty hands.

### Pitfall 6: Over-Engineering Before Working Pipeline
**What goes wrong:** Building confidence systems, multi-model ensembles, and keyframe extraction before the basic pipeline works.
**Why it happens:** Competitive instinct. The hackathon rewards working solutions.
**How to avoid:** Get FOUND-01 working first (single frame -> LLM -> guess). Then add preprocessing (FOUND-04). Then add prompt tuning (FOUND-05, LANG-*). Commit at each step.
**Warning signs:** Complex code that has never been tested in practice mode.

## Code Examples

### Complete Phase 1 analyze() Pattern

```python
# Source: Verified against pydantic-ai v1.73.0 API and CLAUDE.md patterns
import io
from pydantic_ai import Agent, BinaryContent
from PIL import Image
from core import Frame

SYSTEM_PROMPT = """\
You are watching a charades game. A person is acting out or miming a concept.
Focus on their gestures, body language, and movements -- NOT the background.
The person is MIMING and does NOT have real objects.

This game is in the Philippines. Answers may be in English or Tagalog.
Consider Filipino culture and Pinoy Henyo categories:
- Bagay (Things): everyday objects, tools, instruments
- Tao (People): celebrities, occupations, historical figures
- Lugar (Places): landmarks, countries, local spots
- Pangyayari (Events): holidays, activities, sports
- Pagkain (Food): adobo, sinigang, halo-halo, balut, taho, lechon

Common Filipino references: jeepney, tricycle, tinikling, karaoke,
Boracay, Intramuros, mano po, fiesta.

Rules:
- Give your best guess as a short phrase (1-5 words).
- If you are not confident enough, respond with exactly SKIP.
- Be specific: "riding a jeepney" is better than "transportation".
"""

_agent = Agent("google-gla:gemini-2.5-flash", system_prompt=SYSTEM_PROMPT)

_state: dict = {
    "frame_count": 0,
    "guess_history": [],
    "last_guess": None,
}


def _preprocess_frame(image: Image.Image) -> BinaryContent:
    img = image.copy()
    img.thumbnail((512, 512), Image.LANCZOS)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return BinaryContent(data=buf.getvalue(), media_type="image/jpeg")


async def analyze(frame: Frame) -> str | None:
    _state["frame_count"] += 1

    binary = _preprocess_frame(frame.image)
    result = await _agent.run([
        "What is this person acting out? Give your best guess.",
        binary,
    ])

    answer = result.output.strip().strip('"').strip("'")

    if answer.upper() == "SKIP":
        return None

    _state["last_guess"] = answer
    _state["guess_history"].append(answer)
    return answer
```

### Environment Setup

```bash
# In .env file -- MUST use GEMINI_API_KEY (not LLM_API_KEY)
GEMINI_API_KEY=your-google-ai-studio-key
API_URL=https://your-dashboard.example.com
TEAM_TOKEN=your-team-api-key
```

### Testing Workflow

```bash
# Practice mode -- local camera, no network
uv run -m agent --practice

# Practice mode with higher FPS
uv run -m agent --practice --fps 2

# Live mode -- connects to game server
uv run -m agent --live
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generic "what do you see?" prompt | Charades-specific "person is miming" prompt | Domain research 2026 | Reduces hallucinated objects, focuses on gestures |
| Single-language output | Bilingual English/Tagalog prompt | Competition requirement | Matches Manila competition context |
| Full-resolution image upload | 512px thumbnail + JPEG q70 | Best practice for vision APIs | 5-10x payload reduction, faster API calls |
| Agent created per-call | Module-level Agent instance | pydantic-ai pattern | Eliminates cold start penalty per frame |

## Open Questions

1. **LLM_API_KEY vs GEMINI_API_KEY discrepancy**
   - What we know: pydantic-ai v1.73.0 GoogleGLAProvider reads `GEMINI_API_KEY` specifically. The starter template uses `LLM_API_KEY`.
   - What's unclear: Whether the user has already configured their .env with GEMINI_API_KEY or is following the template exactly.
   - Recommendation: The plan should include a step to either set `GEMINI_API_KEY` in .env, or explicitly pass the API key via `GoogleGLAProvider(api_key=os.getenv('LLM_API_KEY'))`. The former is simpler.

2. **thumbnail() produces non-square images**
   - What we know: D-04 says "512x512" but `thumbnail((512, 512))` preserves aspect ratio, producing 512x288 from 16:9 input.
   - What's unclear: Whether the user intended square output or just "fits within 512px".
   - Recommendation: Use thumbnail() as-is. Aspect-ratio preservation is correct for vision LLMs. Non-square images work fine with Gemini. No need to force square.

3. **Gemini 2.5 Flash thinking_budget**
   - What we know: STACK.md mentions `thinking_budget=0` to disable reasoning for lower latency. Phase 1 does not mention this.
   - What's unclear: Whether thinking should be disabled for Phase 1 (speed) or enabled (potentially better reasoning).
   - Recommendation: Leave as default for Phase 1. Thinking budget tuning is an optimization for Phase 2+. Simpler is better for the first working version.

4. **Practice mode has no charades actor**
   - What we know: Practice mode uses local camera. There is no person acting out charades during development.
   - What's unclear: How to meaningfully test the charades-specific prompt without a charades actor.
   - Recommendation: Point camera at objects/people/actions to verify the pipeline works end-to-end. The bilingual prompt tuning is best tested during live rounds.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All code | Needs verification (system Python is 3.9.6) | uv manages Python | uv will use a compatible Python version |
| uv | Package management, running | Yes | 0.10.11 | -- |
| pydantic-ai | Agent framework | Yes | 1.73.0 | -- |
| Pillow | Image processing | Yes | 12.1.1 | -- |
| numpy | Frame differencing (Phase 2) | Yes | 2.4.3 | Not needed for Phase 1 |
| google-genai | Gemini SDK (via pydantic-ai) | Yes | 1.68.0 | -- |
| FFmpeg | Practice mode camera capture | Not found | -- | imageio-ffmpeg fallback |
| GEMINI_API_KEY | Gemini API authentication | Needs user to set | -- | Pass api_key explicitly |

**Missing dependencies with no fallback:**
- `GEMINI_API_KEY` environment variable must be set in .env (or LLM_API_KEY with explicit provider wiring)

**Missing dependencies with fallback:**
- FFmpeg not detected on PATH, but imageio[ffmpeg] (installed) provides a bundled fallback for practice mode
- System Python is 3.9.6, but uv manages its own Python installation that meets >=3.10 requirement

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | No automated test framework installed (no pytest) |
| Config file | None |
| Quick run command | `uv run -m agent --practice` (manual visual verification) |
| Full suite command | `uv run -m agent --practice --fps 2` (higher frame rate test) |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | Frame sent to Gemini, text guess received | manual (practice mode) | `uv run -m agent --practice` | N/A (manual) |
| FOUND-02 | State persists across analyze() calls | manual (check print output) | `uv run -m agent --practice` | N/A (manual) |
| FOUND-03 | Returns None on SKIP | manual (watch for [skip] output) | `uv run -m agent --practice` | N/A (manual) |
| FOUND-04 | Images resized before LLM call | manual (add print of image size) | `uv run -m agent --practice` | N/A (manual) |
| FOUND-05 | Charades-specific prompt in use | code review | N/A | N/A |
| LANG-01 | Guesses are phrases (1-5 words) | manual (watch output) | `uv run -m agent --practice` | N/A (manual) |
| LANG-02 | Filipino context in prompt | code review | N/A | N/A |
| LANG-03 | Tagalog guesses possible | manual (present Filipino items to camera) | `uv run -m agent --practice` | N/A (manual) |
| LANG-04 | Reference list in prompt | code review | N/A | N/A |

### Sampling Rate
- **Per task commit:** `uv run -m agent --practice` -- verify output appears, no crashes
- **Per wave merge:** Run practice mode for 30+ seconds, verify mix of guesses and skips
- **Phase gate:** Successful practice mode run with visible guesses; code review of SYSTEM_PROMPT for all LANG-* requirements

### Wave 0 Gaps
- No automated test infrastructure exists (no pytest, no test directory, no test files)
- For a hackathon project with a single editable file, manual practice mode testing is the pragmatic approach
- Automated tests are not recommended for Phase 1 -- they would test against a live LLM API (expensive, slow, non-deterministic)

## Sources

### Primary (HIGH confidence)
- pydantic-ai v1.73.0 source code -- BinaryContent API, GoogleGLAProvider env var (GEMINI_API_KEY), Agent.run() signature
- Pillow 12.1.1 -- thumbnail() in-place behavior, JPEG save requirements (RGB mode)
- `agent/src/agent/__main__.py` -- orchestrator calling convention, live mode flow
- `core/src/core/frame.py` -- Frame dataclass (image: PIL.Image.Image RGB, timestamp: datetime)

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` -- Gemini 2.5 Flash latency benchmarks (0.53s TTFT, 215 tok/s)
- `.planning/research/PITFALLS.md` -- Hallucination, single-frame blindness, cold start patterns
- `.planning/research/FEATURES.md` -- Pinoy Henyo categories, cultural context requirements

### Tertiary (LOW confidence)
- Gemini 2.5 Flash Tagalog capability -- assumed based on multilingual training data; not formally tested for charades-specific Tagalog output

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified installed with exact versions via uv run
- Architecture: HIGH -- single-file pattern verified against __main__.py calling convention and pydantic-ai API
- Pitfalls: HIGH -- env var issue verified by reading source code; thumbnail behavior verified empirically
- Bilingual prompt: MEDIUM -- Gemini supports Tagalog but quality for charades-specific cultural terms is untested

**Research date:** 2026-03-28
**Valid until:** 2026-04-28 (stable -- pydantic-ai API unlikely to change within 30 days)
