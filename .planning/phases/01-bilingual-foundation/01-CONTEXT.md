# Phase 1: Bilingual Foundation - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a working charades agent that: sends preprocessed frames to Gemini 2.5 Flash, outputs bilingual guesses (English or Tagalog phrases), skips when uncertain, and maintains state across frames. This is the minimum viable competitor — if time runs out here, this version competes.

</domain>

<decisions>
## Implementation Decisions

### Vision Model
- **D-01:** Use Gemini 2.5 Flash as the primary (and only, for Phase 1) model — 0.53s TTFT, 215 tok/s, cheapest per frame. Research validated this as the clear speed winner.
- **D-02:** Use pydantic-ai with BinaryContent to send frames. Pydantic-ai v1.73+ is already installed and supports Gemini natively.
- **D-03:** Set GEMINI_API_KEY in .env to a Google Gemini API key (pydantic-ai reads GEMINI_API_KEY, not LLM_API_KEY). Model string: `google-gla:gemini-2.5-flash`

### Frame Preprocessing
- **D-04:** Resize frames to 512x512 using PIL `Image.thumbnail()` with LANCZOS resampling before sending
- **D-05:** Convert to JPEG bytes (quality 70) for LLM input — reduces payload without losing gesture detail
- **D-06:** No additional preprocessing (no cropping, no color adjustment, no edge detection)

### System Prompt Design
- **D-07:** Charades-specific prompt: "A person is acting out / miming a concept. Focus on gestures and body language, not background."
- **D-08:** Bilingual instruction: "Answers may be in English or Tagalog. Consider Filipino culture, Pinoy Henyo categories (Bagay/Tao/Lugar/Pangyayari/Pagkain), local food, celebrities, landmarks."
- **D-09:** Output format: "Respond with your best guess as a short phrase (1-5 words). If you're not confident, respond with exactly SKIP."
- **D-10:** Include a small reference list of common Filipino charades categories: food (adobo, sinigang, halo-halo, balut, taho), transport (jeepney, tricycle), cultural (tinikling, karaoke), places (Boracay, Intramuros)

### Confidence Gating (SKIP Logic)
- **D-11:** For Phase 1, rely on the LLM's natural SKIP behavior via prompt instruction — no separate confidence scoring yet (that's Phase 2)
- **D-12:** The analyze() function returns None when the LLM outputs "SKIP" (case-insensitive)

### State Management
- **D-13:** Use a module-level dict or class instance in prompt.py for state persistence across analyze() calls
- **D-14:** Track: frame_count (int), guess_history (list of submitted guesses), last_guess (str or None)
- **D-15:** State resets are not needed — each round is a fresh process invocation

### Guess Format
- **D-16:** Guesses are phrases (1-5 words), not single words. "Riding a jeepney" is valid.
- **D-17:** The LLM should output the guess directly, not wrapped in JSON or structured format (Phase 2 adds structured output)
- **D-18:** Strip whitespace and quotes from LLM output before submitting

### Claude's Discretion
- Exact wording of the system prompt (within the constraints above)
- PIL image mode conversion details (RGB handling)
- Error message formatting in practice mode output

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Agent Interface
- `agent/src/agent/prompt.py` — The ONLY file to edit. Contains SYSTEM_PROMPT and analyze() function
- `agent/src/agent/__main__.py` — CLI entry point, practice/live mode loop (DO NOT EDIT, but understand the calling convention)
- `core/src/core/frame.py` — Frame dataclass definition (image: PIL.Image.Image, timestamp: datetime)

### API Contract
- `api/src/api/client.py` — CasperAPI.guess(answer: str) -> GuessResult
- `api/src/api/models.py` — Feed, GuessResult, exception classes

### Research
- `.planning/research/STACK.md` — Vision model comparison, latency benchmarks, pydantic-ai patterns
- `.planning/research/FEATURES.md` — Feature landscape, table stakes, anti-features
- `.planning/research/ARCHITECTURE.md` — Pipeline design, component boundaries
- `.planning/research/PITFALLS.md` — Known failure modes and prevention strategies

### Codebase Maps
- `.planning/codebase/STACK.md` — Current technology stack
- `.planning/codebase/CONVENTIONS.md` — Code style, naming patterns, error handling patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core.Frame` dataclass with `.image` (PIL Image RGB) and `.timestamp` (UTC datetime)
- `core.start_practice()` and `core.start_stream()` async generators yield Frame objects
- pydantic-ai v1.73+ already installed — supports Agent(), BinaryContent, structured output

### Established Patterns
- Async-first: all main functions use `async def`
- Module-level constants use UPPERCASE_WITH_UNDERSCORES
- Private functions prefixed with underscore
- Error handling uses custom exceptions (NoActiveRound, Unauthorized, etc.)
- Type hints use modern `str | None` syntax

### Integration Points
- `analyze(frame: Frame) -> str | None` is called by __main__.py in a loop
- Return str = submit guess, return None = skip frame
- Module-level variables persist across calls within a single process run
- In practice mode: no API calls, just print output
- In live mode: returned string is sent to POST /api/guess as plain text

</code_context>

<specifics>
## Specific Ideas

- User emphasized: answers can be phrases AND can be in Tagalog — this is the "hard criteria"
- Pinoy Henyo categories are the reference framework for Filipino charades
- The person does multiple sequential actions per clue (e.g., pointy head + swimming = shark)
- Phase 1 is single-frame only — multi-frame reasoning comes in Phase 2
- User wants research-backed decisions — all model choices validated by latency benchmarks

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-bilingual-foundation*
*Context gathered: 2026-03-28*
