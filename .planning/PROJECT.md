# Casper Charades Agent

## What This Is

An AI-powered charades guessing agent that watches a livestream of a person acting out clues and submits guesses via the Casper Studios game API. Built on an existing starter template (core/api packages are fixed), with all competitive logic living in `agent/src/agent/prompt.py`. This is a hackathon/demo build — must work flawlessly under pressure.

## Core Value

**Correctly identify what's being acted out with minimal guesses and maximum speed.** Scoring rewards both fewer attempts (max 10) and faster time-to-correct-answer. Every wasted guess and every wasted second costs points.

## Requirements

### Validated

- Frame capture from local camera (practice mode) — existing
- Frame capture from LiveKit stream (live mode) — existing
- HTTP client for guess submission (POST /api/guess) — existing
- CLI with --practice and --live modes — existing
- Exponential backoff retry on judge unavailable (503) — existing

### Active

- [ ] Vision LLM integration — send frames to a vision model and receive guesses
- [ ] Multi-frame temporal reasoning — accumulate evidence across frames, not single-frame guessing
- [ ] Keyframe extraction — detect significant scene changes, skip near-identical frames
- [ ] Confidence-scored candidate tracking — maintain ranked list of guesses with confidence scores
- [ ] Bayesian-style confidence accumulation — merge confidence across frames mathematically (union, never drop)
- [ ] Adaptive guess threshold — submit guess when confidence exceeds optimal threshold given scoring tradeoff
- [ ] Negative evidence feedback loop — feed wrong guesses (409 responses) back as "NOT this" constraints
- [ ] Multi-model strategy — parallel or cascading calls across Gemini Flash (speed), Claude/GPT-4o (accuracy)
- [ ] Bilingual phrase-level guessing — output English or Tagalog phrases, not just single English words
- [ ] Optimized frame preprocessing — resize/compress frames for fastest LLM inference without losing signal

### Out of Scope

- Modifying core/ or api/ packages — locked by starter template rules
- Training custom ML models — hackathon timeframe, use pre-trained/API models only
- Building a UI or dashboard — agent is CLI-only
- Audio processing — video frames only, no audio channel available
- Real-time video model inference (e.g., running a local video transformer) — too complex for demo, use cloud APIs

## Context

**Competition format:**
- One person acts out clues on a livestream (charades-style)
- Multiple sequential actions per clue (e.g., pointy head then swimming = shark)
- **Answers can be phrases, not just single words** (e.g., "riding a jeepney", "kumain ng balut")
- **Answers can be in English OR Tagalog** — bilingual guessing is required
- Categories unknown in advance — could span animals, movies, objects, people, actions, food, places, events
- Maps to Pinoy Henyo categories: Bagay/Tao/Lugar/Pangyayari/Pagkain (Object/Person/Place/Event/Food)
- May include Filipino cultural references, local celebrities, local concepts
- Teams compete on speed and accuracy (fewer guesses + faster = better score)
- Max 10 guesses per round; wrong guesses return HTTP 409
- Judge is likely LLM-based (semantic matching, not exact string) — evidenced by JudgeUnavailable (503) error and retry logic
- Judge likely handles bilingual matching (e.g., "shark" and "pating" for the same clue)

**Technical environment:**
- Python 3.10+ UV workspace, 3 packages (core, api, agent)
- Only `agent/src/agent/prompt.py` is editable (SYSTEM_PROMPT + analyze function)
- Frame object: PIL Image (RGB) + UTC timestamp
- Default 1 FPS from livestream, configurable
- pydantic-ai available as AI agent framework
- All major LLM APIs available: Anthropic (Claude), OpenAI (GPT-4o), Google (Gemini)
- No budget constraints — optimize for speed and accuracy

**Key technical challenges:**
1. Charades is temporal — meaning comes from motion sequences, not single frames
2. LLM confidence scores are not calibrated probabilities — need careful handling
3. Frame similarity detection needed to avoid wasting LLM calls on near-identical frames
4. Optimal guess timing is a mathematical tradeoff between speed bonus and attempt penalty
5. The judge's semantic matching tolerance is unknown — need to test specificity levels

## Constraints

- **Editable surface**: Only `agent/src/agent/prompt.py` — all logic must live there
- **Frame interface**: Must accept Frame (PIL Image + timestamp), return str | None
- **API contract**: Guess submission is plain text string via POST /api/guess
- **Max guesses**: 10 per round (MaxGuessesReached exception on 429)
- **Scoring**: Speed + fewer attempts = better (exact formula unknown)
- **Timeline**: Hackathon — must be demo-ready, not production-hardened
- **No budget limit**: Use best models, parallelize freely

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-frame accumulation over single-frame guessing | Charades is temporal; single frames are ambiguous | -- Pending |
| Bayesian-style confidence merging | Mathematically sound for combining evidence; must validate with research | -- Pending |
| Keyframe extraction before LLM calls | Avoid wasting API calls on similar frames; focus compute on scene changes | -- Pending |
| Multi-model parallel strategy | Different models have different strengths; speed vs accuracy tradeoff | -- Pending |
| LLM-based judge assumption | JudgeUnavailable (503) suggests semantic matching, not string comparison | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-28 after initialization*
