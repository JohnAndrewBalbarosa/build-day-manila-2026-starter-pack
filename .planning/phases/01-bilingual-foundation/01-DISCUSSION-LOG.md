# Phase 1: Bilingual Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 01-bilingual-foundation
**Areas discussed:** Vision Model, Frame Preprocessing, Prompt Design, Confidence Gating, State Management, Guess Format
**Mode:** Autonomous — decisions derived from research findings and user conversation

---

## Vision Model

| Option | Description | Selected |
|--------|-------------|----------|
| Gemini 2.5 Flash | 0.53s TTFT, 215 tok/s, 62x cheaper multi-frame than Claude | ✓ |
| Claude Sonnet 4.6 | Better accuracy, 2x slower, 10x more expensive | |
| GPT-4o | Strong vision, moderate latency | |

**Decision:** Gemini 2.5 Flash — research validated as clear speed winner for Phase 1 primary model.
**Notes:** Claude and GPT reserved for Phase 4 multi-model strategy.

---

## Frame Preprocessing

| Option | Description | Selected |
|--------|-------------|----------|
| 512x512 JPEG q70 | Good balance of quality and speed, 40-60% latency reduction | ✓ |
| 768x768 JPEG q80 | Higher quality, more tokens | |
| 384x384 JPEG q60 | Fastest, may lose gesture detail | |

**Decision:** 512x512 with JPEG quality 70 — research recommended this as sweet spot.
**Notes:** PIL thumbnail with LANCZOS resampling. No additional preprocessing needed.

---

## Prompt Design

**Decision:** Charades-specific bilingual prompt with Pinoy Henyo categories.
**Notes:** User emphasized phrases (not single words) and Tagalog as hard criteria. Filipino cultural references are a competitive advantage in a Manila competition.

---

## Confidence Gating

| Option | Description | Selected |
|--------|-------------|----------|
| LLM SKIP instruction | Simple prompt-based, rely on model judgment | ✓ |
| Structured confidence score | Numerical threshold, more precise | |
| Always guess | Never skip, use all 10 guesses | |

**Decision:** Prompt-based SKIP for Phase 1 — structured confidence scoring deferred to Phase 2.
**Notes:** Research shows LLM self-reported confidence is overconfident but rank-order is preserved.

---

## State Management

**Decision:** Module-level state dict tracking frame_count, guess_history, last_guess.
**Notes:** Python module singleton pattern guarantees persistence across analyze() calls.

---

## Guess Format

**Decision:** Phrases (1-5 words), bilingual (EN/TL), stripped of whitespace/quotes.
**Notes:** User explicitly stated "guesses are not only one words, it can pertain to phrases and it can be english or tagalog."

## Claude's Discretion

- Exact system prompt wording
- PIL image mode conversion details
- Error message formatting
