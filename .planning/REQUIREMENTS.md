# Requirements: Casper Charades Agent

**Defined:** 2026-03-28
**Core Value:** Correctly identify what's being acted out with minimal guesses and maximum speed

## v1 Requirements

Requirements for competition-ready agent. Each maps to roadmap phases.

### Foundation

- [ ] **FOUND-01**: Agent sends frames to a vision LLM and receives text guesses (single-frame baseline)
- [ ] **FOUND-02**: Agent maintains module-level state across analyze() calls (frame history, guess history, candidates)
- [ ] **FOUND-03**: Agent returns None (SKIP) when confidence is below threshold, preserving limited guesses
- [ ] **FOUND-04**: Frames are preprocessed (resized to 512px, JPEG compressed) before LLM calls to reduce latency
- [ ] **FOUND-05**: System prompt is tuned for charades domain: person acting/miming, not scene description

### Bilingual Intelligence

- [ ] **LANG-01**: Agent outputs guesses as phrases (1-5 words), not just single words
- [ ] **LANG-02**: System prompt includes Filipino cultural context (Pinoy Henyo categories: Bagay/Tao/Lugar/Pangyayari/Pagkain)
- [ ] **LANG-03**: Agent can guess in either English or Tagalog based on what fits the action
- [ ] **LANG-04**: Common Filipino charades references included in prompt context (adobo, jeepney, taho, sinigang, etc.)

### Temporal Reasoning

- [ ] **TEMP-01**: Keyframe extraction detects significant scene changes and skips near-identical frames
- [ ] **TEMP-02**: Agent buffers recent keyframes (3-6 frames) for multi-frame context
- [ ] **TEMP-03**: Multi-frame input sent to LLM (image grid or multi-image) so model reasons about action sequences
- [ ] **TEMP-04**: Frame differencing uses lightweight method (PIL thumbnail + numpy MAD or perceptual hash) at <5ms per frame

### Confidence System

- [ ] **CONF-01**: LLM returns structured output with guess, confidence score (0-100), and reasoning
- [ ] **CONF-02**: Rank-weighted vote accumulation across frames (not raw Bayesian multiplication of uncalibrated scores)
- [ ] **CONF-03**: Candidate list is union-only — candidates are never dropped, only re-ranked
- [ ] **CONF-04**: Adaptive guess threshold: starts high (~0.8), decays with time, rises when guesses scarce

### Feedback Loop

- [ ] **FEED-01**: Wrong guesses (409 responses) tracked in rejected_guesses list
- [ ] **FEED-02**: Rejected guesses injected into subsequent prompts as "NOT these" constraints
- [ ] **FEED-03**: Semantic diversity enforced — next guess must be from a different category than rejected guesses
- [ ] **FEED-04**: Agent infers 409 occurred when analyze() is called again after returning a non-None value

### Multi-Model

- [ ] **MULTI-01**: Parallel inference across 2-3 models via asyncio.gather (Gemini Flash primary, Claude secondary)
- [ ] **MULTI-02**: Model agreement used as confidence signal (agreement = boost, disagreement = wait)
- [ ] **MULTI-03**: Different models assigned different roles (fast screener vs deep reasoner) not majority voting
- [ ] **MULTI-04**: Total latency equals slowest model, not sum of all models

### Resilience

- [ ] **RESIL-01**: Graceful handling of API rate limits and timeouts without crashing
- [ ] **RESIL-02**: Fallback to single model if secondary models fail
- [ ] **RESIL-03**: Frame processing continues even if one LLM call fails

## v2 Requirements

Deferred — only if time allows after core agent is competition-ready.

- **OPT-01**: Dynamic threshold tuning based on empirical testing with practice mode
- **OPT-02**: Guess timing optimization using estimated scoring formula
- **OPT-03**: Answer specificity calibration (test if judge prefers "golden retriever" vs "dog")
- **OPT-04**: Frame buffer size optimization (3 vs 5 vs 8 frames empirically tested)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom ML model training | Hackathon timeframe, pre-trained APIs are sufficient |
| Local pose estimation (MediaPipe) | Vision LLMs handle poses natively, adds complexity |
| Local video transformers | Requires GPU infrastructure, impractical for demo |
| Audio processing | No audio channel available |
| OCR pipeline | Person is miming, not showing text |
| Optical flow | 1 FPS too low, LLM multi-frame reasoning is better |
| UI/Dashboard | CLI-only, time better spent on accuracy |
| Logprobs-based confidence | Claude doesn't support it, verbalized confidence works |
| RL/online learning | No training loop during competition |
| Bayesian multiplication of raw LLM scores | Research shows this amplifies errors from uncalibrated inputs |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| FOUND-04 | Phase 1 | Pending |
| FOUND-05 | Phase 1 | Pending |
| LANG-01 | Phase 1 | Pending |
| LANG-02 | Phase 1 | Pending |
| LANG-03 | Phase 1 | Pending |
| LANG-04 | Phase 1 | Pending |
| TEMP-01 | Phase 2 | Pending |
| TEMP-02 | Phase 2 | Pending |
| TEMP-03 | Phase 2 | Pending |
| TEMP-04 | Phase 2 | Pending |
| CONF-01 | Phase 2 | Pending |
| CONF-02 | Phase 3 | Pending |
| CONF-03 | Phase 3 | Pending |
| CONF-04 | Phase 3 | Pending |
| FEED-01 | Phase 2 | Pending |
| FEED-02 | Phase 2 | Pending |
| FEED-03 | Phase 3 | Pending |
| FEED-04 | Phase 2 | Pending |
| MULTI-01 | Phase 4 | Pending |
| MULTI-02 | Phase 4 | Pending |
| MULTI-03 | Phase 4 | Pending |
| MULTI-04 | Phase 4 | Pending |
| RESIL-01 | Phase 4 | Pending |
| RESIL-02 | Phase 4 | Pending |
| RESIL-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0

---
*Requirements defined: 2026-03-28*
*Last updated: 2026-03-28 after research completion*
