# Project Research Summary

**Project:** Casper Charades Agent
**Domain:** Real-time visual action recognition + competitive guessing game AI
**Researched:** 2026-03-28
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project builds an AI agent that watches a live video stream of someone playing charades and submits guesses through a competition API. The agent operates within a strict single-file constraint (`agent/src/agent/prompt.py`) and must balance speed against accuracy under a 10-guess budget per round. The competition scores both speed (guess faster) and efficiency (fewer wrong guesses), creating an optimal-stopping problem at the core of every design decision.

The recommended approach is a **stateful multi-stage pipeline** running entirely inside `prompt.py`: keyframe gating (skip redundant frames), image preprocessing (resize to 512px JPEG), parallel multi-model inference (Gemini 2.5 Flash for speed, Claude Sonnet 4.6 for accuracy, GPT-4.1 mini as tiebreaker), rank-weighted evidence accumulation across frames, wrong-guess elimination via 409 feedback, and a dynamic threshold guess-decision engine. The single most important architectural insight is that **charades is temporal, not spatial** -- multi-frame reasoning dramatically outperforms single-frame classification, and Gemini's native multi-image API (up to 3600 images per request at 258 tokens/frame) is the key enabler at 62x cheaper than sending equivalent frames to Claude.

The primary risks are: (1) over-engineering and running out of time before having a working agent (the most common hackathon failure mode), (2) single-frame blindness causing rapid guess depletion on wrong answers, (3) vision LLM hallucination where models confidently describe objects that are not physically present in a charades mime, and (4) anchoring bias where early wrong hypotheses dominate subsequent reasoning. Mitigation is straightforward: build incrementally with a working agent at every commit, accumulate 3-5 frames before any guess, use charades-specific anti-hallucination prompts ("they are MIMING, no real objects"), and weight recent frames more heavily than early ones.

## Key Findings

### Recommended Stack

All required dependencies are already installed -- no new packages needed. The stack is pydantic-ai 1.73.0 (unified multi-model agent framework with BinaryContent image API), Pillow 12.1.1 (frame processing), and numpy (transitive, for keyframe differencing). Three vision LLM providers are available through pydantic-ai's existing SDK dependencies.

**Core technologies:**
- **Gemini 2.5 Flash** (`google-gla:gemini-2.5-flash`): Primary speed model -- 0.53s TTFT, 215 tok/s, 258 tokens/image, native multi-frame support. Set `thinking_budget=0` to disable reasoning overhead.
- **Claude Sonnet 4.6** (`claude-sonnet-4-6`): Secondary accuracy model -- best practical vision understanding for interpreting human actions and gestures in context. ~1.15s TTFT.
- **GPT-4.1 mini** (`openai:gpt-4.1-mini`): Tertiary tiebreaker -- 0.83s TTFT, different training data for error diversity. Do NOT use GPT-4o mini (degraded to 4.95s TTFT).
- **pydantic-ai 1.73.0**: Unified async agent framework already installed. `BinaryContent` class handles PIL-to-API conversion for all providers. `asyncio.gather()` enables parallel multi-model calls.
- **Pillow 12.1.1 + numpy**: Frame preprocessing (resize, JPEG compress) and keyframe detection (32x32 grayscale mean absolute difference). No OpenCV needed.

**Do NOT use:** Claude Opus 4.6 (too slow), Gemini 2.5 Pro (overkill), OpenCV (heavy dependency for marginal gain), LangChain/litellm (unnecessary abstraction layers), local ML models (out of scope).

### Expected Features

**Must have (table stakes):**
- Vision LLM integration with PIL-to-BinaryContent conversion
- Frame preprocessing (resize to 512px, JPEG q75-85)
- SKIP logic for low-confidence frames (10-guess budget demands discipline)
- Wrong-guess feedback loop (409 elimination injected into subsequent prompts)
- State persistence across frames via module-level variables
- Charades-specific prompt engineering ("person is MIMING, describe gestures not objects")
- Filipino/Tagalog cultural awareness in system prompt (Pinoy Henyo categories, local references)

**Should have (competitive advantage):**
- Multi-frame temporal reasoning (image grid or Gemini multi-image API) -- the single biggest accuracy differentiator
- Keyframe extraction via pixel differencing (skip identical frames, save API calls)
- Confidence-scored candidate tracking with rank-weighted accumulation across frames
- Multi-model parallel inference via `asyncio.gather()` with `asyncio.as_completed` for speed-first responses
- Adaptive guess threshold (decreasing over time, increasing with guess scarcity)
- Semantic diversity in wrong-guess recovery ("guess something from a COMPLETELY DIFFERENT category")

**Defer (not worth hackathon time):**
- Custom ML model training/fine-tuning
- Local pose estimation (MediaPipe/OpenPose) -- vision LLMs handle this natively
- Optical flow computation (1 FPS is too low for meaningful flow)
- Bayesian confidence updating with raw LLM scores (uncalibrated, amplifies errors)
- Complex OCR pipelines, audio processing, UI/dashboard

### Architecture Approach

The architecture is a 6-component stateful pipeline within a single Python file, connected by module-level state. Each `analyze()` call flows through: Keyframe Gate (skip redundant frames) -> Frame Preprocessor (resize/compress) -> Multi-Model Inference (parallel async LLM calls) -> Evidence Accumulator (rank-weighted voting across frames and models with temporal decay) -> Negative Constraint Filter (409 elimination) -> Guess Decision Engine (dynamic threshold optimal stopping). All state persists via module-level variables since Python modules are singletons and the orchestrator calls `analyze()` sequentially. Wrong-guess feedback is detected pessimistically: if `analyze()` is called again after returning a non-None value, the previous guess was wrong.

**Major components:**
1. **KeyframeGate** -- 32x32 grayscale MAD comparison (~5.6ms). Threshold > 5.0 catches body movement while ignoring camera noise. Enforces 0.5s minimum interval.
2. **FramePreprocessor** -- `thumbnail((512, 512))` + JPEG q85. Stateless. Cuts API latency 3-5x vs raw frames.
3. **MultiModelInference** -- `asyncio.gather()` fires Gemini Flash + Claude Sonnet + GPT-4.1 mini in parallel. Each returns a ranked list of 5 guesses. Use `asyncio.as_completed` for speed-first acting on Gemini's faster response.
4. **EvidenceAccumulator** -- Rank-weighted voting (rank 1 = 5 points, rank 5 = 1 point) with temporal decay (0.8x per frame) and model-agreement multiplier (2/3 agree = 1.5x, 3/3 = 2.0x). NOT Bayesian -- LLM confidence is uncalibrated.
5. **NegativeConstraintFilter** -- Tracks eliminated guesses from 409s. Injects "DO NOT repeat these" into prompts. Permanently zeroes accumulator scores for rejected hypotheses.
6. **GuessDecisionEngine** -- Dynamic threshold: `base * (1 + 0.2 * guesses_used) * max(0.3, 1 - t/120s)`. Special rules: never guess on first 3 frames, unanimous model agreement overrides threshold, desperation mode at 8+ guesses used.

### Critical Pitfalls

1. **Single-Frame Blindness** -- Treating each frame independently turns charades into image classification, which is wrong. Charades is temporal; "pointy head + swimming = shark" requires frame sequences. Prevention: accumulate 3-5 frames before any guess, include prior frame summaries in every LLM call, never guess from a single frame alone. CVPR/NeurIPS 2025 research confirms VLMs "perform close to random chance on temporal benchmarks" with individual frames.

2. **Over-Engineering (Hackathon Killer)** -- Building a sophisticated system that has never been tested end-to-end is the most common hackathon failure mode. Prevention: Phase 0 = working agent in 30 minutes, then incremental improvements with a committed working version at every phase. Budget 15 minutes before the live round for full end-to-end testing.

3. **Vision LLM Hallucination** -- Models confidently describe objects that are not physically present (hallucinating a "guitar" when someone mimes guitar-playing with empty hands). GPT-4o shows +0.23 overconfidence deviation. Prevention: add "they are MIMING with NO real objects -- describe GESTURES and BODY POSITIONS only" to the system prompt. Use Claude for critical guesses (better calibrated at -0.04 deviation).

4. **Anchoring Bias in Evidence Accumulation** -- First hypothesis dominates all subsequent reasoning even as gestures change. Bayesian updating on uncalibrated inputs amplifies early errors. Prevention: weight recent frames more than early frames (temporal decay), maintain a diverse top-5 candidate list, hard-reset confidence on 409 rejection, periodically run "fresh eyes" analysis without prior context.

5. **PIL-to-LLM Image Pipeline Failures** -- Raw 1080p frames waste tokens, latency, and may hit API payload limits. Prevention: resize ALL images to 512px JPEG before encoding. This single optimization cuts API latency 3-5x. Pre-compute BytesIO buffer once per frame.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation -- Working Agent (Target: 30-45 min)
**Rationale:** Get a functioning end-to-end agent as fast as possible. This is the safety net -- every subsequent phase is an improvement on something that already works. Addresses the #1 hackathon failure mode (over-engineering without a working baseline).
**Delivers:** An agent that sees frames, preprocesses them, sends to one LLM (Gemini Flash), returns guesses or SKIPs, and tracks wrong guesses via 409 feedback.
**Addresses:** All 6 table-stakes features from FEATURES.md: vision integration, frame preprocessing, SKIP logic, 409 feedback, state persistence, charades prompt engineering + Filipino cultural awareness.
**Avoids:** Pitfall 5 (over-engineering), Pitfall 3 (image pipeline), Pitfall 12 (cold start from per-frame Agent instantiation).
**Stack:** Gemini 2.5 Flash only, pydantic-ai BinaryContent, Pillow thumbnail + JPEG, module-level state.

### Phase 2: Temporal Intelligence (Target: 45-60 min)
**Rationale:** Multi-frame reasoning is the single biggest accuracy improvement for charades. Without it, the agent is doing image classification, not action recognition. This phase transforms the agent from "reflexive" to "deliberative."
**Delivers:** Keyframe extraction (skip redundant frames), multi-frame context window (send last 3-5 frame descriptions to LLM), confidence-scored candidate tracking with rank-weighted accumulation, and adaptive guess threshold.
**Addresses:** Differentiators from FEATURES.md: keyframe extraction, multi-frame temporal reasoning, confidence tracking, adaptive threshold.
**Avoids:** Pitfall 1 (single-frame blindness), Pitfall 4 (anchoring -- uses temporal decay and diverse candidate lists), Pitfall 7 (uncalibrated probabilities -- uses rank voting, not raw scores).
**Architecture:** Implements KeyframeGate, EvidenceAccumulator, and GuessDecisionEngine components.

### Phase 3: Multi-Model Consensus (Target: 30-45 min)
**Rationale:** Different models have different failure modes. Running them in parallel provides both speed (act on Gemini's fast response) and accuracy (cross-validate with Claude/GPT). Research shows 2/3 model agreement boosts accuracy +33% over single model, though correlated errors mean agreement is weak evidence, not proof.
**Delivers:** Parallel multi-model inference with `asyncio.gather()`, model-agreement multiplier in evidence accumulation, fallback provider resilience against rate limiting or API outages.
**Addresses:** Multi-model parallel strategy and Bayesian-style accumulation (simplified to rank-vote) from FEATURES.md.
**Avoids:** Pitfall 6 (correlated errors -- uses models for different roles, not identical queries), Pitfall 8 (frame queue overflow -- Gemini Flash keeps latency < 1s for the critical path), Pitfall 14 (async bugs -- uses `asyncio.gather`, not `create_task`).
**Stack:** Adds Claude Sonnet 4.6 + GPT-4.1 mini alongside Gemini Flash.

### Phase 4: Competition Hardening (Target: 15-30 min)
**Rationale:** Final polish before the live round. Tune thresholds based on practice mode results, add network resilience, verify end-to-end performance.
**Delivers:** Tuned confidence thresholds, semantic diversity in wrong-guess recovery, rate-limit handling with exponential backoff, network resilience (httpx timeouts, retry logic).
**Addresses:** Adaptive threshold tuning, semantic diversity, and all "minor pitfalls" (memory leaks, network issues, rate limiting).
**Avoids:** Pitfall 11 (rate limiting), Pitfall 15 (network issues), Pitfall 16 (guess timing).

### Phase Ordering Rationale

- **Phase 1 first** because the hackathon constraint demands a working baseline before any optimization. Every research file emphasizes this independently.
- **Phase 2 before Phase 3** because multi-frame reasoning is a bigger accuracy win than multi-model consensus. Research shows temporal understanding is the primary weakness of single-frame VLM analysis, and fixing it with one model is higher ROI than running three models on single frames.
- **Phase 3 before Phase 4** because multi-model adds measurable accuracy and resilience. Phase 4 is pure tuning that requires Phases 1-3 to be working.
- **Each phase produces a deployable agent.** If time runs out at any phase boundary, the last committed version competes.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Temporal Intelligence):** The optimal frame buffer size (3 vs 5 vs 8 frames), keyframe threshold (MAD > 5.0 is a starting point), and guess-decision threshold parameters all need empirical tuning in practice mode. The scoring formula is unknown, so adaptive threshold math is theoretical.
- **Phase 3 (Multi-Model):** Gemini's `thinking_budget=0` parameter behavior with multi-image inputs, and whether pydantic-ai correctly passes this config, should be verified against current docs. Correlated error rates across providers are model-version-specific.

Phases with standard patterns (skip deeper research):
- **Phase 1 (Foundation):** Extremely well-documented. pydantic-ai BinaryContent API, Pillow image processing, and basic prompt engineering are all standard patterns with official docs and examples.
- **Phase 4 (Hardening):** Standard async error handling, httpx timeouts, and exponential backoff are well-established patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All dependencies verified in lockfile (pydantic-ai 1.73.0, Pillow 12.1.1). Model latency benchmarks from Artificial Analysis with recent P50 measurements. Pricing from official docs. |
| Features | HIGH | Table stakes derived directly from competition format (10 guesses, speed scoring, 409 feedback). Differentiators grounded in temporal reasoning research and Gemini multi-image docs. |
| Architecture | MEDIUM-HIGH | Pipeline pattern is sound and benchmarked. Evidence accumulation (rank-vote) is a pragmatic heuristic, not theoretically optimal. Optimal-stopping threshold values are educated guesses needing practice-mode tuning. |
| Pitfalls | HIGH | Top pitfalls backed by CVPR/NeurIPS 2025 temporal research, ICML 2025 correlated errors paper, cognitive science anchoring studies, and verified codebase analysis (frame queue maxsize=2 in stream.py). |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Scoring formula unknown:** The competition scoring function (how speed vs. accuracy are weighted) is not documented. The adaptive guess threshold is based on assumptions. Mitigation: test aggressively in practice mode, start conservative (accumulate evidence), and adjust thresholds based on observed scores.
- **Judge semantic matching tolerance unknown:** Does "shark" match "great white shark"? Does "Jaws" match "shark"? The judge appears to be LLM-based (503/JudgeUnavailable pattern) but its matching behavior cannot be tested beforehand. Mitigation: submit single-concept, common-vocabulary answers; if rejected, try broader/narrower variants.
- **Gemini thinking_budget=0 via pydantic-ai untested:** The STACK research recommends disabling Gemini's thinking mode for lower latency, but whether pydantic-ai 1.73.0 exposes this parameter needs verification. Mitigation: check pydantic-ai model settings docs; if not supported, the default thinking mode still works (just slightly slower).
- **Optimal frame buffer size:** Research suggests 3-5 frames for charades but the exact optimal depends on FPS, gesture speed, and the specific Gemini multi-image performance. Mitigation: start with 3 frames, increase if accuracy is low.
- **Claude vs GPT-4o action understanding quality:** No formal benchmark exists for charades-style gesture recognition with commercial APIs. Model rankings are based on general multimodal benchmarks and community reports. Mitigation: test all three models in practice mode, observe which produces better charades guesses.

## Sources

### Primary (HIGH confidence)
- [pydantic-ai Input (BinaryContent) docs](https://ai.pydantic.dev/input/) -- image handling API
- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding) -- multi-image support, token costs
- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding) -- native video processing
- [Artificial Analysis benchmarks](https://artificialanalysis.ai/models/) -- model latency and throughput
- [Anthropic Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview) -- Claude pricing and capabilities
- [OpenAI Vision docs](https://platform.openai.com/docs/guides/images-vision) -- GPT-4.1 vision API
- Verified lockfile: pydantic-ai 1.73.0, Pillow 12.1.1, openai 2.30.0, anthropic 0.86.0, google-genai 1.68.0
- Verified source code: `core/src/core/stream.py` frame queue maxsize=2

### Secondary (MEDIUM confidence)
- [Correlated Errors in LLMs (ICML 2025)](https://arxiv.org/abs/2506.07962) -- multi-model error correlation
- [Majority Rules: LLM Ensemble](https://arxiv.org/html/2511.15714v1) -- ensemble agreement accuracy
- [LLM Confidence Calibration (ICLR 2024)](https://arxiv.org/abs/2306.13063) -- verbalized confidence overconfidence
- [Enhancing Temporal Understanding in Video-LLMs (NeurIPS 2025)](https://arxiv.org/html/2510.26027v1) -- temporal reasoning weaknesses
- [Chain-of-Frames (2025)](https://arxiv.org/html/2506.00318v1) -- frame-aware reasoning
- [Pinoy Henyo categories](https://www.scribd.com/document/421683483/PInoy-Henyo-Categories) -- Filipino cultural context

### Tertiary (LOW confidence -- needs validation)
- Optimal stopping threshold parameters (theoretical, needs practice-mode calibration)
- Gemini `thinking_budget=0` latency improvement claims (not independently benchmarked for multi-image)
- Claude vs GPT-4o relative accuracy for charades gesture recognition (no formal benchmark exists)

---
*Research completed: 2026-03-28*
*Ready for roadmap: yes*
