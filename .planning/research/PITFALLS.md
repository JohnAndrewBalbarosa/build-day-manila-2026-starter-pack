# Domain Pitfalls

**Domain:** Charades AI guessing agent (vision LLM + real-time competition)
**Researched:** 2026-03-28

---

## Critical Pitfalls

Mistakes that cause blown rounds, wasted guesses, or total system failure during the live competition.

---

### Pitfall 1: Single-Frame Blindness — Treating Charades as Image Classification

**What goes wrong:** The agent treats each frame independently and guesses based on a static snapshot. Charades is fundamentally temporal: "swimming motion" looks like "waving" in a single frame. "Pointy head then swimming" means "shark," but a single frame of the pointy-head part yields "unicorn" or "dunce cap." The agent wastes guesses on plausible-but-wrong single-frame interpretations.

**Why it happens:** Vision LLMs process individual images, not video. Research from CVPR 2025 and NeurIPS 2025 confirms that even state-of-the-art Video-LLMs "perform close to random chance on temporal benchmarks" and "struggle when the query requires precise comprehension of temporal progression." Sending one frame at a time to a vision API gives you image classification, not action recognition.

**Consequences:** Rapid depletion of the 10-guess budget on wrong answers. The agent confidently identifies what it sees in a single frame (a person with arms raised) but misses the action sequence (performing a jumping jack vs. surrendering vs. praising).

**Warning signs:**
- Agent guesses immediately on the first frame with high "confidence"
- Guesses change wildly between consecutive frames (no temporal coherence)
- Agent never says "SKIP" -- it always has an answer for what is in a static image

**Prevention:**
- Accumulate 3-5 frames before making any guess. Build a sliding window of recent frame descriptions.
- In the system prompt, explicitly instruct the LLM: "You are watching a charades game. The person is ACTING OUT a concept through gestures and movement. Describe the ACTIONS and MOVEMENTS you see, not just the static scene."
- Include prior frame summaries in each LLM call: "Previous frames showed: [summary]. Now describe what changed."
- Never guess on a single frame alone.

**Phase:** Must be addressed in Phase 1 (core vision integration). This is the single most important architectural decision.

**Confidence:** HIGH -- backed by CVPR/NeurIPS 2025 temporal reasoning research and the Charades dataset literature from Allen AI.

---

### Pitfall 2: Hallucination in Vision -- Confident Descriptions of Things That Are Not There

**What goes wrong:** Vision LLMs confidently describe objects, actions, or details that do not exist in the image. GPT-4o in particular has been tuned to reduce refusals, which means it hallucinates more: a confidence-score deviation of +0.23 from factual accuracy baseline (i.e., it is systematically overconfident). The model says "person is holding a guitar" when the person is miming guitar-playing with empty hands.

**Why it happens:** LLMs have strong language priors that override visual evidence. The "semantic warping" phenomenon means VLM visual representations are "not a faithful physical map, but a semantic map warped by the discrete token space of the LLM." In charades, where people are pantomiming WITHOUT actual objects, this is devastating: the model's language prior fills in objects that should not be there.

**Consequences:** Agent submits confidently wrong guesses. With 10 max guesses, even 2-3 hallucinated guesses significantly reduce chances of getting the right answer. Worse, hallucinated details may anchor subsequent reasoning (see Pitfall 4).

**Warning signs:**
- Agent describes specific objects ("red ball," "wooden guitar") when the charades player has empty hands
- Agent gives very specific/detailed answers on blurry or ambiguous frames
- Agent rarely or never returns "SKIP"

**Prevention:**
- Add explicit anti-hallucination instructions to the system prompt: "The person is playing charades -- they are MIMING and do NOT have real objects. Describe GESTURES and BODY POSITIONS only, never assume real objects are present."
- Require the model to express uncertainty: "Rate your confidence 1-10. Only guess if confidence >= 7."
- Use Claude over GPT-4o for critical guesses: Claude 3.5 shows a -0.04 confidence-to-truth deviation (much better calibrated) compared to GPT-4o's +0.23.
- Cross-validate: if two models disagree on what is in a frame, treat both answers as uncertain.

**Phase:** Phase 1 (prompt engineering) and Phase 2 (confidence scoring).

**Confidence:** HIGH -- backed by 2025 hallucination benchmarks and calibration studies.

---

### Pitfall 3: PIL-to-LLM Image Pipeline Failures

**What goes wrong:** Converting a PIL Image to a format the LLM API accepts (base64-encoded JPEG/PNG via pydantic-ai's `BinaryContent`) introduces multiple failure points: oversized images consuming excessive tokens, wrong media types, memory bloat from repeated encoding, and base64 overhead increasing payload size by ~37%.

**Why it happens:** The `Frame.image` is a raw PIL Image (RGB) with no guaranteed resolution. Camera feeds can produce 1920x1080 images. At that resolution: GPT-4o consumes ~1105 tokens per image (high detail), Gemini consumes ~1290 tokens, and Claude has similar costs. At 1 FPS over a 60-second round, that is 60+ images processed. Base64 encoding a 1080p PNG can be 2-5MB per request payload.

**Consequences:**
- Slow API calls due to large payloads (latency = death in a speed-scored competition)
- Hitting API rate limits or request size limits (10MB JSON limit for OpenAI)
- Token budget exhaustion if accumulating frame history
- Unnecessary cost (though budget is unconstrained, latency is the real cost)

**Warning signs:**
- API calls taking >2 seconds consistently
- HTTP 413 (payload too large) errors
- Token counts in API responses showing 1000+ input tokens per image

**Prevention:**
- Resize ALL images before encoding. 512px on the longest side is the sweet spot -- research shows "most AI vision models perform equally well on images resized to 512px or smaller."
- Use JPEG encoding (not PNG) for photographs/video frames -- 5-10x smaller file size at acceptable quality (quality=75).
- Use OpenAI's `detail: "low"` parameter (85 tokens fixed) for initial screening frames; only use `detail: "high"` for frames where you need detail.
- Pre-compute the BytesIO buffer once per frame, do not re-encode on retry.

```python
# Correct pattern
import io
def prepare_frame(pil_image, max_size=512):
    # Resize
    pil_image.thumbnail((max_size, max_size))
    # Encode as JPEG
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=75)
    return buf.getvalue()
```

**Phase:** Phase 1 (core vision integration). This must be correct from the start.

**Confidence:** HIGH -- based on official API documentation for OpenAI, Claude, and Gemini vision pricing/token models.

---

### Pitfall 4: Anchoring Bias in Confidence Accumulation

**What goes wrong:** The first guess hypothesis dominates all subsequent reasoning. If the agent initially thinks "elephant" based on frame 1, it interprets all subsequent frames through the "elephant" lens, even when the actor transitions to a different part of the charade. Research shows "a highly confident decision will require more disconfirmatory evidence to be overturned" and metacognitive agents "downweight the processing of disconfirmatory evidence when confident."

**Why it happens:** This is a well-documented cognitive bias that applies directly to Bayesian-style confidence accumulation. If you assign "elephant" a 0.7 prior after frame 1, then even mildly compatible evidence from frame 2 pushes it higher. By frame 5, "elephant" is at 0.95 and the agent guesses -- wrong. The mathematical structure of Bayesian updating amplifies early signals unless the prior is carefully calibrated.

**Consequences:** The agent locks onto an early wrong hypothesis and wastes a guess. After receiving a 409 (wrong), it may not recover fast enough because the same anchoring effect makes it hard to shift to a completely different hypothesis.

**Warning signs:**
- Agent's top candidate does not change across frames even as the actor's gestures change
- Confidence in the top candidate monotonically increases (never dips)
- After a 409 rejection, agent quickly re-guesses something very similar to the rejected answer

**Prevention:**
- Maintain a diverse candidate list (top 3-5), not just the single best. Only collapse when multiple candidates converge.
- Apply a "recency bias" -- weight recent frames MORE than early frames. The actor's final gestures are often the most informative.
- After a 409 rejection, HARD RESET confidence for the rejected concept AND semantically similar concepts. Feed back "NOT elephant, NOT any large animal" as a constraint.
- Cap maximum confidence from any single frame's evidence. No single frame should push any candidate above 0.6.
- Periodically re-evaluate from scratch: every 5 frames, run one "fresh eyes" analysis without prior context.

**Phase:** Phase 2 (confidence accumulation system). This is the most subtle and dangerous pitfall in the Bayesian approach.

**Confidence:** HIGH -- backed by cognitive science research on anchoring in sequential decision-making (ACM CHI 2022, PLOS Computational Biology).

---

### Pitfall 5: Over-Engineering in a Hackathon

**What goes wrong:** The team spends hours building a sophisticated Bayesian confidence accumulation system, multi-model ensemble, and keyframe extraction pipeline -- and runs out of time before testing it against the actual game server. The system has never been tested end-to-end. During the live round, an edge case crashes the agent, and the team scores zero.

**Why it happens:** Hackathon-winning advice universally warns: "You're building a demo, not software." The competitive instinct pushes teams toward complex architectures, but hackathon scoring rewards working solutions with fewer guesses and faster time -- not architectural elegance. A simple agent that works reliably beats a sophisticated agent that crashes.

**Consequences:** Total failure during the live round. No guesses submitted = zero score. Worse than a simple agent that occasionally guesses wrong.

**Warning signs:**
- Agent has never been tested in practice mode before going live
- Multiple untested features stacked on top of each other
- Complex state management that has only been tested in unit isolation
- Team is still debugging during the round countdown

**Prevention:**
- Phase 0 (first 30 minutes): Get a WORKING agent that sends frames to an LLM and submits guesses. This is your safety net. Commit it.
- Phase 1 (next 60 minutes): Add frame accumulation and prompt engineering. Test in practice mode. Commit.
- Phase 2 (next 60 minutes): Add confidence scoring and multi-model. Test in practice mode. Commit.
- Each phase must produce a WORKING, COMMITTED agent. Never move to the next phase until the current one works in practice mode.
- Keep a `git stash` or branch of the last known working version. If the current iteration breaks, revert immediately.
- Budget 15 minutes before the live round for a full end-to-end test.

**Phase:** ALL phases. This is a meta-pitfall that affects everything.

**Confidence:** HIGH -- this is the most common failure mode at hackathons, per multiple hackathon strategy guides and retrospectives.

---

## Moderate Pitfalls

---

### Pitfall 6: Correlated Errors in Multi-Model Ensembles

**What goes wrong:** The team runs Gemini, Claude, and GPT-4o in parallel, expecting that if two agree, the answer is likely right. But research from ICML 2025 shows "models agree 60% of the time when both models err." Larger, more accurate models have HIGHLY correlated errors "even with distinct architectures and providers." Three models all saying "dancing" does not mean the answer is "dancing."

**Why it happens:** All major LLMs are trained on overlapping internet data. They share similar visual representations and language priors. They fail on the same ambiguous cases. This is the "algorithmic monoculture" problem.

**Prevention:**
- Do NOT use majority voting as a confidence signal. Treat agreement as weak evidence, not proof.
- Use models for DIFFERENT roles: Gemini Flash for fast initial screening (what general category?), Claude for careful reasoning (what specific action sequence?). Do not ask all models the same question.
- If using multi-model, compare REASONING, not just answers. If two models give the same answer but for different reasons, that is stronger evidence than matching answers alone.
- Consider using one model + multiple prompts (different perspectives on the same frame) rather than multiple models with the same prompt.

**Phase:** Phase 3 (multi-model strategy). Lower priority -- a single well-prompted model may outperform a naive ensemble.

**Confidence:** HIGH -- backed by ICML 2025 paper "Correlated Errors in Large Language Models."

---

### Pitfall 7: Multiplying Uncalibrated "Probabilities" Amplifies Errors

**What goes wrong:** The agent asks the LLM to rate confidence 1-10 and treats these as probabilities, then multiplies them across frames using Bayesian math. But LLM confidence scores are NOT calibrated probabilities. A "9/10" from GPT-4o does not mean 90% likely correct -- it might be 60% correct (due to the +0.23 overconfidence bias). Multiplying overconfident scores across frames creates runaway false certainty.

**Why it happens:** Bayesian updating is mathematically correct IF the inputs are calibrated probabilities. LLM self-reported confidence is not. "If priors are wrong, updates make things worse" is the Bayesian trap.

**Prevention:**
- Do NOT treat LLM confidence scores as probabilities. Treat them as ordinal rankings (high/medium/low).
- Use a simpler accumulation strategy: count how many frames support each candidate. Threshold on count, not on multiplied scores.
- If you must use scores, apply heavy calibration: map LLM 1-10 scores through a sigmoid or log transform that compresses overconfident values (e.g., LLM score 9 -> actual weight 0.55).
- Test calibration in practice mode: does a "9/10" answer actually get correct 90% of the time?

**Phase:** Phase 2 (confidence scoring). Must be understood before building any Bayesian system.

**Confidence:** MEDIUM -- calibration deviation numbers from 2025 benchmarks, but the specific mapping to vision-charades tasks is untested.

---

### Pitfall 8: Frame Queue Overflow Silently Drops Context

**What goes wrong:** The frame queue has `maxsize=2`. If `analyze()` takes >2 seconds (very likely with a vision LLM API call), frames accumulate and are silently dropped -- the oldest frame is discarded to make room. The agent misses the critical gesture transition between "pointy head" and "swimming" because those frames were dropped.

**Why it happens:** The `_pump_video_to_queue` function in `core/src/core/stream.py` (lines 122-129) drops old frames with no logging when the queue is full. The main loop in `__main__.py` processes frames sequentially -- each `analyze()` call blocks the next frame fetch.

**Consequences:** The agent processes frame 1 (person standing), then frame 4 (person swimming) -- missing frames 2-3 where the critical gesture transition happened. The temporal reasoning system has gaps in its evidence.

**Warning signs:**
- Agent seems to "miss" actions that happened quickly
- Frame timestamps have large gaps (>2 seconds between consecutive processed frames)
- Agent behavior in practice mode (low latency, local camera) is much better than live mode

**Prevention:**
- Make `analyze()` as fast as possible. Target <1 second per call.
- Use Gemini Flash (250 tokens/sec, 0.39s TTFT) for frame analysis, not Claude Sonnet (81 tokens/sec).
- Pre-resize images BEFORE the LLM call (see Pitfall 3) to minimize payload and inference time.
- Consider async processing: fire off the LLM call and continue fetching frames, collecting responses when they arrive.
- Log frame timestamps to detect gaps during practice testing.

**Phase:** Phase 1 (core pipeline). Must be fast from the start.

**Confidence:** HIGH -- directly verified from `core/src/core/stream.py` source code.

---

### Pitfall 9: Semantic Judge Matching is Unknown and Untestable

**What goes wrong:** The team assumes the judge does exact string matching and submits overly specific answers ("adult male performing swimming motion") or overly vague answers ("animal"). The judge is likely LLM-based (evidenced by JudgeUnavailable 503 errors and retry logic), but its matching tolerance is completely unknown. Does "shark" match "great white shark"? Does "Jaws" match "shark"?

**Why it happens:** The judge's behavior cannot be tested before the live round. Teams either over-specify (waste the chance by being too narrow) or under-specify (too vague to match).

**Consequences:** Correct identification but wrong format = wasted guess (409). With only 10 guesses, every formatting mismatch is costly.

**Warning signs:**
- Agent submits long, descriptive answers ("person pretending to be a large fish")
- Agent submits compound answers ("shark or dolphin")
- Agent gets 409 on answers that seem correct

**Prevention:**
- Submit single-concept, common-vocabulary answers: "shark" not "great white shark" or "a shark."
- Target the Goldilocks zone: specific enough to be unambiguous, generic enough to match semantic similarity. "Shark" is better than both "fish" and "Carcharodon carcharias."
- If first guess fails (409), try broader/narrower variant: "shark" -> "fish" or "Jaws."
- Strip articles, adjectives, and qualifiers. "A big red ball" -> "ball."
- Test in practice mode: say common words at the camera, see what the LLM produces, and imagine if a semantic matcher would accept it.

**Phase:** Phase 1 (prompt engineering -- output format) and Phase 2 (negative feedback loop from 409s).

**Confidence:** MEDIUM -- the judge being LLM-based is an inference from the codebase (503/JudgeUnavailable pattern), not confirmed.

---

### Pitfall 10: Filipino/Tagalog Cultural References the Model Cannot Recognize

**What goes wrong:** The competition is in Manila. Clues may include Filipino cultural concepts, local celebrities, Tagalog words, or gestures unique to Filipino culture. Vision LLMs are primarily trained on Western (US/European) data and will not recognize:
- **Lip pointing (nguso):** Filipinos point with their lips, not fingers. Models will see "puckered lips" or "kissing" instead of "pointing."
- **Mano po:** Pressing the back of an elder's hand to one's forehead. Models may see "bowing" or "hand kissing."
- **Local celebrities/politicians:** Actors may impersonate figures unknown to the LLM.
- **Filipino food items:** Balut, lechon, halo-halo may not be in the model's visual vocabulary for charades.
- **Tagalog words as answers:** The judge may expect Tagalog terms.

**Why it happens:** AI bias research confirms that models trained on predominantly English/Western data have significant gaps with Filipino cultural context. "Low-resource languages face data scarcity, leading to biases."

**Prevention:**
- Add cultural context to the system prompt: "This game is in the Philippines. Answers may be Filipino cultural concepts, Tagalog words, or local references. Consider: Filipino food (balut, lechon, adobo, halo-halo, sinigang), Filipino gestures (mano po, pagmamano, lip pointing), Filipino cultural events (fiesta, Santo Nino), and popular Filipino figures."
- Include a list of common Filipino charades words in the prompt as a hint vocabulary.
- Prompt the model to consider both English AND Tagalog answers.
- If gestures are unfamiliar to the model, have a fallback prompt that specifically asks "Could this be a Filipino cultural gesture?"

**Phase:** Phase 1 (prompt engineering). Low effort, high impact for this specific competition.

**Confidence:** MEDIUM -- we know the competition is in Manila and may include cultural references, but we do not know the actual clue list. The cultural gap in AI models is well-documented.

---

### Pitfall 11: Rate Limiting Under Competition Load

**What goes wrong:** During the live competition, all teams hit the same LLM APIs simultaneously. API providers rate-limit per-key, and the team's API key hits its request-per-minute (RPM) or tokens-per-minute (TPM) ceiling. API calls start returning 429 (Too Many Requests) with retry-after headers of 30-60 seconds. The agent stalls.

**Why it happens:** Vision API calls are token-heavy (hundreds to thousands of input tokens per image). At 1 FPS with multi-model parallel calls, the agent may be sending 2-6 vision API requests per second. OpenAI Tier 1 limits are ~500 RPM for GPT-4o. Gemini has per-project quotas. Multiple teams sharing the same API provider compound the problem.

**Prevention:**
- Pre-check rate limits for all API keys BEFORE the competition. Upgrade to higher tiers if possible.
- Implement exponential backoff with jitter for 429 responses.
- Use the CHEAPEST/FASTEST model for frame screening (Gemini Flash) and reserve expensive models for high-confidence moments.
- Do NOT send every frame. Skip frames that are visually similar to the previous one (keyframe extraction).
- Have a fallback model on a different provider. If OpenAI is rate-limited, fall back to Gemini or Claude.

**Phase:** Phase 1 (basic error handling) and Phase 3 (multi-model with fallback).

**Confidence:** MEDIUM -- depends on specific API key tier and competition load patterns.

---

## Minor Pitfalls

---

### Pitfall 12: Agent Instantiation Per Frame (Cold Start Penalty)

**What goes wrong:** Following the example code too literally, the developer creates a new `Agent()` object inside `analyze()`, which means the system prompt and model configuration are re-parsed on every frame. While pydantic-ai is lightweight, the overhead adds up at 1 FPS.

**Warning signs:** Unnecessarily high latency even on simple frames. Memory growth over time from discarded Agent objects.

**Prevention:** Instantiate the Agent at MODULE LEVEL, outside `analyze()`. The CLAUDE.md example shows this correctly but it is easy to miss. This is specifically called out in CONCERNS.md under "No Agent Warm-Up or Model Caching."

**Phase:** Phase 1. Trivial fix, important impact.

---

### Pitfall 13: Memory Leak from Accumulating PIL Images

**What goes wrong:** If the agent stores frame history (for temporal reasoning), it accumulates PIL Image objects in a list. Each 1080p RGB image is ~6MB in memory. At 1 FPS, that is 360MB per minute. Over a long round, memory exhaustion causes slowdown or crashes.

**Warning signs:** Python process memory growing steadily. System swap usage increasing. Agent slowing down over time.

**Prevention:**
- Store frame DESCRIPTIONS (text strings), not raw PIL images. A 200-character text summary is ~200 bytes vs. 6MB for the image.
- If you must store images, resize to thumbnails (128x128) and limit the buffer to the last N frames (e.g., 10).
- Use `collections.deque(maxlen=N)` for automatic eviction of old frames.

**Phase:** Phase 1 (if storing images) or Phase 2 (if accumulating evidence).

---

### Pitfall 14: Async Concurrency Bugs in the Single-File Constraint

**What goes wrong:** All logic must live in `prompt.py`. The developer tries to use `asyncio.create_task()` for parallel LLM calls but introduces race conditions, unhandled exceptions in fire-and-forget tasks, or deadlocks. The main loop in `__main__.py` is sequential per-frame -- it `await`s `analyze()` -- so any internal async complexity must be self-contained.

**Warning signs:** Occasional "Task exception was never retrieved" warnings. Agent hanging. Inconsistent behavior between runs.

**Prevention:**
- Use `asyncio.gather()` for parallel calls within `analyze()`, NOT `create_task()`.
- Always `await` all tasks before returning from `analyze()`.
- Use `asyncio.wait_for()` with timeouts on every external API call.
- Keep it simple: sequential calls to 1-2 models are often fast enough and much easier to debug.

**Phase:** Phase 3 (multi-model). Avoid async complexity in earlier phases.

---

### Pitfall 15: Network Issues During Live Demo

**What goes wrong:** WiFi is unreliable at a crowded venue. The LiveKit stream drops. API calls time out. The agent disconnects and cannot reconnect.

**Prevention:**
- The existing code handles `ConnectionError` and has retry logic for JudgeUnavailable.
- Ensure the machine is on wired ethernet if possible.
- Test the agent on a mobile hotspot to simulate degraded network.
- Consider caching the last N frame descriptions locally so the agent can still reason if the stream briefly drops.
- Set httpx timeout to 5s (not the default 10s) so failed calls fail fast.

**Phase:** Phase 1 (hardening). Test network resilience in practice mode.

---

### Pitfall 16: Guessing Too Early vs. Too Late

**What goes wrong:** The scoring formula rewards both speed AND accuracy. Guessing too early wastes attempts on wrong answers (reducing score). Guessing too late means correct answers come slowly (reducing speed bonus). The optimal threshold depends on the unknown scoring formula.

**Prevention:**
- Start conservative: accumulate 3-5 frames before first guess.
- After each 409 (wrong guess), increase the threshold slightly (more frames before next guess).
- After a correct guess, note how many frames it took and calibrate.
- If at 8/10 guesses used with no correct answer, switch to "Hail Mary" mode: guess the top candidate regardless of confidence.

**Phase:** Phase 2 (adaptive threshold). Requires experimentation in practice mode.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: Core Vision | Pitfall 1 (single-frame blindness), Pitfall 3 (image pipeline), Pitfall 12 (cold start) | Multi-frame accumulation from day one, resize images, module-level agent |
| Phase 1: Prompt Engineering | Pitfall 2 (hallucination), Pitfall 9 (judge format), Pitfall 10 (cultural bias) | Anti-hallucination prompts, simple output format, Filipino cultural hints |
| Phase 2: Confidence System | Pitfall 4 (anchoring), Pitfall 7 (uncalibrated scores) | Diverse candidate lists, ordinal rankings not probabilities |
| Phase 2: Negative Feedback | Pitfall 9 (judge matching) | Feed 409 rejections back as constraints, try broader/narrower variants |
| Phase 3: Multi-Model | Pitfall 6 (correlated errors), Pitfall 8 (latency), Pitfall 14 (async bugs) | Different roles per model, Gemini Flash for speed, asyncio.gather() |
| Phase 3: Rate Limiting | Pitfall 11 (API throttling) | Fallback providers, keyframe extraction, exponential backoff |
| ALL Phases | Pitfall 5 (over-engineering) | Working agent at every commit, practice mode testing, time-box each phase |

---

## Summary: Top 5 Pitfalls to Address First

1. **Single-Frame Blindness (Pitfall 1):** Multi-frame accumulation is non-negotiable for charades. Architect for this from frame one.
2. **Over-Engineering (Pitfall 5):** Get a working agent FIRST. Every improvement must be incremental on top of something that works.
3. **Image Pipeline (Pitfall 3):** Resize to 512px JPEG before any LLM call. This single optimization cuts latency by 3-5x.
4. **Hallucination (Pitfall 2):** Charades-specific prompting ("they are MIMING, no real objects") prevents the most common false guesses.
5. **Anchoring (Pitfall 4):** Maintain a diverse candidate list and weight recent frames more than early frames.

---

## Sources

- [Correlated Errors in Large Language Models (ICML 2025)](https://arxiv.org/abs/2506.07962)
- [The Geometry of Representational Failures in VLMs](https://arxiv.org/html/2602.07025)
- [Enhancing Temporal Understanding in Video-LLMs (NeurIPS 2025)](https://arxiv.org/html/2510.26027v1)
- [AI-Moderated Decision-Making: Anchoring Bias (ACM CHI 2022)](https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517443)
- [Confirmation Bias in Perceptual Decision-Making (PLOS)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009517)
- [Charades Dataset - Allen AI](https://prior.allenai.org/projects/charades)
- [LLM Hallucination Benchmarks 2025](https://www.allaboutai.com/resources/llm-hallucination/)
- [Vision Input - Pydantic AI Official Docs](https://ai.pydantic.dev/input/)
- [Claude Vision Documentation](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenAI Images and Vision Guide](https://developers.openai.com/api/docs/guides/images-vision)
- [Filipino Body Language and Gestures](https://turongwika.com/blogs/culture/the-different-filipino-expressions-and-gestures/)
- [Bias in Filipino Language Models (arXiv 2025)](https://arxiv.org/html/2506.07249v1)
- [Artificial Analysis Model Comparisons](https://artificialanalysis.ai/models/)
- [How LLMs See Images (Token Costs)](https://medium.com/@rajeev_ratan/how-llms-see-images-and-what-it-really-costs-you-d982ab8e67ed)
- [Failure Modes in LLM Systems Taxonomy](https://arxiv.org/abs/2511.19933)

---

*Pitfalls audit: 2026-03-28*
