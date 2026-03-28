# Architecture Patterns

**Domain:** Real-time charades AI guessing agent (vision LLM pipeline)
**Researched:** 2026-03-28

## Constraint: Single-File Architecture

All competitive logic must live in `agent/src/agent/prompt.py`. The orchestrator (`__main__.py`) calls `analyze(frame: Frame) -> str | None` once per captured frame at 1 FPS. We cannot modify the orchestrator, core, or API layers. This means `prompt.py` must contain:

- Module-level state (globals, class instances)
- All LLM agent definitions
- All frame processing logic
- All confidence tracking and guess decision logic

The `analyze()` function is our only entry point. It must be stateful across calls via module-level variables.

---

## Recommended Architecture: Stateful Pipeline in prompt.py

```
Frame (PIL Image + timestamp)
       |
       v
  [1] KEYFRAME GATE -----> drop (return None)
       |  (is this frame different enough?)
       |
       v
  [2] FRAME PREPROCESSOR
       |  (resize, compress for LLM)
       |
       v
  [3] MULTI-MODEL INFERENCE (parallel async)
       |  (Gemini Flash + Claude Sonnet + GPT-4o)
       |
       v
  [4] EVIDENCE ACCUMULATOR
       |  (merge new observations into hypothesis tracker)
       |
       v
  [5] NEGATIVE CONSTRAINT FILTER
       |  (remove hypotheses eliminated by 409s)
       |
       v
  [6] GUESS DECISION ENGINE -----> return None (wait)
       |  (is confidence high enough to guess now?)
       |
       v
  return guess_string
```

### Component Boundaries

| Component | Responsibility | State Owned | Communicates With |
|-----------|---------------|-------------|-------------------|
| KeyframeGate | Decide if frame is worth analyzing | `last_analyzed_hash`, `last_analyzed_time` | Preprocessor (pass) or short-circuit (None) |
| FramePreprocessor | Resize/compress for LLM APIs | None (stateless) | Multi-Model Inference |
| MultiModelInference | Call 2-3 vision LLMs in parallel | Agent instances (module-level) | Evidence Accumulator |
| EvidenceAccumulator | Track hypothesis scores across frames | `hypothesis_scores: dict[str, float]`, `frame_count`, `observation_history` | Guess Decision Engine |
| NegativeConstraintFilter | Track and apply wrong-guess eliminations | `eliminated_guesses: set[str]`, `eliminated_categories: set[str]` | Evidence Accumulator (prune) |
| GuessDecisionEngine | Optimal stopping: guess now or wait? | `guess_count`, `round_start_time` | Return value of analyze() |

---

## Component 1: Keyframe Gate

### Purpose
Avoid wasting LLM API calls on near-identical frames. At 1 FPS, consecutive frames of someone holding a pose will be nearly identical. Only forward frames where meaningful motion or scene change occurred.

### Algorithm: Downscaled Mean Absolute Difference (MAD)

**Recommendation: numpy MAD on 32x32 grayscale.** Use this because:

1. **Speed:** ~5.6ms per comparison on 1280x720 source frames (benchmarked on this machine). Well within the 1000ms per-frame budget.
2. **Spatial sensitivity:** Unlike histogram correlation (which missed a 100x100px region change entirely, returning corr=1.0000), MAD on downscaled images detects localized motion.
3. **No external dependencies:** Uses only PIL (already in core) and numpy (already installed).
4. **Simplicity:** ~5 lines of code, easy to tune.

**Alternatives considered and rejected:**

| Method | Speed (1280x720) | Why Not |
|--------|-------------------|---------|
| Histogram correlation | 0.8ms | No spatial sensitivity -- missed localized motion in benchmarks |
| SSIM (scikit-image) | ~50-100ms | Requires scikit-image dependency; overkill for frame gating |
| Perceptual hash (imagehash) | ~6ms | Requires imagehash dependency; designed for duplicate detection, not motion detection |
| PIL dhash-like (manual) | ~1.7ms | Fast but loses spatial magnitude information (binary hash) |

**Confidence:** HIGH (benchmarked on this machine with realistic frame sizes)

### Threshold Tuning

For human action recognition (charades), the threshold needs to catch meaningful body movement while ignoring camera noise and lighting flicker.

**Recommended starting threshold: MAD > 5.0** (on 0-255 scale grayscale difference)

Rationale:
- Camera noise / compression artifacts: typically MAD 0-2
- Slight lighting changes: MAD 2-4
- Meaningful body movement: MAD 5-30
- Scene change (new person, new action): MAD 30+

This threshold should be tuned during practice mode. The system should also enforce a **minimum time gate** of 0.5s between analyzed frames to prevent burst-analyzing during rapid motion (where individual frames are less informative than waiting for the motion to complete).

### Implementation Pattern

```python
# Module-level state
_last_hash: np.ndarray | None = None
_last_analyze_time: float = 0.0
_MIN_INTERVAL_S = 0.5
_CHANGE_THRESHOLD = 5.0

def _is_keyframe(image: Image.Image) -> bool:
    global _last_hash, _last_analyze_time
    now = time.monotonic()
    if now - _last_analyze_time < _MIN_INTERVAL_S:
        return False
    small = np.array(image.resize((32, 32)).convert("L"))
    if _last_hash is None:
        _last_hash = small
        _last_analyze_time = now
        return True
    diff = np.mean(np.abs(small.astype(np.int16) - _last_hash.astype(np.int16)))
    if diff > _CHANGE_THRESHOLD:
        _last_hash = small
        _last_analyze_time = now
        return True
    return False
```

---

## Component 2: Frame Preprocessor

### Purpose
Resize frames for fastest LLM inference without losing visual signal relevant to charades (body pose, hand gestures, facial expressions).

### Recommendation: 512x512 JPEG at quality 85

- Vision LLMs (Gemini, Claude, GPT-4o) all accept images up to ~4096px but process faster with smaller inputs.
- 512x512 preserves enough detail for body pose recognition while significantly reducing token count / upload time.
- JPEG compression at quality 85 reduces payload size ~70% vs PNG with negligible quality loss for charades recognition.
- Convert to bytes in-memory (BytesIO) -- never write to disk.

### Implementation Pattern

```python
import io

def _preprocess(image: Image.Image) -> bytes:
    resized = image.resize((512, 512))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
```

**Confidence:** MEDIUM (image size recommendations based on general LLM vision best practices; exact optimal size depends on specific model tokenization)

---

## Component 3: Multi-Model Parallel Inference

### Architecture

Use `asyncio.gather()` to call 2-3 vision LLMs simultaneously. Each model receives the same preprocessed frame with a structured prompt requesting a ranked list of guesses.

**Recommended model lineup:**

| Model | Role | Latency | Why |
|-------|------|---------|-----|
| Gemini 2.0 Flash | Speed oracle | ~0.5-1s | Fastest vision model; provides quick first-pass hypotheses |
| Claude Sonnet 4 | Accuracy oracle | ~1-2s | Strong visual reasoning; good at describing actions/sequences |
| GPT-4o | Diversity oracle | ~1-2s | Independent training data; catches things others miss |

**Pattern: Fan-out with `asyncio.gather`, first-result-wins for speed, all-results for evidence accumulation.**

```python
import asyncio
from pydantic_ai import Agent

# Module-level agents (instantiated once)
_fast_agent = Agent("google-gla:gemini-2.0-flash", system_prompt=SYSTEM_PROMPT)
_strong_agent = Agent("claude-sonnet-4-20250514", system_prompt=SYSTEM_PROMPT)
_diverse_agent = Agent("openai:gpt-4o", system_prompt=SYSTEM_PROMPT)

async def _multi_model_infer(image_bytes: bytes) -> list[list[str]]:
    """Returns list of ranked guess lists, one per model."""
    tasks = [
        _run_agent(_fast_agent, image_bytes),
        _run_agent(_strong_agent, image_bytes),
        _run_agent(_diverse_agent, image_bytes),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, list)]
```

### Prompt Design for Structured Output

The prompt should request a JSON array of ranked guesses to enable programmatic parsing:

```
Analyze this frame from a charades game. Someone is acting out a word or phrase.

Context from previous frames: {context_summary}
Previously wrong guesses (DO NOT repeat these): {eliminated}

Return a JSON array of your top 5 guesses, ranked by confidence:
["most likely guess", "second guess", "third guess", "fourth guess", "fifth guess"]

If you cannot determine anything meaningful, return: ["SKIP"]
```

### Model Agreement as Confidence Signal

Research confirms that multi-model agreement is a strong confidence signal (Prabhu et al., 2025; "Majority Rules" paper, arxiv 2511.15714). When 2 of 3 models agree on a hypothesis, the expected accuracy jumps significantly:

- Single model accuracy: baseline
- 2/3 agreement: +33% over single model
- 3/3 agreement: highest confidence signal

**Implementation: Track per-hypothesis agreement count as a multiplier on evidence scores.**

**Confidence:** HIGH (asyncio.gather pattern verified in pydantic-ai docs; multi-model agreement validated by research)

**Sources:**
- [Majority Rules: LLM Ensemble is a Winning Approach](https://arxiv.org/html/2511.15714v1)
- [Pydantic AI Parallel Execution](https://ai.pydantic.dev/graph/beta/parallel/)
- [Async LLM Pipelines in Python](https://dasroot.net/posts/2026/02/async-llm-pipelines-python-bottlenecks/)

---

## Component 4: Evidence Accumulator

### The Core Problem

LLM "confidence" is not a calibrated probability. Asking a model "how confident are you?" produces unreliable numbers. We need an alternative mathematical framework for accumulating evidence across frames.

### Recommendation: Weighted Rank-Vote Accumulation (not Bayesian)

Pure Bayesian updating requires calibrated likelihoods P(frame|hypothesis), which LLMs cannot provide. Research confirms LLMs systematically violate Bayesian coherence properties (arxiv 2507.11768). Instead, use a **rank-weighted voting system** that is robust to uncalibrated model outputs.

**Algorithm:**

For each frame `t` and each model `m`, the model returns a ranked list of k guesses. Assign points based on rank position:

```
score(hypothesis, frame_t, model_m) = k - rank_position
```

For k=5 guesses: rank 1 gets 5 points, rank 2 gets 4 points, ..., rank 5 gets 1 point.

**Accumulation across frames and models:**

```
total_score(hypothesis) = sum over all (frame, model) pairs of score(hypothesis, frame, model)
```

**With temporal decay** (recent frames matter more):

```
total_score(h) = SUM_t SUM_m [ score(h, t, m) * decay^(T - t) ]
```

where `decay = 0.8` and `T` is current frame index. This gives recent evidence ~5x the weight of evidence from 8 frames ago, reflecting that charades actions evolve and earlier frames may be from a different action phase.

**With model agreement bonus:**

If `n` models agree on hypothesis `h` in the same frame, multiply that frame's contribution by `agreement_multiplier(n)`:

```
agreement_multiplier(1) = 1.0
agreement_multiplier(2) = 1.5  (50% bonus for 2-model agreement)
agreement_multiplier(3) = 2.0  (100% bonus for unanimous agreement)
```

### Implementation Pattern

```python
from collections import defaultdict

class EvidenceAccumulator:
    def __init__(self, decay: float = 0.8):
        self.scores: defaultdict[str, float] = defaultdict(float)
        self.frame_count: int = 0
        self.decay = decay
        self.history: list[dict[str, float]] = []  # per-frame snapshots

    def update(self, model_results: list[list[str]]) -> None:
        """Integrate ranked guess lists from multiple models."""
        self.frame_count += 1

        # Count per-hypothesis agreement across models for this frame
        frame_votes: defaultdict[str, int] = defaultdict(int)
        frame_scores: defaultdict[str, float] = defaultdict(float)

        for ranked_guesses in model_results:
            k = len(ranked_guesses)
            for rank, guess in enumerate(ranked_guesses):
                guess_normalized = guess.lower().strip()
                frame_votes[guess_normalized] += 1
                frame_scores[guess_normalized] += (k - rank)

        # Apply agreement multiplier
        agreement_map = {1: 1.0, 2: 1.5, 3: 2.0}
        for h, raw_score in frame_scores.items():
            n_agree = frame_votes[h]
            multiplier = agreement_map.get(n_agree, 2.0)
            frame_scores[h] = raw_score * multiplier

        # Decay all existing scores, then add new evidence
        for h in self.scores:
            self.scores[h] *= self.decay
        for h, s in frame_scores.items():
            self.scores[h] += s

        self.history.append(dict(frame_scores))

    def top_hypotheses(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top-n hypotheses sorted by accumulated score."""
        return sorted(self.scores.items(), key=lambda x: -x[1])[:n]

    def confidence_ratio(self) -> float:
        """Ratio of top hypothesis score to second-best. Higher = more confident."""
        top = self.top_hypotheses(2)
        if len(top) < 2 or top[1][1] == 0:
            return float('inf') if top and top[0][1] > 0 else 0.0
        return top[0][1] / top[1][1]
```

### Why Not Full Bayesian?

| Approach | Requires | Problem |
|----------|----------|---------|
| Bayesian updating | Calibrated P(frame\|hypothesis) | LLMs are not calibrated (arxiv 2503.15850) |
| Bayesian updating | Prior over all possible answers | Answer space is unbounded (any word/phrase) |
| Bayesian updating | Independence between observations | LLM outputs are correlated (same prompt, similar images) |
| Rank-vote accumulation | Ranked list from each model | Easy to extract; robust to miscalibration |

**Confidence:** MEDIUM (rank-vote is a pragmatic heuristic, not theoretically optimal; but Bayesian is unworkable given constraints)

**Sources:**
- [LLMs are Bayesian, In Expectation, Not in Realization](https://arxiv.org/html/2507.11768v1)
- [Uncertainty Quantification and Confidence Calibration in LLMs](https://arxiv.org/html/2503.15850)
- [Sequential Multi-Hypothesis Testing in Multi-Armed Bandit Problems](https://arxiv.org/abs/2007.12961)

---

## Component 5: Negative Constraint Filter

### Purpose

When a guess returns HTTP 409 (wrong), feed that information back to:
1. Permanently eliminate that exact guess from future consideration.
2. Inform subsequent LLM calls ("it is NOT X").
3. Optionally eliminate semantically adjacent guesses.

### Research Finding: Elimination Prompting Works

Research on "Process of Elimination" prompting (Ma & Du, 2023) demonstrates that explicitly telling LLMs to eliminate wrong options before guessing significantly improves accuracy on reasoning tasks. This is directly applicable: adding "Previously wrong guesses (do NOT guess these): [list]" to the prompt is a validated technique.

However, **automated semantic adjacency elimination is risky.** If "shark" is wrong, should we eliminate "great white shark" or "hammerhead"? The judge uses semantic matching, so "shark" might have been rejected because the answer is "great white shark" specifically. Aggressive elimination could eliminate the correct answer.

### Recommendation: Conservative Elimination

1. **Exact match elimination:** Always. Track in a `set[str]`.
2. **Prompt-based soft elimination:** Include wrong guesses in the LLM prompt. Let the LLM decide what is "too similar" -- it handles semantic reasoning better than hard-coded rules.
3. **No automated category elimination:** Too risky. The judge's semantic matching tolerance is unknown.

### Implementation Pattern

```python
class NegativeConstraintTracker:
    def __init__(self):
        self.eliminated: list[str] = []

    def add_wrong_guess(self, guess: str) -> None:
        self.eliminated.append(guess)

    def format_for_prompt(self) -> str:
        if not self.eliminated:
            return ""
        return (
            "IMPORTANT - These guesses were WRONG. Do NOT repeat them "
            "or guess anything too similar:\n"
            + "\n".join(f"  - {g}" for g in self.eliminated)
        )

    def is_eliminated(self, guess: str) -> bool:
        return guess.lower().strip() in {g.lower().strip() for g in self.eliminated}
```

**Integration with Evidence Accumulator:** When a guess is marked wrong, zero out its score and all future evidence for it:

```python
def eliminate(self, hypothesis: str):
    normalized = hypothesis.lower().strip()
    self.scores[normalized] = -float('inf')  # permanent elimination
```

**Confidence:** HIGH for exact elimination, MEDIUM for prompt-based soft elimination (validated by POE research but not in this exact domain)

**Sources:**
- [Elimination-based reasoning with LLM for MCQA](https://link.springer.com/article/10.1007/s44443-025-00122-2)
- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347)

---

## Component 6: Guess Decision Engine (Optimal Stopping)

### The Mathematical Framework

This is a variant of the **optimal stopping problem** (related to the Secretary Problem). The agent must decide at each frame: **guess now, or wait for more evidence?**

**Key variables:**
- `g` = number of guesses used so far (max 10)
- `t` = time elapsed since round start
- `c(h)` = confidence in top hypothesis `h` (confidence_ratio from accumulator)
- `S(g, t)` = scoring function (unknown, but rewards fewer guesses and faster time)

### Formalization

At each frame, compute the **expected value of guessing now vs. waiting:**

```
E[guess_now] = P(correct | current_evidence) * S(g+1, t)
             + P(wrong | current_evidence) * E[future | one_fewer_guess, negative_feedback]

E[wait] = E[value | one_more_frame_of_evidence, t + dt]
```

**Guess when:** `E[guess_now] > E[wait]`

Since we do not know `P(correct)` precisely (LLM confidence is uncalibrated), we use the **confidence ratio** as a proxy and set thresholds empirically.

### Practical Decision Rule (Threshold Policy)

Use a **dynamic threshold** that decreases over time and increases with guess count:

```
threshold(g, t) = base_threshold * guess_penalty(g) / time_pressure(t)
```

Where:
- `base_threshold = 2.0` (confidence ratio: top hypothesis must be 2x the second-best)
- `guess_penalty(g) = 1.0 + 0.3 * g` (more conservative as guesses are spent; at g=5 we require 2.5x)
- `time_pressure(t) = max(0.5, 1.0 - t/120)` (after 60s, start lowering threshold; after 120s, threshold halves)

This yields:
- **Early game (g=0, t<30s):** threshold = 2.0, requiring strong confidence before first guess
- **Mid game (g=3, t=60s):** threshold = 1.9 * 1.9 / 0.75 = ~2.53 -- actually more conservative because guesses are scarce
- **Late game (g=7, t=90s):** threshold = 2.0 * 3.1 / 0.625 = ~9.92 -- very conservative, nearly out of guesses
- **Desperation (g=8, t>100s):** Override to `1.0` -- just guess the top hypothesis

Wait -- the formula above makes the agent *more* conservative as time passes when guesses are spent, which is wrong. Let me reformulate:

### Corrected Decision Rule

The correct tradeoff is:
- **Time pressure increases urgency** (lower threshold over time)
- **Guess scarcity increases caution** (higher threshold as guesses deplete)
- **These forces oppose each other** -- the agent must balance them

```
threshold(g, t) = base * (1 + 0.2 * g) * max(0.3, 1.0 - t / T_max)
```

With `base = 1.5`, `T_max = 120s`:

| Guesses Used | t=10s | t=30s | t=60s | t=90s |
|-------------|-------|-------|-------|-------|
| 0 | 1.38 | 1.13 | 0.75 | 0.38 |
| 3 | 2.21 | 1.80 | 1.20 | 0.60 |
| 6 | 3.04 | 2.48 | 1.65 | 0.83 |
| 9 | 3.87 | 3.15 | 2.10 | 1.05 |

Reading this table: at t=10s with 0 guesses used, the top hypothesis must be 1.38x the second-best to trigger a guess. At t=90s with 0 guesses remaining... we still have 10 guesses so we can afford lower thresholds. At t=90s with 9 guesses used, threshold is 1.05 -- essentially "guess anything you have."

### Special Rules

1. **First-frame bypass:** Never guess on the very first frame. The actor may still be setting up.
2. **Minimum frames rule:** Require at least 3 analyzed frames before first guess (accumulate some evidence).
3. **Unanimous agreement override:** If all 3 models agree on the same answer in a single frame AND it has been the top hypothesis for 2+ frames, guess immediately regardless of threshold.
4. **Desperation mode:** If `g >= 8` (only 2 guesses left), switch to "best available" -- guess the top hypothesis if score > 0.
5. **Duplicate prevention:** Never return a guess that was already submitted (check against NegativeConstraintTracker).

### Implementation Pattern

```python
import time

class GuessDecisionEngine:
    def __init__(self, base_threshold: float = 1.5, t_max: float = 120.0):
        self.base = base_threshold
        self.t_max = t_max
        self.guess_count = 0
        self.round_start: float | None = None
        self.frames_analyzed = 0

    def should_guess(
        self,
        top_hypothesis: str,
        confidence_ratio: float,
        model_agreement: int,       # how many models agree on top
        total_models: int,           # total models queried
        eliminated: set[str],
    ) -> str | None:
        if self.round_start is None:
            self.round_start = time.monotonic()

        self.frames_analyzed += 1
        t = time.monotonic() - self.round_start

        # Rule: minimum frames before first guess
        if self.frames_analyzed < 3:
            return None

        # Rule: never guess something already eliminated
        if top_hypothesis.lower().strip() in eliminated:
            return None

        # Rule: desperation mode
        if self.guess_count >= 8:
            return top_hypothesis

        # Rule: unanimous agreement override
        if (model_agreement == total_models and
            total_models >= 2 and
            self.frames_analyzed >= 4):
            return top_hypothesis

        # Dynamic threshold
        time_factor = max(0.3, 1.0 - t / self.t_max)
        threshold = self.base * (1 + 0.2 * self.guess_count) * time_factor

        if confidence_ratio >= threshold:
            return top_hypothesis

        return None
```

**Confidence:** MEDIUM (threshold values are theoretically motivated but need empirical tuning in practice mode)

**Sources:**
- [Optimal Stopping - Wikipedia](https://en.wikipedia.org/wiki/Optimal_stopping)
- [Secretary Problem - Wikipedia](https://en.wikipedia.org/wiki/Secretary_problem)
- [Speed, Accuracy, and the Optimal Timing of Choices (MIT)](https://dspace.mit.edu/bitstream/handle/1721.1/134916/aer.20150742.pdf)
- [Optimal Stopping Problem (Cornell, 2024)](https://physiology.med.cornell.edu/faculty/skrabanek/lab/qbio/resources_2024/2024_PS.2_optstop.pdf)

---

## Data Flow: Complete Pipeline

```
analyze(frame) called by orchestrator
  |
  |-- [KeyframeGate] Is frame different enough?
  |     NO --> return None (skip)
  |     YES --> continue
  |
  |-- [FramePreprocessor] Resize to 512x512, JPEG compress
  |     output: image_bytes
  |
  |-- [NegativeConstraintTracker] Build context string
  |     output: eliminated_text, context_summary
  |
  |-- [MultiModelInference] asyncio.gather(fast, strong, diverse)
  |     input: image_bytes + prompt(context_summary, eliminated_text)
  |     output: list[list[str]] (ranked guesses per model)
  |
  |-- [EvidenceAccumulator] Update hypothesis scores
  |     input: model_results
  |     output: top_hypothesis, confidence_ratio
  |
  |-- [GuessDecisionEngine] Should we guess now?
  |     input: top_hypothesis, confidence_ratio, agreement, eliminated
  |     NO --> return None (wait)
  |     YES --> return top_hypothesis
  |
  (orchestrator submits guess to API)
  |
  |-- If 409 (wrong):
  |     [NegativeConstraintTracker].add_wrong_guess(guess)
  |     [EvidenceAccumulator].eliminate(guess)
  |
  |-- If 201 correct=true: round over (orchestrator exits)
```

### Important: Handling 409 Feedback

The orchestrator in `__main__.py` handles the API response but does NOT call back into prompt.py with the result. The `analyze()` function just returns a string and has no knowledge of whether it was correct or wrong.

**Solution:** Track the guess we returned and detect the 409 indirectly. Since `analyze()` is called again on the next frame after a wrong guess, and the orchestrator prints `[guess #N]` for submissions, we can:

1. Track what we last returned as a non-None value.
2. Assume the guess was submitted (since analyze returned non-None).
3. Since the orchestrator only calls `analyze()` again if the guess was wrong OR it was a skip, if we returned a non-None value last time and `analyze()` is called again, the guess was wrong.

**Better solution:** The orchestrator does not inform us directly, but we can use a **pessimistic assumption**: any guess we submit that does not end the round (i.e., we get called again) was wrong. This is safe because if the guess was correct, the orchestrator breaks the loop and never calls `analyze()` again.

```python
# Module-level state
_last_submitted_guess: str | None = None

async def analyze(frame: Frame) -> str | None:
    global _last_submitted_guess

    # If we submitted a guess last call and we're being called again,
    # that guess was wrong (or judge was unavailable, but assume wrong)
    if _last_submitted_guess is not None:
        _negative_tracker.add_wrong_guess(_last_submitted_guess)
        _accumulator.eliminate(_last_submitted_guess)
        _last_submitted_guess = None

    # ... pipeline logic ...

    if decision is not None:
        _last_submitted_guess = decision
    return decision
```

---

## State Management Architecture

All state lives at module level in `prompt.py`. The `analyze()` function reads and mutates this state on each call.

```python
# === Module-Level State (initialized once when module loads) ===

# Agents (stateless after init)
_fast_agent = Agent(...)
_strong_agent = Agent(...)
_diverse_agent = Agent(...)

# Stateful components
_keyframe_gate = KeyframeGate(threshold=5.0, min_interval=0.5)
_accumulator = EvidenceAccumulator(decay=0.8)
_negative_tracker = NegativeConstraintTracker()
_decision_engine = GuessDecisionEngine(base_threshold=1.5)

# Inter-call tracking
_last_submitted_guess: str | None = None
_context_buffer: list[str] = []  # recent frame descriptions for context
```

### Why Module-Level State Works

- Python modules are singletons -- `import agent.prompt` always returns the same module object.
- The orchestrator imports `analyze` once and calls it repeatedly in the same process.
- Module-level objects persist across calls because the module is never reloaded.
- No concurrency issues: the orchestrator calls `analyze()` sequentially (one frame at a time, awaiting the result before proceeding to the next frame).

---

## Temporal Context: Multi-Frame Reasoning

Single frames are ambiguous in charades. "Someone pointing at their head" could be part of "smart", "headache", "thinking", or dozens of other charades. Temporal sequence resolves ambiguity.

### Context Window Strategy

Maintain a rolling buffer of the last 5 frame descriptions (from LLM outputs). Include this context in subsequent prompts:

```
Recent observations from previous frames:
- Frame 1 (5s ago): "Person pointing at their head repeatedly"
- Frame 2 (4s ago): "Person making swimming motions with arms"
- Frame 3 (3s ago): "Person still swimming, moving forward"
- Frame 4 (1s ago): "Person combining: pointed head + swimming = shark?"

Current hypotheses:
1. shark (score: 23.5)
2. dolphin (score: 8.2)
3. swimming (score: 6.1)
```

This gives the LLM temporal context without requiring it to process multiple images simultaneously (which would be slower and more expensive).

---

## Suggested Build Order

Dependencies flow downward. Build in this order:

### Phase 1: Foundation (no LLM calls yet)
1. **FramePreprocessor** -- stateless, no dependencies, testable immediately
2. **KeyframeGate** -- depends only on PIL/numpy, testable with practice mode camera
3. **Module-level state scaffold** -- define the globals, wire up analyze() skeleton

### Phase 2: Single-Model Intelligence
4. **Single LLM Agent** -- get one model (Gemini Flash) working end-to-end
5. **NegativeConstraintTracker** -- simple set tracking, integrate with prompt
6. **Basic guess return** -- analyze() returns the top guess from one model, practice mode works

### Phase 3: Multi-Model and Accumulation
7. **MultiModelInference** -- add Claude/GPT-4o in parallel with asyncio.gather
8. **EvidenceAccumulator** -- rank-vote scoring across frames and models
9. **Context buffer** -- rolling window of frame descriptions

### Phase 4: Optimization
10. **GuessDecisionEngine** -- dynamic threshold with optimal stopping math
11. **409 feedback loop** -- pessimistic wrong-guess detection
12. **Threshold tuning** -- practice mode experimentation
13. **Prompt engineering** -- iterate on SYSTEM_PROMPT for charades specifics

### Build Order Rationale

- Phases 1-2 get a working end-to-end agent as fast as possible (critical for hackathon).
- Phase 3 adds the competitive differentiators (multi-model consensus, evidence accumulation).
- Phase 4 is pure optimization that only matters once the foundation is solid.
- Each phase produces a working agent that can be used in practice mode for testing.

---

## Scalability Considerations

| Concern | At 1 FPS | At 2 FPS | At 5 FPS |
|---------|----------|----------|----------|
| Keyframe gate | Filters ~50% of frames | Filters ~75% | Filters ~90% |
| LLM latency bottleneck | ~1-2s per inference (OK at 1 FPS) | May queue frames; need keyframe gate | Mandatory aggressive gating |
| API rate limits | ~1 call/model/sec | May hit Gemini rate limits | Likely hit rate limits |
| Evidence accumulation | ~3-5 frames before confident | ~6-10 frames | ~15-25 frames (more noise) |
| Memory | Negligible | Negligible | Context buffer grows; cap at 10 |

**Recommendation:** Stay at 1 FPS (the default). Higher FPS does not provide proportionally more information for charades, where actions take 2-5 seconds. The keyframe gate handles any redundancy.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Single-Frame Stateless Guessing
**What:** Treat each frame independently, guess based on one frame alone.
**Why bad:** Charades is inherently temporal. A single frame of "swimming arms" tells you nothing without knowing the previous "pointed head" gesture. Single-frame guessing leads to wildly inconsistent guesses and wastes all 10 attempts.
**Instead:** Accumulate evidence across frames; never guess until multiple frames agree.

### Anti-Pattern 2: Trusting LLM Self-Reported Confidence
**What:** Ask the LLM "on a scale of 1-10, how confident are you?" and use that number directly.
**Why bad:** LLM confidence scores are not calibrated probabilities. Research shows systematic violations of Bayesian coherence (arxiv 2507.11768). A model saying "9/10 confident" may be wrong 50% of the time.
**Instead:** Use rank position and multi-model agreement as confidence proxies.

### Anti-Pattern 3: Guessing on Every Frame
**What:** Return a non-None value from analyze() on every call.
**Why bad:** With 10 max guesses and a ~60s round, guessing every second exhausts attempts in 10 seconds with no time for evidence accumulation.
**Instead:** Use keyframe gating + confidence threshold + minimum frame count before first guess.

### Anti-Pattern 4: Blocking Sequential Model Calls
**What:** Call Model A, wait for response, then call Model B, then Model C.
**Why bad:** Total latency = sum of all model latencies (~4-5s). At 1 FPS, you miss 4-5 frames while waiting.
**Instead:** Use asyncio.gather() for parallel calls. Total latency = max of model latencies (~2s).

### Anti-Pattern 5: Hardcoded Guess Timing
**What:** "Guess after exactly 5 frames" or "guess every 10 seconds."
**Why bad:** Does not adapt to confidence level or remaining guesses. May guess too early (low evidence) or too late (wasted time).
**Instead:** Dynamic threshold based on confidence ratio, guess count, and elapsed time.

---

## Sources

### Keyframe Detection
- [Frame Differencing for Motion Detection](https://medium.com/@itberrios6/introduction-to-motion-detection-part-1-e031b0bb9bb2)
- [Adaptive Keyframe Sampling for Long Video Understanding (CVPR 2025)](https://github.com/ncTimTang/AKS)
- [Video Keyframe Detector](https://github.com/joelibaceta/video-keyframe-detector)
- [Duplicate Image Detection with Perceptual Hashing](https://benhoyt.com/writings/duplicate-image-detection/)

### Multi-Model Inference
- [Async LLM Pipelines in Python Without Bottlenecks](https://dasroot.net/posts/2026/02/async-llm-pipelines-python-bottlenecks/)
- [Python Asyncio for LLM Concurrency](https://www.newline.co/@zaoyang/python-asyncio-for-llm-concurrency-best-practices--bc079176)
- [Pydantic AI Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/)
- [Pydantic AI Parallel Execution](https://ai.pydantic.dev/graph/beta/parallel/)

### Evidence Accumulation and Confidence
- [Majority Rules: LLM Ensemble](https://arxiv.org/html/2511.15714v1)
- [Beyond Majority Voting: LLM Aggregation](https://arxiv.org/abs/2510.01499)
- [Voting or Consensus? Decision-Making in Multi-Agent Debate](https://aclanthology.org/2025.findings-acl.606.pdf)
- [LLMs are Bayesian, In Expectation, Not in Realization](https://arxiv.org/html/2507.11768v1)
- [Uncertainty Quantification and Confidence Calibration in LLMs](https://arxiv.org/html/2503.15850)

### Sequential Testing and Optimal Stopping
- [Sequential Multi-Hypothesis Testing in Multi-Armed Bandit Problems](https://arxiv.org/abs/2007.12961)
- [Optimal Stopping](https://en.wikipedia.org/wiki/Optimal_stopping)
- [Speed, Accuracy, and the Optimal Timing of Choices](https://dspace.mit.edu/bitstream/handle/1721.1/134916/aer.20150742.pdf)
- [The Optimal Stopping Problem (Cornell 2024)](https://physiology.med.cornell.edu/faculty/skrabanek/lab/qbio/resources_2024/2024_PS.2_optstop.pdf)

### Negative Evidence and Elimination
- [Elimination-based Reasoning with LLM for MCQA](https://link.springer.com/article/10.1007/s44443-025-00122-2)
- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347)

---

*Architecture research: 2026-03-28*
