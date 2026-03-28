# Feature Landscape: Charades AI Guessing Agent

**Domain:** Real-time visual action recognition + competitive guessing game AI
**Researched:** 2026-03-28
**Competition context:** Casper Studios charades-style game, livestream input, max 10 guesses per round, scoring rewards speed + fewer attempts

---

## Table Stakes

Features the agent must have or it will lose every round. Without these, the agent either cannot participate or will be trivially outperformed.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Vision LLM integration (single-frame)** | Cannot guess without seeing. The minimum viable agent sends one frame to a vision LLM and returns a guess. | Low | Use pydantic-ai `BinaryContent` to send PIL Image bytes to any supported model. All three major providers (OpenAI GPT-4o, Anthropic Claude, Google Gemini) support image input. |
| **SKIP logic for low-confidence frames** | Wasting guesses on garbage frames (e.g., person between gestures, camera adjusting) burns limited attempts. With only 10 guesses, a 10% false-positive rate means 1 wasted guess. | Low | Current template already has SKIP in system prompt. Must be reliable -- "say SKIP if uncertain" in system prompt is the minimum. |
| **Wrong-guess feedback loop (409 elimination)** | The API returns HTTP 409 on wrong guesses. Not using this feedback means the agent may guess the same wrong answer repeatedly or guess near-synonyms of already-rejected answers. | Low | Maintain a `rejected_guesses: list[str]` in state. Inject "Do NOT guess: {rejected}" into every subsequent prompt. Simple, high-value. |
| **Frame preprocessing (resize/compress)** | Raw camera frames are often 1920x1080+. Sending full-resolution images wastes latency and tokens without improving accuracy. Gemini uses 258 tokens/frame at standard resolution, 66 at low. | Low | Resize to 512x512 or 768x768 before sending. PIL `Image.thumbnail()` with LANCZOS resampling. Reduces API latency by 40-60% with minimal accuracy loss for action recognition. |
| **Prompt engineering for charades domain** | Generic "what do you see?" prompts perform poorly on charades. The model needs to understand it is watching someone act out a concept, not describe a scene. | Low | System prompt must specify: (1) a person is acting out / miming a concept, (2) look at gestures and body language not background, (3) answers could be objects, animals, actions, movies, people, food, places (match Pinoy Henyo categories: Bagay/Tao/Lugar/Pangyayari/Pagkain), (4) be specific but concise. |
| **State persistence across frames** | The `analyze()` function is called per-frame with no built-in memory. Without module-level state, every frame is independent -- the agent cannot accumulate evidence or track what it already guessed. | Low | Use module-level variables or a class instance in `prompt.py`. Python closures or a global dict work within the single-file constraint. |

### Confidence: HIGH
These are derived directly from the competition format (max 10 guesses, speed scoring, 409 feedback) and the technical interface (PIL Image input, pydantic-ai BinaryContent). All are implementable within the single-file constraint.

---

## Differentiators

Features that create competitive advantage. Not every team will implement these, and quality of implementation matters enormously.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Multi-frame temporal reasoning** | Charades is inherently temporal. A single frame of someone with arms up could be "surrender," "airplane," "tree," or "tall." Three frames showing arms-up then swimming motion = "shark" (pointy head + swimming). Single-frame agents will lose to temporal agents on every compound clue. | Medium | **Two approaches:** (1) *Image grid*: stitch 4-6 recent frames into a 2x3 grid, send as one image with labels "Frame 1...Frame 6." Proven to work with GPT-4o and Gemini. Cheap (one API call). (2) *Multiple image input*: send as separate images in the content array. Gemini supports up to 3,600 images per request; GPT-4o supports up to 500. Pydantic-ai supports multiple BinaryContent items in a list. Grid approach is simpler and more token-efficient. |
| **Confidence-scored candidate tracking** | Instead of binary guess/skip, maintain a ranked list of candidate answers with confidence scores. Only submit when top candidate exceeds a threshold. This turns the agent from a reflex system into a deliberative one. | Medium | **Implementation:** Ask the LLM to return structured output: `{"guess": "shark", "confidence": 0.85, "reasoning": "..."}`. Use pydantic-ai structured output with a Pydantic model. Accumulate across frames. The research literature (ICLR 2024, Xiong et al.) shows verbalized confidence from LLMs is consistently overconfident but becomes better calibrated with larger models (GPT-4 class). Use relative ranking rather than absolute thresholds. |
| **Bayesian-style confidence accumulation** | When Frame 1 says 70% "shark" and Frame 2 independently says 60% "shark," the combined evidence is stronger than either alone. Naive approaches just take the latest frame's answer. Bayesian updating provides principled evidence merging. | Medium-High | **Formula:** `P(shark|frame1,frame2) = P(shark|frame1) * P(shark|frame2) / P(shark)` (assuming conditional independence). In practice, use log-odds for numerical stability: `log_odds += log(p/(1-p))` for each frame's estimate. The optimal stopping literature (Jedynak et al., "Bayes Optimal Policies for Entropy Loss") shows that a time-dependent threshold on maximum posterior probability provides approximately optimal stopping. Since scoring rewards speed, the threshold should DECREASE over time (guess sooner as time passes, even with lower confidence). |
| **Keyframe extraction / scene change detection** | At 1 FPS, many consecutive frames look nearly identical (person holding same pose). Sending identical frames wastes API calls, money, and latency without adding information. Detecting significant changes focuses compute on moments that matter. | Medium | **Use perceptual hashing (imagehash library):** compute `imagehash.phash(frame.image.resize((64,64)))` for each frame. If hamming distance from previous keyframe's hash < threshold (research suggests 5-10 bits for "same scene"), skip the LLM call. This is O(1) per frame, requires only PIL + imagehash. **Alternative:** SSIM via SSIM-PIL library, but more compute-intensive. phash is sufficient and faster for this use case. |
| **Multi-model parallel strategy** | Different models have different strengths. Gemini 2.5 Flash is 2.5x faster than GPT-4o mini (212 vs 85 tokens/sec) but may be less accurate. Claude excels at reasoning. Running them in parallel and merging results gives both speed and accuracy. | Medium-High | **Strategy:** Fire Gemini Flash (fast, cheap) and Claude/GPT-4o (accurate) in parallel via `asyncio.gather()`. Gemini returns first (speed bonus for easy clues). If answers agree, high confidence -- guess immediately. If they disagree, wait for more frames. Pydantic-ai supports different models per agent instance. Cost is not a constraint per PROJECT.md. |
| **Adaptive guess threshold (speed vs. accuracy tradeoff)** | The scoring formula rewards both speed AND accuracy. Guessing too early wastes attempts; guessing too late loses speed bonus. The optimal threshold depends on elapsed time and remaining guesses. | Medium | **Decision rule:** `should_guess = (confidence > threshold(t, guesses_remaining))` where threshold decreases with time (speed bonus decaying) and increases with fewer remaining guesses (each guess more precious). Start threshold at ~0.8, decay toward ~0.5 over 30 seconds. With 2-3 guesses remaining, raise threshold to ~0.9 (preserve attempts). This is the "optimal stopping" problem from sequential decision theory. |
| **Filipino/Tagalog cultural awareness** | The competition is in Manila. Clues may reference Filipino culture, local celebrities, Pinoy Henyo-style categories (Bagay, Tao, Lugar, Pangyayari, Pagkain), or Tagalog concepts. Teams that only consider English/Western references will miss these. | Low-Medium | **Implementation:** Add to system prompt: "This game is played in the Philippines. Answers may be in Tagalog or reference Filipino culture. Consider: Filipino food (adobo, sinigang, halo-halo), local celebrities, Philippine landmarks, Pinoy Henyo categories." Provide a small reference list of common Filipino charades answers. Low effort, high value in a Manila competition. |
| **Semantic diversity in guessing** | When wrong, the next guess should be maximally different from rejected guesses to maximize information gain. Guessing "dog" then "puppy" wastes a guess. Guessing "dog" then "airplane" covers more of the possibility space. | Low-Medium | **Implementation:** Include rejected guesses in prompt with instruction: "Previous wrong guesses: {list}. Your next guess should be from a COMPLETELY DIFFERENT category." This is the Expected Information Gain principle from 20-questions research (Jedynak et al.) -- each guess should maximally reduce entropy over the remaining possibility space. |

### Confidence Assessment

| Feature | Confidence | Source |
|---------|------------|--------|
| Multi-frame temporal reasoning | HIGH | Gemini docs confirm multi-image support (up to 3,600 images); GPT-4o supports 500; Chain-of-Frames research (arxiv 2506.00318) validates frame-aware reasoning |
| Confidence-scored candidates | MEDIUM | ICLR 2024 paper (Xiong et al., arxiv 2306.13063) confirms verbalized confidence is overconfident but rank-order is useful; structured output via pydantic-ai is well-documented |
| Bayesian accumulation | MEDIUM | Mathematically sound, but LLM confidence scores are not calibrated probabilities -- Bayesian math on miscalibrated inputs can amplify errors. Need empirical calibration. |
| Keyframe extraction | HIGH | imagehash library is mature (PyPI), phash hamming distance thresholds well-studied (Krawetz: >10 bits = different image) |
| Multi-model parallel | HIGH | pydantic-ai multi-agent docs confirm different models per agent; asyncio.gather for parallelism is standard Python |
| Adaptive threshold | MEDIUM | Optimal stopping theory is sound; exact scoring formula unknown, so threshold tuning requires empirical testing |
| Filipino cultural awareness | HIGH | Pinoy Henyo categories documented (Scribd, Quizlet); competition explicitly in Manila per PROJECT.md |
| Semantic diversity | HIGH | Information gain / entropy reduction well-established (Jedynak et al., arxiv); trivial to implement in prompt |

---

## Anti-Features

Things to deliberately NOT build. These are time sinks that sound good but provide negative or negligible value given the constraints.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Custom ML model training / fine-tuning** | Hackathon timeframe makes this impossible. Even LoRA fine-tuning on charades data would take hours of setup + training + debugging. Pre-trained foundation models (GPT-4o, Gemini, Claude) already understand human actions from web-scale training. | Use foundation model APIs directly. Their zero-shot action recognition is already very strong. |
| **Local pose estimation pipeline (MediaPipe/OpenPose)** | Running a separate pose estimation model adds complexity, latency, and another failure point. Vision LLMs already perceive body poses implicitly. The Charades dataset research shows that end-to-end models outperform pipeline approaches. Pose keypoints would need to be converted to text descriptions to feed to the LLM anyway. | Let the vision LLM handle pose understanding natively. It already knows what "arms raised" looks like. |
| **Real-time video model inference (Video-LLaVA, VideoChat)** | Running video transformer models locally requires GPU infrastructure, model downloading, and CUDA setup. Completely impractical for a hackathon. Cloud API models process frames faster with no infrastructure overhead. | Use Gemini's native video understanding or send frame sequences to cloud APIs. |
| **Audio processing** | No audio channel is available per PROJECT.md. Building ASR or audio analysis is wasted effort. | Focus entirely on visual frames. |
| **Complex OCR pipeline** | The person is acting/miming, not holding up text. OCR would fire on background text (signs, shirts) creating noise. Vision LLMs handle incidental text reading naturally when relevant. | Let the vision LLM read any text it naturally notices. Do not add a dedicated OCR step. |
| **Optical flow computation** | Classical optical flow (OpenCV Farneback, Lucas-Kanade) adds computational overhead and produces dense motion fields that are hard to interpret semantically. At 1 FPS, temporal resolution is too low for meaningful optical flow anyway. | Use multi-frame reasoning in the LLM instead. The model can compare "what changed between Frame 1 and Frame 6" directly from the image grid. |
| **Building a UI or dashboard** | Agent is CLI-only per PROJECT.md. Time spent on visualization is time not spent on accuracy. | Use print statements for debugging. The CLI loop already shows guess/skip status. |
| **Logprobs-based confidence (for Claude)** | Anthropic Claude API does not support logprobs as of the latest documentation. GPT-4o and Gemini do support logprobs, but extracting and interpreting logprobs from multimodal inputs is complex and the correlation between token logprobs and answer correctness is not straightforward for open-ended generation. | Use verbalized confidence (ask the model to rate 0-100) or structured output with a confidence field. Research shows this is overconfident but rank-order-preserving with GPT-4 class models. |
| **Exhaustive category enumeration** | Pre-building a complete dictionary of all possible charades answers wastes time and constrains the agent. The answer space is essentially unbounded. | Let the LLM's world knowledge handle the answer space. Focus prompt engineering on category hints (objects, animals, actions, movies, food, places, people). |
| **Reinforcement learning / online learning** | There is no training loop during the competition. Each round is independent. RL requires many iterations to converge. | Use prompt-based adaptation within a round (rejected guesses as negative evidence) rather than weight updates. |

---

## Feature Dependencies

```
State Persistence ──> Everything else (all features need cross-frame state)
     |
     v
Vision LLM Integration (single-frame) ──> Multi-frame Temporal Reasoning
     |                                           |
     v                                           v
SKIP Logic ──────────────────────────> Confidence-Scored Candidates
     |                                           |
     v                                           v
Wrong-Guess Feedback (409) ──────────> Semantic Diversity in Guessing
     |                                           |
     v                                           v
Frame Preprocessing ─────────────────> Keyframe Extraction
                                                 |
                                                 v
                                       Adaptive Guess Threshold
                                                 |
                                                 v
                                       Bayesian Confidence Accumulation
                                                 |
                                                 v
                                       Multi-Model Parallel Strategy
```

### Critical path (must be built in order):
1. **State persistence** -- everything else depends on cross-frame memory
2. **Vision LLM integration** -- the foundation; nothing works without it
3. **Frame preprocessing** -- reduces latency for everything downstream
4. **SKIP logic + wrong-guess feedback** -- prevents wasting limited guesses
5. **Keyframe extraction** -- reduces unnecessary API calls
6. **Multi-frame temporal reasoning** -- the single biggest accuracy improvement
7. **Confidence tracking + adaptive threshold** -- optimizes when to guess
8. **Multi-model parallel** -- the cherry on top for speed + accuracy

### Independent features (can be added at any time):
- **Filipino cultural awareness** -- pure prompt engineering, no code dependencies
- **Semantic diversity** -- pure prompt engineering, needs only rejected_guesses list

---

## MVP Recommendation

**Prioritize (Phase 1 -- must work for any round):**
1. State persistence + Vision LLM integration (without this, agent does nothing)
2. Frame preprocessing (immediate latency win)
3. SKIP logic with confidence field in structured output
4. Wrong-guess feedback loop (409 elimination)
5. Filipino cultural prompt engineering

**This MVP can compete.** It sees frames, makes educated guesses, does not waste attempts on uncertain frames, learns from wrong guesses, and understands the Manila context.

**Build next (Phase 2 -- competitive advantage):**
1. Keyframe extraction via perceptual hashing
2. Multi-frame temporal reasoning (image grid approach)
3. Confidence-scored candidate tracking with accumulation
4. Adaptive guess threshold

**This level wins.** The agent now reasons across time, accumulates evidence, and makes mathematically-informed decisions about when to guess.

**Build if time allows (Phase 3 -- optimization):**
1. Multi-model parallel strategy
2. Bayesian confidence merging
3. Semantic diversity in guessing

**This level dominates.** But the marginal value is smaller than Phase 1 and 2.

**Defer indefinitely:**
- Custom model training (wrong timeframe)
- Local pose estimation (redundant with VLM capabilities)
- Optical flow (too low temporal resolution at 1 FPS)
- Real-time video models (infrastructure overhead)

---

## Key Research Papers and Sources

### Action Recognition
- Sigurdsson et al. (2016). ["Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding"](https://arxiv.org/abs/1604.01753) -- Original Charades dataset paper. 9,848 videos, 157 action classes.
- [Charades dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/charades)
- [Temporal Grounding of Activities using Multimodal LLMs](https://arxiv.org/html/2407.06157v1) -- 2024 work on using VLMs for activity recognition on Charades.

### Confidence Calibration
- Xiong et al. (2024). ["Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"](https://arxiv.org/abs/2306.13063) -- ICLR 2024. Key finding: verbalized confidence is overconfident but improves with model scale. Sample consistency (multiple generations) is more reliable than single verbalized scores.
- ["On Verbalized Confidence Scores for LLMs"](https://openreview.net/pdf?id=CVRdNQvFPE) -- Reliability depends heavily on prompt formulation. Large models benefit from more complex prompt methods.
- ["Confidence Calibration in Vision-Language-Action Models"](https://arxiv.org/html/2507.17383v1) -- Extends calibration research to vision-language models.

### Keyframe Extraction
- ["Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis"](https://arxiv.org/abs/2506.00667) -- 2025. Recommends min scene length 12s, detection threshold 15, weighted scoring (0.7 sharpness + 0.3 brightness).
- [imagehash Python library](https://github.com/JohannesBuchner/imagehash) -- Perceptual hashing for frame deduplication. phash with hamming distance threshold 5-10 for duplicate detection.
- [SSIM-PIL library](https://pypi.org/project/SSIM-PIL/) -- SSIM computation directly on PIL Images. Values: 1.0 = identical, 0.0 = completely different.

### Multi-Frame Temporal Reasoning
- ["Chain-of-Frames: Advancing Video Understanding in Multimodal LLMs"](https://arxiv.org/html/2506.00318v1) -- Proposes frame-aware chain-of-thought reasoning. InternVL uses Frame-1, Frame-2 text identifiers for temporal structure.
- ["M-LLM Based Video Frame Selection for Efficient Video Understanding"](https://arxiv.org/html/2502.19680v2) -- Adaptive frame selection outperforms uniform sampling.
- [Gemini Video Understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding) -- 258 tokens/frame standard, 66 tokens/frame low resolution. Supports dynamic FPS (0.1 to 60). Up to 3,600 images per request.

### Optimal Guessing Strategy
- Jedynak et al. ["Bayes Optimal Policies for Entropy Loss"](https://people.orie.cornell.edu/pfrazier/pub/2011_JedynakFrazierSznitman_20questions.pdf) -- Optimal stopping with time-dependent threshold on maximum posterior probability.
- ["Learning to Ask Informative Questions: Enhancing LLMs with Expected Information Gain"](https://arxiv.org/html/2406.17453v3) -- Each question (or guess) should maximally reduce entropy over remaining possibilities.
- ["Information-directed sampling for bandits"](https://arxiv.org/html/2512.20096) -- Balancing immediate regret against information gain in sequential decisions.

### Vision LLM Capabilities
- [Pydantic-AI Image Input docs](https://ai.pydantic.dev/input/) -- BinaryContent class for sending images. Supports OpenAI, Anthropic, Google models.
- [Pydantic-AI Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/) -- Different models per agent, parallel tool calls via asyncio.
- [HuggingFace VLM 2025 Blog](https://huggingface.co/blog/vlms-2025) -- Overview of vision language model evolution.

### Filipino Cultural Context
- [Pinoy Henyo Categories](https://www.scribd.com/document/421683483/PInoy-Henyo-Categories) -- Five categories: Bagay (Things), Tao (People), Lugar (Places), Pangyayari (Events), Pagkain (Food).
- [Pinoy Henyo Word Lists](https://thepinoyofw.com/pinoy-henyo-words-list/) -- Common Filipino word game answers spanning all categories.

### Model Speed Benchmarks
- [Gemini 2.5 Flash analysis](https://artificialanalysis.ai/models/gemini-2-5-flash) -- 212 tokens/sec output, 0.54s TTFT.
- [GPT-4o mini vs Gemini 2.5 Flash comparison](https://llm-stats.com/models/compare/gemini-2.5-flash-vs-gpt-4o-mini-2024-07-18) -- Gemini 2.5x faster throughput.
