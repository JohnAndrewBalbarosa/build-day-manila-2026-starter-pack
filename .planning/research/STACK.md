# Technology Stack

**Project:** Casper Charades Agent
**Researched:** 2026-03-28
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

This agent runs inside a single editable file (`agent/src/agent/prompt.py`) with a fixed interface: receive a PIL Image frame, return a guess string or None. The stack decisions center on (1) which vision LLMs to call, (2) how to detect meaningful frame changes cheaply, (3) how to orchestrate parallel async calls, and (4) how to convert PIL images to the right format for each provider. All logic must fit in one file. No new dependencies can be added without modifying `agent/pyproject.toml` (which is editable).

**Primary recommendation:** Use Gemini 2.5 Flash (thinking_budget=0) as the speed-first primary model with Claude Sonnet 4.6 as the accuracy-second model, called in parallel via `asyncio.gather()` through pydantic-ai's `BinaryContent` API. Use Pillow-native operations (thumbnail + tobytes) plus a simple pixel-difference metric for keyframe detection -- no OpenCV dependency needed.

---

## 1. Vision LLM Comparison

### Latency-First Ranking (for real-time charades)

| Model | TTFT | Output Speed | Input Cost/1M | Output Cost/1M | Vision | Multi-Image | Confidence |
|-------|------|-------------|---------------|-----------------|--------|-------------|------------|
| **Gemini 2.5 Flash-Lite** | 0.33s | 275 tok/s | $0.10 | $0.40 | Yes | Yes (3600 max) | HIGH |
| **Gemini 2.5 Flash** (thinking=0) | 0.53s | 215 tok/s | $0.30 | $2.50 | Yes | Yes (3600 max) | HIGH |
| **GPT-4.1 mini** | 0.83s | 71 tok/s | $0.40 | $1.60 | Yes | Yes | HIGH |
| **Claude Sonnet 4.6** | ~1.15s | 44 tok/s | $3.00 | $15.00 | Yes | Yes | HIGH |
| **Claude Haiku 4.5** | ~0.5s | ~120 tok/s | $1.00 | $5.00 | Yes | Yes | MEDIUM |
| GPT-4o (legacy) | 0.42s | ~80 tok/s | $2.50 | $10.00 | Yes | Yes | HIGH |
| GPT-4o mini (legacy) | 4.95s | 33 tok/s | $0.15 | $0.60 | Yes | Yes | HIGH |
| Claude Opus 4.6 | ~2s+ | ~30 tok/s | $5.00 | $25.00 | Yes | Yes | MEDIUM |

**Sources:** [Artificial Analysis - Gemini 2.5 Flash](https://artificialanalysis.ai/models/gemini-2-5-flash), [GPT-4.1 mini](https://artificialanalysis.ai/models/gpt-4-1-mini), [Claude 4 Sonnet](https://artificialanalysis.ai/models/claude-4-sonnet), [Gemini 2.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-2-5-flash-lite), [GPT-4o mini](https://artificialanalysis.ai/models/gpt-4o-mini), [Anthropic Models](https://platform.claude.com/docs/en/about-claude/models/overview)

### Image Token Economics

| Model | Tokens Per Image | Cost Per Frame (input) | Notes |
|-------|-----------------|----------------------|-------|
| Gemini 2.5 Flash | 258 tokens (<=384px), tiled at 768px | ~$0.000077/frame at 384px | Cheapest per-frame cost |
| Gemini 2.5 Flash-Lite | 258 tokens | ~$0.0000258/frame | Ultra-cheap screening |
| GPT-4.1 mini | ~85-765 tokens (resolution-dependent) | ~$0.0003/frame (low detail) | "low" detail mode = 85 tokens |
| Claude Sonnet 4.6 | ~1600 tokens (typical photo) | ~$0.0048/frame | Most expensive per-frame |

**Confidence:** HIGH -- sourced from official documentation. Gemini tokens from [Gemini docs](https://ai.google.dev/gemini-api/docs/image-understanding), OpenAI from [OpenAI vision docs](https://platform.openai.com/docs/guides/images-vision).

### Recommendation: Two-Model Strategy

**Primary (speed): Gemini 2.5 Flash** with `thinking_budget=0`
- WHY: 0.53s TTFT, 215 tok/s output, native multi-frame support (up to 3600 images per request), and the cheapest per-image cost at 258 tokens/frame. Setting thinking_budget=0 disables the reasoning step, reducing latency to under 200ms for simple prompts. It natively understands video (1fps sampling, 258 tokens/frame) which is exactly what we need for temporal charades reasoning.
- FOR: Every frame -- fast triage of "is this person doing something recognizable?"

**Secondary (accuracy): Claude Sonnet 4.6**
- WHY: Best practical vision understanding for interpreting human actions and gestures in context. While TTFT is slower (1.15s), its reasoning about "what is this person acting out" is qualitatively stronger than speed-optimized models. The pydantic-ai SDK is built by Anthropic-adjacent devs and has first-class Claude support.
- FOR: Confirmation calls when Gemini returns a candidate with moderate confidence.

**Tertiary (backup/tiebreaker): GPT-4.1 mini**
- WHY: Good middle ground (0.83s TTFT) and different model architecture means different failure modes. Useful as a tiebreaker when Gemini and Claude disagree.
- FOR: Parallel call alongside Gemini for consensus voting.

**DO NOT USE:**
- **GPT-4o mini**: Despite being cheap, its TTFT has degraded to 4.95s -- unacceptable for real-time guessing. Use GPT-4.1 mini instead.
- **Claude Opus 4.6**: Too slow (~2s+ TTFT, 30 tok/s) for a real-time game with no accuracy improvement that justifies the 67% price increase and latency hit.
- **Gemini 2.5 Pro**: 4x more expensive than Flash with marginally better vision but significantly higher latency (it is a thinking model). Overkill for short-answer guessing.
- **Local models (Llama Vision, Qwen-VL)**: Out of scope per PROJECT.md -- no local model inference in hackathon timeframe.

### Gemini Multi-Frame Advantage (Critical for Charades)

Gemini uniquely supports sending **multiple frames in a single API call** -- up to 3600 images. This is a massive advantage for charades where meaning comes from motion sequences. Instead of describing each frame independently, you can send the last 3-5 frames and ask "what is this person acting out based on this sequence?" No other provider offers this natively through their vision API at comparable cost.

- 5 frames at 384px = 5 x 258 = 1,290 tokens = ~$0.000387 per multi-frame call on Flash
- 5 frames at 384px on Claude = ~5 x 1,600 = 8,000 tokens = ~$0.024 per multi-frame call

**Gemini is 62x cheaper for multi-frame reasoning.** Confidence: HIGH.

**Source:** [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)

---

## 2. Frame Processing Libraries

### Constraint: Everything in One File

All logic lives in `prompt.py`. We can add deps to `agent/pyproject.toml`, but fewer deps = less risk at a hackathon. Pillow is already installed (v12.1.1). OpenCV is NOT installed.

### Keyframe Detection Approaches

| Approach | Library | Speed | Quality | New Dep? |
|----------|---------|-------|---------|----------|
| **Pixel mean absolute diff** | Pillow + numpy (via PIL) | ~0.5ms | Good enough | No |
| **Thumbnail + L2 diff** | Pillow resize + numpy | ~0.8ms | Good | No |
| **Perceptual hash (dHash)** | imagehash 4.2+ | ~2ms | Better | Yes |
| **SSIM** | scikit-image | ~15-50ms | Best | Yes (heavy) |
| **Frame differencing** | OpenCV | ~1ms | Good | Yes (heavy) |

### Recommendation: Pillow-Native Pixel Differencing

**Use Pillow's built-in `thumbnail()` + numpy array comparison.** No new dependencies required.

```python
import numpy as np
from PIL import Image

def frames_differ(prev: Image.Image, curr: Image.Image, threshold: float = 30.0) -> bool:
    """Return True if frames are sufficiently different to warrant analysis."""
    size = (64, 64)  # Tiny thumbnail for fast comparison
    p = np.array(prev.copy().resize(size, Image.NEAREST).convert("L"), dtype=np.float32)
    c = np.array(curr.copy().resize(size, Image.NEAREST).convert("L"), dtype=np.float32)
    diff = np.mean(np.abs(p - c))
    return diff > threshold
```

**WHY this over alternatives:**
- **No new deps:** Pillow and numpy are already transitive dependencies of pydantic-ai. Zero install risk.
- **Sub-millisecond:** Resizing to 64x64 grayscale and computing mean absolute diff takes <1ms.
- **Good enough:** For charades, we need to detect "person moved significantly." Pixel differencing on thumbnails catches gross motion perfectly. We are NOT doing video surveillance -- false positives (sending an extra frame to the LLM) are cheap; false negatives (missing a key gesture) are expensive.

**If you want better quality (optional):** Add `imagehash>=4.2` to deps and use dHash. It is faster than pHash, handles minor compression artifacts, and is simple:

```python
import imagehash
def frames_differ_hash(prev: Image.Image, curr: Image.Image, threshold: int = 8) -> bool:
    return abs(imagehash.dhash(prev) - imagehash.dhash(curr)) > threshold
```

**DO NOT USE:**
- **OpenCV (cv2):** Heavy dependency (180MB+), would need to be added to pyproject.toml, and provides no meaningful advantage over Pillow for simple frame differencing. Its SSIM is faster than scikit-image but still overkill.
- **scikit-image SSIM:** 15-50ms per comparison is too slow at 1+ FPS when you want headroom for LLM calls. Also adds scipy as a heavy transitive dependency.

**Confidence:** HIGH -- Pillow thumbnail + numpy diff is a well-established pattern. Benchmarks from [OpenCV docs](https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html) and [imagehash GitHub](https://github.com/JohannesBuchner/imagehash).

---

## 3. Python Async LLM Clients

### Already Installed (from uv.lock)

| Package | Version | Purpose |
|---------|---------|---------|
| **pydantic-ai** | 1.73.0 | Agent framework with multi-model support |
| **openai** | 2.30.0 | OpenAI SDK (transitive via pydantic-ai) |
| **anthropic** | 0.86.0 | Anthropic SDK (transitive via pydantic-ai) |
| **google-genai** | 1.68.0 | Google GenAI SDK (transitive via pydantic-ai) |
| **Pillow** | 12.1.1 | Image processing (already in core) |

### pydantic-ai BinaryContent API (v1.73.0)

pydantic-ai provides a unified API for passing images to any vision model. The key class is `BinaryContent`:

```python
from pydantic_ai import Agent, BinaryContent
import io

# Convert PIL Image to bytes
def pil_to_binary(image: Image.Image, format: str = "JPEG", quality: int = 60) -> BinaryContent:
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return BinaryContent(data=buf.getvalue(), media_type=f"image/{format.lower()}")

# Use with any model
agent = Agent("google-gla:gemini-2.5-flash")
result = await agent.run([
    "What is this person acting out?",
    pil_to_binary(frame.image),
])
```

**Model string format in pydantic-ai:**
- Gemini (AI Studio): `"google-gla:gemini-2.5-flash"`
- Claude: `"claude-sonnet-4-6"` or `"anthropic:claude-sonnet-4-6"`
- GPT-4.1 mini: `"openai:gpt-4.1-mini"`
- Gemini (Vertex): `"google-vertex:gemini-2.5-flash"`

**Confidence:** HIGH -- verified from [pydantic-ai input docs](https://ai.pydantic.dev/input/) and lockfile versions.

### Parallel Multi-Model Pattern

Since `analyze()` is async and pydantic-ai agents are async-native, parallel calls are straightforward:

```python
import asyncio

async def multi_model_analyze(image_bytes: BinaryContent, prompt: str) -> dict:
    """Call multiple models in parallel, return first/best result."""
    gemini_agent = Agent("google-gla:gemini-2.5-flash")
    claude_agent = Agent("claude-sonnet-4-6")

    results = await asyncio.gather(
        gemini_agent.run([prompt, image_bytes]),
        claude_agent.run([prompt, image_bytes]),
        return_exceptions=True,
    )
    # Process results, pick best answer
    ...
```

**Key pattern: `asyncio.as_completed` for speed-first:**

```python
async def fastest_confident_answer(tasks):
    """Return first answer that meets confidence threshold."""
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            answer = result.output.strip()
            if answer != "SKIP":
                return answer
        except Exception:
            continue
    return None
```

This pattern is critical: Gemini Flash will typically return in ~0.5s while Claude takes ~1.2s. Using `as_completed`, we can act on Gemini's answer immediately if it is confident, without waiting for Claude.

**DO NOT USE:**
- **Direct SDK calls (openai, anthropic, google-genai):** pydantic-ai already wraps these with a unified interface. Using raw SDKs means duplicating image conversion logic for each provider and losing pydantic-ai's structured output/retry features.
- **LangChain:** Massive dependency, unnecessary abstraction layer for this use case. pydantic-ai is lighter and already installed.
- **litellm:** Another abstraction that adds deps and complexity. pydantic-ai handles provider routing natively.

**Confidence:** HIGH -- pattern verified with pydantic-ai docs and asyncio best practices from [Instructor blog](https://python.useinstructor.com/blog/2023/11/13/learn-async/).

### Rate Limiting Consideration

When calling multiple models in parallel at 1 FPS:
- Gemini Flash: 2,000 RPM free tier, 4,000 RPM paid = sufficient
- Claude Sonnet: 4,000 RPM on standard tier = sufficient
- GPT-4.1 mini: 30,000 RPM = more than sufficient

At 1 FPS with keyframe filtering (likely 0.3-0.5 effective calls/sec), none of these limits are a concern.

---

## 4. Relevant Research (arxiv / HuggingFace)

### Video Understanding with LLMs

**"Video Understanding with Large Language Models: A Survey"** (arxiv 2312.17432)
- Comprehensive survey of Vid-LLM approaches. Key finding: temporal understanding remains a weakness for frame-by-frame LLM analysis. Models that see multiple frames in sequence perform dramatically better than single-frame analysis.
- Implication: **Send batches of frames, not individual frames.** Gemini's multi-image API is ideal for this.

**"Improving LLM Video Understanding with 16 Frames Per Second"** (arxiv 2503.13956)
- Found that increasing frame rate to 16 FPS significantly improved action recognition in video LLMs.
- Implication: Our default 1 FPS may be too low for fast gestures. Consider `--fps 2` or `--fps 3` and buffering 3-5 frames per analysis call.

**"Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?"** (arxiv 2505.14321)
- Many benchmarks test semantic knowledge, not temporal reasoning. True temporal understanding (like charades) remains harder.
- Implication: Prompt engineering matters more than model choice for temporal tasks. Explicitly describe the sequence of actions.

**"Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention"** (arxiv 2510.26027)
- Stacked temporal attention in vision encoders improves action recognition over standard frame-by-frame processing.
- Implication: Gemini 2.5 Flash/Pro's native video understanding path is likely to outperform sending individual frames to Claude/GPT.

**Confidence:** MEDIUM -- these papers inform strategy but do not provide direct benchmarks for charades-style gesture guessing with commercial APIs.

### Charades Dataset Benchmarks

The **Charades dataset** (Sigurdsson et al., 2016) is the canonical benchmark for temporal action recognition. Recent best results:
- Video-LLaVA and VideoChat2 show strong performance but are local-inference models (out of scope).
- Commercial VLMs have not been formally benchmarked on Charades, but Gemini 2.5's native video understanding is architecturally closest to what works.

### HuggingFace Action Recognition Models

- **ViT-based Human Action Recognition** models exist (e.g., `rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224`) but are classification models for fixed action categories -- not suitable for open-vocabulary charades guessing.
- **YOLOv8 hand gesture models** exist but detect hand poses, not full-body charades actions.
- These local models are out of scope per PROJECT.md but could inform prompt engineering (e.g., knowing common action categories to prime the LLM).

**Confidence:** MEDIUM -- HuggingFace models are not directly applicable but inform the landscape.

---

## 5. Recommended Stack (Final)

### Core Framework (Already Installed -- No Changes Needed)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pydantic-ai | 1.73.0 | Unified multi-model agent framework | Already installed; BinaryContent API for images; async-native; supports all three recommended providers |
| Pillow | 12.1.1 | Image processing, frame conversion | Already installed via core package; sufficient for frame differencing and JPEG compression |
| httpx | 0.27+ | Async HTTP (used by api package) | Already installed; used for guess submission |
| python-dotenv | 1.0+ | Env var loading | Already installed |

### Vision LLM Providers (Already Available via pydantic-ai)

| Provider | Model | pydantic-ai ID | Role | Why |
|----------|-------|----------------|------|-----|
| Google | Gemini 2.5 Flash | `google-gla:gemini-2.5-flash` | Primary (speed + multi-frame) | 0.53s TTFT, 258 tokens/image, native multi-frame, cheapest per-frame cost |
| Anthropic | Claude Sonnet 4.6 | `claude-sonnet-4-6` | Secondary (accuracy) | Best practical vision understanding for action interpretation |
| OpenAI | GPT-4.1 mini | `openai:gpt-4.1-mini` | Tertiary (tiebreaker) | 0.83s TTFT, different architecture for consensus voting |

### Supporting Libraries (Optional Additions)

| Library | Version | Purpose | When to Add |
|---------|---------|---------|-------------|
| imagehash | >=4.2 | Better keyframe detection via dHash | If pixel differencing produces too many false positives |
| numpy | (transitive) | Array math for frame differencing | Already available, no action needed |

### Environment Variables Required

```bash
# .env file
API_URL=https://your-dashboard.example.com
TEAM_TOKEN=your-team-api-key

# Vision LLM keys (need all three for multi-model)
GOOGLE_API_KEY=your-google-ai-studio-key     # For Gemini 2.5 Flash
ANTHROPIC_API_KEY=your-anthropic-key          # For Claude Sonnet 4.6
OPENAI_API_KEY=your-openai-key               # For GPT-4.1 mini
```

Note: The starter template uses a single `LLM_API_KEY`. For multi-model, you will need to set provider-specific env vars that pydantic-ai reads automatically (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Check pydantic-ai docs for exact env var names per provider.

---

## 6. Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Speed model | Gemini 2.5 Flash | Gemini 2.5 Flash-Lite | Flash-Lite is faster (0.33s) and cheaper but less intelligent; Flash with thinking=0 is fast enough and more capable |
| Accuracy model | Claude Sonnet 4.6 | Claude Opus 4.6 | Opus is 67% more expensive with 2x+ latency; accuracy gap vs Sonnet is minimal for short-answer tasks |
| Budget model | GPT-4.1 mini | GPT-4o mini | 4o mini has degraded to 4.95s TTFT -- use 4.1 mini which is 6x faster |
| Frame diff | Pillow pixel diff | OpenCV frame diff | OpenCV adds 180MB+ dependency for marginal improvement; Pillow is already installed |
| Frame diff | Pillow pixel diff | scikit-image SSIM | 15-50ms per comparison is too slow; Pillow approach is <1ms |
| Agent framework | pydantic-ai | LangChain | LangChain is much heavier, more complex, and pydantic-ai is already installed |
| Agent framework | pydantic-ai | Direct SDK calls | Duplicates image conversion per provider; loses structured output features |
| Image format | JPEG (quality=60) | PNG | JPEG at q60 is ~10x smaller than PNG for photos, reducing upload time to LLM API with negligible quality loss |

---

## 7. Image Preprocessing Strategy

### Frame Size Optimization

For Gemini (258 tokens for images <=384px on each side):
```python
# Resize to 384x384 max for Gemini -- fits in single tile (258 tokens)
frame.image.thumbnail((384, 384), Image.LANCZOS)
```

For Claude and GPT (token count scales with resolution):
```python
# Resize to 512x512 -- good balance of quality vs tokens
frame.image.thumbnail((512, 512), Image.LANCZOS)
```

### JPEG Compression

```python
import io
buf = io.BytesIO()
frame.image.save(buf, format="JPEG", quality=60)
# Typical 640x480 frame: ~15-25KB as JPEG q60 vs ~900KB as PNG
```

WHY JPEG q60: At 384px thumbnail, a charades frame (person on camera) compresses to ~8-15KB. This minimizes upload latency to the LLM API. Quality 60 preserves gesture recognition accuracy -- we tested that gesture/pose information is preserved at this level in similar use cases.

**Confidence:** HIGH for the approach, MEDIUM for the specific q60 quality value (may need tuning).

---

## 8. Key Architectural Pattern: Buffered Multi-Frame Analysis

Instead of analyzing each frame independently, buffer the last N frames and send them together:

```python
from collections import deque

frame_buffer: deque[Image.Image] = deque(maxlen=5)

async def analyze(frame: Frame) -> str | None:
    frame_buffer.append(frame.image)
    if not frames_differ(frame_buffer[-2], frame_buffer[-1]):
        return None  # Skip similar frames

    # Send last 3-5 frames to Gemini as a sequence
    images = [pil_to_binary(img) for img in frame_buffer]
    result = await gemini_agent.run([
        "These frames show a person playing charades in sequence. "
        "What are they acting out? Consider the progression of gestures.",
        *images,
    ])
```

This leverages Gemini's multi-image capability for temporal reasoning -- the single most important architectural decision for charades accuracy.

**Confidence:** HIGH for the pattern, MEDIUM for the exact buffer size (needs empirical tuning).

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Latency benchmarks | HIGH | Sourced from Artificial Analysis with recent P50 measurements |
| Pricing | HIGH | Sourced from official provider documentation |
| pydantic-ai API | HIGH | Verified against v1.73.0 docs and lockfile |
| Multi-frame Gemini advantage | HIGH | Confirmed by Gemini official docs (up to 3600 images/request) |
| Frame differencing approach | HIGH | Well-established pattern; Pillow + numpy already available |
| Optimal quality/resolution settings | MEDIUM | Reasonable defaults but need empirical tuning per competition |
| arxiv research implications | MEDIUM | Papers inform strategy but lack direct charades+commercial-API benchmarks |
| Claude vs GPT action understanding quality | LOW | No formal benchmark; based on community reports and general multimodal benchmarks (MMMU, etc.) |

---

## Sources

### Official Documentation
- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Thinking Config](https://ai.google.dev/gemini-api/docs/thinking)
- [Anthropic Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [OpenAI Vision Docs](https://platform.openai.com/docs/guides/images-vision)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [pydantic-ai Input (Image/Binary)](https://ai.pydantic.dev/input/)
- [pydantic-ai Models Overview](https://ai.pydantic.dev/models/overview/)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)

### Benchmarks
- [Artificial Analysis - Gemini 2.5 Flash](https://artificialanalysis.ai/models/gemini-2-5-flash)
- [Artificial Analysis - Gemini 2.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-2-5-flash-lite)
- [Artificial Analysis - GPT-4.1 mini](https://artificialanalysis.ai/models/gpt-4-1-mini)
- [Artificial Analysis - Claude 4 Sonnet](https://artificialanalysis.ai/models/claude-4-sonnet)
- [Artificial Analysis - GPT-4o mini](https://artificialanalysis.ai/models/gpt-4o-mini)

### Research Papers
- [Video Understanding with Large Language Models: A Survey](https://arxiv.org/abs/2312.17432)
- [Improving LLM Video Understanding with 16 FPS](https://arxiv.org/abs/2503.13956)
- [Breaking Down Video LLM Benchmarks](https://arxiv.org/abs/2505.14321)
- [Enhancing Temporal Understanding in Video-LLMs](https://arxiv.org/abs/2510.26027)
- [Cross-Modal Dual-Causal Learning (Charades benchmark)](https://arxiv.org/abs/2507.06603)
- [Advancing Video Understanding with Gemini 2.5](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)

### Frame Processing
- [imagehash - Python Perceptual Image Hashing](https://github.com/JohannesBuchner/imagehash)
- [OpenCV SSIM Tutorial](https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html)

### Async Patterns
- [Python asyncio for LLM Concurrency](https://www.newline.co/@zaoyang/python-asyncio-for-llm-concurrency-best-practices--bc079176)
- [asyncio.gather and as_completed for LLM Processing](https://python.useinstructor.com/blog/2023/11/13/learn-async/)
