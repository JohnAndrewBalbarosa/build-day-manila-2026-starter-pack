# Codebase Concerns

**Analysis Date:** 2026-03-28

## Tech Debt

**Incomplete LLM Integration in Agent:**
- Issue: `agent/src/agent/prompt.py` contains a TODO comment indicating the `analyze()` function is a placeholder with no actual LLM configured. The function always returns `None` (skips all frames) with only print statements for debugging.
- Files: `agent/src/agent/prompt.py` (lines 45-63)
- Impact: Agent cannot make any guesses in practice or live mode until LLM is implemented. The core functionality is non-operational.
- Fix approach: Users must implement the `analyze()` function by instantiating a pydantic-ai Agent and making actual LLM calls. Example skeleton is provided in comments but must be filled in.

**Broad Exception Handling:**
- Issue: `agent/src/agent/__main__.py` line 101 uses bare `except Exception` which masks unexpected errors and makes debugging difficult.
- Files: `agent/src/agent/__main__.py` (line 101)
- Impact: Unexpected errors during feed connection may be silently swallowed, making it hard to troubleshoot actual failures vs. expected error conditions.
- Fix approach: Replace with specific exception types that feed connection may raise, or add detailed logging before re-raising.

**Silent FFmpeg Detection Fallback:**
- Issue: `core/src/core/practice.py` lines 21-25 catches all exceptions when trying to import `imageio_ffmpeg` and silently passes, only to fail with FileNotFoundError later if system ffmpeg is not found.
- Files: `core/src/core/practice.py` (lines 21-25)
- Impact: If imageio_ffmpeg exists but is broken, user gets confusing error message about ffmpeg not being found instead of the actual import error. Debugging is harder.
- Fix approach: Either let the exception propagate with context, or catch specific expected exceptions and log what was attempted.

**Queue Drop Behavior Undocumented:**
- Issue: `core/src/core/stream.py` lines 122-129 drops old frames silently when the queue is full. If consuming LLM calls are slow, frames are discarded without logging.
- Files: `core/src/core/stream.py` (lines 122-129)
- Impact: Agent may miss important visual context if LLM inference is slow. No visibility into dropped frames makes performance debugging difficult.
- Fix approach: Add debug logging when frames are dropped, or expose dropped frame count to caller for monitoring.

## Known Bugs

**No Data Validation on Environment Variables:**
- Symptoms: If API_URL or TEAM_TOKEN are malformed or contain typos, user only discovers the problem after calling `get_feed()`.
- Files: `api/src/api/client.py` (lines 41-49)
- Current mitigation: `from_env()` checks for presence but not validity. URL format is not validated.
- Workaround: Manually validate before instantiating client.
- Fix approach: Add URL validation (e.g., urllib parse) and require https scheme in production.

**Frame Dimension Detection is Brittle:**
- Symptoms: If ffmpeg returns a raw frame size that doesn't match hardcoded resolutions (640x480, 1280x720, 1920x1080, 320x240, 800x600), capture fails with generic error.
- Files: `core/src/core/practice.py` (lines 82-89)
- Trigger: Camera or ffmpeg configuration returns unusual resolution.
- Workaround: Force ffmpeg `-vf scale=640:480` in the capture command.
- Fix approach: Parse actual frame dimensions from ffmpeg output or try to infer from aspect ratio.

**Race Condition in Stream Track Subscription:**
- Symptoms: If a remote track is published before the room fully connects, it may be missed.
- Files: `core/src/core/stream.py` (lines 43-71)
- Current state: There is a loop over existing participants after connect (lines 64-71) and a subscription handler (lines 53-59), but timing is not guaranteed.
- Impact: User may not receive any frames if admin publishes before agent subscribes.
- Mitigation: The code does handle already-published tracks, but synchronization is implicit.

## Security Considerations

**Bearer Token in HTTP Headers:**
- Risk: TEAM_TOKEN is sent as Bearer token in Authorization header. If HTTPS is not enforced, token is exposed in transit.
- Files: `api/src/api/client.py` (line 34)
- Current mitigation: httpx client is created with token, but no validation that base_url is https.
- Recommendations:
  1. Validate that API_URL starts with `https://` in `from_env()` before instantiating client.
  2. Add a warning in documentation if running in practice mode against insecure server.
  3. Consider certificate pinning for production if available.

**Environment Variable Exposure in Logs:**
- Risk: `python-dotenv` loads .env file. If `.env` is accidentally committed, or if logs print frame data with metadata, secrets could leak.
- Files: `agent/src/agent/__main__.py` (line 170)
- Current mitigation: dotenv is only loaded, not logged. But developer code in `analyze()` could log secrets.
- Recommendations: Document best practices for LLM API key usage (e.g., use env var, never log it).

**No Rate Limiting or Backoff for Unauthorized Requests:**
- Risk: If TEAM_TOKEN is wrong, agent will repeatedly fail requests in live mode.
- Files: `agent/src/agent/__main__.py` (lines 132-134)
- Current mitigation: Unauthorized breaks the loop immediately.
- Workaround: Fixed by exiting on Unauthorized.
- Recommendations: Consider rate limiting if retrying in future versions.

## Performance Bottlenecks

**Frame Capture Serialized with Network I/O:**
- Problem: In live mode, each call to `analyze()` must complete before the next frame is fetched from the queue. If LLM inference takes >1s, frames pile up or are dropped.
- Files: `agent/src/agent/__main__.py` (lines 112-161), `core/src/core/stream.py` (lines 122-129)
- Cause: The main loop is synchronous per-frame. LLM calls are awaited sequentially.
- Improvement path: Consider batching frames or running LLM inference in parallel with frame capture (separate asyncio task).

**Hardcoded Timeouts Without Tuning:**
- Problem: `_FIRST_FRAME_TIMEOUT_S = 120.0` (core/src/core/stream.py line 18) and httpx timeout=10.0 are hardcoded and may be too conservative for slow networks or too aggressive for unreliable ones.
- Files: `core/src/core/stream.py` (line 18), `api/src/api/client.py` (line 35)
- Impact: May fail prematurely on poor connections, or hang if not set high enough.
- Improvement path: Make timeouts configurable via environment or CLI args.

**Exponential Backoff for Judge Unavailability Unbounded:**
- Problem: `_JUDGE_UNAVAILABLE_BACKOFF_CAP_S = 30.0` caps backoff at 30 seconds, but with 5 retries maximum, agent waits up to 31 seconds total before giving up on a single guess.
- Files: `agent/src/agent/__main__.py` (lines 16-18, 126-130)
- Impact: If judge service is down, agent stalls for 30+ seconds per guess attempt, then skips to next frame. Can cause long pauses in live gameplay.
- Improvement path: Consider shorter initial backoff or fewer retries for responsiveness.

**No Connection Pooling or Keep-Alive:**
- Problem: `httpx.AsyncClient` is created once but closed at the end of live mode. No keep-alive or connection reuse across multiple requests.
- Files: `api/src/api/client.py` (lines 32-36)
- Impact: Each request may establish a new connection if server doesn't support connection reuse.
- Improvement path: httpx should handle this automatically, but verify TCP_NODELAY and keep-alive settings.

## Fragile Areas

**FFmpeg Subprocess Dependency:**
- Files: `core/src/core/practice.py` (lines 16-90)
- Why fragile: Relies on external ffmpeg binary with platform-specific format strings. Command construction is manual string building. If ffmpeg version changes, codec names or options may differ.
- Safe modification: Use `ffmpeg.compile()` or similar abstraction if available. Add version detection or fallback formats. Test on all three platforms (Linux, macOS, Windows).
- Test coverage: No unit tests for FFmpeg integration. Platform-specific code paths are not tested.

**LiveKit SDK Integration:**
- Files: `core/src/core/stream.py` (lines 34-98)
- Why fragile: Relies on undocumented livekit rtc API internals (e.g., `.on("track_subscribed")`, `VideoStream`, `VideoBufferType`). API may change between SDK versions.
- Safe modification: Pin livekit SDK version in pyproject.toml. Document which SDK version was tested. Add error handling for SDK exceptions.
- Test coverage: No integration tests with LiveKit server. Only practice mode is safe to test locally.

**Frame Queue Overflow Handling:**
- Files: `core/src/core/stream.py` (lines 122-129)
- Why fragile: When queue is full, code drops the oldest frame and enqueues the new one. If this pattern repeats, user loses all video context. No feedback loop to slow down video capture.
- Safe modification: Add backpressure by pausing video stream when queue is full, or expose queue size to caller.
- Test coverage: No tests for queue overflow behavior.

**Guess Result Parsing from Plain Text:**
- Files: `api/src/api/client.py` (lines 104-112)
- Why fragile: Server returns guess ID as plain text in response body on 201. If server format changes, parsing fails silently (guess_id becomes None).
- Safe modification: Migrate server to return JSON (e.g., `{"id": 123}`) so Pydantic can validate. Add validation that guess_id is always returned.
- Test coverage: No tests for API response parsing.

## Scaling Limits

**Single AsyncClient per Game:**
- Current capacity: One httpx.AsyncClient handles all API calls in a round.
- Limit: If agent must make multiple simultaneous requests (e.g., batch guess validation), client is not shared or pooled.
- Scaling path: Implement a connection pool or request queue. httpx already handles this internally, but no explicit configuration.

**Queue Size Fixed at 2 Frames:**
- Current capacity: `frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)` (core/src/core/stream.py line 41)
- Limit: If LLM inference latency exceeds 2 frame intervals, frames are dropped.
- Scaling path: Make queue size configurable. Monitor queue depth and adjust based on LLM latency.

## Dependencies at Risk

**Pydantic-AI Pre-Release:**
- Risk: Dependency is `pydantic-ai>=0.1`, which is a pre-release version. API may break in 0.2+.
- Files: `agent/pyproject.toml` (line 9)
- Impact: Agent code using pydantic-ai may need updates for API changes.
- Migration plan: Pin to `>=0.1,<1.0` when pydantic-ai reaches stable release. Test against new versions before upgrading.

**imageio_ffmpeg Fallback:**
- Risk: `imageio_ffmpeg` is an optional dependency (installed via imageio, not explicitly listed). If it breaks, silent fallback to system ffmpeg.
- Impact: Behavior differs depending on which ffmpeg is available.
- Migration plan: Either list as explicit optional dependency or remove fallback and require system ffmpeg.

**LiveKit SDK Compatibility:**
- Risk: LiveKit SDK version is not pinned (inherited from workspace). Breaking changes in SDK could cause stream failures.
- Impact: Live mode breaks silently if SDK API changes.
- Migration plan: Pin LiveKit SDK version in `core/pyproject.toml` and test on upgrades.

## Missing Critical Features

**No Agent Warm-Up or Model Caching:**
- Problem: Every frame requires instantiating a new LLM Agent (if following the example code). Model loading and initialization overhead may be significant.
- Impact: Slow first-frame response, especially for large models.
- Workaround: Agent should be instantiated once at module level, not per-frame (see CLAUDE.md example).
- Note: This is documented but easy to miss.

**No Fallback or Default Guesses:**
- Problem: If LLM API fails (rate limit, quota exceeded, etc.), agent has no fallback strategy.
- Impact: No guesses submitted during API outage.
- Improvement: Consider storing previous successful guesses or fallback heuristics.

**No Telemetry or Guessing Statistics:**
- Problem: No way to measure agent performance (accuracy, response time, frame drop rate, etc.).
- Impact: Cannot optimize agent strategy without manual inspection of logs.
- Recommendation: Add optional telemetry collection (guesses, correctness, latency).

**No Support for Multiple Simultaneous Rounds:**
- Problem: Agent can only play one round at a time (single feed and stream).
- Impact: Cannot test or run multiple agents in parallel.
- Note: May be intentional (one round per team), but limits experimentation.

## Test Coverage Gaps

**No Unit Tests for API Client:**
- What's not tested: Response parsing (especially guess ID extraction), error handling, status code routing.
- Files: `api/src/api/client.py`
- Risk: Regressions in API response handling go undetected. Server API changes break silently.
- Priority: High — API client is critical path.

**No Unit Tests for Frame Capture:**
- What's not tested: FFmpeg command construction, frame dimension detection, subprocess error handling.
- Files: `core/src/core/practice.py`
- Risk: Platform-specific bugs (e.g., wrong format on Windows) are discovered only in production.
- Priority: High — practice mode is entry point for users.

**No Integration Tests for LiveKit Streaming:**
- What's not tested: Track subscription, queue overflow, frame timestamp accuracy, cleanup on disconnect.
- Files: `core/src/core/stream.py`
- Risk: Stream failures (missing frames, stuck tasks, resource leaks) only appear during actual live games.
- Priority: Medium — live mode is less frequently tested during development.

**No E2E Tests for Agent Main Loop:**
- What's not tested: Full cycle of practice/live mode with mocked LLM and API.
- Files: `agent/src/agent/__main__.py`
- Risk: Main loop bugs (frame processing order, error recovery, graceful shutdown) go undetected.
- Priority: Medium.

**No Tests for Error Recovery:**
- What's not tested: Behavior when ffmpeg fails, when judge is unavailable, when stream disconnects mid-round.
- Risk: Agent crashes or hangs instead of gracefully handling transient failures.
- Priority: Medium-High — critical for robustness in live games.

---

*Concerns audit: 2026-03-28*
