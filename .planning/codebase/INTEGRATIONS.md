# External Integrations

**Analysis Date:** 2026-03-28

## APIs & External Services

**Game Server Dashboard API:**
- Service: Casper Studios game server (HTTP API)
- What it's used for: Fetch round state and submit guesses; retrieve LiveKit connection credentials
- SDK/Client: Custom `CasperAPI` class in `api/src/api/client.py`
- Auth: Bearer token (TEAM_TOKEN env var, sent via `Authorization: Bearer {token}` header)
- Endpoints:
  - `GET /api/feed` - Fetch active round info and LiveKit credentials (returns Feed model with livekit_url, token, round_id)
  - `POST /api/guess` - Submit a guess as plain text body; returns 201 (correct), 409 (incorrect), 401 (unauthorized), 404 (no round), 429 (max guesses), or 503 (judge unavailable)

**Vision LLM Provider:**
- Service: Pluggable via pydantic-ai (supports OpenAI, Anthropic, Google, etc.)
- What it's used for: Vision analysis of camera frames to generate guesses
- SDK/Client: pydantic-ai v0.1+ (`from pydantic_ai import Agent`)
- Auth: LLM_API_KEY environment variable (provider-specific key format)
- Usage location: `agent/src/agent/prompt.py` - `analyze(frame)` function (currently a stub with TODO comment; users implement vision LLM calls here)
- Note: Default implementation does not call any LLM; returns None (skips all frames)

## Data Storage

**Databases:**
- Type/Provider: Not directly accessed by agent
- Note: Game server maintains round state and guess history; agent only submits guesses via REST API

**File Storage:**
- Local filesystem only - frame images stored in memory as PIL Image objects
- No persistent file storage integration

**Caching:**
- None configured

## Authentication & Identity

**Auth Provider:**
- Custom bearer token scheme
  - Team API key (TEAM_TOKEN) - unique per team, matches `team.api_key` in dashboard database
  - Sent as HTTP header: `Authorization: Bearer {TEAM_TOKEN}`
  - Implementation: `api/src/api/client.py` lines 29-35 (httpx.AsyncClient with auth header)

**Auth Failures Handled:**
- 401 Unauthorized: Invalid or missing TEAM_TOKEN (raises `Unauthorized` exception in `api/src/api/client.py` lines 63-64, 95-96)
- Error handling in live mode: `agent/src/agent/__main__.py` lines 95-96, 132-134

## Monitoring & Observability

**Error Tracking:**
- None configured (no Sentry, Datadog, etc.)

**Logs:**
- Approach: Console output via print() statements
- Log locations:
  - `agent/src/agent/__main__.py` - Status messages during practice and live mode
  - `core/src/core/practice.py` - Camera initialization and capture status
  - `core/src/core/stream.py` - Implicit logging via frame queue operations
- Note: Errors are printed to stdout; no structured logging framework integrated

## CI/CD & Deployment

**Hosting:**
- Game Server: Deployed via Cloudflare Workers (`https://your-app.workers.dev` mentioned in README.md)
- Agent: Runs locally on developer machine or deployment environment with camera/LiveKit access

**CI Pipeline:**
- Not detected - no GitHub Actions, pytest config, or test files present

**Build Process:**
- UV workspace build using hatchling backend
- Build command: `uv sync` (installs dependencies)
- Packages built to wheels via `[tool.hatch.build.targets.wheel]` in each pyproject.toml

## Environment Configuration

**Required env vars:**
- `API_URL` - Base URL of game server dashboard (scheme + host only, no trailing slash or /api)
  - Example: `https://your-dashboard-host.example.com`
  - Used in: `api/src/api/client.py` line 41 (CasperAPI.from_env())

- `TEAM_TOKEN` - Team API key for authentication
  - Matches `team.api_key` in dashboard database
  - Used in: `api/src/api/client.py` line 42, sent as Bearer token

- `LLM_API_KEY` - Vision LLM provider API key
  - Provider-specific format (OpenAI, Anthropic, Google, etc.)
  - Used in: `agent/src/agent/prompt.py` - users pass to their pydantic-ai Agent initialization
  - Currently unused in default stub implementation

**Secrets location:**
- `.env` file in project root (copy from `.env.example`, should be gitignored)
- Loaded via `python-dotenv` in `agent/src/agent/__main__.py` line 170: `load_dotenv()`

## Webhooks & Callbacks

**Incoming:**
- None configured

**Outgoing:**
- None configured
- Note: Agent submits guesses to game server via synchronous POST requests (not webhooks); server returns results immediately

## Frame Capture & Video Streaming

**Practice Mode (Local Camera):**
- Mechanism: FFmpeg subprocess for frame capture via `imageio[ffmpeg]`
- Implementation: `core/src/core/practice.py` lines 64-89 (`_capture_one_frame`)
- Platform-specific ffmpeg commands:
  - Linux: v4l2 input format, `/dev/video{index}` device
  - macOS: avfoundation input format, 30 fps framerate
  - Windows: dshow input format, video device index
- Yields PIL Image.Image (RGB, raw bytes) frames via async generator

**Live Mode (Remote Stream):**
- Mechanism: WebRTC via LiveKit SDK
- Implementation: `core/src/core/stream.py` (start_stream async generator)
- Connection: Connects to LiveKit server URL with subscribe-only JWT token from `/api/feed`
- Track subscription: Subscribes to video tracks published by remote participants (admin publishing camera)
- Output: PIL Image.Image (RGB, converted from LiveKit VideoBufferType.RGB24) frames at ~1 FPS min interval
- Timeout: 120s to receive first remote video track (raises ConnectionError if timeout)

## Rate Limiting & Quotas

**API Guess Submissions:**
- HTTP 429 (MaxGuessesReached) - Per-round guess limit enforced by server
  - Error handling: `api/src/api/client.py` line 101, raises `MaxGuessesReached` exception
  - Live mode behavior: `agent/src/agent/__main__.py` lines 138-140 prints error and breaks loop

**Judge Availability:**
- HTTP 503 (JudgeUnavailable) - Server cannot judge the guess (judge LLM failure)
  - Error handling: `api/src/api/client.py` line 118, raises `JudgeUnavailable` exception
  - Retry strategy: Live mode implements exponential backoff (1s base, 2x multiplier, 30s cap) with max 5 retries (`agent/src/agent/__main__.py` lines 16-18, 119-131)
  - If all retries fail: Continues to next frame and retries same guess later

---

*Integration audit: 2026-03-28*
