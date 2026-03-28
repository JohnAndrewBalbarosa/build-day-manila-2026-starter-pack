# Architecture

**Analysis Date:** 2026-03-28

## Pattern Overview

**Overall:** Modular three-tier pipeline architecture with optional vision LLM integration

**Key Characteristics:**
- Layered design: frame capture → vision analysis → API submission
- Dual-mode operation: practice (local camera) and live (streaming server)
- Async-first implementation using Python asyncio
- Clear separation between infrastructure (core, api) and customization (agent)
- Frame-based processing with configurable sampling rates

## Layers

**Capture Layer (Core - Frame Source):**
- Purpose: Acquire video frames from local camera or remote LiveKit stream
- Location: `core/src/core/practice.py`, `core/src/core/stream.py`
- Contains: Frame capture abstractions, device handling, video format conversion
- Depends on: PIL (Pillow) for image processing, ffmpeg for local capture, LiveKit SDK for streaming
- Used by: Agent layer via async iteration pattern

**Model Layer (Core - Data Structure):**
- Purpose: Define the Frame contract shared across practice and live modes
- Location: `core/src/core/frame.py`
- Contains: Immutable Frame dataclass with PIL Image and UTC timestamp
- Depends on: PIL.Image, datetime
- Used by: All other layers for frame transport

**Analysis Layer (Agent - Vision Inference):**
- Purpose: Transform frames into guesses using vision LLM (user-defined)
- Location: `agent/src/agent/prompt.py`
- Contains: System prompt, analyze() function (customization point)
- Depends on: Frame model, vision LLM (pydantic-ai or user's choice)
- Used by: CLI orchestrator

**API Integration Layer (API - Server Communication):**
- Purpose: Type-safe HTTP communication with the game server
- Location: `api/src/api/client.py`, `api/src/api/models.py`
- Contains: Async HTTP client, Pydantic models for requests/responses, exception types
- Depends on: httpx (async HTTP), Pydantic (validation)
- Used by: Live mode orchestrator for guess submission and status queries

**Orchestration Layer (Agent - Control Flow):**
- Purpose: Coordinate frame flow and guess submission logic, handle errors
- Location: `agent/src/agent/__main__.py`
- Contains: CLI argument parsing, practice/live mode implementations, retry logic
- Depends on: Core, API, and Agent layers
- Used by: Entry point via asyncio.run()

## Data Flow

**Practice Mode:**

1. User starts: `uv run -m agent --practice`
2. CLI parses arguments, loads .env
3. `start_practice(camera_index, fps)` opens local camera via ffmpeg subprocess
4. Ffmpeg captures raw RGB frame → PIL Image conversion
5. Frame with timestamp yielded to `analyze(frame)`
6. User's vision LLM analyzes frame → text guess or "SKIP"
7. If guess: printed locally
8. Loop continues at FPS rate until Ctrl+C

**Live Mode:**

1. User starts: `uv run -m agent --live`
2. CLI loads .env (API_URL, TEAM_TOKEN)
3. `CasperAPI.from_env()` creates authenticated HTTP client
4. `client.get_feed()` → GET /api/feed returns LiveKit credentials (401/404/403 handled)
5. `start_stream(url, token)` connects to LiveKit room, subscribes to remote video
6. Remote frame from admin's camera → PIL Image conversion
7. Frame with timestamp yielded to `analyze(frame)`
8. User's vision LLM analyzes frame → text guess or "SKIP"
9. If guess: `client.guess(answer)` → POST /api/guess (plain text body)
   - 201: correct=True → print victory, exit loop
   - 409: correct=False → continue to next frame
   - 401: Unauthorized → exit with error (bad token)
   - 404: NoActiveRound → exit (round ended)
   - 429: MaxGuessesReached → exit (limit hit)
   - 503: JudgeUnavailable → exponential backoff retry (up to 5x), then skip this guess
10. Loop continues until correct answer, round ends, or user interrupts

## Key Abstractions

**Frame:**
- Purpose: Immutable container for video frame data shared across all modes
- Examples: `core/src/core/frame.py`
- Pattern: Frozen dataclass with PIL.Image and datetime timestamp
- Used by: All layers as unit of processing

**AsyncIterator[Frame]:**
- Purpose: Unified async streaming interface for both capture sources
- Examples: `start_practice()` and `start_stream()` both return AsyncIterator[Frame]
- Pattern: Allows downstream code to treat practice/live modes identically
- Used by: Orchestrator loops with `async for frame in source`

**CasperAPI Client:**
- Purpose: Isolate server communication with type safety and error handling
- Examples: `api/src/api/client.py`
- Pattern: Typed async client with factory method (from_env), exception-based error handling
- Used by: Live mode orchestrator for all game server interactions

**Pydantic Models:**
- Purpose: Validate and parse API responses
- Examples: `Feed` (round credentials), `GuessResult` (outcome)
- Pattern: BaseModel subclasses with field validation
- Used by: CasperAPI internally and returned to caller

**Custom Exceptions:**
- Purpose: Domain-specific error handling without magic numbers
- Examples: `Unauthorized`, `NoActiveRound`, `MaxGuessesReached`, `JudgeUnavailable`
- Pattern: Exception subclasses with __str__() for user messages
- Used by: Live mode orchestrator for conditional logic and error reporting

## Entry Points

**CLI Entry Point:**
- Location: `agent/src/agent/__main__.py`
- Triggers: `uv run -m agent` (invokes __main__.py main() function)
- Responsibilities: Parse CLI args, load environment, dispatch to practice/live modes

**Practice Mode Entry:**
- Location: `run_practice()` in `agent/src/agent/__main__.py`
- Triggers: `--practice` flag
- Responsibilities: Open local camera, iterate frames, call analyze() per frame, print results

**Live Mode Entry:**
- Location: `run_live()` in `agent/src/agent/__main__.py`
- Triggers: `--live` flag (default if no mode specified)
- Responsibilities: Authenticate API, fetch feed, connect LiveKit, iterate frames, submit guesses, handle server errors

**Analysis Function:**
- Location: `analyze(frame)` in `agent/src/agent/prompt.py`
- Triggers: Called once per captured frame
- Responsibilities: Accept Frame, invoke vision LLM, return guess string or None

## Error Handling

**Strategy:** Exception-based with specific exception types, exponential backoff for transient errors

**Patterns:**

**Frame Capture Errors (Practice Mode):**
- FFmpeg subprocess fails: RuntimeError with stderr output
- Camera timeout or not found: FileNotFoundError or RuntimeError
- No frame data: RuntimeError with guidance on resolution
- Handled: Try/except in loop, break on persistent error

**Stream Connection Errors (Live Mode):**
- Room connection fails: ConnectionError with diagnostic message
- First frame timeout: ConnectionError after 120s wait
- Track subscription issues: Handled silently, retried on new tracks
- Handled: Try/except with clear error messages before exit

**API Errors (Live Mode):**
- 401 Unauthorized: Raise Unauthorized exception, exit with message
- 404 NoActiveRound: Raise NoActiveRound exception, exit with message
- 429 MaxGuessesReached: Raise MaxGuessesReached exception, exit with message
- 503 JudgeUnavailable: Raise JudgeUnavailable exception, retry with exponential backoff (1s, 2s, 4s, 8s, 16s, cap 30s), skip guess if all retries exhausted
- Other HTTP errors: httpx raise_for_status() on unexpected codes
- Handled: Named exception catching, selective retries, human-readable messages

**Frame Analysis Errors:**
- Analyze function raises exception: Uncaught (bubbles to main, crashes CLI)
- Analyze function returns None: Treated as "skip this frame" (normal)
- Handled: User responsible for error handling in their LLM integration

## Cross-Cutting Concerns

**Logging:** No structured logging; uses print() for user-facing output prefixed with [status], [guess], [skip], [!] for errors, [+] for success, [practice], [agent]

**Validation:** Pydantic models validate API responses (Feed, GuessResult). Frame data assumed valid from PIL/ffmpeg (errors surfaced as exceptions). User input (team token, API URL) validated via environment var presence checks.

**Authentication:** Bearer token passed in HTTP Authorization header. Set via environment variable (TEAM_TOKEN) in .env file. Checked before any API request; 401 response indicates mismatch.

**Async Coordination:** Entire pipeline is async-native. Frame sources are async generators. Frame queue used in live mode with bounded maxsize (2) to prevent memory buildup. All I/O (camera subprocess, HTTP, LiveKit connection) uses asyncio methods. Main entry uses asyncio.run() to execute.

---

*Architecture analysis: 2026-03-28*
