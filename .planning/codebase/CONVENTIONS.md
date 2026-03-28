# Coding Conventions

**Analysis Date:** 2026-03-28

## Naming Patterns

**Files:**
- Module files: lowercase with underscores (`practice.py`, `client.py`, `models.py`)
- Packages: lowercase single words (`core`, `api`, `agent`)
- All files use lowercase naming convention exclusively

**Functions:**
- Public functions: lowercase with underscores (`start_practice()`, `start_stream()`, `analyze()`)
- Private/internal functions: leading underscore prefix (`_detect_ffmpeg()`, `_build_capture_cmd()`, `_pump_video_to_queue()`, `_on_track_subscribed()`)
- Async functions: same naming as sync functions, use `async def` prefix
- Class methods: lowercase with underscores, following standard Python conventions

**Variables:**
- Local variables: lowercase with underscores (`frame_queue`, `last_emit_monotonic`, `guess_count`, `token`, `base_url`)
- Instance variables: lowercase with underscores, prefixed with underscore if "private" (`self._base_url`, `self._token`, `self._client`)
- Module-level constants: UPPERCASE with underscores (`_JUDGE_UNAVAILABLE_BACKOFF_CAP_S`, `_MAX_JUDGE_UNAVAILABLE_RETRIES`, `_MIN_FRAME_INTERVAL_S`, `_FIRST_FRAME_TIMEOUT_S`)

**Types:**
- Classes: PascalCase (`Frame`, `CasperAPI`, `Feed`, `GuessResult`, `NoActiveRound`, `Unauthorized`, `MaxGuessesReached`, `JudgeUnavailable`)
- Exception classes: PascalCase, no "Exception" suffix (inherit from `Exception`) (`NoActiveRound`, `Unauthorized`, `MaxGuessesReached`, `JudgeUnavailable`)
- Type hints use modern Python syntax: `str | None` instead of `Optional[str]`

## Code Style

**Formatting:**
- No explicit formatter configured
- Code follows standard Python style patterns (spaces around operators, clear naming)
- Consistent indentation (4 spaces implied by readable code)

**Linting:**
- No explicit linter configured in project
- Code adheres to PEP 8 style conventions

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations` (always first, seen in all modules)
2. Standard library: `import asyncio`, `import os`, `import sys`, `import platform`, etc.
3. Third-party libraries: `import httpx`, `from PIL import Image`, `from pydantic import BaseModel`, etc.
4. Local imports: `from core import ...`, `from core.frame import ...`, `from api import ...`, etc.

**Path Aliases:**
- No path aliases configured
- Relative imports within packages: `from core.frame import Frame` (from `core/src/core/__init__.py`)
- Package-absolute imports: `from api.models import ...` (when importing from `api` package)
- Workspace packages imported as top-level: `from core import Frame, start_practice, start_stream`

**Barrel Files:**
- Core package exports public API: `core/__init__.py` exports `Frame`, `start_practice`, `start_stream`
- API package exports public API: `api/__init__.py` exports `CasperAPI` and all exception classes
- Agent package minimal exports: `agent/__init__.py` is empty

## Error Handling

**Patterns:**
- Custom exceptions for domain-specific errors (all inherit from `Exception`): `NoActiveRound`, `Unauthorized`, `MaxGuessesReached`, `JudgeUnavailable`
- Custom exceptions include custom `__str__()` methods with user-friendly messages (see `api/src/api/models.py`)
- Standard exceptions used when appropriate: `FileNotFoundError`, `ValueError`, `EnvironmentError`, `RuntimeError`, `ConnectionError`, `TimeoutError`
- Exception re-raising with context: `raise ConnectionError(...) from exc` (preserves stack trace)
- Nested try-except blocks for specific error handling: Retry logic in `agent/__main__.py` for `JudgeUnavailable` with exponential backoff
- Broad exception handling (`except Exception`) used only at top-level entry points, followed by user-facing error messages
- Specific exception handling for asyncio: `except asyncio.CancelledError: raise` (re-raise immediately, see `core/src/core/stream.py:130-131`)
- Queue-specific exception handling: `asyncio.QueueFull` and `asyncio.QueueEmpty` caught separately (see `core/src/core/stream.py:122-129`)
- HTTP status code exceptions: API client inspects status codes and raises appropriate custom exceptions before calling `resp.raise_for_status()`

## Logging

**Framework:** `print()` for console output (no logging library configured)

**Patterns:**
- User-facing messages use bracket prefixes: `[practice]`, `[!]`, `[+]`, `[guess]`, `[skip]`, `[agent]`
- Success messages: `[+]` prefix (see `agent/__main__.py:105-106`)
- Error messages: `[!]` prefix (see `agent/__main__.py:96-103`)
- Agent output: `[agent]` prefix (see `agent/src/agent/prompt.py:59`)
- Practice mode output: `[practice]` prefix (see `core/src/core/practice.py:107-114`)
- Guesses logged with frame number: `[guess #{number}]` format (see `agent/__main__.py:152`)
- Skip notifications: `[skip]` prefix (see `agent/__main__.py:70, 161`)
- State changes printed with separator lines (`"=" * 50`) for visual clarity (see `agent/__main__.py:58-61, 156-158`)

## Comments

**When to Comment:**
- Explaining non-obvious algorithm choices: "Match default practice sampling (~1 FPS)" (see `core/src/core/stream.py:14`)
- Explaining platform-specific behavior: "avfoundation defaults to ~29.97 fps; many Mac cameras only allow 30.0" (see `core/src/core/practice.py:42`)
- Explaining why exceptions are caught: "No remote video track received within..." (see `core/src/core/stream.py:79-82`)
- Comments use clear, complete sentences
- Inline comments separated by two spaces from code

**JSDoc/TSDoc:**
- Google-style docstrings for public functions and classes
- Docstrings include one-line summary followed by `Args:`, `Returns:`, and `Raises:` sections
- Example: `api/src/api/client.py:51-70` (get_feed method)
- Dataclass field documentation: One-line docstrings after field definition (see `core/src/core/frame.py:15-19`)
- Exception docstrings explain when raised and any safe behaviors (see `api/src/api/models.py:52-57`)
- Class docstrings include usage examples in `Usage::` code blocks (see `api/src/api/client.py:19-27`)

## Function Design

**Size:**
- Functions are focused and reasonably sized (e.g., `_detect_ffmpeg()` is 13 lines, `_build_capture_cmd()` is 28 lines)
- Main async entry points are longer (e.g., `run_live()` is 75 lines due to state management and loop)
- Utility functions extracted when logic is reused or complex (`_pump_video_to_queue()` for frame processing)

**Parameters:**
- Use explicit parameters with type hints: `async def start_stream(url: str, token: str) -> AsyncIterator[Frame]`
- No **kwargs or *args used in public APIs
- Defaults provided for optional parameters: `start_practice(camera_index: int = 0, fps: int = 1)`
- Environment variables loaded via class method: `CasperAPI.from_env()` instead of constructor

**Return Values:**
- Explicit return type hints on all functions: `-> None`, `-> str`, `-> Feed`, `-> AsyncIterator[Frame]`
- Union types using modern syntax: `str | None`, `int | None`
- Async generators return `AsyncIterator[T]` (see `core/src/core/stream.py:21`)
- Functions return None explicitly when appropriate (see async functions that don't return values)
- Custom exceptions raised instead of returning error codes

## Module Design

**Exports:**
- Public API defined in `__init__.py` files using `__all__` list: `core/__init__.py` exports `["Frame", "start_practice", "start_stream"]`
- All public exports documented in their respective modules
- Private module internals use leading underscore: `_detect_ffmpeg`, `_build_capture_cmd`

**Barrel Files:**
- Core exposes three public items: `Frame`, `start_practice`, `start_stream`
- API exposes client and all exceptions: `CasperAPI`, `Feed`, `GuessResult`, plus exception classes
- Enables clean imports: `from core import Frame` rather than `from core.frame import Frame`

---

*Convention analysis: 2026-03-28*
