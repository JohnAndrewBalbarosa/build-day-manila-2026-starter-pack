# Testing Patterns

**Analysis Date:** 2026-03-28

## Test Framework

**Runner:**
- No test framework detected or configured
- No pytest, unittest, or other test runner found in dependencies
- No test configuration files (pytest.ini, setup.cfg, tox.ini)

**Assertion Library:**
- Not applicable (no testing framework in place)

**Run Commands:**
- No test execution commands available
- Project uses `uv run -m agent --practice` and `uv run -m agent --live` for manual testing

## Test File Organization

**Location:**
- No test files found in the codebase
- No `tests/` directory
- No `*_test.py` or `test_*.py` files

**Naming:**
- Not applicable (no test files present)

**Structure:**
- Not applicable (no test files present)

## Test Structure

**Suite Organization:**
- No test suites defined
- Code is designed for manual integration testing via `--practice` and `--live` modes

**Patterns:**
- Manual testing mode for development: `uv run -m agent --practice` with local camera
- Live integration testing: `uv run -m agent --live` against actual game server
- Frame sampling for manual verification: Camera feed visible during practice mode
- No automated unit test suite

## Mocking

**Framework:**
- No mocking library used
- No mock objects in dependencies

**Patterns:**
- Not applicable for this project structure

**What to Mock:**
- Not applicable

**What NOT to Mock:**
- Not applicable

## Fixtures and Factories

**Test Data:**
- No fixtures or test data factories found
- Manual testing uses real camera input or real game server
- Configuration uses environment variables for test/prod distinction

**Location:**
- Not applicable

## Coverage

**Requirements:**
- No coverage requirements enforced
- No coverage configuration files

**View Coverage:**
- Coverage not tracked

## Test Types

**Unit Tests:**
- Not implemented
- Code would benefit from unit tests for:
  - Frame capture and parsing (`core/src/core/practice.py`)
  - HTTP client error handling (`api/src/api/client.py`)
  - Async stream processing (`core/src/core/stream.py`)

**Integration Tests:**
- Manual integration testing via `--live` mode
- Game server handles validation of guesses
- Frame capture integration tested manually with real camera

**E2E Tests:**
- Live mode (`uv run -m agent --live`) is the E2E test flow
- Tests full pipeline: frame capture → analysis → API submission → result handling
- No automated E2E test suite

## Common Patterns

**Async Testing:**
- No automated async tests
- Manual testing of async functions via `--practice` and `--live` modes
- Code heavily uses `async def` and `await` (see `core/src/core/stream.py:21-132`, `api/src/api/client.py:51-127`)
- Async iteration tested manually: `async for frame in start_practice(...)`
- Async generator yield patterns not unit tested

**Error Testing:**
- Manual error testing via invalid credentials, network failures, etc.
- Exception handling verified through live mode error paths (see `agent/__main__.py:93-103`, `111-163`)
- No automated tests for custom exceptions: `NoActiveRound`, `Unauthorized`, `MaxGuessesReached`, `JudgeUnavailable`
- Retry logic with exponential backoff not tested automatically (see `agent/__main__.py:119-147`)

## Current Testing Strategy

**Manual Testing:**
- `--practice` mode: Developers test locally with camera
  - Verifies frame capture works
  - Tests agent analysis logic
  - No network required
  - Immediate visual feedback

- `--live` mode: Integration testing against server
  - Tests full end-to-end flow
  - Verifies error handling (connection, auth, game state)
  - Tests retry logic for judge failures
  - Validates guess submission

**Recommended Improvements:**
- Add pytest with asyncio support for unit tests of core modules
- Create fixtures for Frame objects (see `core/src/core/frame.py`)
- Mock httpx.AsyncClient for API client tests (see `api/src/api/client.py`)
- Test exception handling with pytest.raises
- Add async test patterns for generators and stream processing
- Create integration test utilities for game server interaction

---

*Testing analysis: 2026-03-28*
