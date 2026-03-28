# Casper Agent Starter

Starter template for building AI agents in the Casper Studios guessing game.

## Quick Start

```bash
# 1. Install uv (if you don't have it)
brew install uv        # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Copy the env file
cp .env.example .env

# 4. Run in practice mode (local camera)
uv run -m core.app --practice
```

## Project Structure

```
├── core/src/core/     🔒 Frame capture & streaming (don't edit)
├── api/src/api/       🔒 HTTP client for the game server (don't edit)
├── agent/src/agent/   ✏️  Your AI agent (edit this!)
│   ├── __main__.py         CLI entry point
│   └── prompt.py           Your prompt & analysis logic
└── .env                    Your team config
```

The main app entrypoint is `core/src/core/app.py`.

## Architecture

The main runtime entrypoint is `core/src/core/app.py`.

- `core` owns frame capture, LiveKit streaming, and the shared `Frame` type.
- `agent` owns runtime orchestration and frame analysis.
- `api` is only used in live mode to fetch feed credentials and submit guesses.

`agent/src/agent/__main__.py` remains in place as a compatibility wrapper, so
existing `uv run -m agent ...` commands still work.

The intended flows are:

- Practice mode: `core.app` -> `core.start_practice()` -> `agent.prompt.analyze(frame)`
- Live mode: `core.app` -> `api.CasperAPI.get_feed()` -> `core.start_stream()` -> `agent.prompt.analyze(frame)` -> `api.CasperAPI.guess()`

If you add frame preprocessing later, keep it in `agent.prompt.analyze()` or
an agent-side helper that consumes `Frame`, not in `core.stream`.

## Modes

### Practice Mode (offline)

Uses your local camera. No server connection needed. Great for tuning your prompts.

```bash
uv run -m core.app --practice
uv run -m core.app --practice --camera 1    # use a different camera
uv run -m core.app --practice --fps 2       # sample 2 frames/sec
```

### Live Mode (event day)

Connects to the dashboard’s HTTP API and LiveKit stream. Set these in `.env` (see `.env.example`):

- **`API_URL`** — dashboard origin only, e.g. `https://your-app.workers.dev` (no trailing slash).
- **`TEAM_TOKEN`** — your team’s API key (same as the dashboard `team.api_key`).

The admin must be on the **live** dashboard with the camera publishing to the room (so your agent can subscribe to video).

```bash
uv run -m core.app --live
```

## How It Works

1. A frame is captured (from your camera or the live stream)
2. `core/src/core/app.py` passes that frame to your `analyze()` function in `agent/prompt.py`
3. You send it to a vision LLM and return a guess (or `None` to skip)
4. If in live mode, `api` submits the guess via `POST /api/guess` (plain text body)
5. If correct (HTTP 201), you win. If wrong (HTTP 409), keep guessing. The server may return 401 (bad token), 404 (no round), 429 (max guesses), or 503 (judge unavailable — live mode retries with exponential backoff, then continues on the next frame if the judge stays down).
6. The round ends when the admin closes the stream.

## What to Edit

Open `agent/src/agent/prompt.py`. That's it. Customize:

- **`SYSTEM_PROMPT`** — the instructions for your vision LLM
- **`analyze(frame)`** — your logic for turning a frame into a guess

See [AGENTS.md](./AGENTS.md) for tips.
