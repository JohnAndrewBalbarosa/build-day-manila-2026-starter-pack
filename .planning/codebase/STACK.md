# Technology Stack

**Analysis Date:** 2026-03-28

## Languages

**Primary:**
- Python 3.10+ - All source code; primary runtime for agent, API client, and frame capture
- Plain text/TOML - Configuration and project metadata

## Runtime

**Environment:**
- Python 3.10+ (requires-python = ">=3.10" in all pyproject.toml files)

**Package Manager:**
- UV (Astral package manager) - Modern Python package and project manager
- Lockfile: `uv.lock` present (managed automatically by UV)

## Frameworks

**Core:**
- Pydantic v2.0+ - Data validation and serialization via pydantic-ai and API models
- pydantic-ai v0.1+ - Async AI agent framework for vision LLM integration (in `agent/src/agent/`)
- python-dotenv v1.0+ - Environment variable loading from `.env` files

**Video Capture & Streaming:**
- Pillow (PIL) v10.0+ - Image processing and frame manipulation (`core/src/core/frame.py`)
- imageio[ffmpeg] v2.34+ - FFmpeg wrapper for video capture subprocess handling
- LiveKit SDK v1.0+ - WebRTC streaming and video room connectivity for live game mode

**HTTP & API:**
- httpx v0.27+ - Async HTTP client for API communication (`api/src/api/client.py`)

## Key Dependencies

**Critical:**
- pydantic-ai v0.1+ - Core dependency for vision LLM agent implementation; enables async AI agent patterns with model flexibility (OpenAI, Anthropic, Google, etc.)
- livekit v1.0+ - Enables subscription to remote video streams in live game mode; handles WebRTC connectivity
- httpx v0.27+ - Async HTTP client required for communicating with the game server API

**Infrastructure:**
- Pillow v10.0+ - PIL Image type is core to frame representation throughout codebase
- imageio[ffmpeg] v2.34+ - FFmpeg integration for local camera capture via subprocess
- python-dotenv v1.0+ - Loads API credentials and configuration from environment

## Configuration

**Environment:**
- Configuration via `.env` file (copy from `.env.example`)
- Required variables: `API_URL`, `TEAM_TOKEN`, `LLM_API_KEY`

**Build:**
- `pyproject.toml` - Root workspace configuration at `/Users/adelavega/Developer/build-day-manila-2026-starter-pack/pyproject.toml`
- Per-package `pyproject.toml` files in:
  - `core/pyproject.toml` - Frame capture utilities
  - `api/pyproject.toml` - HTTP API client
  - `agent/pyproject.toml` - User-editable AI agent

## Platform Requirements

**Development:**
- UV package manager (install: `brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- FFmpeg binary (system install preferred; fallback to imageio-ffmpeg)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg`
- Python 3.10 or later

**Production:**
- Python 3.10+ runtime
- FFmpeg binary (for practice mode) or LiveKit connectivity (for live mode)
- Network access to game server API and LiveKit WebRTC servers

## Project Structure

The project is a UV workspace with three Python packages:

- **`core`** - Frame capture and streaming abstractions (locked; do not edit)
  - `src/core/frame.py` - Frame dataclass
  - `src/core/practice.py` - Local camera capture via ffmpeg subprocess
  - `src/core/stream.py` - LiveKit room subscription for remote video

- **`api`** - Typed HTTP client for game server (locked; do not edit)
  - `src/api/client.py` - CasperAPI async HTTP client with Bearer token auth
  - `src/api/models.py` - Pydantic models and custom exceptions

- **`agent`** - User-editable AI agent (primary customization target)
  - `src/agent/__main__.py` - CLI entry point with --practice and --live modes
  - `src/agent/prompt.py` - SYSTEM_PROMPT and analyze() function (main customization point)

---

*Stack analysis: 2026-03-28*
