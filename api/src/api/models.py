"""Pydantic models and exceptions for the Casper API."""

from __future__ import annotations

from pydantic import BaseModel


class Feed(BaseModel):
    """Response from GET /feed when a round is active."""

    livekit_url: str
    """LiveKit server URL (e.g. wss://project.livekit.cloud)."""

    token: str
    """Subscribe-only LiveKit JWT for the current room."""

    round_id: str
    """Identifier for the current round."""


class GuessResult(BaseModel):
    """Result of a POST /guess submission."""

    correct: bool
    """Whether the guess was correct (200 = True, 400 = False)."""

    guess_number: int | None = None
    """How many guesses this team has made this round."""


class NoActiveRound(Exception):
    """Raised when GET /feed returns 404 (no round in progress)."""

    def __str__(self) -> str:
        return "No active round. Wait for the admin to start a new round."
