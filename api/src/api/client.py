"""Typed HTTP client for the Casper guessing game API."""

from __future__ import annotations

import os

import httpx

from api.models import Feed, GuessResult, NoActiveRound


class CasperAPI:
    """Client for interacting with the Casper game server.

    Usage::

        client = CasperAPI.from_env()
        feed = await client.get_feed()
        result = await client.guess("golden retriever")
    """

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10.0,
        )

    @classmethod
    def from_env(cls) -> CasperAPI:
        """Create a client from API_URL and TEAM_TOKEN environment variables."""
        base_url = os.environ.get("API_URL")
        token = os.environ.get("TEAM_TOKEN")

        if not base_url:
            raise EnvironmentError("API_URL is not set. Check your .env file.")
        if not token:
            raise EnvironmentError("TEAM_TOKEN is not set. Check your .env file.")

        return cls(base_url=base_url, token=token)

    async def get_feed(self) -> Feed:
        """Get LiveKit credentials for the current round.

        Returns:
            Feed with livekit_url, token, and round_id.

        Raises:
            NoActiveRound: If no round is currently active (404).
        """
        resp = await self._client.get("/feed")

        if resp.status_code == 404:
            raise NoActiveRound()

        resp.raise_for_status()
        return Feed.model_validate(resp.json())

    async def guess(self, answer: str) -> GuessResult:
        """Submit a guess for the current round.

        Args:
            answer: The text guess to submit.

        Returns:
            GuessResult indicating whether the guess was correct.
            - 200 OK → correct = True  (you got it!)
            - 400 Bad Request → correct = False (keep trying)
        """
        resp = await self._client.post("/guess", json={"answer": answer})

        if resp.status_code == 200:
            data = resp.json() if resp.text else {}
            return GuessResult(
                correct=True,
                guess_number=data.get("guess_number"),
            )

        if resp.status_code == 400:
            data = resp.json() if resp.text else {}
            return GuessResult(
                correct=False,
                guess_number=data.get("guess_number"),
            )

        if resp.status_code == 404:
            raise NoActiveRound()

        resp.raise_for_status()
        # Unreachable but keeps type checker happy
        return GuessResult(correct=False)  # pragma: no cover

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
