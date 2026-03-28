"""HTTP client SDK for the Casper guessing game API.

`api` owns live-mode transport to the game server. It fetches feed
credentials and submits guesses, but it does not capture frames or run
analysis.
"""

from api.client import CasperAPI
from api.models import (
    Feed,
    GuessResult,
    JudgeUnavailable,
    MaxGuessesReached,
    NoActiveRound,
    Unauthorized,
)

__all__ = [
    "CasperAPI",
    "Feed",
    "GuessResult",
    "JudgeUnavailable",
    "MaxGuessesReached",
    "NoActiveRound",
    "Unauthorized",
]
