"""Main application entrypoint for the Casper starter workspace.

This module centralizes the runtime orchestration for both:
- practice mode: local camera -> analysis
- live mode: game API -> LiveKit stream -> analysis -> guess submission

Run with:
    uv run -m core.app --practice
    uv run -m core.app --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from core.practice import start_practice
from core.stream import start_stream

_JUDGE_UNAVAILABLE_BACKOFF_CAP_S = 30.0
_MAX_JUDGE_UNAVAILABLE_RETRIES = 5
_JUDGE_UNAVAILABLE_BACKOFF_S = 1.0


def parse_args(
    argv: list[str] | None = None,
    *,
    prog: str = "core.app",
) -> argparse.Namespace:
    """Parse CLI arguments for practice and live modes."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Casper guessing game application",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--practice",
        action="store_true",
        help="Use local camera for offline development",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Connect to a live game round",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for practice mode (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to sample (default: 1)",
    )
    return parser.parse_args(argv)


async def run_practice(camera: int, fps: int) -> None:
    """Run the practice pipeline: core capture -> agent analysis."""
    from agent.prompt import analyze

    print("=" * 50)
    print("  PRACTICE MODE")
    print("  Local camera - no network required")
    print("=" * 50)
    print()

    async for frame in start_practice(camera_index=camera, fps=fps):
        guess = await analyze(frame)
        if guess:
            print(f"  [guess] {guess}")
        else:
            print("  [skip]  No guess this frame")


async def run_live() -> None:
    """Run the live pipeline: api feed -> core stream -> agent -> api guess."""
    from api import (
        CasperAPI,
        JudgeUnavailable,
        MaxGuessesReached,
        NoActiveRound,
        Unauthorized,
    )
    from agent.prompt import analyze

    print("=" * 50)
    print("  LIVE MODE")
    print("  Connecting to game server...")
    print("=" * 50)
    print()

    client = CasperAPI.from_env()

    try:
        feed = await client.get_feed()
    except Unauthorized:
        print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
        sys.exit(1)
    except NoActiveRound:
        print("[!] No active round. Wait for the admin to start one.")
        sys.exit(1)
    except Exception as exc:
        print(f"[!] Could not connect to game server: {exc}")
        sys.exit(1)

    print(f"[+] Joined round: {feed.round_id}")
    print(f"[+] LiveKit URL:  {feed.livekit_url}")
    print()

    guess_count = 0

    try:
        async for frame in start_stream(feed.livekit_url, feed.token):
            guess = await analyze(frame)

            if guess:
                result = None
                n_503 = 0
                try:
                    while True:
                        try:
                            result = await client.guess(guess)
                            break
                        except JudgeUnavailable:
                            if n_503 >= _MAX_JUDGE_UNAVAILABLE_RETRIES:
                                break
                            delay = min(
                                _JUDGE_UNAVAILABLE_BACKOFF_S * (2**n_503),
                                _JUDGE_UNAVAILABLE_BACKOFF_CAP_S,
                            )
                            await asyncio.sleep(delay)
                            n_503 += 1
                except Unauthorized:
                    print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
                    break
                except NoActiveRound:
                    print("[!] No active round (round may have ended).")
                    break
                except MaxGuessesReached:
                    print("[!] Maximum guesses reached for this round.")
                    break

                if result is None:
                    attempts = 1 + _MAX_JUDGE_UNAVAILABLE_RETRIES
                    print(
                        f"[!] Judge unavailable (503) after {attempts} attempt(s). "
                        "Skipping this guess; will try again on the next frame."
                    )
                    continue

                guess_count += 1
                id_suffix = f" id={result.guess_id}" if result.guess_id is not None else ""
                print(f"  [guess #{guess_count}{id_suffix}] {guess}")

                if result.correct:
                    print()
                    print("=" * 50)
                    print(f"  CORRECT! Solved in {guess_count} guesses.")
                    print("=" * 50)
                    break
            else:
                print("  [skip] No guess this frame")

    except (KeyboardInterrupt, ConnectionError):
        print("\n[!] Disconnected from stream.")
    finally:
        await client.close()


async def main(
    argv: list[str] | None = None,
    *,
    prog: str = "core.app",
) -> None:
    """Load environment configuration and run the selected mode."""
    load_dotenv()
    args = parse_args(argv, prog=prog)

    if args.practice:
        await run_practice(camera=args.camera, fps=args.fps)
    else:
        await run_live()


def run(
    argv: list[str] | None = None,
    *,
    prog: str = "core.app",
) -> None:
    """Synchronous wrapper for the main async application entrypoint."""
    try:
        asyncio.run(main(argv, prog=prog))
    except KeyboardInterrupt:
        print("\nBye!")
        os._exit(0)


if __name__ == "__main__":
    run()
