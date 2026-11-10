"""Connect a local mock environment to a Rung API and keep the episode locally."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("RUNG_API_URL"))
    parser.add_argument("--out", required=True, type=Path, help="fresh local recording directory")
    parser.add_argument("--instruction", default="put the banana in the bowl")
    parser.add_argument("--max-usd", type=positive_float, default=0.50)
    parser.add_argument("--max-model-calls", type=positive_int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.url:
        parser.error("set RUNG_API_URL or pass --url with your provisioned endpoint")
    url = urlsplit(args.url)
    if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password:
        parser.error("--url must be a ws/wss endpoint without embedded credentials")
    if url.scheme == "ws" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("use wss for remote endpoints; ws is supported only for loopback testing")
    key = os.environ.get("RUNG_API_KEY")
    if not key:
        parser.error("set RUNG_API_KEY; it is sent in the authorization header")
    if args.out.exists():
        parser.error("--out must be a fresh directory")

    try:
        from rung.api.client import RungApiClient, run_remote_episode
        from rung.mock import SCENES
    except ModuleNotFoundError as exc:
        parser.error(
            f"missing dependency {exc.name}; install the api extra: pip install -e '.[api]'"
        )

    print("Connecting. Each model decision may take seconds to minutes.", flush=True)
    try:
        with RungApiClient(args.url, key) as client:
            if client.metadata.get("backend") == "fake":
                print("Server reports a fake backend: token usage and costs below are synthetic.")
            result = run_remote_episode(
                client,
                SCENES["banana_in_bowl"](),
                args.instruction,
                limits={
                    "max_usd": args.max_usd,
                    "max_model_calls": args.max_model_calls,
                    "horizon_s": 120,
                },
                episode_seed=args.seed,
                record_dir=args.out,
            )
    except Exception as exc:
        print(
            f"Run failed ({type(exc).__name__}). Any saved partial recording remains at {args.out}."
        )
        return 2
    print(f"Caller grade: {result.get('success')}; end reason: {result.get('end_reason')}")
    print(f"Targets executed: {result['targets_executed']}; usage: {result.get('usage', {})}")
    print(f"Recording: {args.out}")
    print(f"Inspect: python examples/inspect_recording.py {args.out}")
    return 0 if result.get("success") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
