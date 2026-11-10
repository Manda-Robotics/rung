"""Run a scripted pick-and-place episode. No GPU, simulator or API key required."""

from __future__ import annotations

import argparse
from pathlib import Path

from rung.contracts.trace import TraceWriter
from rung.loop import Limits, run_episode
from rung.mock import SCENES
from rung.record import Recorder
from rung.reference import ScriptedReference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="optional fresh directory for trace and images")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error("--out must be a fresh directory")

    embodiment = SCENES["banana_in_bowl"]()
    trace = TraceWriter(args.out / "trace.jsonl" if args.out else None)
    result = run_episode(
        embodiment,
        ScriptedReference(embodiment),
        task="banana_in_bowl",
        limits=Limits(horizon_s=120, max_decisions=20, max_dollars=0.10),
        trace=trace,
        recorder=Recorder(args.out / "episodes") if args.out else None,
        episode_seed=0,
    )
    print(result.summary())
    print(f"Recorded {len(trace.events)} trace events; no model calls were made.")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
