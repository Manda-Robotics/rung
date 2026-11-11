#!/usr/bin/env python3
"""Run episodes on the mock tabletop: one scene, one interface, one model.

    # scripted reference, no model or API key
    rung --scene banana_in_bowl --policy scripted

    # L2: the model calls semantic primitives
    rung --scene cluttered_bin --policy l2 --model gemini-3.7-flash

    # L0: the same model emits TCP pose targets
    rung --scene cluttered_bin --policy l0 --model gemini-3.7-flash

    # Direct: the model moves the gripper through tools, as on the RoboLab leaderboard.
    # --model fake replays a fixed tool-call script, so no API key is needed.
    rung --scene banana_in_bowl --policy direct --model fake
    rung --scene banana_in_bowl --policy direct --model gemini-4-argon \\
        --profile configs/gemini-4-argon-direct.json

    # every scene, three episodes each, with traces, recordings and provenance
    rung --all --episodes 3 --policy l2 --model gemini-3.7-flash \\
        --out runs/gemini-l2

With --out, a run writes trace.jsonl, per-episode recordings, results.json and
provenance.json. See docs/protocol.md for how results are interpreted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rung import __version__
from rung.backends.prices import VERSION as PRICES_VERSION
from rung.contracts.trace import RunProvenance, TraceWriter
from rung.direct_profile import DirectProfile, build_direct_policy, load_profile
from rung.executor import Executor
from rung.factory import VENDORS, build_backend, build_grounding
from rung.grasp import AnalyticGraspPlanner
from rung.interfaces import L0Config, L0Policy, L2Config, L2Policy
from rung.loop import EpisodeResult, Limits, run_episode
from rung.mock import SCENES
from rung.perception import ObjectMemory, OracleGrounding, PerceptionService
from rung.record import Recorder
from rung.reference import RandomReference, ScriptedReference

BACKENDS = VENDORS


def build_policy(args, embodiment):
    if args.policy == "scripted":
        return ScriptedReference(embodiment)
    if args.policy == "random":
        return RandomReference(embodiment, seed=args.seed or 0)
    if args.policy == "l0":
        if not args.model:
            raise SystemExit("--policy l0 needs --model")
        return L0Policy(
            build_backend(args.model, args.vendor),
            L0Config(effort=args.effort, send_wrist=not args.no_wrist),
        )
    if args.policy == "l2":
        if not args.model:
            raise SystemExit("--policy l2 needs --model")
        grounding = (
            build_grounding(args.grounding_model)
            if args.grounding_model
            else OracleGrounding(embodiment)
        )
        executor = Executor(
            embodiment,
            PerceptionService(grounding),
            AnalyticGraspPlanner(embodiment.capabilities),
            ObjectMemory(),
        )
        return L2Policy(
            build_backend(args.model, args.vendor),
            executor,
            L2Config(effort=args.effort, send_wrist=not args.no_wrist),
        )
    if args.policy == "direct":
        if not args.model:
            raise SystemExit("--policy direct needs --model (fake needs no API key)")
        profile = load_profile(args.profile) if args.profile else DirectProfile()
        profile.model = args.model
        if args.effort:
            profile.effort = args.effort
        return build_direct_policy(profile)
    raise SystemExit(f"unknown policy {args.policy!r}")


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", default="banana_in_bowl", choices=sorted(SCENES))
    ap.add_argument("--all", action="store_true", help="every scene")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument(
        "--policy",
        default="scripted",
        choices=["scripted", "random", "l0", "l2", "direct"],
        help="which rung: direct moves the gripper through tools over a native Gemini "
        "session, l0 names poses and does its own metric reasoning, l2 names "
        "objects and primitives and the harness supplies the geometry. scripted and "
        "random are references, not policies under test.",
    )
    ap.add_argument("--model", help="model id; with --policy direct, 'fake' needs no key")
    ap.add_argument(
        "--profile",
        help="direct only: JSON profile with model settings and DirectConfig fields",
    )
    ap.add_argument("--vendor", choices=sorted(BACKENDS))
    ap.add_argument("--effort", help="low, medium, high; swept on validation tasks only")
    ap.add_argument(
        "--grounding-model",
        help="ask this model for object boxes instead of using the mock's ground truth. "
        "Held fixed across acting models in a controlled comparison. Note that the mock's "
        "objects are featureless boxes, so this tests the plumbing, not the grounding.",
    )
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="directory for traces, recordings and provenance")
    ap.add_argument("--max-decisions", type=int, default=25)
    ap.add_argument("--max-dollars", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=180.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes must be at least 1")

    scenes = sorted(SCENES) if args.all else [args.scene]
    limits = Limits(
        horizon_s=args.horizon,
        max_decisions=args.max_decisions,
        max_dollars=args.max_dollars,
    )
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out = Path(args.out) if args.out else None
    trace = TraceWriter(out / "trace.jsonl" if out else None)
    recorder = Recorder(out / "episodes") if out else None

    results: list[EpisodeResult] = []
    for scene in scenes:
        for episode in range(args.episodes):
            embodiment = SCENES[scene]()
            try:
                policy = build_policy(args, embodiment)
            except ValueError as exc:
                if "credential" in str(exc).lower() or "api key" in str(exc).lower():
                    print(
                        f"rung: {str(exc).rstrip('.')}. Set the model's API key "
                        "(for example GEMINI_API_KEY), or use --policy scripted or "
                        "--policy direct --model fake.",
                        file=sys.stderr,
                    )
                    return 2
                raise
            seed = args.seed if args.seed is not None else (episode or None)
            result = run_episode(
                embodiment,
                policy,
                task=scene,
                limits=limits,
                trace=trace,
                recorder=recorder,
                episode_seed=seed,
            )
            results.append(result)
            if not args.quiet:
                print(result.summary())
                for note in result.notes:
                    print(f"    {note}")

    solved = sum(r.success for r in results)
    spent = sum(r.usage.dollars for r in results)
    decisions = sum(r.decisions for r in results)
    print(
        f"\n{solved}/{len(results)} episodes solved, {decisions} model decisions, "
        f"${spent:.3f}, {sum(r.wall_seconds for r in results):.0f} s wall"
    )

    if out:
        model = getattr(
            getattr(policy, "backend", None) or getattr(policy, "session", None), "model", None
        )
        RunProvenance(
            run_id=run_id,
            started_wall=time.time(),
            harness_revision=f"{__version__}+{git_revision()}",
            embodiment="mock-tabletop",
            embodiment_revision=None,
            interface=results[0].interface if results else args.policy,
            model=model,
            effort=args.effort,
            prompt_hashes={},
            perception_version=f"{PerceptionService.version}/{args.grounding_model or 'oracle'}",
            grasp_version=AnalyticGraspPlanner.version,
            task_role="development",
            limits=vars(limits),
            prices_version=PRICES_VERSION,
            notes={
                "scenes": scenes,
                "episodes_per_scene": args.episodes,
                "solved": solved,
                "dollars": spent,
            },
        ).write(out / "provenance.json")
        (out / "results.json").write_text(
            json.dumps([r.summary() for r in results], indent=2), encoding="utf-8"
        )
        print(f"wrote {out}")
    return 0 if solved == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
