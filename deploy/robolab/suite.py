#!/usr/bin/env python3
"""Run a task manifest sequentially, isolating Isaac crashes per task.

Unknown arguments are forwarded to run.py. A task that exits zero without a matching
complete status JSON counts as failed. --resume extends completed tasks. Uses only the
standard library, so it can run outside the simulator interpreter.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from run import write_json


def load_tasks(path):
    tasks = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    tasks = [task for task in tasks if task]
    if not tasks or any(not re.fullmatch(r"[A-Za-z0-9_]+Task", t) for t in tasks):
        raise ValueError("manifest must contain one task class name per line")
    if len(set(tasks)) != len(tasks):
        raise ValueError("manifest contains duplicate tasks")
    return tasks


def complete_status(path, expected):
    try:
        status = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return (
        status.get("state") == "complete"
        and status.get("completed_episodes") == expected
        and status.get("expected_episodes") == expected
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-dollars", type=float, default=0.60)
    parser.add_argument("--max-total-dollars", type=float, default=10.0)
    parser.add_argument("--task-timeout-s", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args, forwarded = parser.parse_known_args()
    tasks = load_tasks(args.tasks_file)
    if min(args.episodes, args.num_envs, args.task_timeout_s) <= 0:
        parser.error("episodes, num-envs and task timeout must be positive")
    if args.max_dollars < 0 or args.max_total_dollars < 0:
        parser.error("dollar limits must be non-negative")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    marker = out / "run_complete.json"
    if marker.exists():
        if not args.resume:
            parser.error("completed suite exists; use --resume to extend it")
        marker.rename(out / "previous_run_complete.json")
    manifest = dict(
        tasks=tasks,
        episodes_per_env=args.episodes,
        num_envs=args.num_envs,
        max_total_dollars=args.max_total_dollars,
        forwarded=forwarded,
    )
    write_json(out / "suite_manifest.json", manifest)
    outcomes = {}
    for task in tasks:
        rows_path = out / "episode_results.jsonl"
        rows = (
            [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            if rows_path.exists()
            else []
        )
        spent = sum(r.get("rung", {}).get("usage", {}).get("dollars", 0.0) for r in rows)
        already = sum(r.get("task_name") == task for r in rows) if args.resume else 0
        remaining = max(0, args.episodes * args.num_envs - already)
        if spent + remaining * args.max_dollars > args.max_total_dollars + 1e-9:
            outcomes[task] = "budget_not_reserved"
            continue
        command = [
            args.python,
            str(Path(__file__).with_name("run.py")),
            "--task",
            task,
            "--out",
            str(out),
            "--episodes",
            str(args.episodes),
            "--num-envs",
            str(args.num_envs),
            "--max-dollars",
            str(args.max_dollars),
            *forwarded,
        ]
        if args.resume:
            command.append("--resume")
        print(f"START {task}", flush=True)
        with (out / "logs" / f"{task}.log").open("a") as log:
            try:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.task_timeout_s
                )
                complete = result.returncode == 0 and complete_status(
                    out / "status" / f"{task}.json", args.episodes * args.num_envs
                )
                outcomes[task] = "complete" if complete else "failed"
            except subprocess.TimeoutExpired:
                outcomes[task] = "timeout"
        write_json(out / "suite_status.json", dict(outcomes=outcomes))
        print(f"{outcomes[task].upper()} {task}", flush=True)
    passed = all(outcomes.get(t) == "complete" for t in tasks)
    write_json(
        out / "suite_status.json",
        dict(state="complete" if passed else "incomplete", outcomes=outcomes),
    )
    if passed:
        write_json(
            marker, dict(tasks=len(tasks), episodes=len(tasks) * args.episodes * args.num_envs)
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
