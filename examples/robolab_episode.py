"""Calibrate, probe and record one RoboLab episode inside the GPU container."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "deploy" / "robolab"


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def verdict(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("passed") is not True:
        raise ValueError(f"qualification failed; inspect {path}")
    return result


def probe_command(out: Path, offset: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "check.py"),
        "--task",
        "BananaInBowlTask",
        "--wrist-calibration",
        "--wrist-visual-fiducial",
        "--fiducial-position",
        "0.45",
        "0.1",
        "0.15",
        "--tcp-offset-m",
        offset,
        "--out",
        str(out / "probe.json"),
        "--headless",
    ]


def episode_command(out: Path, task: str, dollars: float) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "run.py"),
        "--task",
        task,
        "--num-runs",
        "1",
        "--num-envs",
        "1",
        "--interface",
        "direct",
        "--campaign-profile",
        "frontier-capability-dev",
        "--model",
        "gemini-4-argon",
        "--effort",
        "medium",
        "--model-max-tokens",
        "8192",
        "--model-timeout-s",
        "900",
        "--direct-prices",
        str(ROOT / "configs/direct-gemini-4-argon-prices.json"),
        "--motion-tools",
        "both",
        "--max-translation-m",
        "0.35",
        "--max-rotation-rad",
        "1.57",
        "--max-speed-mps",
        "0.3",
        "--max-ang-speed-rps",
        "1.5",
        "--wrist-calibration",
        "--table-projection",
        "--stereo-triangulation",
        "--camera-geometry-prompt",
        "--recent-frames",
        "--history-steps",
        "14,7",
        "--right-shoulder",
        "--image-scale",
        "0.3",
        "--action-sequence",
        "--max-sequence-steps",
        "15",
        "--done-awaits-confirmation",
        "--time-mode",
        "paused",
        "--native-horizon",
        "--max-decisions",
        "180",
        "--max-dollars",
        str(dollars),
        "--probe",
        str(out / "probe.json"),
        "--calibration",
        str(out / "calibration/calibration.json"),
        "--out",
        str(out / "episode"),
        "--headless",
        "--kit_args=--/UJITSO/enabled=false",
    ]


def episode_result(out: Path, task: str) -> dict:
    status_path = out / "episode/status" / f"{task}.json"
    status = json.loads(status_path.read_text())
    if (
        status.get("state") != "complete"
        or status.get("completed_episodes") != 1
        or status.get("expected_episodes") != 1
    ):
        raise ValueError(f"episode did not complete; inspect {status_path}")
    rows = [
        json.loads(line)
        for line in (out / "episode/episode_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("success"), bool):
        raise ValueError("expected one native result with a boolean success grade")
    recordings = list((out / "episode").rglob("*.hdf5"))
    if not recordings or any(path.stat().st_size == 0 for path in recordings):
        raise ValueError("episode result exists but native HDF5 recording is missing or empty")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="fresh mounted output directory")
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--max-dollars", type=positive_float, default=1.00)
    parser.add_argument("--qualify-only", action="store_true", help="no model call")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*Task", args.task):
        parser.error("--task must be a RoboLab task name, for example BananaInBowlTask")
    out = args.out.resolve()
    calibrate = [
        sys.executable,
        str(SCRIPTS / "calibrate_tcp.py"),
        "--out",
        str(out / "calibration"),
        "--headless",
    ]
    if args.dry_run:
        for command in [calibrate, probe_command(out, "<calibrated-tcp-offset>")]:
            print(shlex.join(command))
        if not args.qualify_only:
            print(shlex.join(episode_command(out, args.task, args.max_dollars)))
        return 0
    if out.exists():
        parser.error("--out must be a fresh directory")
    if not args.qualify_only and not os.environ.get("GEMINI_API_KEY"):
        parser.error("set GEMINI_API_KEY; use --qualify-only to test without paid model calls")
    cwd = Path(os.environ.get("ROBOLAB_ROOT", "/opt/robolab"))
    if not cwd.is_dir():
        parser.error("run inside the GPU container, or set ROBOLAB_ROOT to your RoboLab checkout")
    out.mkdir(parents=True)
    try:
        print("Calibrating TCP offset...", flush=True)
        subprocess.run(calibrate, cwd=cwd, check=True)
        calibration = verdict(out / "calibration/calibration.json")
        offset = float(calibration["tcp_offset_m"])
        if not math.isfinite(offset) or not 0.06 < offset < 0.20:
            raise ValueError("TCP offset outside the gripper envelope")
        print("Checking adapter and cameras...", flush=True)
        subprocess.run(probe_command(out, str(offset)), cwd=cwd, check=True)
        verdict(out / "probe.json")
        if args.qualify_only:
            print(f"Qualification passed. Evidence: {out}")
            return 0
        print("Running one episode. Simulation pauses during model inference.", flush=True)
        subprocess.run(episode_command(out, args.task, args.max_dollars), cwd=cwd, check=True)
        result = episode_result(out, args.task)
        print(f"Native task success: {result['success']}")
        print(f"Results, videos, model traces and HDF5 recordings: {out / 'episode'}")
        return 0 if result["success"] else 1
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Run incomplete: {exc}\nEvidence remains at {out}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
