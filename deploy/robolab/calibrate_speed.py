#!/usr/bin/env python3
"""Measure closed-loop arrival speed and tracking quality at several tracking gains.

The DROID rel-IK controller realises about 28 percent of each commanded Cartesian step
(TRACKING_GAIN); uncompensated, a 26 cm approach takes about 8 s of robot time. This
drives the bridge's closed loop, one re-plan per control step at the executor's speed, and
records per gain: control steps to arrive, overshoot past the target, drift while holding
afterwards, and the realised share of each planned step. A gain is usable if it arrives
faster without overshoot or ringing.

    PYTHONPATH=/path/to/rung/src <robolab>/.venv/bin/python calibrate_speed.py \\
        --task BananaInBowlTask --gains 1.0,0.5,0.35,0.283 --out speed.json
"""

from __future__ import annotations

import argparse
import json

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--gains", default="1.0,0.6,0.45,0.35,0.283")
parser.add_argument("--tcp-offset-m", type=float, default=0.14883)
parser.add_argument("--lin-speed", type=float, default=0.25)
parser.add_argument("--ang-speed", type=float, default=1.0)
parser.add_argument("--budget", type=int, default=240)
parser.add_argument("--out", default="/workspace/rung_speed.json")
parser.add_argument("--renderer", default="realtime")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from registration import register  # noqa: E402

from rung.adapters.robolab.translate import (  # noqa: E402
    CONTROL_HZ,
    SCENE_CAMERA,
    action_chunk,
    capabilities_from_observation,
    steps_needed,
    to_observation,
)
from rung.contracts.frames import (  # noqa: E402
    Pose,
    quat_angle_between,
    quat_from_axis_angle,
    quat_mul,
)

ARRIVED_M, ARRIVED_RAD = 0.006, 0.05


def main() -> int:
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env

    register(args.task)
    env = create_env(
        scene=get_envs(task=[args.task])[0], device="cuda:0", num_envs=1, renderer=args.renderer
    )
    if isinstance(env, tuple):
        env = env[0]
    dt = 1.0 / CONTROL_HZ
    results = {
        "task": args.task,
        "lin_speed": args.lin_speed,
        "ang_speed": args.ang_speed,
        "gains": {},
    }

    def observe(raw, caps):
        return to_observation(
            raw, t=0.0, env_id=0, camera=SCENE_CAMERA, capabilities=caps, convention="opengl"
        )

    def step(action):
        raw, *_ = env.step(
            torch.as_tensor(np.asarray(action, dtype=np.float32)[None], device=env.device)
        )
        return raw

    # Reset once. A second env.reset() in this loop leaves the arm unresponsive to later
    # commands, which is indistinguishable from a gain that stalls the controller. Each
    # gain's move sequence ends at home instead.
    raw, _ = env.reset()
    caps = capabilities_from_observation(raw, tcp_offset_m=args.tcp_offset_m)
    for _ in range(15):
        raw = step(np.zeros(7))
    home = observe(raw, caps).tcp
    for gain in [float(g) for g in args.gains.split(",")]:
        yawed = quat_mul(quat_from_axis_angle([0, 0, 1], 0.8), home.quat)
        moves = [
            ("down 8 cm", Pose(home.pos + [0, 0, -0.08], home.quat)),
            ("across 25 cm", Pose(home.pos + [0, 0.25, -0.08], home.quat)),
            ("diagonal 25 cm + 0.8 rad yaw", Pose(home.pos + [0.10, 0.05, 0.02], yawed)),
            ("back home", home),
        ]
        rows = []
        for name, target in moves:
            obs = observe(raw, caps)
            start = obs.tcp
            direction = target.pos - start.pos
            length = float(np.linalg.norm(direction))
            unit = direction / length if length > 1e-9 else np.zeros(3)
            planned = steps_needed(
                start,
                target,
                args.lin_speed * dt,
                args.ang_speed * dt,
                tcp_offset_m=args.tcp_offset_m,
            )
            track, arrived_at, overshoot, shares = [], None, 0.0, []
            prev = start.pos.copy()
            for i in range(args.budget):
                now = obs.tcp
                n = max(
                    1,
                    steps_needed(
                        now,
                        target,
                        args.lin_speed * dt,
                        args.ang_speed * dt,
                        tcp_offset_m=args.tcp_offset_m,
                    ),
                )
                command = action_chunk(
                    now,
                    target,
                    grip=0.0,
                    horizon=n,
                    max_dxyz=args.lin_speed * dt,
                    max_drot=args.ang_speed * dt,
                    tcp_offset_m=args.tcp_offset_m,
                    gain=gain,
                )[0]
                gap = float(np.linalg.norm(target.pos - now.pos))
                planned_travel = min(args.lin_speed * dt, gap)
                raw = step(command)
                obs = observe(raw, caps)
                moved = float(np.linalg.norm(obs.tcp.pos - prev))
                if planned_travel > 0.005 and n > 2:
                    shares.append(moved / planned_travel)
                prev = obs.tcp.pos.copy()
                along = float(np.dot(obs.tcp.pos - start.pos, unit))
                overshoot = max(overshoot, along - length)
                err = float(np.linalg.norm(obs.tcp.pos - target.pos))
                rot = quat_angle_between(obs.tcp.quat, target.quat)
                track.append([round(err, 5), round(rot, 4)])
                if arrived_at is None and err <= ARRIVED_M and rot <= ARRIVED_RAD:
                    arrived_at = i + 1
                    break
            hold_start = obs.tcp.pos.copy()
            peak_drift = 0.0
            for _ in range(15):
                raw = step(np.zeros(7))
                obs = observe(raw, caps)
                peak_drift = max(peak_drift, float(np.linalg.norm(obs.tcp.pos - hold_start)))
            final_err = float(np.linalg.norm(obs.tcp.pos - target.pos))
            row = dict(
                move=name,
                distance_m=round(length, 4),
                planned_steps=planned,
                arrived_steps=arrived_at,
                arrived_s=None if arrived_at is None else round(arrived_at * dt, 2),
                overshoot_m=round(overshoot, 4),
                drift_after_hold_m=round(peak_drift, 4),
                error_after_hold_m=round(final_err, 4),
                realised_share_median=round(float(np.median(shares)), 3) if shares else None,
                track=track,
            )
            rows.append(row)
            print(
                f"gain {gain:.3f} {name:32s} planned {planned:3d} arrived "
                f"{arrived_at} overshoot {overshoot * 100:.1f} cm drift {peak_drift * 100:.1f} cm "
                f"after-hold err {final_err * 100:.1f} cm share {row['realised_share_median']}",
                flush=True,
            )
        results["gains"][str(gain)] = rows
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    raise SystemExit(code)
