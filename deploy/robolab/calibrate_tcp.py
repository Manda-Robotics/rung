#!/usr/bin/env python3
"""Simulation-only TCP offset calibration by table contact, from flange poses only.

Each trial descends onto an observed flat, clear table patch, stalls on contact, and must
lift away again. Task termination invalidates a trial. The result is diagnostic ground
truth, never policy input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = args.enable_cameras = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = {"passed": False, "trials": []}
    app = AppLauncher(args).app
    env = writer = None
    try:
        import cv2
        import numpy as np
        import torch
        from registration import register
        from robolab.constants import set_output_dir
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env

        from rung.adapters.robolab.translate import (
            action_chunk,
            capabilities_from_observation,
            to_observation,
        )
        from rung.contracts.frames import Pose, depth_image_to_points, quat_angle_between
        from rung.grasp.analytic import top_down_pose

        set_output_dir(str(out))
        register("BananaInBowlTask")
        cfg = parse_env_cfg(get_envs(task=["BananaInBowlTask"])[0], device="cuda:0", num_envs=1)
        cfg.episode_length_s = 600.0
        env, _ = create_env(cfg, device="cuda:0", num_envs=1)
        raw, _ = env.reset()
        caps = capabilities_from_observation(raw, tcp_offset_m=0.0)
        steps = 0

        def observation():
            return to_observation(raw, t=steps / 15, capabilities=caps, convention="opengl")

        def step(target):
            nonlocal raw, steps, writer
            if env.all_terminated:
                raise RuntimeError(f"calibration episode terminated: {env.get_env_results()}")
            current = observation()
            command = action_chunk(
                current.tcp,
                target,
                grip=0,
                horizon=1,
                max_dxyz=0.006,
                max_drot=0.035,
                tcp_offset_m=0,
            )[0]
            raw, *_ = env.step(torch.as_tensor(command[None], device=env.device))
            steps += 1
            current = observation()
            rgb = current.scene.rgb
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "contact.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    15,
                    (rgb.shape[1], rgb.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("video writer failed")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if env.all_terminated:
                raise RuntimeError("trial froze while measuring contact")
            return current

        for x, y, yaw in ((0.45, -0.28, 0.0), (0.53, -0.28, np.pi / 2)):
            env.reset_eval_state()
            raw, _ = env.reset()
            steps = 0
            start = observation()
            cloud, _ = depth_image_to_points(
                start.scene.depth, start.scene.K, start.scene.pose, stride=2
            )
            patch = cloud[(abs(cloud[:, 0] - x) < 0.045) & (abs(cloud[:, 1] - y) < 0.045)]
            if len(patch) < 40 or np.ptp(np.percentile(patch[:, 2], [10, 90])) > 0.008:
                raise RuntimeError("contact location is not a visible, flat table patch")
            table_z = float(np.median(patch[:, 2]))
            target = top_down_pose([x, y, 0.30], yaw)
            for _ in range(400):
                current = step(target)
                if (
                    np.linalg.norm(current.tcp.pos - target.pos) < 0.002
                    and quat_angle_between(current.tcp.quat, target.quat) < 0.02
                ):
                    break
            else:
                raise RuntimeError("failed to reach calibration approach")
            history = []
            contact = None
            for _ in range(400):
                # Continuous small downward demand; bound penetration and total steps.
                target = Pose(current.tcp.pos + [0, 0, -0.006], target.quat)
                current = step(target)
                history.append(current.tcp.pos.tolist())
                if current.tcp.pos[2] < table_z + 0.06:
                    raise RuntimeError("no table contact within the expected gripper envelope")
                if len(history) >= 25:
                    movement = np.linalg.norm(np.array(history[-1]) - history[-25])
                    if movement < 0.0005:
                        contact = current.tcp.pos.copy()
                        break
            if contact is None:
                raise RuntimeError("no stationary contact measured")
            target = top_down_pose(contact + [0, 0, 0.05], yaw)
            for _ in range(100):
                current = step(target)
                if np.linalg.norm(current.tcp.pos - target.pos) < 0.003:
                    break
            recovered = float(current.tcp.pos[2] - contact[2]) > 0.04
            trial = dict(
                x=x,
                y=y,
                yaw_rad=float(yaw),
                table_z=table_z,
                contact_flange=contact.tolist(),
                offset_m=float(contact[2] - table_z),
                recovered=recovered,
                terminated=env.all_terminated,
                steps=steps,
                descent=history,
            )
            status["trials"].append(trial)
            print(
                "CONTACT " + json.dumps({k: v for k, v in trial.items() if k != "descent"}),
                flush=True,
            )
            (out / "calibration.json").write_text(json.dumps(status, indent=2))
            if not recovered:
                raise RuntimeError("arm did not lift away: stall is not a valid contact reference")
        values = [t["offset_m"] for t in status["trials"]]
        status["tcp_offset_m"] = float(np.median(values))
        status["spread_m"] = float(np.ptp(values))
        status["passed"] = status["spread_m"] < 0.003
        status["method"] = "open-finger table contact, two locations/yaws, lift-away verified"
    except BaseException:
        import traceback

        status["error"] = traceback.format_exc()
        print(status["error"], flush=True)
    finally:
        if writer:
            writer.release()
        (out / "calibration.json").write_text(json.dumps(status, indent=2))
        print(
            "CALIBRATION " + json.dumps({k: v for k, v in status.items() if k != "trials"}),
            flush=True,
        )
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
