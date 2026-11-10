#!/usr/bin/env python3
"""Probe the rung RoboLab adapter against a live Isaac environment.

Requires RoboLab and Isaac Sim, and covers what the offline tests cannot. Each check
targets an adapter fault that would otherwise look like a model failure: an unconverted
camera convention, the flange mistaken for the fingertips, depth in the wrong units, or
motions at the wrong scale.

Run once per tree and host before trusting any result, and keep the JSON output with the
run.

    PYTHONPATH=/path/to/rung/src <robolab>/.venv/bin/python check.py --task BananaInBowlTask

The harness goes on PYTHONPATH rather than being installed, because installing it would
replace the numpy version Isaac pins.
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument(
    "--tcp-offset-m",
    type=float,
    default=0.14883,
    help="base_link to the open finger plane, metres. Prefer the "
    "value calibrate_tcp.py measured on this rig.",
)
parser.add_argument("--out", default="/workspace/rung_check.json")
parser.add_argument("--renderer", default="realtime")
parser.add_argument("--wrist-calibration", action="store_true")
parser.add_argument(
    "--wrist-visual-fiducial",
    action="store_true",
    help="Qualification-only visual marker; never an evaluation episode",
)
parser.add_argument(
    "--fiducial-position",
    type=float,
    nargs=3,
    default=[0.45, 0.0, 0.02],
    metavar=("X", "Y", "Z"),
    help=(
        "Qualification-only visible marker centre in root metres. Select a point "
        "above the actual table, clear of task objects; never used by policies."
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

from rung.adapters.robolab.translate import (  # noqa: E402
    REL_IK_SCALE,
    action_chunk,
    capabilities_from_observation,
    detect_camera_convention,
    to_numpy,
    to_observation,
)
from rung.contracts.frames import (  # noqa: E402
    Pose,
    deproject,
    depth_image_to_points,
    project,
    quat_angle_between,
    quat_from_axis_angle,
    quat_mul,
    quat_to_rot,
)
from rung.contracts.validate import ContractError, validate_observation  # noqa: E402

SCENE_CAMERA = "over_shoulder_left_camera"
findings: list[dict] = []


def record(name: str, passed: bool, detail: str, measured=None, unmeasurable: bool = False) -> None:
    """Record one finding.

    ``unmeasurable`` is a third state beside pass and fail: the rig does not expose what
    the check needs, so the result says nothing about the adapter. A stated gap is
    preferable to a confident failure computed from a degenerate reference.
    """
    findings.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "measured": measured or {},
            "unmeasurable": bool(unmeasurable),
        }
    )
    mark = "n/a " if unmeasurable else ("pass" if passed else "FAIL")
    print(f"[{mark}] {name}: {detail}", flush=True)


from registration import register  # noqa: E402


def record_visual_fiducial(obs, phase):
    """Independent rendered marker centroid versus FK projection, not a round-trip test."""
    from pathlib import Path

    from PIL import Image, ImageDraw

    frame = obs.wrist
    uv, z = project(np.array([args.fiducial_position]), frame.K, frame.pose)
    rgb = frame.rgb.astype(float)
    mask = (rgb[:, :, 0] > 130) & (rgb[:, :, 2] > 110) & (rgb[:, :, 1] < 90)
    ys, xs = np.nonzero(mask)
    actual = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])
    error = float(np.linalg.norm(actual - uv[0]))
    path = Path(args.out).with_suffix(f".{phase}-fiducial.png")
    image = Image.fromarray(frame.rgb.copy())
    draw = ImageDraw.Draw(image)
    if np.isfinite(uv).all() and z[0] > 0:
        u, v = uv[0]
        draw.line((u - 12, v, u + 12, v), fill=(0, 255, 255), width=2)
        draw.line((u, v - 12, u, v + 12), fill=(0, 255, 255), width=2)
    image.save(path)
    record(
        f"wrist_rendered_fiducial_{phase}",
        len(xs) > 3 and error < 5,
        "Cyan predicted point / independently rendered magenta marker; qualification only",
        {
            "expected_uv": uv[0].tolist(),
            "actual_uv": actual.tolist() if np.isfinite(actual).all() else None,
            "pixel_error": error if np.isfinite(error) else None,
            "marker_pixels": len(xs),
            "image": str(path),
        },
    )


def main() -> int:
    if not np.isfinite(args.fiducial_position).all():
        raise ValueError("fiducial position must be finite")
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env

    register(args.task, wrist_calibration=args.wrist_calibration)
    envs = get_envs(task=[args.task])
    print("registered envs:", envs, flush=True)
    if not envs:
        record("registration", False, f"no environment registered for {args.task}")
        return 1
    record("registration", True, f"{len(envs)} env(s) registered for {args.task}")

    env = create_env(scene=envs[0], device="cuda:0", num_envs=1, renderer=args.renderer)
    if isinstance(env, tuple):
        env = env[0]
    if args.wrist_visual_fiducial:
        if not args.wrist_calibration:
            raise ValueError("fiducial probe requires wrist calibration")
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        sphere = UsdGeom.Sphere.Define(stage, "/World/RungQualificationVisualMarker")
        sphere.CreateRadiusAttr(0.007)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*args.fiducial_position))
        material = UsdShade.Material.Define(stage, "/World/RungQualificationMarkerMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/RungQualificationMarkerMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 1))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 1))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        # No rigid-body/collision API: qualification-only, absent from evaluation.
    raw, _ = env.reset()
    if args.wrist_visual_fiducial:
        import torch

        for _ in range(3):
            raw, *_ = env.step(torch.zeros((1, 7), device=env.device))

    groups = {k: (sorted(v) if isinstance(v, dict) else type(v).__name__) for k, v in raw.items()}
    print("observation groups:", json.dumps(groups, indent=2)[:1500], flush=True)

    images = raw.get("image_obs", {})
    have = [k for k in images if k.startswith(SCENE_CAMERA)]
    record(
        "depth_terms_present",
        f"{SCENE_CAMERA}_depth" in images,
        f"scene camera terms: {sorted(have)}",
    )

    caps = capabilities_from_observation(raw, tcp_offset_m=args.tcp_offset_m)

    # 1. Camera convention. It differs between trees (upstream on Isaac Sim 5 reports
    # OpenGL axes, the Isaac Sim 6 port ROS axes), so it is measured against a point the
    # camera must be viewing and recorded in the run's provenance.
    table_centre = np.array([0.45, 0.0, 0.0])
    try:
        convention, scores = detect_camera_convention(
            raw, looks_at=table_centre, env_id=0, camera=SCENE_CAMERA
        )
        record(
            "camera_convention_detected",
            convention is not None,
            f"measured {convention!r} from the extrinsics; scores "
            + ", ".join(f"{k}={v:+.2f}" for k, v in scores.items())
            + " (the right one is near +1, the wrong one near -1)",
            {"convention": convention, "scores": scores},
        )
    except Exception as exc:  # noqa: BLE001
        record("camera_convention_detected", False, f"{type(exc).__name__}: {exc}")
        convention = "opengl"
    convention = convention or "opengl"

    # 2. Does the translation produce a contract-valid observation?
    try:
        obs = to_observation(
            raw, t=0.0, env_id=0, camera=SCENE_CAMERA, capabilities=caps, convention=convention
        )
        validate_observation(obs, caps)
        record(
            "observation_translates",
            True,
            f"scene {obs.scene.width}x{obs.scene.height}, depth "
            f"{'yes' if obs.scene.depth is not None else 'no'}, tcp "
            f"{np.round(obs.tcp.pos, 3).tolist()}, travel {obs.gripper.travel:.2f}",
        )
    except (ContractError, KeyError, Exception) as exc:  # noqa: BLE001
        record("observation_translates", False, f"{type(exc).__name__}: {exc}")
        env.close()
        return 1

    wrist_to_tcp = None
    if args.wrist_calibration:
        wrist = obs.wrist
        valid = (
            wrist is not None
            and wrist.K is not None
            and wrist.pose is not None
            and np.isfinite(wrist.K).all()
            and wrist.K[0, 0] > 0
            and wrist.K[1, 1] > 0
        )
        record(
            "wrist_calibration_present",
            valid,
            "RGB wrist has public pose/intrinsics without a depth annotator",
        )
        if valid:
            wrist_to_tcp = wrist.pose.inverse().compose(obs.tcp)
            uv, z = project(obs.tcp.pos[None], wrist.K, wrist.pose)
            from pathlib import Path

            from PIL import Image, ImageDraw

            marker = Image.fromarray(wrist.rgb.copy())
            draw = ImageDraw.Draw(marker)
            u, v = uv[0]
            if np.isfinite(uv).all() and z[0] > 0:
                draw.line((u - 12, v, u + 12, v), fill=(255, 0, 255), width=3)
                draw.line((u, v - 12, u, v + 12), fill=(255, 0, 255), width=3)
            path = Path(args.out).with_suffix(".wrist-tcp.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            marker.save(path)
            record(
                "wrist_tcp_projection",
                bool(z[0] > 0 and np.isfinite(uv).all()),
                "Measured TCP projected into wrist; review saved physical marker image",
                {
                    "uv": uv[0].tolist(),
                    "optical_depth_m": float(z[0]),
                    "image": str(path),
                    "wrist_to_tcp_pos": wrist_to_tcp.pos.tolist(),
                },
            )
            if args.wrist_visual_fiducial:
                record_visual_fiducial(obs, "initial")

    # 3. With that convention applied, +z must point into the scene.
    forward = quat_to_rot(obs.scene.pose.quat) @ np.array([0.0, 0.0, 1.0])
    to_table = np.array([0.45, 0.0, 0.0]) - obs.scene.pose.pos
    to_table = to_table / np.linalg.norm(to_table)
    alignment = float(np.dot(forward, to_table))
    record(
        "camera_points_at_the_scene",
        alignment > 0.7,
        f"converted camera +z points at the table, alignment {alignment:+.2f}",
        {"alignment": alignment, "convention": convention, "cam_pos": obs.scene.pose.pos.tolist()},
    )

    # 4. Depth scale and extrinsics: the table must reconstruct near z = 0.
    pts, _ = depth_image_to_points(obs.scene.depth, obs.scene.K, obs.scene.pose, stride=4)
    finite = pts[np.isfinite(pts).all(axis=1)]
    near = finite[(np.abs(finite[:, 0] - 0.5) < 0.35) & (np.abs(finite[:, 1]) < 0.35)]
    if len(near) > 200:
        hist, edges = np.histogram(near[:, 2], bins=120, range=(-0.3, 0.9))
        peak = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
        share = float(np.mean(np.abs(near[:, 2] - peak) < 0.02))
        record(
            "depth_reconstructs_the_table",
            abs(peak) < 0.06,
            f"most common reconstructed height {peak:+.3f} m over the table area, "
            f"{share * 100:.0f} percent of points within 2 cm of it",
            {"peak_z": peak, "share": share, "n_points": int(len(near))},
        )
    else:
        record(
            "depth_reconstructs_the_table",
            False,
            f"only {len(near)} deprojected points over the table area",
        )

    # 5. The TCP must project into the image and deproject back to itself.
    uv, z = project(obs.tcp.pos[None], obs.scene.K, obs.scene.pose)
    u, v = float(uv[0, 0]), float(uv[0, 1])
    in_frame = 0 <= u < obs.scene.width and 0 <= v < obs.scene.height and z[0] > 0
    if in_frame:
        back = deproject(np.array([u, v]), float(z[0]), obs.scene.K, obs.scene.pose)
        err = float(np.linalg.norm(back - obs.tcp.pos))
        record(
            "tcp_projects_and_returns",
            err < 1e-6,
            f"the fingertips project to ({u:.0f}, {v:.0f}) and return within {err * 1000:.4f} mm",
            {"uv": [u, v], "depth": float(z[0])},
        )
    else:
        record(
            "tcp_projects_and_returns",
            False,
            f"the fingertips project to ({u:.1f}, {v:.1f}) depth {float(z[0]):.2f}, "
            f"outside the {obs.scene.width}x{obs.scene.height} image",
        )

    # 6. Derived TCP against an independent finger-body midpoint. Probe-only; never part
    # of the model's observation.
    #
    # The reference must be validated before use. On the DROID rig all nine Robotiq bodies
    # report base_link's pose, so their midpoint is base_link and the comparison returns
    # exactly the configured offset as the "error", which looks like a precise measurement.
    from rung.adapters.robolab.compat import isaaclab_compat

    compat = isaaclab_compat()
    as_torch, quat_isaaclab_to_wxyz = compat.as_torch, compat.quat_isaaclab_to_wxyz

    robot = env.scene["robot"]
    indices = [robot.body_names.index(name) for name in ("left_inner_finger", "right_inner_finger")]
    body_pos = to_numpy(robot.data.body_pos_w)[0, indices]
    root_pos = to_numpy(robot.data.root_pos_w)[0]
    root_quat = to_numpy(quat_isaaclab_to_wxyz(as_torch(robot.data.root_quat_w)))[0]
    jaw_separation = float(np.linalg.norm(body_pos[0] - body_pos[1]))
    midpoint = quat_to_rot(root_quat).T @ (body_pos.mean(axis=0) - root_pos)
    flange = to_numpy(raw["proprio_obs"]["ee_pos"])[0]
    local_offset = quat_to_rot(obs.tcp.quat).T @ (midpoint - flange)
    tcp_error = float(np.linalg.norm(midpoint - obs.tcp.pos))
    measured = {
        "error_m": tcp_error,
        "measured_offset_local_m": local_offset.tolist(),
        "configured_offset_m": args.tcp_offset_m,
        "jaw_separation_m": jaw_separation,
    }
    # An open 2F-85 holds its fingers about 8 cm apart. Coincident bodies mean the
    # articulation is not publishing them, not that the jaw is shut.
    if jaw_separation < 0.01:
        record(
            "tcp_matches_finger_midpoint",
            True,
            "cannot be measured on this rig: the left and right finger bodies are "
            f"{jaw_separation * 1000:.3f} mm apart, so the articulation is reporting "
            "base_link for every gripper body, and the USD stage holds authored rest "
            "poses because Fabric owns the live transforms. Use calibrate_tcp.py, "
            "which drives the open hand onto the table at two places and two yaws and "
            "verifies it can lift away again. Do not fall back to the Robotiq figure "
            "quoted in DroidRelIKActionCfg: that is the gripper's physical height, not "
            "the distance to the control point, and it put every grasp 5 cm high.",
            measured,
            unmeasurable=True,
        )
    else:
        record(
            "tcp_matches_finger_midpoint",
            tcp_error < 0.01,
            f"derived TCP is {tcp_error * 1000:.1f} mm from the finger-body midpoint; "
            f"measured flange-local offset {local_offset.tolist()}",
            measured,
        )

    # 7. Closed-loop reach, as the harness drives the arm. The rel-IK controller (damped
    # least squares) undershoots: one open-loop chunk of a 6 cm move travels 1.7 cm, a
    # gain of 0.283 identical to five significant figures on Isaac Sim 5.1 and 6.0.1.
    # Re-issuing toward a held target closes about 29 percent of the remaining distance
    # per round, so the arm converges geometrically. The open-loop gain is recorded;
    # arrival is graded.
    import torch

    def drive_to(observation, target_pose, budget=160):
        """Re-plan toward a fixed target until it is reached or the budget runs out."""
        nonlocal raw
        steps = 0
        while steps < budget:
            chunk = action_chunk(
                observation.tcp, target_pose, grip=0.0, horizon=10, tcp_offset_m=args.tcp_offset_m
            )
            for step in chunk:
                raw, *_ = env.step(torch.as_tensor(step[None], device=env.device))
                steps += 1
            observation = to_observation(
                raw, t=0.0, env_id=0, camera=SCENE_CAMERA, capabilities=caps, convention=convention
            )
            if (
                float(np.linalg.norm(observation.tcp.pos - target_pose.pos)) < 0.002
                and quat_angle_between(observation.tcp.quat, target_pose.quat) < 0.02
            ):
                break
        return observation, steps

    start = obs.tcp.pos.copy()
    target = Pose(start + np.array([0.0, 0.0, -0.06]), obs.tcp.quat)

    # One open-loop chunk, recorded as a measurement and not graded.
    for step in action_chunk(
        obs.tcp, target, grip=0.0, horizon=args.steps, tcp_offset_m=args.tcp_offset_m
    ):
        raw, *_ = env.step(torch.as_tensor(step[None], device=env.device))
    after = to_observation(
        raw, t=1.0, env_id=0, camera=SCENE_CAMERA, capabilities=caps, convention=convention
    )
    gain = float(-(after.tcp.pos[2] - start[2]) / 0.06)
    record(
        "open_loop_tracking_gain",
        True,
        f"one open-loop chunk of a 6 cm move travelled {gain * 6:.1f} cm, gain "
        f"{gain:.3f}. Recorded, not graded: this is the controller's, and TRACKING_GAIN "
        f"in the adapter carries it. A gain near 0.5 or 2.0 would mean REL_IK_SCALE is "
        f"wrong; this is neither.",
        {"gain": gain, "rel_ik_scale": REL_IK_SCALE},
        unmeasurable=True,
    )

    # Finish the same move closed loop; this is graded.
    after, used = drive_to(after, target)
    error = float(np.linalg.norm(after.tcp.pos - target.pos))
    drift = float(np.linalg.norm((after.tcp.pos - start)[:2]))
    record(
        "reaches_a_commanded_pose",
        error < 0.002 and drift < 0.01,
        f"6 cm down, closed loop: arrived within {error * 1000:.1f} mm after {used} "
        f"further control steps, off-axis drift {drift * 1000:.1f} mm",
        {
            "error_m": error,
            "drift_m": drift,
            "steps": used,
            "start": start.tolist(),
            "end": after.tcp.pos.tolist(),
        },
    )

    # Check independent root axes and a full-orientation target, not just displacement
    # magnitude. Small motions here are diagnostic, not task actions.
    for axis in (0, 1):
        start_obs = after
        delta = np.zeros(3)
        delta[axis] = 0.03
        target = Pose(start_obs.tcp.pos + delta, start_obs.tcp.quat)
        after, used = drive_to(start_obs, target)
        actual = after.tcp.pos - start_obs.tcp.pos
        cross = np.delete(actual, axis)
        record(
            f"root_axis_{'xy'[axis]}",
            abs(actual[axis] - 0.03) < 0.002 and np.linalg.norm(cross) < 0.01,
            f"requested {np.round(delta, 4).tolist()}, arrived at "
            f"{np.round(actual, 4).tolist()} in {used} control steps",
            {"requested": delta.tolist(), "observed": actual.tolist(), "steps": used},
        )
    start_obs = after
    target = Pose(after.tcp.pos, quat_mul(quat_from_axis_angle([0, 1, 0], 0.15), after.tcp.quat))
    after, used = drive_to(start_obs, target)
    residual = quat_angle_between(after.tcp.quat, target.quat)
    drift = float(np.linalg.norm(after.tcp.pos - start_obs.tcp.pos))
    record(
        "rotation_about_root_y",
        residual < 0.02 and drift < 0.01,
        f"0.15 rad about root y, closed loop: residual {residual:.4f} rad after {used} "
        f"control steps, TCP drift {drift * 1000:.1f} mm",
        {"residual_rad": residual, "drift_m": drift, "steps": used},
    )
    if wrist_to_tcp is not None:
        current = after.wrist.pose.inverse().compose(after.tcp)
        position_error = float(np.linalg.norm(current.pos - wrist_to_tcp.pos))
        rotation_error = float(quat_angle_between(current.quat, wrist_to_tcp.quat))
        record(
            "wrist_mount_tracks_tcp",
            position_error < 0.002 and rotation_error < 0.01,
            "Camera/TCP rigid transform after translation and root rotation",
            {"position_error_m": position_error, "rotation_error_rad": rotation_error},
        )
        if args.wrist_visual_fiducial:
            record_visual_fiducial(after, "after_motion")
    for command in (1.0, 0.0):
        for _ in range(int(np.ceil(caps.gripper.close_time_s / caps.control_period_s)) + 3):
            step = np.zeros((1, 7), dtype=np.float32)
            step[0, 6] = command
            raw, *_ = env.step(torch.as_tensor(step, device=env.device))
        after = to_observation(raw, t=0, capabilities=caps, convention=convention)
        record(
            "gripper_close" if command else "gripper_open",
            abs(after.gripper.travel - command) < 0.15,
            f"command {command}, observed travel {after.gripper.travel:.3f}",
        )

    env.close()
    graded = [f for f in findings if not f.get("unmeasurable")]
    passed = sum(f["passed"] for f in graded)
    skipped = len(findings) - len(graded)
    print(
        f"\n{passed}/{len(graded)} adapter checks passed on {args.task}"
        + (f", {skipped} unmeasurable on this rig" if skipped else ""),
        flush=True,
    )
    with open(args.out, "w") as fh:
        json.dump(
            {"task": args.task, "camera_convention": convention, "findings": findings}, fh, indent=2
        )
    print(f"wrote {args.out}", flush=True)
    return 0 if passed == len(graded) else 1


try:
    code = main()
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    code = 1
finally:
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "task": args.task,
                "passed": code == 0,
                "wrist_calibration": args.wrist_calibration,
                "wrist_visual_fiducial": args.wrist_visual_fiducial,
                "fiducial_position_root_m": args.fiducial_position
                if args.wrist_visual_fiducial
                else None,
                "tcp_offset_m": args.tcp_offset_m,
                "findings": findings,
            },
            indent=2,
        )
    )
    simulation_app.close()
sys.exit(code)
