"""Deterministic embodiment probes.

Run these against any adapter before a policy touches it, and again after any adapter
change. They test the things that fail silently: a commanded distance that is not the
distance travelled, a rotation that goes the wrong way, a camera convention that puts
deprojected points behind the robot, a gripper that reports holding when it is empty,
and a clock that does not advance.

    rung-probes --embodiment mock
    rung-probes --embodiment mock --scene tall_bottle_and_crate --json out.json

Exit code is 0 only if every probe passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
from rung.contracts.frames import Pose, deproject, depth_image_to_points, project, yaw_of
from rung.contracts.validate import validate_observation
from rung.embodiment import Embodiment
from rung.mock.world import top_down_pose

TOL_M = 5e-3
TOL_RAD = 0.02


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: str
    measured: dict[str, Any]

    def line(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


def _move(
    emb: Embodiment,
    pose: Pose,
    grip: GripperAction = GripperAction.HOLD,
    timeout: float = 10.0,
):
    return emb.execute(
        TCPTarget(
            pose=pose,
            gripper=GripperCommand(grip),
            max_lin_speed_mps=min(0.25, emb.capabilities.max_lin_speed_mps),
            max_ang_speed_rps=min(1.0, emb.capabilities.max_ang_speed_rps),
            timeout_s=timeout,
        )
    )


def probe_contract(emb: Embodiment) -> ProbeResult:
    obs = emb.reset()
    validate_observation(obs, emb.capabilities)
    caps = emb.capabilities
    names = {c.name for c in caps.cameras}
    ok = "scene" in names and caps.gripper.max_opening_m > 0
    return ProbeResult(
        "contract",
        ok,
        f"observation validates; cameras {sorted(names)}; "
        f"jaw {caps.gripper.max_opening_m * 100:.1f} cm",
        {"cameras": sorted(names), "control_hz": round(1 / caps.control_period_s, 2)},
    )


def probe_commanded_equals_travelled(emb: Embodiment) -> ProbeResult:
    """Check that the arm travels the commanded distance along each axis.

    A scale error silently halves or doubles every commanded motion and presents as a
    policy failure rather than an adapter bug.
    """
    emb.reset()
    start = emb.observe().tcp.pos.copy()
    errors = []
    for delta in ([0.06, 0, 0], [0, 0.06, 0], [0, 0, -0.05], [-0.06, -0.06, 0]):
        before = emb.observe().tcp.pos.copy()
        want = before + np.asarray(delta, dtype=float)
        rep = _move(emb, Pose(want, emb.observe().tcp.quat))
        after = emb.observe().tcp.pos
        commanded = float(np.linalg.norm(np.asarray(delta)))
        travelled = float(np.linalg.norm(after - before))
        errors.append(
            (commanded, travelled, rep.outcome.value, float(np.max(np.abs(after - want))))
        )
    worst = max(e[3] for e in errors)
    ratios = [round(t / c, 4) for c, t, _, _ in errors]
    return ProbeResult(
        "commanded_equals_travelled",
        worst < TOL_M,
        f"worst axis error {worst * 1000:.2f} mm over 4 moves; travelled/commanded {ratios}",
        {"errors_m": [e[3] for e in errors], "ratios": ratios, "start": start.tolist()},
    )


def probe_rotation_direction(emb: Embodiment) -> ProbeResult:
    """A positive yaw command must turn the closing axis the positive way about world z."""
    emb.reset()
    here = emb.observe().tcp.pos.copy()
    _move(emb, top_down_pose(here, 0.0))
    before = yaw_of(emb.observe().tcp.quat)
    _move(emb, top_down_pose(here, 0.5))
    after = yaw_of(emb.observe().tcp.quat)
    delta = float((after - before + np.pi) % (2 * np.pi) - np.pi)
    return ProbeResult(
        "rotation_direction",
        abs(delta - 0.5) < TOL_RAD,
        f"commanded +0.500 rad yaw, measured {delta:+.3f} rad",
        {"delta_rad": delta},
    )


def probe_camera_geometry(emb: Embodiment) -> ProbeResult:
    """Deproject the TCP's own pixel and check it lands back at the TCP.

    This catches an OpenGL/OpenCV mix-up, a transposed extrinsic, and intrinsics that do
    not match the delivered image size, all at once.
    """
    emb.reset()
    _move(emb, top_down_pose([0.45, 0.0, 0.22]))
    obs = emb.observe()
    cam = obs.scene
    if not cam.has_geometry:
        return ProbeResult("camera_geometry", False, "scene camera has no depth or calibration", {})
    tcp = obs.tcp.pos
    uv, z = project(tcp[None], cam.K, cam.pose)
    u, v = float(uv[0, 0]), float(uv[0, 1])
    in_frame = 0 <= u < cam.width and 0 <= v < cam.height and z[0] > 0
    if not in_frame:
        return ProbeResult(
            "camera_geometry",
            False,
            f"TCP projects to ({u:.1f}, {v:.1f}) depth {float(z[0]):.3f}, outside the "
            f"{cam.width}x{cam.height} image; check the camera convention",
            {"uv": [u, v], "z": float(z[0])},
        )
    back = deproject(np.array([u, v]), float(z[0]), cam.K, cam.pose)
    err = float(np.linalg.norm(back - tcp))
    return ProbeResult(
        "camera_geometry",
        err < 1e-6,
        f"TCP projects to ({u:.0f}, {v:.0f}) and deprojects back within {err * 1000:.4f} mm",
        {"uv": [u, v], "z": float(z[0]), "error_m": err},
    )


def probe_depth_reconstructs_the_table(emb: Embodiment) -> ProbeResult:
    """Deprojected depth must put the table at the height the capabilities declare.

    Catches millimetre/metre scale errors, ray distance reported as optical-axis depth,
    transposed extrinsics and OpenGL camera conventions. Needs only the declared table
    height, so it also runs on real hardware.

    Depth at the TCP pixel is not the distance to the TCP: the TCP is the midpoint between
    the fingers, and with the gripper open its pixel sees through the gap.
    """
    emb.reset()
    _move(emb, top_down_pose([0.45, 0.0, 0.30]))
    obs = emb.observe()
    cam = obs.scene
    if not cam.has_geometry:
        return ProbeResult(
            "depth_reconstructs_the_table", False, "scene camera has no depth or calibration", {}
        )
    pts, _ = depth_image_to_points(cam.depth, cam.K, cam.pose, stride=2)
    if len(pts) < 100:
        return ProbeResult(
            "depth_reconstructs_the_table", False, f"only {len(pts)} valid depth pixels", {}
        )
    table_z = emb.capabilities.workspace.table_z
    heights = pts[:, 2]
    lo, hi = table_z - 0.25, table_z + 0.60
    hist, edges = np.histogram(heights[(heights > lo) & (heights < hi)], bins=85, range=(lo, hi))
    peak = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    share = float(np.mean(np.abs(heights - table_z) < 0.01))
    ok = abs(peak - table_z) < 0.015 and share > 0.10
    return ProbeResult(
        "depth_reconstructs_the_table",
        bool(ok),
        f"the most common reconstructed height is {peak:+.3f} m against a declared table at "
        f"{table_z:+.3f} m; {share * 100:.0f} percent of pixels are within 1 cm of it",
        {"peak_z": peak, "table_z": table_z, "share_at_table": share, "n_points": int(len(pts))},
    )


def probe_clock(emb: Embodiment) -> ProbeResult:
    emb.reset()
    t0 = emb.observe().t
    same = emb.observe().t
    rep = _move(emb, top_down_pose([0.45, 0.05, 0.25]))
    t1 = emb.observe().t
    ok = same == t0 and t1 > t0 and abs(rep.t_end - t1) < 1e-9
    return ProbeResult(
        "clock",
        bool(ok),
        f"observing does not advance time; a move advanced it {t1 - t0:.2f} s "
        f"and the report agrees",
        {"t_start": t0, "t_after": t1, "report_end": rep.t_end},
    )


def probe_gripper_states(emb: Embodiment) -> ProbeResult:
    """Closing on empty air must read as fully closed, not as holding.

    Otherwise a policy carries an empty gripper to the container and believes it succeeded.
    """
    emb.reset()
    caps = emb.capabilities
    empty_spot = np.array(
        [caps.workspace.min_xyz[0] + 0.05, caps.workspace.max_xyz[1] - 0.05, 0.06]
    )
    _move(emb, top_down_pose(empty_spot), grip=GripperAction.OPEN)
    open_travel = emb.observe().gripper.travel
    _move(emb, top_down_pose(empty_spot), grip=GripperAction.CLOSE)
    closed_travel = emb.observe().gripper.travel
    ok = open_travel < 0.05 and closed_travel > 0.95
    return ProbeResult(
        "gripper_states",
        bool(ok),
        f"open travel {open_travel:.2f}, closed-on-nothing travel {closed_travel:.2f} "
        "(a closed-empty gripper must read as fully closed, not as holding)",
        {"open": open_travel, "closed_empty": closed_travel},
    )


def probe_timeout_is_honoured(emb: Embodiment) -> ProbeResult:
    emb.reset()
    far = np.array([0.70, 0.35, 0.50])
    rep = emb.execute(TCPTarget(pose=top_down_pose(far), max_lin_speed_mps=0.02, timeout_s=1.0))
    ok = rep.outcome is ExecutionOutcome.EXPIRED and rep.duration_s <= 1.0 + 1e-6
    return ProbeResult(
        "timeout_is_honoured",
        bool(ok),
        f"a slow long move returned {rep.outcome.value} after {rep.duration_s:.2f} s "
        f"with {rep.residual_m * 100:.1f} cm remaining",
        {"outcome": rep.outcome.value, "duration_s": rep.duration_s, "residual_m": rep.residual_m},
    )


def probe_reset_restores(emb: Embodiment) -> ProbeResult:
    a = emb.reset()
    _move(emb, top_down_pose([0.5, 0.1, 0.15]), grip=GripperAction.CLOSE)
    b = emb.reset()
    same_pose = np.allclose(a.tcp.pos, b.tcp.pos, atol=1e-9)
    same_pixels = np.array_equal(a.scene.rgb, b.scene.rgb)
    same_clock = b.t == 0.0
    return ProbeResult(
        "reset_restores",
        bool(same_pose and same_pixels and same_clock),
        f"reset returns the same TCP ({same_pose}), the same image ({same_pixels}), "
        f"and t=0 ({same_clock})",
        {"pose": bool(same_pose), "pixels": bool(same_pixels), "clock": bool(same_clock)},
    )


PROBES: list[Callable[[Embodiment], ProbeResult]] = [
    probe_contract,
    probe_reset_restores,
    probe_clock,
    probe_commanded_equals_travelled,
    probe_rotation_direction,
    probe_camera_geometry,
    probe_depth_reconstructs_the_table,
    probe_gripper_states,
    probe_timeout_is_honoured,
]


def run_probes(emb: Embodiment) -> list[ProbeResult]:
    results = []
    for probe in PROBES:
        try:
            results.append(probe(emb))
        except Exception as exc:  # a probe that raises is a failure, not a crash
            results.append(
                ProbeResult(probe.__name__.removeprefix("probe_"), False, f"raised {exc!r}", {})
            )
    return results


def build_embodiment(kind: str, scene: str) -> Embodiment:
    if kind == "mock":
        from rung.mock import SCENES

        if scene not in SCENES:
            raise SystemExit(f"unknown scene {scene!r}; have {sorted(SCENES)}")
        return SCENES[scene]()
    raise SystemExit(f"unknown embodiment {kind!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--embodiment", default="mock")
    ap.add_argument("--scene", default="banana_in_bowl")
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args(argv)

    emb = build_embodiment(args.embodiment, args.scene)
    results = run_probes(emb)
    for r in results:
        print(r.line())
    failed = [r for r in results if not r.passed]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} probes passed on "
        f"{args.embodiment}:{args.scene}"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embodiment": args.embodiment,
                    "scene": args.scene,
                    "results": [asdict(r) for r in results],
                },
                f,
                indent=2,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
