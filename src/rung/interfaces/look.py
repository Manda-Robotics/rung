"""``look``: move the wrist camera to a chosen viewpoint on a target and capture views.

Re-localisation for the direct interface. "Target lost" was Gemini's second-largest
failure class in Screen 2b (6 of 13 counted failures); the model can already move the
wrist, but it has to plan the whole motion itself, one bounded step per call. ``look``
takes a point (root metres, or a pixel resolved through public calibration) and a
viewpoint on a sphere around it, and plans a guarded TCP path whose wrist camera ends
looking straight at the point:

* the viewpoint is the camera position ``target + distance * [cos(el) cos(az),
  cos(el) sin(az), sin(el)]``; elevation 90 looks straight down, azimuth 180 places the
  camera on the robot's side of the target looking away from the base;
* the camera turns by the smallest rotation that puts its optical axis on the target,
  so image roll follows the current view and the wrist turns as little as possible;
* with a calibrated wrist camera (``frame.pose`` in OpenCV axes) the hand-to-camera mount
  is measured from the current frame and ``distance`` is camera-to-target. Without one,
  the TCP approach axis (+z) is used as the optical axis and ``distance`` is
  TCP-to-target (``approach_axis_approximation``), which the receipt says;
* the path rises first, then rotates in place at a transit height, then translates at
  that height, then descends (lift, rotate, translate, descend), so no horizontal motion
  happens lower than either end or a declared transit clearance above the table, and no
  step combines translation with rotation; each leg is split into steps inside the
  per-call translation and rotation bounds, with rotation further capped at
  ``LOOK_STEP_MAX_ROTATION_RAD`` per step (DROID relative IK did not track a live step
  that combined 0.17 m with 0.75 rad), and every step keeps the speed limits and the
  blocked-motion guard;
* a look may start from a TCP outside the declared workspace when the nearest point
  inside it is within one bounded translation step: the first step moves straight there
  without rotating, so no commanded pose is ever outside the workspace. From further out
  the plan is rejected with the distance and the nearest inside point.

Defaults (live qualification 2026-10-27): ``distance_m`` 0.30, ``elevation_deg`` 65,
``azimuth_deg`` 180. Views below ``SIDE_VIEW_BELOW_DEG`` are side views that may not be
reachable near the robot base.

Nothing here reads privileged simulator state: inputs are the measured TCP, the public
camera calibration already in the observation, and the declared workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rung.contracts.frames import Pose, quat_angle_between, rot_to_quat

DISTANCE_RANGE_M = (0.10, 0.60)
ELEVATION_RANGE_DEG = (20.0, 90.0)
DEFAULT_DISTANCE_M = 0.30
DEFAULT_ELEVATION_DEG = 65.0
DEFAULT_AZIMUTH_DEG = 180.0
"""Viewpoint filled in when the model omits it: every 65 to 80 degree live look reached its
pose, and 0.30 to 0.35 m framed a mug-sized object whole with its neighbours."""
SIDE_VIEW_BELOW_DEG = 45.0
"""Below this elevation a look is a side view; near the robot base it may not be reachable."""
LOOK_STEP_MAX_ROTATION_RAD = 0.35
"""Rotation per planned look step, below the per-call bound."""
WORKSPACE_ENTRY_MARGIN_M = 0.01
"""An out-of-workspace start enters the box this far inside its faces, so the achieved
pose (within the adapter's arrival tolerance) is still inside for the next step."""
MIN_TCP_TABLE_CLEARANCE_M = 0.05
"""Fingertip midpoint at the look pose stays at least this far above the table plane."""
MIN_TCP_TARGET_CLEARANCE_M = 0.06
"""The fingertips stop at least this far from the target point (camera behind the hand)."""
TRANSIT_CLEARANCE_M = 0.15
"""Horizontal travel happens at least this far above the table, or higher when either end
of the path is higher. Clamped to the workspace ceiling."""
SETTLE_S = 0.2
"""Dwell at the look pose before the capture, so the view is not taken mid-motion."""
STEP_TRANSLATION_MARGIN_M = 0.015
STEP_ROTATION_MARGIN_RAD = 0.06
"""Planned steps stay this far inside the per-call bounds so that a step revalidated from
the achieved pose (up to the adapter's arrival tolerance away) still passes."""

CALIBRATED = "calibrated_wrist_camera"
APPROACH_AXIS = "approach_axis_approximation"


@dataclass
class LookPlan:
    target_xyz: np.ndarray
    look_tcp: Pose
    camera_pose: Pose
    """Predicted camera pose at the look pose, OpenCV axes (+z toward the target)."""
    mount: Pose
    """Camera in the TCP frame (identity under the approach-axis approximation)."""
    view_model: str
    waypoints: list[Pose]
    """TCP poses to visit in order; the last is ``look_tcp``."""
    transit_z: float
    notes: list[str] = field(default_factory=list)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("degenerate direction")
    return v / n


def align_rotation(a, b, fallback_axis=None):
    """Smallest rotation matrix taking unit vector ``a`` onto unit vector ``b``.

    For opposite vectors the half turn is about ``fallback_axis`` (projected to be
    perpendicular to ``a``), so the choice is deterministic.
    """
    a, b = _unit(a), _unit(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        k = np.asarray([1.0, 0.0, 0.0] if fallback_axis is None else fallback_axis, float)
        k = k - (k @ a) * a
        if np.linalg.norm(k) < 1e-6:
            k = np.cross(a, [0.0, 1.0, 0.0])
            if np.linalg.norm(k) < 1e-6:
                k = np.cross(a, [0.0, 0.0, 1.0])
        k = _unit(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = math.atan2(s, c)
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def camera_mount(tcp: Pose, wrist_frame) -> tuple[Pose, str]:
    """The wrist camera in the TCP frame, and which view model that is.

    Uses the frame's own TCP at capture when the adapter recorded it, so a frame taken a
    control step before the measured TCP does not skew the mount.
    """
    if (
        wrist_frame is not None
        and wrist_frame.pose is not None
        and wrist_frame.convention == "opencv"
    ):
        at_capture = wrist_frame.state_at_capture or tcp
        return at_capture.inverse().compose(wrist_frame.pose), CALIBRATED
    return Pose.identity(), APPROACH_AXIS


def viewpoint(target, distance_m, elevation_deg, azimuth_deg):
    """Camera position on the viewing sphere around ``target``."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(target, float) + distance_m * direction


def _number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must be in [{low:g}, {high:g}]")
    return float(value)


def _slerp(q0, q1, frac):
    a, b = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        out = a + frac * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(1.0, d))
    return (math.sin((1 - frac) * theta) * a + math.sin(frac * theta) * b) / math.sin(theta)


def _interpolate(a: Pose, b: Pose, n: int) -> list[Pose]:
    return [
        Pose(a.pos + (b.pos - a.pos) * (i / n), _slerp(a.quat, b.quat, i / n))
        for i in range(1, n + 1)
    ]


def plan_look(
    start: Pose,
    wrist_frame,
    target,
    distance_m,
    elevation_deg,
    azimuth_deg,
    workspace,
    max_step_m,
    max_step_rad,
    max_steps,
) -> LookPlan:
    """A guarded TCP path from ``start`` to a pose whose wrist camera looks at ``target``.

    Raises ``ValueError`` (nothing executed) when the viewpoint or any waypoint leaves the
    declared workspace, the fingertips would come too close to the table or the target,
    or the path needs more than ``max_steps`` bounded steps.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target)):
        raise ValueError("target must be three finite numbers")
    distance_m = _number(distance_m, "distance_m", *DISTANCE_RANGE_M)
    elevation_deg = _number(elevation_deg, "elevation_deg", *ELEVATION_RANGE_DEG)
    azimuth_deg = _number(azimuth_deg, "azimuth_deg", -180.0, 180.0)
    lo, hi = np.asarray(workspace.min_xyz, float), np.asarray(workspace.max_xyz, float)
    if not (
        np.all(target[:2] >= lo[:2] - 0.10)
        and np.all(target[:2] <= hi[:2] + 0.10)
        and workspace.table_z - 0.05 <= target[2] <= hi[2]
    ):
        raise ValueError("target outside the declared workspace; nothing executed")

    mount, view_model = camera_mount(start, wrist_frame)
    camera_now = start.compose(mount)
    eye = viewpoint(target, distance_m, elevation_deg, azimuth_deg)
    forward = _unit(target - eye)
    r_cam = (
        align_rotation(camera_now.rot[:, 2], forward, fallback_axis=camera_now.rot[:, 0])
        @ camera_now.rot
    )
    camera_pose = Pose(eye, rot_to_quat(r_cam))
    look_tcp = camera_pose.compose(mount.inverse())

    if not workspace.contains(look_tcp.pos):
        raise ValueError(
            f"look pose {np.round(look_tcp.pos, 3).tolist()} is outside the "
            "reachable workspace; choose a closer or higher viewpoint"
        )
    if look_tcp.pos[2] < workspace.table_z + MIN_TCP_TABLE_CLEARANCE_M:
        raise ValueError(
            "look pose puts the fingertips within "
            f"{MIN_TCP_TABLE_CLEARANCE_M * 100:.0f} cm of the table; raise "
            "elevation_deg or distance_m"
        )
    if np.linalg.norm(look_tcp.pos - target) < MIN_TCP_TARGET_CLEARANCE_M:
        raise ValueError(
            "look pose puts the fingertips within "
            f"{MIN_TCP_TARGET_CLEARANCE_M * 100:.0f} cm of the target; "
            "increase distance_m"
        )

    step_m = max(max_step_m - STEP_TRANSLATION_MARGIN_M, 0.5 * max_step_m)
    step_rad = min(
        max(max_step_rad - STEP_ROTATION_MARGIN_RAD, 0.5 * max_step_rad), LOOK_STEP_MAX_ROTATION_RAD
    )
    ceiling = hi[2] - 1e-3
    notes = []
    entry = start
    if not workspace.contains(start.pos):
        entry = Pose(workspace_entry(start.pos, workspace), start.quat)
        outside = float(np.linalg.norm(entry.pos - start.pos))
        if outside > step_m:
            inside = np.round(entry.pos, 3).tolist()
            if outside <= max_step_m:
                todo = f"move_to {inside} with the current orientation first, then call look again"
            else:
                todo = (
                    f"that is more than one bounded motion ({max_step_m:g} m) away, and "
                    "every motion tool rejects targets outside the workspace, so no "
                    "single call can bring the TCP back from here"
                )
            raise ValueError(
                f"look rejected, nothing executed: the TCP {np.round(start.pos, 3).tolist()} "
                f"is {outside:.3f} m outside the declared workspace. look only starts inside "
                f"it or within one look step ({step_m:.3f} m) of it, and never commands a "
                f"pose outside it. Nearest point inside: {inside}; {todo}"
            )
        notes.append(
            f"started {outside:.3f} m outside the workspace; the first step moves "
            f"straight in to {np.round(entry.pos, 3).tolist()} without rotating"
        )
    transit_z = min(
        max(entry.pos[2], look_tcp.pos[2], workspace.table_z + TRANSIT_CLEARANCE_M), ceiling
    )
    legs = []
    lifted = Pose(np.array([entry.pos[0], entry.pos[1], transit_z]), start.quat)
    turned = Pose(lifted.pos, look_tcp.quat)
    over = Pose(np.array([look_tcp.pos[0], look_tcp.pos[1], transit_z]), look_tcp.quat)
    for pose in (entry, lifted, turned, over, look_tcp):
        previous = legs[-1] if legs else start
        if (
            np.linalg.norm(pose.pos - previous.pos) > 1e-3
            or quat_angle_between(previous.quat, pose.quat) > 1e-3
        ):
            legs.append(pose)
    waypoints, previous = [], start
    for pose in legs:
        n = max(
            1,
            math.ceil(np.linalg.norm(pose.pos - previous.pos) / step_m - 1e-9),
            math.ceil(quat_angle_between(previous.quat, pose.quat) / step_rad - 1e-9),
        )
        waypoints += _interpolate(previous, pose, n)
        previous = pose
    if not waypoints:
        waypoints = [look_tcp]
    if len(waypoints) > max_steps:
        raise ValueError(
            f"look needs {len(waypoints)} bounded steps, more than the "
            f"{max_steps} allowed; choose a viewpoint nearer the current pose"
        )
    for pose in waypoints:
        if not workspace.contains(pose.pos):
            raise ValueError("look path leaves the reachable workspace; nothing executed")
    if view_model == APPROACH_AXIS:
        notes.append(
            "no wrist calibration: the TCP approach axis (+z) is aimed at the target "
            "and distance_m is fingertip-to-target; the camera sits behind the hand, "
            "so the true camera distance is larger"
        )
    if transit_z >= ceiling and max(start.pos[2], look_tcp.pos[2]) < ceiling:
        notes.append("transit height clamped to the workspace ceiling")
    return LookPlan(target, look_tcp, camera_pose, mount, view_model, waypoints, transit_z, notes)


def workspace_entry(pos, workspace):
    """Nearest point to ``pos`` that is at least ``WORKSPACE_ENTRY_MARGIN_M`` inside every
    face of the workspace box (the box centre on an axis narrower than twice the margin)."""
    lo, hi = np.asarray(workspace.min_xyz, float), np.asarray(workspace.max_xyz, float)
    margin = np.minimum(WORKSPACE_ENTRY_MARGIN_M, 0.5 * (hi - lo))
    return np.clip(np.asarray(pos, float).reshape(3), lo + margin, hi - margin)


def view_span(K, width, height, distance_m):
    """Width and height of the area the camera sees at ``distance_m`` on its axis."""
    K = np.asarray(K, float)
    return [round(distance_m * width / K[0, 0], 3), round(distance_m * height / K[1, 1], 3)]


def expected_pixel(frame, target):
    """Where ``target`` projects in a calibrated frame, normalized [u, v], or None.

    Returns the in-image flag as well, so the model can tell a target that is outside the
    view (or behind the camera) from one it should be able to see.
    """
    if frame is None or frame.K is None or frame.pose is None or frame.convention != "opencv":
        return None
    from rung.contracts.frames import project

    uv, z = project(np.asarray(target, float).reshape(1, 3), frame.K, frame.pose)
    u, v, z = float(uv[0, 0]), float(uv[0, 1]), float(z[0])
    nu, nv = u / max(frame.width - 1, 1), v / max(frame.height - 1, 1)
    return {
        "normalized_uv": [round(nu, 3), round(nv, 3)] if z > 0 else None,
        "in_view": bool(z > 0 and 0 <= nu <= 1 and 0 <= nv <= 1),
        "camera_to_target_m": round(
            float(np.linalg.norm(np.asarray(target, float) - frame.pose.pos)), 3
        ),
    }
