"""Geometric checks a grasp must pass before it is offered.

Every check returns a reason, never a bare False. Reasons become the rejection text a
policy sees, so they are written to inform its next choice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import Pose, quat_to_rot

TABLE_MARGIN_M = 0.004
"""Fingertips never go closer than this to the table."""


def finger_volumes(
    pose: Pose, opening_m: float, caps: EmbodimentCapabilities
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """The two finger boxes at a given opening, as ``(centre, half_extents, rotation)``.

    Local axes: +y is the closing direction, +z is the approach direction, and the fingers
    extend backwards from the TCP along -z.
    """
    rot = quat_to_rot(pose.quat)
    g = caps.gripper
    half = np.array([g.pad_width_m / 2, g.pad_width_m / 2, g.finger_length_m / 2])
    out = []
    for sign in (-1.0, 1.0):
        local = np.array([0.0, sign * (opening_m / 2 + g.pad_width_m / 2), -g.finger_length_m / 2])
        out.append((pose.pos + rot @ local, half, rot))
    return out


PALM_DEPTH_M = 0.075
PALM_HEIGHT_M = 0.10
WRIST_HALF_M = 0.05
WRIST_HEIGHT_M = 0.15
"""The Robotiq 2F-85 body above the fingers is about 7.5 cm deep and as wide as the open
fingers, and reaches the flange 15 cm above the fingertips; the Franka wrist and camera
continue above that. Checking only the finger boxes misses body collisions, e.g. with a
pail rim 20 cm up while the fingers are clear."""


def hand_volumes(
    pose: Pose, opening_m: float, caps: EmbodimentCapabilities
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], str]]:
    """Fingers, palm and wrist as ``(centre, half_extents, rotation, part)`` boxes."""
    rot = quat_to_rot(pose.quat)
    g = caps.gripper
    out = [(c, h, r, "fingers") for c, h, r in finger_volumes(pose, opening_m, caps)]
    palm_half = np.array([PALM_DEPTH_M / 2, opening_m / 2 + g.pad_width_m, PALM_HEIGHT_M / 2])
    palm_centre = pose.pos + rot @ np.array([0.0, 0.0, -(g.finger_length_m + PALM_HEIGHT_M / 2)])
    out.append((palm_centre, palm_half, rot, "gripper body"))
    wrist_half = np.array([WRIST_HALF_M, WRIST_HALF_M, WRIST_HEIGHT_M / 2])
    wrist_centre = pose.pos + rot @ np.array(
        [0.0, 0.0, -(g.finger_length_m + PALM_HEIGHT_M + WRIST_HEIGHT_M / 2)]
    )
    out.append((wrist_centre, wrist_half, rot, "wrist"))
    return out


def points_inside_box(
    points: npt.ArrayLike,
    centre: npt.ArrayLike,
    half: npt.ArrayLike,
    rot: npt.ArrayLike,
    margin: float = 0.0,
) -> npt.NDArray[np.bool_]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    local = (pts - np.asarray(centre, dtype=np.float64)) @ np.asarray(rot, dtype=np.float64)
    return np.all(np.abs(local) <= np.asarray(half, dtype=np.float64) + margin, axis=1)


def check_opening(
    width_m: float, caps: EmbodimentCapabilities, margin_m: float = 0.008
) -> str | None:
    need = width_m + 2 * margin_m
    if need > caps.gripper.max_opening_m:
        return (
            f"too wide for the jaw: {width_m * 100:.1f} cm across plus clearance needs "
            f"{need * 100:.1f} cm, the jaw opens to {caps.gripper.max_opening_m * 100:.1f} cm"
        )
    return None


def check_reach(pose: Pose, caps: EmbodimentCapabilities) -> str | None:
    if not caps.workspace.contains(pose.pos):
        return f"out of reach: {np.round(pose.pos, 3).tolist()} is outside the arm's workspace"
    return None


def check_table(pose: Pose, caps: EmbodimentCapabilities) -> str | None:
    if pose.pos[2] < caps.workspace.table_z + TABLE_MARGIN_M:
        return f"would drive the fingertips into the table: grasp height {pose.pos[2] * 100:.1f} cm"
    return None


def check_fingers_above_table(
    pose: Pose, opening_m: float, caps: EmbodimentCapabilities
) -> str | None:
    """Neither finger box may dip below the table. For a top-down grasp this is the
    fingertip check; for a horizontal one the finger's own width is what reaches down."""
    for centre, half, rot in finger_volumes(pose, opening_m, caps):
        reach_down = float(np.sum(np.abs(rot[2]) * half))
        lowest = float(centre[2]) - reach_down
        if lowest < caps.workspace.table_z + TABLE_MARGIN_M:
            return (
                f"the fingers would touch the table: their lowest point would be "
                f"{(lowest - caps.workspace.table_z) * 100:.1f} cm above it"
            )
    return None


def check_finger_collision(
    pose: Pose,
    opening_m: float,
    caps: EmbodimentCapabilities,
    obstacles: npt.ArrayLike,
    margin_m: float = 0.004,
) -> tuple[str | None, float]:
    """Do the open hand's fingers, body or wrist intersect anything that is not the target?

    Returns ``(reason_or_none, clearance_m)``. Clearance is the distance from the finger
    volumes to the nearest obstacle point, which the policy sees even when the grasp is
    feasible: a 4 mm clearance is worth knowing about before committing.
    """
    obs = np.asarray(obstacles, dtype=np.float64)
    if len(obs) == 0:
        return None, float("inf")
    hits: dict[str, int] = {}
    min_dist = float("inf")
    for centre, half, rot, part in hand_volumes(pose, opening_m, caps):
        inside = points_inside_box(obs, centre, half, rot, margin=margin_m)
        if inside.any():
            hits[part] = hits.get(part, 0) + int(inside.sum())
        if part == "fingers":
            local = np.abs((obs - centre) @ rot) - half
            outside = np.clip(local, 0.0, None)
            min_dist = min(min_dist, float(np.min(np.linalg.norm(outside, axis=1))))
    if hits:
        where = ", ".join(f"the {part} ({n} points)" for part, n in hits.items())
        if set(hits) == {"fingers"}:
            return (
                f"the fingers would collide with something else ({hits['fingers']} "
                "points inside the finger volume)"
            ), 0.0
        return f"the hand would collide with something else: {where}", 0.0
    return None, min_dist


def check_approach_path(
    pre_pose: Pose,
    pose: Pose,
    opening_m: float,
    caps: EmbodimentCapabilities,
    obstacles: npt.ArrayLike,
    samples: int = 6,
) -> str | None:
    """Sample the straight line from the standoff to the grasp and check each pose."""
    obs = np.asarray(obstacles, dtype=np.float64)
    if len(obs) == 0:
        return None
    for i in range(1, samples + 1):
        f = i / samples
        p = Pose(pre_pose.pos + (pose.pos - pre_pose.pos) * f, pose.quat)
        reason, _ = check_finger_collision(p, opening_m, caps, obs, margin_m=0.002)
        if reason is not None:
            return f"the approach is blocked {int((1 - f) * 100)} percent of the way down"
    return None


def check_retreat(pose: Pose, lift_m: float, caps: EmbodimentCapabilities) -> str | None:
    top = pose.pos + np.array([0.0, 0.0, lift_m])
    if not caps.workspace.contains(top):
        return f"cannot lift {lift_m * 100:.0f} cm from here, the top of the lift is out of reach"
    return None
