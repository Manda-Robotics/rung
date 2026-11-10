"""Translation between RoboLab and the harness contracts. Pure functions, no Isaac.

Everything here is testable with plain arrays. The silent failure modes, frame conventions
and scale factors, need no simulator to check.

RoboLab observation terms and their harness counterparts:

============================  ==========================================================
RoboLab                       harness
============================  ==========================================================
``image_obs[cam]``            ``scene.rgb``, batched over envs, uint8
``image_obs[cam + "_depth"]`` ``scene.depth``; already distance-to-image-plane, which is
                              the optical-axis depth the contract wants, not ray length
``image_obs[cam + "_K"]``     ``scene.K``
``image_obs[cam + "_pos"]``,  ``scene.pose``, after converting the tree's camera axes
``image_obs[cam + "_quat"]``  (see DEFAULT_CAMERA_CONVENTION) to the OpenCV axes the
                              harness uses everywhere
``proprio_obs["ee_pos"]``     base_link (the flange), not the fingertips. ``tcp``
                              composes the flange-to-TCP transform, FLANGE_TO_FINGERTIP_M
``proprio_obs["gripper_pos"]``  ``gripper.travel``, 0 open to 1 fingers touching
============================  ==========================================================

Actions go back as ``(N, 7)``: ``dx dy dz droll dpitch dyaw grip`` on root axes.
``DroidRelIKActionCfg`` multiplies the first six by its ``scale`` of 0.5 before applying
them, so commanded deltas are divided by that scale on the way out. Omitting the division
halves every motion, which looks like a model failure; ``rung.probes`` checks it first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rung.contracts.capabilities import (
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import (
    Pose,
    convert_camera_pose,
    quat_angle_between,
    quat_from_axis_angle,
    quat_to_rot,
    rot_to_quat,
)
from rung.contracts.observation import CameraFrame, GripperState, Observation

SCENE_CAMERA = "over_shoulder_left_camera"
WRIST_CAMERA = "wrist_cam"
RIGHT_SHOULDER_CAMERA = "over_shoulder_right_camera"
EXTRA_VIEW_NAMES = {RIGHT_SHOULDER_CAMERA: "scene_right", SCENE_CAMERA: "scene_left"}
"""Role names for extra views, so prompts and tool enums never carry rig key names."""

FLANGE_TO_FINGERTIP_M = 0.14883
"""base_link to the open finger plane, in metres. Measured, not quoted.

0.148832 from ``calibrate_tcp.py`` on Isaac Sim 6.0: the open hand is driven onto two
visible table patches at two yaw angles, with lift-away verified so a wedged arm cannot
register as contact. The two trials agree to 0.28 mm. Deployment reads the offset from a
calibration file; this constant is the fallback and a sanity reference.

The 0.1628 m Robotiq 2F-85 figure quoted in ``DroidRelIKActionCfg`` is the gripper's
physical height, not the distance to the control point; using it places grasps about
5 cm high.

The USD collision geometry extends along base_link +X, not +Z. Tool +Z is approach and
tool +Y is jaw closing, so the tool frame is rotated +90 degrees about base_link Y. The
fixed TCP is the open finger plane; finger motion is articulated.
"""


def droid_tcp_transform(offset_m: float = FLANGE_TO_FINGERTIP_M) -> Pose:
    return Pose(np.array([offset_m, 0.0, 0.0]), quat_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2))


STEPS_PER_EFOLD = 30.0
"""Control steps for the remaining distance to shrink by a factor of e.

Measured alongside TRACKING_GAIN: re-issuing toward a held target leaves 71.3 percent of
the gap after each ten-step round, so a factor of e takes about 30 steps and a 6 cm move
about 90. Motion budgets derive from this. Execution through the bridge is slower than the
loop it was measured in; ``time_to_converge`` applies a safety factor for that.
"""

TRACKING_GAIN = 0.283
"""Fraction of a commanded Cartesian step the controller realises. Measured, not assumed.

The DROID rel-IK controller is damped least squares and undershoots: a 6 cm commanded move
travels 1.7 cm, identical to five significant figures on Isaac Sim 5.1 and 6.0.1, across
root axes and for a rotation. Re-issuing toward a held target closes 28.7 percent of the
*remaining* distance each round, so motion converges geometrically rather than stalling;
reaching within 2 mm of a 6 cm target takes about 100 control steps. Time or step budgets
computed from ``steps_needed`` alone are optimistic by roughly 3.5x.
"""

REL_IK_SCALE = 0.5
"""``DroidRelIKActionCfg.scale``. Commanded deltas are divided by it."""

MAX_DXYZ_M = 0.02
MAX_DROT_RAD = 0.10
"""Per-step ceilings at 15 Hz. Larger steps make the differential IK solver overshoot
rather than move faster."""

CONTROL_HZ = 15.0

DEFAULT_CAMERA_CONVENTION = "opengl"
"""Axis convention of the simulator's camera extrinsics, before conversion.

It differs between RoboLab trees, and a wrong value sends every deprojected point away
from the scene. Upstream NVlabs RoboLab on Isaac Sim 5 reports OpenGL axes (looking down
local -z, +y up). The Isaac Sim 6 port reports ROS axes, which equal the harness's OpenCV
axes and need no rotation.

``detect_camera_convention`` measures it against a point the camera is known to view;
record the result in the run's provenance.
"""


def to_numpy(value) -> npt.NDArray | None:
    """A torch tensor, a numpy array, or None. The adapter never imports torch."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _grab(group: dict, key: str, env_id: int) -> npt.NDArray | None:
    if not isinstance(group, dict) or key not in group:
        return None
    array = to_numpy(group[key])
    if array is None:
        return None
    return array[env_id] if array.ndim >= 1 and array.shape[0] > env_id else array


def droid_capabilities(
    scene_size: tuple[int, int] = (1280, 720),
    wrist_size: tuple[int, int] | None = (1280, 720),
    table_z: float = 0.0,
    tcp_offset_m: float = FLANGE_TO_FINGERTIP_M,
) -> EmbodimentCapabilities:
    """The DROID Franka with a Robotiq 2F-85, as RoboLab configures it.

    The workspace is a conservative box around the table rather than the arm's true
    reachable set. It is used to reject obviously bad targets early; the differential IK
    solver is the real authority, and a target it cannot reach comes back as a residual.
    """
    # Cameras are named by role, not by rig key: every embodiment calls its main view
    # "scene", so prompts, tool enums and traces agree across rigs. RoboLab key names
    # stay in this module.
    cameras = [CameraSpec("scene", *scene_size, has_depth=True, mounted_on="world")]
    if wrist_size is not None:
        cameras.append(CameraSpec("wrist", *wrist_size, has_depth=False, mounted_on="tcp"))
    return EmbodimentCapabilities(
        name="robolab-droid",
        cameras=tuple(cameras),
        gripper=GripperGeometry(
            max_opening_m=0.085, finger_length_m=0.0555, pad_width_m=0.022, close_time_s=1.5
        ),
        workspace=Workspace(
            min_xyz=np.array([0.15, -0.55, table_z]),
            max_xyz=np.array([0.90, 0.55, 0.75]),
            table_z=table_z,
            table_extent=TABLE_EXTENT_ROOT_M,
        ),
        control_period_s=1.0 / CONTROL_HZ,
        tcp_from_ee=droid_tcp_transform(tcp_offset_m),
        has_joint_state=True,
        max_lin_speed_mps=MAX_DXYZ_M * CONTROL_HZ,
        max_ang_speed_rps=MAX_DROT_RAD * CONTROL_HZ,
        steps_per_efold=STEPS_PER_EFOLD,
        notes={
            "action": "DroidRelIKActionCfg, 6 relative pose deltas on root axes plus grip",
            "camera_convention": "OpenGL in RoboLab, converted to OpenCV here",
            "depth": "distance_to_image_plane, which is optical-axis depth",
        },
    )


# RoboLab-Verified's table fixture (assets/fixtures/franka_table.usd, pose x -0.087 m, yaw
# 180 degrees, scene_fixture.FRANKA_TABLE_FIXTURE) has a top of 0.900 x 0.759 m; in the
# robot root frame (+x across the table) it spans x -0.089 to 0.811 and y -0.379 to
# 0.379. Measured 2026-10-24 from the fixture's USD bounds. Declared rig geometry, like
# the camera calibration; it is not a perception result. Seven of 32 counted Gemini
# failures in the 2026-10-23 subset ended with the target off the table, most released
# at or past these edges.
TABLE_EXTENT_ROOT_M = (-0.089, 0.811, -0.379, 0.379)


def flange_to_tcp(
    flange_pos: npt.ArrayLike, flange_quat: npt.ArrayLike, offset_m: float = FLANGE_TO_FINGERTIP_M
) -> Pose:
    """Where the fingertips are, given the flange pose RoboLab reports."""
    pos = np.asarray(flange_pos, dtype=np.float64).reshape(3)
    quat = np.asarray(flange_quat, dtype=np.float64).reshape(4)
    return Pose(pos, quat).compose(droid_tcp_transform(offset_m))


def tcp_to_flange(tcp: Pose, offset_m: float = FLANGE_TO_FINGERTIP_M) -> Pose:
    """The inverse, for turning a TCP target into something the flange controller wants."""
    return tcp.compose(droid_tcp_transform(offset_m).inverse())


def to_observation(
    raw_obs: dict,
    t: float,
    env_id: int = 0,
    camera: str = SCENE_CAMERA,
    capabilities: EmbodimentCapabilities | None = None,
    convention: str = DEFAULT_CAMERA_CONVENTION,
    extra_cameras: tuple[str, ...] = (),
) -> Observation:
    """RoboLab's nested batched observation to one flat per-env observation.

    ``extra_cameras`` names further rig cameras (for example
    ``over_shoulder_right_camera``) delivered as calibrated RGB views under
    ``observation.extra["views"]``; each needs its ``_K``, ``_pos`` and ``_quat`` terms.

    Raises on a missing term rather than returning ``None``, which would let the
    controller run silently without depth, intrinsics or camera pose.
    """
    images = raw_obs.get("image_obs")
    proprio = raw_obs.get("proprio_obs")
    if not isinstance(images, dict) or not isinstance(proprio, dict):
        raise KeyError(
            "expected image_obs and proprio_obs groups; got keys "
            f"{sorted(raw_obs)!r}. This observation is not from RoboLab."
        )

    rgb = _grab(images, camera, env_id)
    if rgb is None:
        available = sorted(k for k in images if not k.endswith(("_depth", "_K", "_pos", "_quat")))
        raise KeyError(f"no camera called {camera!r}; available: {available}")

    depth = _grab(images, f"{camera}_depth", env_id)
    intrinsics = _grab(images, f"{camera}_K", env_id)
    cam_pos = _grab(images, f"{camera}_pos", env_id)
    cam_quat = _grab(images, f"{camera}_quat", env_id)
    if depth is not None and (intrinsics is None or cam_pos is None or cam_quat is None):
        raise KeyError(
            f"{camera} has depth but not all of _K, _pos and _quat. Register the "
            "environment with depth enabled and lazy_sensor_update off; see "
            "docs/contracts.md, the adapter checklist."
        )

    scene = CameraFrame(
        name="scene",
        rgb=_as_uint8(rgb),
        t=t,
        depth=None if depth is None else np.asarray(depth, dtype=np.float32).squeeze(),
        K=None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64).reshape(3, 3),
        pose=None if cam_pos is None else convert_camera_pose(Pose(cam_pos, cam_quat), convention),
        convention="opencv",
    )

    wrist_rgb = _grab(images, WRIST_CAMERA, env_id)
    wrist_k = _grab(images, f"{WRIST_CAMERA}_K", env_id)
    wrist_pos = _grab(images, f"{WRIST_CAMERA}_pos", env_id)
    wrist_quat = _grab(images, f"{WRIST_CAMERA}_quat", env_id)
    if any(value is not None for value in (wrist_k, wrist_pos, wrist_quat)) and any(
        value is None for value in (wrist_k, wrist_pos, wrist_quat)
    ):
        raise KeyError("wrist camera calibration requires all of _K, _pos and _quat")
    wrist = (
        None
        if wrist_rgb is None
        else CameraFrame(
            name="wrist",
            rgb=_as_uint8(wrist_rgb),
            t=t,
            convention="opencv",
            K=None if wrist_k is None else np.asarray(wrist_k, dtype=np.float64),
            pose=None
            if wrist_pos is None
            else convert_camera_pose(Pose(wrist_pos, wrist_quat), convention),
        )
    )

    # Verified's eef_frame has EEF_OFFSET_ROT applied and is not base_link. The relative
    # IK controller targets base_link, so its paired pose terms are required.
    flange = _grab(proprio, "ee_pos", env_id)
    quat = _grab(proprio, "ee_quat", env_id)
    if flange is None or quat is None:
        raise KeyError(
            "proprio_obs requires base_link pose ee_pos and ee_quat; "
            "eef_pos/eef_quat use a different control frame"
        )

    travel = _grab(proprio, "gripper_pos", env_id)
    if travel is None:
        raise KeyError("proprio_obs requires gripper_pos")
    travel_value = float(np.asarray(travel).reshape(-1)[0])
    caps = capabilities or droid_capabilities()

    observation = Observation(
        t=t,
        scene=scene,
        wrist=wrist,
        tcp=Pose(flange, quat).compose(caps.tcp_from_ee),
        gripper=GripperState(
            travel=float(np.clip(travel_value, 0.0, 1.0)),
            opening_m=caps.gripper.opening_for_travel(float(np.clip(travel_value, 0.0, 1.0))),
        ),
    )
    scene.state_at_capture = observation.tcp
    if wrist is not None:
        wrist.state_at_capture = observation.tcp
    views = []
    for name in extra_cameras:
        rgb_extra = _grab(images, name, env_id)
        if rgb_extra is None:
            suffixes = ("_depth", "_K", "_pos", "_quat")
            available = sorted(k for k in images if not k.endswith(suffixes))
            raise KeyError(f"no extra camera called {name!r}; available: {available}")
        k_extra, pos_extra, quat_extra = (
            _grab(images, f"{name}_{term}", env_id) for term in ("K", "pos", "quat")
        )
        if k_extra is None or pos_extra is None or quat_extra is None:
            raise KeyError(f"extra camera {name} needs _K, _pos and _quat terms")
        depth_extra = _grab(images, f"{name}_depth", env_id)
        view = CameraFrame(
            name=EXTRA_VIEW_NAMES.get(name, name),
            rgb=_as_uint8(rgb_extra),
            t=t,
            depth=None if depth_extra is None else np.asarray(depth_extra, np.float32).squeeze(),
            K=np.asarray(k_extra, dtype=np.float64).reshape(3, 3),
            pose=convert_camera_pose(Pose(pos_extra, quat_extra), convention),
            convention="opencv",
            state_at_capture=observation.tcp,
        )
        views.append(view)
    if views:
        observation.extra["views"] = views
    return observation


def _as_uint8(array: npt.NDArray) -> npt.NDArray[np.uint8]:
    """Convert to uint8 RGB. RoboLab returns float or RGBA images in some configurations."""
    out = np.asarray(array)
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[..., :3]
    if out.dtype == np.uint8:
        return out
    if np.issubdtype(out.dtype, np.floating):
        peak = float(np.nanmax(out)) if out.size else 1.0
        out = out * 255.0 if peak <= 1.001 else out
    return np.clip(out, 0, 255).astype(np.uint8)


def action_chunk(
    current: Pose,
    target_pose: Pose,
    grip: float,
    horizon: int,
    max_dxyz: float = MAX_DXYZ_M,
    max_drot: float = MAX_DROT_RAD,
    scale: float = REL_IK_SCALE,
    tcp_offset_m: float = FLANGE_TO_FINGERTIP_M,
    gain: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Relative-IK action steps that walk the flange from ``current`` toward ``target_pose``.

    ``max_dxyz`` and ``max_drot`` bound the travel planned per step. ``gain`` is the
    fraction of a commanded step the controller realises; commands are divided by it so the
    arm follows the planned path. 1.0 means no compensation: the arm then closes only a
    fixed share of the remaining gap per chunk and converges geometrically, so a 26 cm
    approach takes about 8 s. See TRACKING_GAIN.

    Rotation is the axis-angle vector taking one orientation to the other, split evenly
    across the chunk. That is what ``droll dpitch dyaw`` means to the differential IK
    controller: a small rotation vector on root axes, not ordered Euler angles. The
    difference appears only for large rotations, which the per-step ceiling excludes.
    """
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain is the realised fraction of a commanded step, in (0, 1]")
    horizon = max(1, int(horizon))
    displacement = target_pose.pos - current.pos
    omega = _rotation_vector(current.quat, target_pose.quat)
    angle = float(np.linalg.norm(omega))
    # Bound flange travel too: rotation of the TCP offset adds up to offset * angle.
    travel_bound = float(np.linalg.norm(displacement)) + abs(tcp_offset_m) * angle
    fraction = min(
        1.0 / horizon,
        max_dxyz / travel_bound if travel_bound > 0 else 1.0,
        max_drot / angle if angle > 0 else 1.0,
    )
    chunk = np.zeros((horizon, 7), dtype=np.float32)
    previous = tcp_to_flange(current, tcp_offset_m)
    for i in range(horizon):
        f = min(1.0, (i + 1) * fraction)
        rotation = quat_to_rot(current.quat)
        if angle > 1e-12:
            rotation = quat_to_rot(quat_from_axis_angle(omega / angle, angle * f)) @ rotation
        tcp = Pose(current.pos + displacement * f, rot_to_quat(rotation))
        flange = tcp_to_flange(tcp, tcp_offset_m)
        chunk[i, :3] = flange.pos - previous.pos
        chunk[i, 3:6] = _rotation_vector(previous.quat, flange.quat)
        previous = flange
    chunk[:, 6] = float(np.clip(grip, 0.0, 1.0))
    chunk[:, :6] /= scale * gain
    return chunk


def steps_needed(
    current: Pose,
    target_pose: Pose,
    max_dxyz: float = MAX_DXYZ_M,
    max_drot: float = MAX_DROT_RAD,
    tcp_offset_m: float = FLANGE_TO_FINGERTIP_M,
) -> int:
    """How many control steps the move needs at the per-step ceiling."""
    rot = quat_angle_between(current.quat, target_pose.quat)
    lin = float(np.linalg.norm(target_pose.pos - current.pos)) + abs(tcp_offset_m) * rot
    # The epsilon keeps an exactly-one-step move from rounding up: 0.32 - 0.30 is
    # 0.020000000000000018 in floating point, and ceil of 1.0000000000000009 is 2.
    return max(1, int(np.ceil(max(lin / max_dxyz, rot / max_drot) - 1e-9)))


def _rotation_vector(q_from: npt.ArrayLike, q_to: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Axis-angle vector taking one orientation to the other, in the root frame."""
    r = quat_to_rot(q_to) @ quat_to_rot(q_from).T
    quat = rot_to_quat(r)
    angle = 2.0 * float(np.arccos(np.clip(quat[0], -1.0, 1.0)))
    axis_norm = float(np.linalg.norm(quat[1:]))
    if axis_norm < 1e-9 or angle < 1e-9:
        return np.zeros(3)
    if angle > np.pi:
        angle -= 2 * np.pi
    return (quat[1:] / axis_norm) * angle


def _clip_norm(vector: npt.NDArray[np.float64], limit: float) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit or norm == 0 else vector * (limit / norm)


def detect_camera_convention(
    raw_obs: dict,
    looks_at,
    env_id: int = 0,
    camera: str = SCENE_CAMERA,
    candidates: tuple[str, ...] = ("opengl", "ros"),
) -> tuple[str | None, dict[str, float]]:
    """Which convention makes this camera point at something it is known to be looking at.

    ``looks_at`` is a point in the robot root frame that must be in view, in practice the
    middle of the table. For each candidate the camera's converted +z is compared with the
    direction from the camera to that point; the right convention scores near +1 and the
    wrong one near -1.

    Returns ``(best, scores)``, with ``best`` None when no candidate scores above 0.5.
    Measure once per tree and record the result.
    """
    images = raw_obs.get("image_obs")
    if not isinstance(images, dict):
        raise KeyError("expected an image_obs group")
    cam_pos = _grab(images, f"{camera}_pos", env_id)
    cam_quat = _grab(images, f"{camera}_quat", env_id)
    if cam_pos is None or cam_quat is None:
        raise KeyError(f"{camera} has no pose; cannot measure the convention")

    target = np.asarray(looks_at, dtype=np.float64).reshape(3)
    to_target = target - np.asarray(cam_pos, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(to_target))
    if norm < 1e-9:
        raise ValueError("looks_at is at the camera's own position")
    to_target /= norm

    scores: dict[str, float] = {}
    for candidate in candidates:
        pose = convert_camera_pose(Pose(cam_pos, cam_quat), candidate)
        forward = quat_to_rot(pose.quat) @ np.array([0.0, 0.0, 1.0])
        scores[candidate] = float(np.dot(forward, to_target))
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0.5 else None), scores


def capabilities_from_observation(
    raw_obs: dict, env_id: int = 0, camera: str = SCENE_CAMERA, **kwargs
):
    """Declare camera sizes from delivered data, including the optional wrist view."""
    images = raw_obs["image_obs"]
    scene = _grab(images, camera, env_id)
    if scene is None:
        raise KeyError(f"missing scene camera {camera}")
    wrist = _grab(images, WRIST_CAMERA, env_id)
    wrist_size = None if wrist is None else (wrist.shape[1], wrist.shape[0])
    return droid_capabilities(
        scene_size=(scene.shape[1], scene.shape[0]), wrist_size=wrist_size, **kwargs
    )
