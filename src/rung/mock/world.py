"""A dependency-free tabletop world: oriented boxes on a plane, one camera, one gripper.

Not a physics simulator. Exercises the full harness without a GPU or Isaac Sim, with
perspective projection, optical-axis depth and occlusion, a gripper that only grasps what
fits between its jaws, and objects that drop to the table on release.

Not modelled: contact dynamics, friction, toppling, deformable objects, and collisions
between a held object and the scene during transit.

Grounding cannot be evaluated on the mock. Objects are featureless coloured boxes, so a
vision model asked for "the apple" correctly reports a red block. Use ``OracleGrounding``
here; grounding quality needs photorealistic renders or real images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from rung.contracts.frames import Pose, quat_from_axis_angle, quat_to_rot


@dataclass
class Body:
    """An oriented box resting on the table, yawed about the world z axis."""

    name: str
    pos: npt.NDArray[np.float64]  # centre, root frame, metres
    size: npt.NDArray[np.float64]  # full extents (length, width, height)
    yaw: float = 0.0
    color: tuple[int, int, int] = (180, 180, 180)
    graspable: bool = True
    container: bool = False
    wall_thickness_m: float = 0.012

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)

    @property
    def rot(self) -> npt.NDArray[np.float64]:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + self.size[2] / 2)

    @property
    def bottom_z(self) -> float:
        return float(self.pos[2] - self.size[2] / 2)

    @property
    def footprint(self) -> tuple[float, float]:
        """Length and width in the xy plane, longer side first."""
        a, b = float(self.size[0]), float(self.size[1])
        return (a, b) if a >= b else (b, a)

    @property
    def min_width(self) -> float:
        return min(float(self.size[0]), float(self.size[1]))

    def contains_xy(self, p: npt.ArrayLike, margin: float = 0.0) -> bool:
        q = self.rot.T @ (np.asarray(p, dtype=np.float64).reshape(3) - self.pos)
        return bool(
            abs(q[0]) <= self.size[0] / 2 + margin and abs(q[1]) <= self.size[1] / 2 + margin
        )


@dataclass
class Camera:
    name: str
    pose: Pose  # camera in root frame, OpenCV axes
    width: int
    height: int
    fov_y_deg: float = 50.0

    @property
    def K(self) -> npt.NDArray[np.float64]:
        f = (self.height / 2) / np.tan(np.radians(self.fov_y_deg) / 2)
        return np.array(
            [[f, 0.0, self.width / 2 - 0.5], [0.0, f, self.height / 2 - 0.5], [0.0, 0.0, 1.0]]
        )


def look_at(eye: npt.ArrayLike, target: npt.ArrayLike, up: npt.ArrayLike = (0, 0, 1)) -> Pose:
    """Camera pose with OpenCV axes: +z toward the target, +y down, +x right."""
    e = np.asarray(eye, dtype=np.float64).reshape(3)
    t = np.asarray(target, dtype=np.float64).reshape(3)
    u = np.asarray(up, dtype=np.float64).reshape(3)
    z = t - e
    z /= np.linalg.norm(z)
    x = np.cross(z, u)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # looking straight along up; pick any perpendicular
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    from rung.contracts.frames import rot_to_quat

    return Pose(e, rot_to_quat(np.stack([x, y, z], axis=1)))


TABLE_COLOR = np.array([196, 186, 170], dtype=np.float64)
BACKDROP_COLOR = np.array([88, 92, 98], dtype=np.float64)


class Scene:
    """Bodies on a table plane, rendered by ray casting."""

    def __init__(self, bodies: list[Body], table_z: float = 0.0, table_half: float = 0.6):
        self.bodies = bodies
        self.table_z = table_z
        self.table_half = table_half

    def body(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def render(
        self, cam: Camera, extra: list[Body] | None = None
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Returns ``(rgb, depth, body_index)``.

        ``depth`` is optical-axis depth in metres, NaN where nothing is hit.
        ``body_index`` is the index into ``bodies + extra``, -1 for table, -2 for background.
        The index map is ground truth and is used only by diagnostics and by the mock
        perception source, never handed to a policy.
        """
        bodies = list(self.bodies) + list(extra or [])
        k = cam.K
        h, w = cam.height, cam.width
        vs, us = np.mgrid[0:h, 0:w]
        dirs_cam = np.stack(
            [(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1], np.ones_like(us, dtype=float)],
            axis=-1,
        )
        rot = cam.pose.rot
        dirs = dirs_cam @ rot.T  # world directions, unnormalised (z_cam == 1)
        origin = cam.pose.pos

        best_t = np.full((h, w), np.inf)
        idx = np.full((h, w), -2, dtype=np.int32)

        # table plane z = table_z, bounded square
        with np.errstate(divide="ignore", invalid="ignore"):
            t_tab = (self.table_z - origin[2]) / dirs[..., 2]
        hit = np.isfinite(t_tab) & (t_tab > 0)
        p = origin + dirs * t_tab[..., None]
        hit &= (np.abs(p[..., 0]) < self.table_half + 0.4) & (np.abs(p[..., 1]) < self.table_half)
        upd = hit & (t_tab < best_t)
        best_t = np.where(upd, t_tab, best_t)
        idx = np.where(upd, np.int32(-1), idx)

        for i, b in enumerate(bodies):
            t_hit = _ray_box(origin, dirs, b)
            upd = np.isfinite(t_hit) & (t_hit < best_t)
            best_t = np.where(upd, t_hit, best_t)
            idx = np.where(upd, np.int32(i), idx)

        # shade
        rgb = np.zeros((h, w, 3), dtype=np.float64)
        rgb[idx == -2] = BACKDROP_COLOR
        rgb[idx == -1] = TABLE_COLOR
        for i, b in enumerate(bodies):
            m = idx == i
            if m.any():
                rgb[m] = np.asarray(b.color, dtype=np.float64)
        # cheap depth cue so edges are visible after JPEG encoding
        finite = np.isfinite(best_t)
        near = float(best_t[finite].min()) if finite.any() else 0.0
        shade = np.where(finite, np.clip(1.15 - (best_t - near) * 0.35, 0.6, 1.15), 1.0)
        rgb = np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8)

        depth = np.where(np.isfinite(best_t), best_t, np.nan).astype(np.float32)
        return rgb, depth, idx


def _ray_box(origin: npt.NDArray[np.float64], dirs: npt.NDArray[np.float64], b: Body):
    """Slab intersection in the body's local frame. ``dirs`` is (..., 3), not normalised.

    Returns the ray parameter t (which equals optical-axis depth because the camera-frame
    z component of every direction is 1), inf where there is no hit.
    """
    r = b.rot
    o = (origin - b.pos) @ r  # world -> local: r.T @ v, done as row-vector @ r
    d = dirs @ r
    half = b.size / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.maximum.reduce(np.minimum(t1, t2), axis=-1)
    tmax = np.minimum.reduce(np.maximum(t1, t2), axis=-1)
    ok = (tmax >= np.maximum(tmin, 0)) & np.isfinite(tmin)
    return np.where(ok & (tmin > 0), tmin, np.inf)


@dataclass
class GripperModel:
    """Jaw geometry used both for rendering the hand and for deciding what it can hold."""

    max_opening_m: float = 0.085
    finger_length_m: float = 0.055
    pad_width_m: float = 0.022
    close_time_s: float = 1.0


@dataclass
class RobotState:
    tcp: Pose
    travel: float = 0.0  # 0 open, 1 closed
    held: str | None = None
    held_offset: npt.NDArray[np.float64] | None = None
    held_yaw_offset: float = 0.0
    contacts: list[str] = field(default_factory=list)


def gripper_bodies(tcp: Pose, travel: float, g: GripperModel) -> list[Body]:
    """Two finger boxes and a palm, so the hand occludes the scene like a real one."""
    opening = (1.0 - travel) * g.max_opening_m
    rot = quat_to_rot(tcp.quat)
    yaw = float(np.arctan2(rot[1, 1], rot[0, 1]))  # yaw of the local +y (closing) axis
    out: list[Body] = []
    for sign in (-1.0, 1.0):
        offset = rot @ np.array(
            [0.0, sign * (opening / 2 + g.pad_width_m / 2), -g.finger_length_m / 2]
        )
        out.append(
            Body(
                name=f"_finger{int(sign)}",
                pos=tcp.pos + offset,
                size=np.array([g.pad_width_m, g.pad_width_m, g.finger_length_m]),
                yaw=yaw,
                color=(70, 74, 80),
                graspable=False,
            )
        )
    palm = rot @ np.array([0.0, 0.0, -g.finger_length_m - 0.02])
    out.append(
        Body(
            name="_palm",
            pos=tcp.pos + palm,
            size=np.array([0.06, g.max_opening_m + 0.05, 0.04]),
            yaw=yaw,
            color=(52, 56, 62),
            graspable=False,
        )
    )
    return out


def grasp_check(tcp: Pose, travel_target: float, body: Body, g: GripperModel) -> tuple[bool, str]:
    """Would closing here hold this body? Returns ``(holds, reason)``.

    Purely geometric and strict, so harness bugs surface: the fingertips must straddle the
    body, the body's extent along the closing axis must fit the jaw, and the fingers must
    be around the body rather than above or below it.
    """
    rot = quat_to_rot(tcp.quat)
    local = rot.T @ (body.pos - tcp.pos)
    close_axis_extent = _extent_along(body, rot[:, 1])
    if close_axis_extent > g.max_opening_m:
        return False, f"too wide for the jaw ({close_axis_extent * 100:.1f} cm > 8.5 cm)"
    if abs(local[1]) > g.max_opening_m / 2:
        return False, "not between the fingers"
    if abs(local[0]) > max(0.03, _extent_along(body, rot[:, 0]) / 2):
        return False, "off to the side of the fingers"
    # fingertips are at tcp; the body must not be entirely below the fingertips
    if body.top_z < tcp.pos[2] - 0.005:
        return False, "fingers are above the object"
    if body.bottom_z > tcp.pos[2] + g.finger_length_m:
        return False, "fingers are below the object"
    if not body.graspable:
        return False, "not graspable"
    if travel_target < 0.2:
        return False, "gripper not closing"
    return True, "grasped"


def _extent_along(body: Body, axis: npt.NDArray[np.float64]) -> float:
    """Full width of an oriented box measured along a unit axis."""
    r = body.rot
    return float(np.sum(np.abs(r.T @ axis) * body.size))


def top_down_pose(pos: npt.ArrayLike, yaw: float = 0.0) -> Pose:
    """TCP pointing straight down with the closing axis at ``yaw`` about world z."""
    down = quat_from_axis_angle([1.0, 0.0, 0.0], np.pi)  # +z local -> -z world
    about_z = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    from rung.contracts.frames import quat_mul

    return Pose(np.asarray(pos, dtype=np.float64).reshape(3), quat_mul(about_z, down))
