"""Frames, rotations and camera geometry.

Every convention the harness relies on is stated here once, and every other module
imports from here rather than restating it.

Conventions
-----------
* Positions are metres. Angles are radians.
* Quaternions are ``(w, x, y, z)``, unit length, representing the rotation that maps
  a vector expressed in the child frame into the parent frame.
* A ``Pose`` of frame B "in" frame A holds ``pos`` = origin of B expressed in A and
  ``quat`` = rotation that maps B-coordinates into A-coordinates. Composition is
  ``pose_A_C = pose_A_B * pose_B_C``.
* The robot root frame is the fixed base frame of the arm. The table top is at a
  height stated by the embodiment's capabilities, never assumed.
* Cameras use the OpenCV convention inside the harness: ``+x`` right, ``+y`` down,
  ``+z`` forward along the optical axis. Adapters convert from whatever the source
  uses (Isaac / USD cameras are OpenGL: ``+x`` right, ``+y`` up, ``-z`` forward).
* Depth images are optical-axis depth (the ``z`` of the point in camera coordinates),
  not the Euclidean ray distance, in metres, ``float32``, ``NaN`` where invalid.
* Intrinsics ``K`` are valid for the image at the resolution delivered. Resizing an
  image rescales ``fx, fy, cx, cy`` by the same factors; cropping shifts ``cx, cy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]
Quat = npt.NDArray[np.float64]
Mat3 = npt.NDArray[np.float64]

CAMERA_CONVENTIONS = ("opencv", "opengl", "ros")
"""``ros`` is the same axes as ``opencv``: +x right, +y down, +z along the optical axis.
Both names appear in simulator documentation, so both are accepted and mean one thing."""

# Rotation applied to camera-frame coordinates to go from OpenGL axes to OpenCV axes
# (and back; it is its own inverse): flip y and z.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Pose:
    """A rigid transform: child frame expressed in a parent frame."""

    pos: Vec3
    quat: Quat  # (w, x, y, z)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=np.float64).reshape(3))
        q = np.asarray(self.quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError("quaternion must be finite and non-zero")
        object.__setattr__(self, "quat", q / n)

    @staticmethod
    def identity() -> Pose:
        return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    @property
    def rot(self) -> Mat3:
        return quat_to_rot(self.quat)

    def apply(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points from the child frame into the parent frame. Shape (..., 3)."""
        p = np.asarray(points, dtype=np.float64)
        return p @ self.rot.T + self.pos

    def inverse(self) -> Pose:
        r_t = self.rot.T
        return Pose(-(r_t @ self.pos), rot_to_quat(r_t))

    def compose(self, other: Pose) -> Pose:
        """``self * other``: other is expressed in self's child frame."""
        return Pose(self.apply(other.pos), quat_mul(self.quat, other.quat))

    def __mul__(self, other: Pose) -> Pose:
        return self.compose(other)


def quat_to_rot(q: npt.ArrayLike) -> Mat3:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(r: npt.ArrayLike) -> Quat:
    """Rotation matrix to (w, x, y, z). Numerically stable branch selection."""
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        x = 0.25 * s
        w, y, z = (m[2, 1] - m[1, 2]) / s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        y = 0.25 * s
        w, x, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        z = 0.25 * s
        w, x, y = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_mul(a: npt.ArrayLike, b: npt.ArrayLike) -> Quat:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_axis_angle(axis: npt.ArrayLike, angle: float) -> Quat:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * a])


def quat_angle_between(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Smallest rotation angle in radians taking orientation a to b."""
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    d = abs(float(np.dot(qa, qb)) / (np.linalg.norm(qa) * np.linalg.norm(qb)))
    return 2.0 * float(np.arccos(min(1.0, d)))


def yaw_of(quat: npt.ArrayLike, axis_local: npt.ArrayLike = (0.0, 1.0, 0.0)) -> float:
    """Yaw (about root +z) of a local axis expressed in the root frame."""
    v = quat_to_rot(quat) @ np.asarray(axis_local, dtype=np.float64)
    return float(np.arctan2(v[1], v[0]))


def convert_camera_pose(pose: Pose, source_convention: str) -> Pose:
    """Return the camera pose with OpenCV axes, given the axes the source uses.

    The camera origin does not move; only which way its axes point changes.
    """
    if source_convention not in CAMERA_CONVENTIONS:
        raise ValueError(f"unknown camera convention {source_convention!r}")
    if source_convention in ("opencv", "ros"):
        return pose
    return Pose(pose.pos, rot_to_quat(pose.rot @ _GL_TO_CV))


def deproject(
    uv: npt.ArrayLike, depth: npt.ArrayLike, K: npt.ArrayLike, cam_pose: Pose | None = None
) -> npt.NDArray[np.float64]:
    """Pixels plus optical-axis depth to 3D points. Camera frame, or root frame if a pose is given.

    ``uv`` is (..., 2) with ``u`` along image columns and ``v`` along rows, pixel centres at
    integer coordinates. ``depth`` broadcasts against ``uv[..., 0]``.
    """
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    uv_arr = np.asarray(uv, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    x = (uv_arr[..., 0] - k[0, 2]) * z / k[0, 0]
    y = (uv_arr[..., 1] - k[1, 2]) * z / k[1, 1]
    pts = np.stack([x, y, z], axis=-1)
    return cam_pose.apply(pts) if cam_pose is not None else pts


def project(
    points: npt.ArrayLike, K: npt.ArrayLike, cam_pose: Pose | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """3D points to pixels. Returns ``(uv, z)`` where ``z`` is optical-axis depth.

    Points behind the camera get ``z <= 0``; callers must mask them.
    """
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    p = np.asarray(points, dtype=np.float64)
    if cam_pose is not None:
        p = cam_pose.inverse().apply(p)
    z = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * p[..., 0] / z + k[0, 2]
        v = k[1, 1] * p[..., 1] / z + k[1, 2]
    return np.stack([u, v], axis=-1), z


def depth_image_to_points(
    depth: npt.ArrayLike, K: npt.ArrayLike, cam_pose: Pose | None = None, stride: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Every valid depth pixel to a 3D point.

    Returns ``(points (N, 3), flat pixel indices (N,))``.
    """
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    dd = d[::stride, ::stride]
    valid = np.isfinite(dd) & (dd > 0)
    uv = np.stack([us[valid], vs[valid]], axis=-1)
    pts = deproject(uv, dd[valid], K, cam_pose)
    idx = (vs[valid] * w + us[valid]).astype(np.intp)
    return pts, idx


def rescale_intrinsics(K: npt.ArrayLike, sx: float, sy: float) -> Mat3:
    """Intrinsics after resizing the image by ``sx`` horizontally and ``sy`` vertically."""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def crop_intrinsics(K: npt.ArrayLike, x0: int, y0: int) -> Mat3:
    """Intrinsics after cropping so that pixel ``(x0, y0)`` becomes ``(0, 0)``."""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    return k
