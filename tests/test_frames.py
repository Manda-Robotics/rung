"""Frame, rotation and camera-geometry conventions.

These cover the conventions that fail silently: quaternion order, transform direction,
camera axis convention, and intrinsics that no longer match a resized image.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.contracts.frames import (
    Pose,
    convert_camera_pose,
    crop_intrinsics,
    deproject,
    depth_image_to_points,
    project,
    quat_angle_between,
    quat_from_axis_angle,
    quat_mul,
    quat_to_rot,
    rescale_intrinsics,
    rot_to_quat,
    yaw_of,
)


def test_quat_rot_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        assert np.allclose(rot_to_quat(quat_to_rot(q)), q, atol=1e-9)


def test_rot_to_quat_all_branches():
    """Each branch of the trace-based selection must round trip."""
    for axis, angle in [
        ((0, 0, 1), 0.1),
        ((1, 0, 0), np.pi - 1e-3),
        ((0, 1, 0), np.pi - 1e-3),
        ((0, 0, 1), np.pi - 1e-3),
    ]:
        q = quat_from_axis_angle(axis, angle)
        assert np.allclose(quat_to_rot(rot_to_quat(quat_to_rot(q))), quat_to_rot(q), atol=1e-7)


def test_quat_is_wxyz_not_xyzw():
    """A 90 degree turn about z maps +x to +y. If the order were (x, y, z, w) it would not."""
    q = quat_from_axis_angle([0, 0, 1], np.pi / 2)
    assert np.allclose(q[0], np.cos(np.pi / 4))
    assert np.allclose(quat_to_rot(q) @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)


def test_pose_maps_child_into_parent():
    """A pose of B in A takes B-coordinates to A-coordinates, not the other way."""
    p = Pose(np.array([1.0, 2.0, 3.0]), quat_from_axis_angle([0, 0, 1], np.pi / 2))
    assert np.allclose(p.apply([1.0, 0.0, 0.0]), [1.0, 3.0, 3.0])
    assert np.allclose(p.inverse().apply(p.apply([0.3, -0.2, 0.5])), [0.3, -0.2, 0.5])


def test_pose_compose_order():
    a = Pose(np.array([1.0, 0, 0]), quat_from_axis_angle([0, 0, 1], np.pi / 2))
    b = Pose(np.array([0.0, 0.5, 0]), quat_from_axis_angle([0, 0, 1], np.pi / 2))
    v = np.array([0.2, 0.1, 0.0])
    assert np.allclose((a * b).apply(v), a.apply(b.apply(v)))


def test_quat_mul_matches_matrix_product():
    a = quat_from_axis_angle([0.3, 0.5, 0.8], 0.7)
    b = quat_from_axis_angle([1.0, 0.1, -0.2], -1.1)
    assert np.allclose(quat_to_rot(quat_mul(a, b)), quat_to_rot(a) @ quat_to_rot(b), atol=1e-9)


def test_quat_angle_between():
    q = quat_from_axis_angle([0, 0, 1], 0.6)
    assert quat_angle_between(np.array([1.0, 0, 0, 0]), q) == pytest.approx(0.6, abs=1e-9)


def test_yaw_of_default_axis_is_closing_axis():
    q = quat_from_axis_angle([0, 0, 1], 0.4)
    assert yaw_of(q) == pytest.approx(np.pi / 2 + 0.4, abs=1e-9)


def _camera_looking_at(eye, target):
    """The OpenCV basis for a camera at ``eye`` pointing at ``target``: +z forward, +y down."""
    e = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - e
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return Pose(e, rot_to_quat(np.stack([x, np.cross(z, x), z], axis=1)))


def test_project_deproject_round_trip():
    K = np.array([[240.0, 0, 160.0], [0, 240.0, 120.0], [0, 0, 1.0]])
    cam = _camera_looking_at([-0.2, -0.4, 0.5], [0.4, 0.0, 0.05])
    pts = np.array([[0.4, 0.0, 0.05], [0.5, 0.2, 0.1], [0.3, -0.15, 0.02]])
    uv, z = project(pts, K, cam)
    assert np.all(z > 0)
    back = deproject(uv, z, K, cam)
    assert np.allclose(back, pts, atol=1e-9)


def test_deproject_uses_optical_axis_depth_not_ray_distance():
    """An off-centre pixel at depth d is further than d from the camera. If the harness
    treated depth as ray distance the point would land short."""
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    p = deproject(np.array([90.0, 50.0]), 2.0, K)
    assert p[2] == pytest.approx(2.0)
    assert np.linalg.norm(p) > 2.0


def test_convert_camera_pose_opengl_to_opencv():
    """OpenGL looks along -z with +y up; OpenCV along +z with +y down."""
    gl = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    cv = convert_camera_pose(gl, "opengl")
    assert np.allclose(cv.rot @ np.array([0, 0, 1.0]), [0, 0, -1.0], atol=1e-9)
    assert np.allclose(cv.rot @ np.array([0, 1.0, 0]), [0, -1.0, 0], atol=1e-9)
    assert np.allclose(cv.rot @ np.array([1.0, 0, 0]), [1.0, 0, 0], atol=1e-9)


def test_convert_camera_pose_is_involution_and_noop_for_opencv():
    p = Pose(np.array([0.1, 0.2, 0.3]), quat_from_axis_angle([0.2, 1.0, 0.3], 0.8))
    assert convert_camera_pose(p, "opencv") is p
    twice = convert_camera_pose(convert_camera_pose(p, "opengl"), "opengl")
    assert np.allclose(twice.rot, p.rot, atol=1e-9)


def test_ros_is_an_accepted_name_for_the_opencv_axes():
    """``ros`` and ``opencv`` name the same axes. The Isaac Sim 6 RoboLab port reports camera
    extrinsics in ROS axes, so no rotation is applied."""
    p = Pose(np.array([0.1, 0.2, 0.3]), quat_from_axis_angle([0.2, 1.0, 0.3], 0.8))
    assert convert_camera_pose(p, "ros") is p


def test_convert_camera_pose_rejects_unknown():
    with pytest.raises(ValueError):
        convert_camera_pose(Pose.identity(), "left_handed_z_up")


def test_rescale_intrinsics_keeps_projection_consistent():
    K = np.array([[240.0, 0, 160.0], [0, 240.0, 120.0], [0, 0, 1.0]])
    p = np.array([[0.1, 0.05, 0.8]])
    uv, _ = project(p, K)
    uv_half, _ = project(p, rescale_intrinsics(K, 0.5, 0.5))
    assert np.allclose(uv_half, uv * 0.5, atol=1e-9)


def test_crop_intrinsics_shifts_principal_point():
    K = np.array([[240.0, 0, 160.0], [0, 240.0, 120.0], [0, 0, 1.0]])
    k2 = crop_intrinsics(K, 40, 20)
    assert k2[0, 2] == pytest.approx(120.0)
    assert k2[1, 2] == pytest.approx(100.0)


def test_depth_image_to_points_masks_invalid():
    K = np.array([[100.0, 0, 15.5], [0, 100.0, 11.5], [0, 0, 1.0]])
    d = np.full((24, 32), 1.5, dtype=np.float32)
    d[0, 0] = np.nan
    d[1, 1] = 0.0
    pts, idx = depth_image_to_points(d, K)
    assert len(pts) == 24 * 32 - 2
    assert len(idx) == len(pts)
    assert np.all(np.isfinite(pts))
