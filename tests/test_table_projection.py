import numpy as np
import pytest

from rung.contracts.frames import Pose, project
from rung.contracts.observation import CameraFrame
from rung.interfaces.table_projection import project_table_pixel


def camera():
    return CameraFrame(
        "scene",
        np.zeros((101, 101, 3), np.uint8),
        2.0,
        K=np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
        pose=Pose([0.5, 0, 1], [0, 1, 0, 0]),
    )


def test_selected_ray_uses_only_public_calibration_and_table():
    frame = camera()
    # No depth exists; the selected pixel comes from the model, not a scene query.
    uv, _ = project(np.array([[0.7, 0.1, 0]]), frame.K, frame.pose)
    result = project_table_pixel(frame, *uv[0] / 100, 2.0, 0.0)
    np.testing.assert_allclose(result["table_intersection_xyz_m"], [0.7, 0.1, 0])
    assert "no depth or object lookup" in result["method"]


@pytest.mark.parametrize(
    "u,v,t", [(True, 0.5, 2.0), (0.5, 0.5, 1.0), (1.1, 0.5, 2.0), (float("nan"), 0.5, 2.0)]
)
def test_rejects_invalid_or_stale_selection(u, v, t):
    with pytest.raises(ValueError):
        project_table_pixel(camera(), u, v, t, 0.0)


def test_two_rgb_correspondences_recover_a_raised_point_without_depth():
    from rung.interfaces.table_projection import triangulate_selected_pixels

    scene, wrist = camera(), camera()
    scene.pose = Pose([0, 0, 1], [0, 1, 0, 0])
    wrist.pose = Pose([0.4, 0, 1], [0, 1, 0, 0])
    wrist.name = "wrist"
    point = np.array([[0.2, 0.1, 0.3]])
    uv1, _ = project(point, scene.K, scene.pose)
    uv2, _ = project(point, wrist.K, wrist.pose)
    result = triangulate_selected_pixels(
        scene, wrist, (uv1[0] / 100).tolist(), (uv2[0] / 100).tolist(), 2.0
    )
    np.testing.assert_allclose(result["selected_point_xyz_m"], point[0], atol=1e-12)
    assert result["ray_gap_m"] < 1e-12
    assert scene.depth is None and wrist.depth is None
    with pytest.raises(ValueError, match="nearly parallel"):
        triangulate_selected_pixels(scene, wrist, [0.5, 0.5], [0.5, 0.5], 2.0)
    with pytest.raises(ValueError, match="miss by"):
        triangulate_selected_pixels(scene, wrist, (uv1[0] / 100).tolist(), [0.2, 0.2], 2.0)


def test_public_reprojection_roundtrip_and_no_hidden_depth_dependency():
    from rung.interfaces.table_projection import reproject_public_point

    frame = camera()
    selected = [0.7, 0.6]
    point = project_table_pixel(frame, *selected, 2.0, 0.0)["table_intersection_xyz_m"]
    result = reproject_public_point(point, [frame, None])
    view = result["views"]["scene"]
    np.testing.assert_allclose(view["normalized_uv"], selected, atol=1e-12)
    assert view["inside_image"] and view["in_front_of_camera"]
    assert frame.depth is None
    assert "not proof of object identity" in result["limitation"]


def test_public_reprojection_labels_behind_camera_and_off_image_without_clipping():
    from rung.interfaces.table_projection import reproject_public_point

    frame = camera()
    behind = reproject_public_point([0.5, 0, 2], [frame])["views"]["scene"]
    assert not behind["in_front_of_camera"] and behind["normalized_uv"] is None
    outside = reproject_public_point([10, 0, 0], [frame])["views"]["scene"]
    assert outside["in_front_of_camera"] and not outside["inside_image"]
    assert not all(0 <= x <= 1 for x in outside["normalized_uv"])
