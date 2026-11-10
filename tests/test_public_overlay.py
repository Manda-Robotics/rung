import numpy as np
import pytest

from rung.contracts.capabilities import Workspace
from rung.contracts.frames import Pose
from rung.contracts.observation import CameraFrame
from rung.interfaces.public_overlay import public_geometry_overlay, public_tcp_marker


def test_overlay_uses_public_calibration_and_does_not_mutate_rgb():
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    frame = CameraFrame(
        "scene", rgb, 0, K=np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]]), pose=Pose.identity()
    )
    workspace = Workspace(np.array([-0.5, -0.5, 1]), np.array([0.5, 0.5, 2]), 1.0)
    tcp = Pose(np.array([0, 0, 1.5]), np.array([1, 0, 0, 0]))
    result = public_geometry_overlay(frame, tcp, workspace)
    assert result.shape == rgb.shape and result.dtype == np.uint8
    assert result[100, 100, 0] > 200 and result[100, 100, 2] > 150
    assert not rgb.any()
    assert frame.depth is None  # no object depth or segmentation needed
    frame.K = None
    with pytest.raises(ValueError, match="calibrated"):
        public_geometry_overlay(frame, tcp, workspace)


def test_tcp_marker_leaves_selected_centre_and_original_pixels_visible():
    rgb = np.full((200, 200, 3), 70, dtype=np.uint8)
    frame = CameraFrame(
        "wrist", rgb, 0, K=np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]]), pose=Pose.identity()
    )
    tcp = Pose(np.array([0, 0, 1.0]), np.array([1, 0, 0, 0]))
    shown = public_tcp_marker(frame, tcp)
    np.testing.assert_array_equal(shown[100, 100], rgb[100, 100])
    assert shown[100, 105, 0] == 255 and shown[100, 105, 2] == 220
    assert np.all(rgb == 70) and frame.depth is None
    behind = Pose(np.array([0, 0, -1.0]), np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(public_tcp_marker(frame, behind), rgb)
    frame.K = None
    with pytest.raises(ValueError, match="calibrated"):
        public_tcp_marker(frame, tcp)
