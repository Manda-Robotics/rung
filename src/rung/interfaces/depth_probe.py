"""Calibrated surface point under a model-selected pixel, from the rig's own depth image.

Declared-input geometry: the harness reads its depth camera at the pixel the model chose.
It returns the visible surface point, not an object centre, not a grasp pose, and not an
object identity. Missing depth, depth edges and stale captures are reported, never
silently patched with the table plane.
"""

from __future__ import annotations

import numpy as np

from rung.interfaces.table_projection import _selected_ray

EDGE_SPREAD_M = 0.02
"""Depth range inside the sampling window above which the pixel sits on a depth edge:
the median then mixes two surfaces (an object rim and the table behind it)."""


def probe_depth_pixel(frame, u, v, capture_time_s, table_z=None, window_px=5):
    """Surface point in the robot root frame under normalized pixel ``(u, v)``.

    The depth image is optical-axis depth, so the camera-frame point is
    ``depth * K^-1 [col, row, 1]``. A window median rejects single bad pixels; the
    window's depth spread flags edges. Raises on stale captures, missing calibration,
    and windows without any valid depth, so the model learns the probe failed rather
    than receiving a fabricated point.
    """
    if window_px < 1 or window_px % 2 == 0:
        raise ValueError("window must be an odd positive pixel count")
    # Reuses the public-ray validation: finite coordinates, [0,1] range, capture time,
    # OpenCV calibration. The ray itself is not used; depth gives the range.
    _selected_ray(frame, u, v, capture_time_s)
    depth = frame.depth
    if depth is None:
        raise ValueError("selected camera has no depth image")
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (frame.height, frame.width):
        raise ValueError("depth image does not match the RGB resolution")
    col = int(round(u * (frame.width - 1)))
    row = int(round(v * (frame.height - 1)))
    half = window_px // 2
    window = depth[max(0, row - half) : row + half + 1, max(0, col - half) : col + half + 1]
    finite = window[np.isfinite(window) & (window > 0)]
    if finite.size == 0:
        raise ValueError("no valid depth in the sampling window; reselect the pixel")
    z = float(np.median(finite))
    spread = float(finite.max() - finite.min())
    try:
        point_cam = z * np.linalg.solve(frame.K, np.array([col, row, 1.0]))
    except np.linalg.LinAlgError:
        raise ValueError("camera intrinsics singular") from None
    point = frame.pose.pos + frame.pose.rot @ point_cam
    if not np.all(np.isfinite(point)):
        raise ValueError("invalid surface point")
    result = {
        "surface_point_xyz_m": point.tolist(),
        "depth_m": z,
        "camera": frame.name,
        "capture_time_s": frame.t,
        "normalized_uv": [u, v],
        "pixel": [col, row],
        "window_px": window_px,
        "valid_depth_fraction": float(finite.size / window.size),
        "window_depth_spread_m": spread,
        "on_depth_edge": spread > EDGE_SPREAD_M,
        "method": "rig depth image at the selected pixel; no object detection",
        "limitation": "This is the visible surface under the pixel, not an object centre, "
        "grasp point or identity. On a depth edge the point may mix two "
        "surfaces; reselect away from the edge. No motion executed.",
    }
    if table_z is not None:
        result["height_above_table_m"] = float(point[2] - table_z)
    return result
