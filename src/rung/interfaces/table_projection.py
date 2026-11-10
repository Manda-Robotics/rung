"""Public rig geometry only: selected image ray intersecting a declared table plane."""

import numpy as np


def reproject_public_point(point_xyz_m, frames):
    """Project a computed point into qualified RGB views; never query object state.

    Off-image and behind-camera projections remain explicit, not clipped into a
    plausible target. This is consistency feedback, not semantic/depth validation.
    """
    from rung.contracts.frames import project

    point = np.asarray(point_xyz_m, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("reprojection point must contain three finite coordinates")
    views = {}
    for frame in frames:
        if (
            frame is None
            or frame.K is None
            or frame.pose is None
            or frame.convention != "opencv"
            or frame.width < 2
            or frame.height < 2
        ):
            continue
        uv, depth = project(point[None, :], frame.K, frame.pose)
        valid = bool(depth[0] > 0 and np.all(np.isfinite(uv[0])))
        normalized = (uv[0] / [frame.width - 1, frame.height - 1]).tolist() if valid else None
        views[frame.name] = {
            "normalized_uv": normalized,
            "in_front_of_camera": valid,
            "inside_image": bool(valid and all(0 <= x <= 1 for x in normalized)),
            "width_px": frame.width,
            "height_px": frame.height,
            "capture_time_s": frame.t,
        }
    return {
        "views": views,
        "coordinate_convention": "normalized [u,v]; 0 left/top, 1 right/bottom",
        "limitation": "Calibration-only consistency check; not proof of object identity, "
        "surface depth or a correct grasp point.",
    }


def _selected_ray(frame, u, v, capture_time_s):
    for value in (u, v, capture_time_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("pixel coordinates/time must be finite numbers")
    if not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError("normalized pixel coordinates must be in [0,1]")
    if frame is None:
        raise ValueError("requested camera unavailable")
    if abs(capture_time_s - frame.t) > 1e-6:
        raise ValueError("stale camera capture time")
    if frame.K is None or frame.pose is None or frame.convention != "opencv":
        raise ValueError("qualified OpenCV camera calibration required")
    pixel = np.array([u * (frame.width - 1), v * (frame.height - 1), 1.0])
    try:
        ray = frame.pose.rot @ np.linalg.solve(frame.K, pixel)
    except np.linalg.LinAlgError:
        raise ValueError("camera intrinsics singular") from None
    if not np.all(np.isfinite(ray)):
        raise ValueError("invalid calibrated ray")
    return ray


def project_table_pixel(frame, u, v, capture_time_s, table_z):
    ray = _selected_ray(frame, u, v, capture_time_s)
    if abs(ray[2]) < 1e-9:
        raise ValueError("ray parallel to table")
    distance = (table_z - frame.pose.pos[2]) / ray[2]
    if distance <= 0:
        raise ValueError("table intersection behind camera")
    point = frame.pose.pos + distance * ray
    if not np.all(np.isfinite(point)):
        raise ValueError("invalid table intersection")
    return {
        "table_intersection_xyz_m": point.tolist(),
        "camera": frame.name,
        "capture_time_s": frame.t,
        "normalized_uv": [u, v],
        "method": "calibrated ray / declared table plane; no depth or object lookup",
        "limitation": "This is the table under the selected ray, not object surface height; "
        "occlusion and raised objects cause parallax. No motion executed.",
    }


def triangulate_selected_pixels(scene, wrist, scene_uv, wrist_uv, capture_time_s):
    """Two model-selected corresponding RGB pixels; calibration only, no depth lookup."""
    rays = []
    for frame, uv in ((scene, scene_uv), (wrist, wrist_uv)):
        if not isinstance(uv, list) or len(uv) != 2:
            raise ValueError("normalized image coordinate needs two numbers")
        ray = _selected_ray(frame, *uv, capture_time_s)
        rays.append(ray / np.linalg.norm(ray))
    d1, d2 = rays
    angle = float(np.arccos(np.clip(abs(d1 @ d2), 0, 1)))
    if angle < np.deg2rad(5):
        raise ValueError("selected rays nearly parallel; triangulation ill-conditioned")
    distances, *_ = np.linalg.lstsq(
        np.column_stack((d1, -d2)), wrist.pose.pos - scene.pose.pos, rcond=None
    )
    if np.any(distances <= 0):
        raise ValueError("selected correspondence triangulates behind camera")
    p1 = scene.pose.pos + distances[0] * d1
    p2 = wrist.pose.pos + distances[1] * d2
    gap = float(np.linalg.norm(p1 - p2))
    if gap > 0.025:
        raise ValueError(f"selected rays miss by {gap:.4f} m; reselect corresponding pixels")
    return {
        "selected_point_xyz_m": ((p1 + p2) / 2).tolist(),
        "ray_gap_m": gap,
        "ray_angle_rad": angle,
        "capture_time_s": capture_time_s,
        "method": "two model-selected RGB rays / public camera calibration; no depth",
        "limitation": "Pixel correspondence is your choice, not an object detector. "
        "A small ray gap does not establish object identity. No motion executed.",
    }
