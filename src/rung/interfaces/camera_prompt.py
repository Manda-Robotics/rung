"""Camera calibration as prompt text: what Astranomy found to be the largest single effect.

NVIDIA's Astranomy report (2026-10-14) raised GPT-6 Astra's direct-policy success on a
30-task RoboLab subset from 19.3 percent (base prompt) to 62.7 percent by writing the
camera poses, intrinsics and wrist-camera tilt into the prompt, and to 72.7 percent with
gripper geometry as well. Rung already states gripper geometry; this module states the
cameras. Public rig calibration only: no object state, no depth values, no scorer.
"""

from __future__ import annotations

import numpy as np

from rung.contracts.frames import Pose


def _round(values, digits=4):
    return [round(float(v), digits) for v in np.asarray(values).reshape(-1)]


def camera_geometry_text(frames, tcp: Pose | None = None, image_scale: float = 1.0) -> str:
    """One paragraph per calibrated camera, in the robot-root frame the actions use.

    For each view: where the camera is, where it looks (optical axis, image right and
    image down directions in root axes), focal lengths and principal point at the
    delivered resolution, and for a camera mounted on the hand its pose relative to the
    TCP. Uncalibrated views are named so the model knows they are view-only.
    """
    lines = [
        "CAMERA GEOMETRY (robot-root frame: +x forward across the table, +y robot's "
        "left, +z up; OpenCV image axes: u right, v down; optical +z forward):"
    ]
    for frame in frames:
        if frame is None:
            continue
        if frame.K is None or frame.pose is None or frame.convention != "opencv":
            lines.append(
                f"- {frame.name}: RGB view only, no public calibration; use it for "
                "visual alignment, not for metric estimates."
            )
            continue
        rot = frame.pose.rot
        scale = image_scale if image_scale else 1.0
        fx, fy = frame.K[0, 0] * scale, frame.K[1, 1] * scale
        cx, cy = frame.K[0, 2] * scale, frame.K[1, 2] * scale
        width, height = round(frame.width * scale), round(frame.height * scale)
        forward, right, down = rot[:, 2], rot[:, 0], rot[:, 1]
        tilt = float(np.degrees(np.arcsin(np.clip(-forward[2], -1, 1))))
        line = (
            f"- {frame.name}: position {_round(frame.pose.pos)} m; optical axis "
            f"{_round(forward)} (tilted {tilt:.0f} deg below horizontal); image right "
            f"{_round(right)}; image down {_round(down)}; quaternion wxyz "
            f"{_round(frame.pose.quat)}; intrinsics fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} "
            f"cy {cy:.1f} px at {width}x{height}."
        )
        if frame.name == "wrist" and tcp is not None:
            relative = tcp.inverse().compose(frame.pose)
            line += (
                f" Mounted on the hand: {_round(relative.pos)} m from the TCP in TCP "
                f"axes, optical axis {_round(relative.rot[:, 2])} in TCP axes; it moves "
                "with every command."
            )
        lines.append(line)
    lines.append(
        "A pixel (u, v) maps to the ray K^-1 [u, v, 1] in camera axes, then into root "
        "axes through the rotation above; the TCP's projection in each view follows "
        "from its root position and these numbers."
    )
    return "\n".join(lines)
