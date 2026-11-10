"""Deterministic rig-only coordinates over RGB: no object/depth/grade information.

Grid points lie on the declared table plane. Projection may overlay nearer objects;
the RGB image, not the grid, determines whether a location has physical support.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rung.contracts.frames import Pose, project


def public_tcp_marker(frame, tcp: Pose) -> np.ndarray:
    """Mark the calibrated fingertip midpoint, with no inferred object/contact data."""
    if frame.K is None or frame.pose is None or frame.convention != "opencv":
        raise ValueError("TCP marker requires calibrated opencv camera intrinsics and pose")
    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    base = Image.fromarray(rgb).copy()
    uv, depth = project(tcp.pos[None], frame.K, frame.pose)
    u, v = uv[0]
    if depth[0] > 0.005 and np.isfinite(u + v) and 0 <= u < frame.width and 0 <= v < frame.height:
        x, y = int(round(u)), int(round(v))
        draw = ImageDraw.Draw(base)
        # Four short arms leave the central pixel visible for feature selection.
        for a, b in (
            ((x - 8, y), (x - 3, y)),
            ((x + 3, y), (x + 8, y)),
            ((x, y - 8), (x, y - 3)),
            ((x, y + 3), (x, y + 8)),
        ):
            draw.line([a, b], fill=(255, 30, 220), width=2)
        draw.text(
            (x + 10, y - 14), "TCP", fill=(255, 30, 220), stroke_width=1, stroke_fill=(0, 0, 0)
        )
    return np.asarray(base)


def public_geometry_overlay(frame, tcp: Pose, workspace) -> np.ndarray:
    if frame.K is None or frame.pose is None:
        raise ValueError("public overlay requires calibrated camera intrinsics and pose")
    base = Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=max(12, min(18, frame.height // 40)))

    def pixel(point):
        uv, depth = project(np.asarray(point)[None], frame.K, frame.pose)
        u, v = uv[0]
        if depth[0] <= 0.005 or not np.isfinite(u + v):
            return None
        if not -frame.width <= u <= 2 * frame.width or not (-frame.height <= v <= 2 * frame.height):
            return None
        return (int(round(u)), int(round(v)))

    step = 0.1
    xs = np.arange(np.ceil(workspace.min_xyz[0] / step) * step, workspace.max_xyz[0] + 1e-6, step)
    ys = np.arange(np.ceil(workspace.min_xyz[1] / step) * step, workspace.max_xyz[1] + 1e-6, step)
    z = workspace.table_z
    for x in xs:
        a, b = pixel([x, workspace.min_xyz[1], z]), pixel([x, workspace.max_xyz[1], z])
        if a is not None and b is not None:
            draw.line([a, b], fill=(0, 220, 255, 95), width=1)
    for y in ys:
        a, b = pixel([workspace.min_xyz[0], y, z]), pixel([workspace.max_xyz[0], y, z])
        if a is not None and b is not None:
            draw.line([a, b], fill=(0, 220, 255, 95), width=1)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            p = pixel([x, y, z])
            if (
                ix % 2
                or iy % 2
                or p is None
                or not (0 <= p[0] < frame.width - 75 and 20 <= p[1] < frame.height - 20)
            ):
                continue
            draw.text(
                p,
                f"({x:.1f},{y:.1f})",
                font=font,
                fill=(0, 230, 255, 240),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 220),
            )
    p = pixel(tcp.pos)
    if p is not None:
        draw.line([(p[0] - 9, p[1]), (p[0] + 9, p[1])], fill=(255, 30, 220, 255), width=3)
        draw.line([(p[0], p[1] - 9), (p[0], p[1] + 9)], fill=(255, 30, 220, 255), width=3)
        draw.text(
            (p[0] + 10, p[1] - 20),
            "TCP",
            font=font,
            fill=(255, 30, 220, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
    draw.rectangle([0, 0, min(frame.width, 560), 25], fill=(0, 0, 0, 190))
    draw.text(
        (5, 3),
        f"Rig calibration: cyan (root x,y) metres at table z={z:.3f}; magenta TCP",
        font=font,
        fill=(255, 255, 255, 255),
    )
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))
