"""From a detection to metric geometry: masks, point clouds, extents, principal axis.

Two failure modes are guarded against:

* Neighbour merging. A box around a bottle can catch part of a jug 13 cm behind it, and
  the merged cluster measures 24 by 10 cm instead of 5 by 5. Every cluster is flood
  filled from a seed and unconnected points are dropped.
* Hand contamination. With the gripper over the object the camera sees the fingers, and a
  naive cluster floods into them. Points near the current TCP are removed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rung.contracts.frames import Pose, deproject
from rung.contracts.observation import CameraFrame

LINK_M = 0.02
"""Two points belong to the same object if they are within this distance. At 4 cm a
bottle and an adjacent jug merge; 2 cm keeps them apart."""

HAND_EXCLUSION_M = 0.055
"""Points this close to the TCP are the gripper, not the object."""


def mask_from_box(
    box: tuple[float, float, float, float], height: int, width: int
) -> npt.NDArray[np.bool_]:
    u0, v0, u1, v1 = box
    m = np.zeros((height, width), dtype=bool)
    x0, x1 = sorted((int(np.floor(u0)), int(np.ceil(u1))))
    y0, y1 = sorted((int(np.floor(v0)), int(np.ceil(v1))))
    m[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    return m


def points_from_mask(
    cam: CameraFrame,
    mask: npt.NDArray[np.bool_],
    exclude_near: npt.ArrayLike | None = None,
    exclude_radius: float = HAND_EXCLUSION_M,
) -> tuple[npt.NDArray[np.float64], float]:
    """Deproject the masked pixels. Returns ``(points, visible_fraction)``.

    ``visible_fraction`` is the share of masked pixels that had usable depth, before the
    hand exclusion, and is the signal that the object is partly hidden.
    """
    if cam.depth is None or cam.K is None or cam.pose is None:
        raise ValueError(f"camera {cam.name} has no depth or calibration")
    d = np.asarray(cam.depth, dtype=np.float64)
    valid = mask & np.isfinite(d) & (d > 0)
    total = int(mask.sum())
    if total == 0 or not valid.any():
        return np.zeros((0, 3)), 0.0
    vs, us = np.nonzero(valid)
    uv = np.stack([us, vs], axis=-1).astype(np.float64)
    pts = deproject(uv, d[valid], cam.K, cam.pose)
    fraction = float(valid.sum() / total)
    if exclude_near is not None and len(pts):
        tcp = np.asarray(exclude_near, dtype=np.float64).reshape(3)
        pts = pts[np.linalg.norm(pts - tcp, axis=1) > exclude_radius]
    return pts, fraction


def component_labels(points: npt.ArrayLike, link_m: float = LINK_M) -> npt.NDArray[np.int64] | None:
    """Connected-component label per point, linking points within ``link_m``. Exact.

    None when scipy is not installed; callers fall back to the pure-Python fill. On a
    12-object scene the KD-tree takes about 0.1 s where the Python fill takes about 4.5 s,
    time the robot spends stationary in real-time runs.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(pts).query_pairs(link_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def connected_component(
    points: npt.ArrayLike, seed: npt.ArrayLike, link_m: float = LINK_M, max_points: int = 20000
) -> npt.NDArray[np.float64]:
    """Points reachable from the seed by hops of at most ``link_m``.

    Uses ``component_labels`` when scipy is present, and otherwise a grid-bucketed flood
    fill: linking distance sets the cell size, so each point only ever compares against its
    27 neighbouring cells. Both give the same set.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    if len(pts) > max_points:  # deterministic thinning, never random
        pts = pts[:: int(np.ceil(len(pts) / max_points))]
    seed = np.asarray(seed, dtype=np.float64).reshape(3)
    start = int(np.argmin(np.linalg.norm(pts - seed, axis=1)))
    if np.linalg.norm(pts[start] - seed) > link_m * 3:
        return np.zeros((0, 3))

    labels = component_labels(pts, link_m)
    if labels is not None:
        return pts[labels == labels[start]]
    return _flood_fill(pts, start, link_m)


def _flood_fill(pts: npt.NDArray[np.float64], start: int, link_m: float) -> npt.NDArray[np.float64]:
    cell = max(link_m, 1e-6)
    keys = np.floor(pts / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    seen = np.zeros(len(pts), dtype=bool)
    seen[start] = True
    stack = [start]
    neighbourhood = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    while stack:
        i = stack.pop()
        ki = keys[i]
        for off in neighbourhood:
            for j in buckets.get((ki[0] + off[0], ki[1] + off[1], ki[2] + off[2]), ()):
                if not seen[j] and np.linalg.norm(pts[j] - pts[i]) <= link_m:
                    seen[j] = True
                    stack.append(j)
    return pts[seen]


def principal_axis_xy(
    points: npt.ArrayLike, min_points: int = 25, min_elongation: float = 1.3
) -> npt.NDArray[np.float64] | None:
    """The dominant horizontal direction, or None when the shape is too round to have one.

    Forcing an axis onto a round object would produce a jaw angle driven by noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < min_points:
        return None
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov = np.cov(xy.T)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[1] <= 1e-12:
        return vecs[:, 0]
    if np.sqrt(vals[0] / vals[1]) < min_elongation:
        return None
    return vecs[:, 0]


def extents(
    points: npt.ArrayLike, axis: npt.ArrayLike | None = None
) -> tuple[float, float, float, float, float]:
    """``(top_z, bottom_z, length, width, height)`` measured along ``axis`` and across it.

    With no axis, length and width are the axis-aligned spans, which is the right answer
    for a round object and a harmless approximation for anything else.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    top = float(np.percentile(pts[:, 2], 98))
    bottom = float(np.percentile(pts[:, 2], 2))
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    if axis is None:
        span_a = float(np.ptp(xy[:, 0]))
        span_b = float(np.ptp(xy[:, 1]))
    else:
        a = np.asarray(axis, dtype=np.float64).reshape(2)
        a = a / np.linalg.norm(a)
        b = np.array([-a[1], a[0]])
        span_a = float(np.ptp(xy @ a))
        span_b = float(np.ptp(xy @ b))
    length, width = max(span_a, span_b), min(span_a, span_b)
    return (top, bottom, length, width, float(top - bottom))


def support_height(points: npt.ArrayLike, table_z: float) -> float:
    """Height of the surface a released object would land on, from a container's cloud.

    Uses a low percentile of the interior rather than the minimum, because the lowest
    returned point inside a bin is usually a speck of the table seen through a gap.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return float(table_z)
    return float(max(table_z, np.percentile(pts[:, 2], 5)))


def rim_model(points: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], float] | None:
    """A container's ``(centre_xy_z, rim_height)``.

    The centre comes from the rim ring, area-weighted. From one oblique camera the far
    inner wall is visible and the near one is not, so the cloud is denser on the far side
    and point-weighted statistics drift away from the camera (the whole-cloud median is 2
    to 3 cm off on a real bowl). The rim edge is visible all round, and counting occupied
    cells rather than points removes the density bias (0.1 cm error on the same bowl).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 50:
        return None
    rim = float(np.percentile(pts[:, 2], 95))
    ring = top_slab(pts, rim)
    centre_xy = footprint_centre(ring if len(ring) >= 8 else pts)
    return np.array([centre_xy[0], centre_xy[1], rim]), rim


TABLE_CUT_M = 0.008
"""Points this close above the table plane are the table, not an object standing on it.

Every object touches the table, so a flood fill linking points 2 cm apart walks from the
object's base out across the table inside the detection box; a 6 cm onion then measures
26 by 11 cm. The margin is slightly above the plane's depth noise, measured under 0.5 cm
on the real rig.
"""

CELL_M = 0.005
"""Cell size for area-weighted statistics."""

TOP_SLAB_M = 0.015
"""How far below an object's top the top slab reaches: the part of an object a single
camera sees completely."""


def above_table(points: npt.ArrayLike, table_z: float, margin_m: float = TABLE_CUT_M):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    return pts[pts[:, 2] > table_z + margin_m]


def top_slab(points: npt.ArrayLike, top_z: float, slab_m: float = TOP_SLAB_M):
    """Points within ``slab_m`` below ``top_z``, excluding any above it.

    ``top_z`` is a high percentile; the few returns above it are edge pixels mixing the
    object and the background, which widen a rim by several millimetres."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    return pts[(pts[:, 2] >= top_z - slab_m) & (pts[:, 2] <= top_z + 1e-6)]


def footprint_centre(points: npt.ArrayLike, cell_m: float = CELL_M) -> npt.NDArray[np.float64]:
    """The horizontal centre of a cloud, weighted by area rather than by point count.

    A depth camera returns many points from surfaces it faces and few from surfaces seen
    edge-on, so the plain mean is biased by sampling density (on a real rig, 3.6 cm short
    of a banana's top slab and 1.8 cm past a bowl). Counting each occupied cell once makes
    the centre independent of sampling.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(2)
    cells = np.unique(np.floor(pts[:, :2] / cell_m).astype(np.int64), axis=0)
    return (cells + 0.5).mean(axis=0) * cell_m


LOCAL_HALF_M = 0.03
"""How far along the finger pads, either side of the grasp point, the jaw's width is
measured: the 2.2 cm pad width plus centring error."""


def grasp_point(crown: npt.ArrayLike, cell_m: float = CELL_M) -> npt.NDArray[np.float64]:
    """Horizontal aim point on an object's top: the area-weighted centre, moved onto the object.

    The footprint centre of a curved object such as a banana lies inside the curve, off
    the object, and a grasp aimed there closes on one edge. If the centre is more than
    1 cm from any top point, the aim moves to the nearest one and re-centres on the stretch
    around it.
    """
    pts = np.asarray(crown, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(2)
    centre = footprint_centre(pts, cell_m)
    d = np.linalg.norm(pts[:, :2] - centre, axis=1)
    if float(d.min()) > 0.01:
        centre = pts[int(np.argmin(d)), :2]
        near = pts[np.linalg.norm(pts[:, :2] - centre, axis=1) <= LOCAL_HALF_M]
        if len(near) >= 3:
            centre = footprint_centre(near, cell_m)
    return centre


def local_span(
    points: npt.ArrayLike,
    centre_xy: npt.ArrayLike,
    closing: npt.ArrayLike,
    half_m: float = LOCAL_HALF_M,
) -> float | None:
    """The object's extent along ``closing`` over the stretch the pads touch.

    Uses points within ``half_m`` of the grasp point along the pad direction; None if
    fewer than 8. The whole-object span overstates curved or tapered objects: a curved
    banana spans 7.7 cm over its full length and 4.5 cm where the fingers close."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return None
    c = np.asarray(closing, dtype=np.float64)[:2]
    c = c / max(1e-9, np.linalg.norm(c))
    perp = np.array([-c[1], c[0]])
    xy = pts[:, :2] - np.asarray(centre_xy, dtype=np.float64)[:2]
    sel = np.abs(xy @ perp) <= half_m
    if sel.sum() < 8:
        return None
    return float(np.ptp(xy[sel] @ c))


def estimate_table_z(
    cam: CameraFrame, centre_xy=(0.5, 0.0), half_m: float = 0.35, fallback: float = 0.0
) -> tuple[float, float]:
    """``(table_z, share)`` of the dominant horizontal surface in front of the robot.

    Estimated from one depth image; ``share`` is the fraction of nearby points within 2 cm
    of the plane. The table need not be at the robot's base height (some tasks place it
    5.5 cm up). Assuming zero merges the table into every object, inflating footprints
    past the jaw and offsetting grasp floors and collision checks by the same amount.
    """
    from rung.contracts.frames import depth_image_to_points

    if cam.depth is None or cam.K is None or cam.pose is None:
        return fallback, 0.0
    pts, _ = depth_image_to_points(cam.depth, cam.K, cam.pose, stride=4)
    pts = pts[np.isfinite(pts).all(axis=1)]
    near = pts[
        (np.abs(pts[:, 0] - centre_xy[0]) < half_m)
        & (np.abs(pts[:, 1] - centre_xy[1]) < half_m)
        & (pts[:, 2] > -0.10)
        & (pts[:, 2] < 0.40)
    ]
    if len(near) < 200:
        return fallback, 0.0
    hist, edges = np.histogram(near[:, 2], bins=100, range=(-0.10, 0.40))
    peak = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    band = near[np.abs(near[:, 2] - peak) < 0.01]
    z = float(np.median(band[:, 2])) if len(band) else peak
    share = float(np.mean(np.abs(near[:, 2] - z) < 0.02))
    if share < 0.2:
        return fallback, share
    return z, share


def view_direction_xy(camera_pos: npt.ArrayLike | None, at: npt.ArrayLike):
    """Unit horizontal direction from the camera to a point, or None if undefined."""
    if camera_pos is None:
        return None
    d = np.asarray(at, dtype=np.float64)[:2] - np.asarray(camera_pos, dtype=np.float64)[:2]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else None


def view_elevation(camera_pos: npt.ArrayLike | None, at: npt.ArrayLike) -> float | None:
    """How steeply the camera looks down at a point, in radians above horizontal."""
    if camera_pos is None:
        return None
    cam = np.asarray(camera_pos, dtype=np.float64)
    at = np.asarray(at, dtype=np.float64)
    flat = float(np.linalg.norm(at[:2] - cam[:2]))
    return float(np.arctan2(cam[2] - at[2], max(flat, 1e-6)))


def unseen_fraction(elevation_rad: float | None) -> float:
    """Relative amount a convex object can extend beyond its visible span along the view.

    For a round object of radius r seen at elevation e, the surface beyond the tangent
    point hides r(1 - sin e): a quarter of the visible span at 30 degrees, nothing from
    directly above. Unknown elevation is treated as 30 degrees."""
    if elevation_rad is None:
        return 0.25
    e = float(np.clip(elevation_rad, 0.0, np.pi / 2))
    return 0.5 * (1.0 - np.sin(e))


ACROSS_VIEW_DEG = 55.0
"""A closing direction at least this far from the camera's line of sight measures the
object's silhouette, which one camera sees completely. Nearer the line of sight than
this, a ball's hidden far side causes significant error: 10 percent at 45 degrees on a
synthetic 7 cm ball."""

FAR_BAND_FRACTION = 0.25
FAR_BAND_MIN_M = 0.005
"""The far end of the top slab is judged over its far quarter (at least 5 mm), not its
last point: a single edge pixel can sit below the top on a perfectly flat rim."""

FAR_SLOPE = -0.15
"""How steeply the top surface may fall, going away from the camera, over the far part
of the top slab before it counts as curving out of view. A flat top has slope zero to
its edge; a rim's inner wall makes the slope positive; a dome's far edge falls at the
tangent of the camera's elevation, about -0.5 at 30 degrees, and a 4 cm banana's
ridge falls at -0.3. Height alone cannot separate these: near its top a small dome is
flat to under a millimetre."""

FLAT_TOP_FRACTION = 0.7
"""Minimum ratio of top-slab span to visible band span for the top face to count as
reaching the far edge: a top face well short of the footprint is a cap, not a lid."""


def trusted_width_direction(
    band: npt.ArrayLike,
    top: npt.ArrayLike,
    view_dir,
    n_angles: int = 180,
    fits_m: float | None = None,
    elevation_rad: float | None = None,
) -> tuple[float, float, npt.NDArray[np.float64], bool]:
    """``(width, length, closing_direction, bounded)`` from a single viewpoint.

    One camera sees an object's near side and top; the far side is hidden. The span across
    the line of sight is the silhouette and is exact. The span along it is a lower bound,
    unless the top face reaches the far edge (a box, a mug, a bottle), in which case the
    top face's extent is used. A dome-topped object (an onion, a ball) has no such edge.

    Partly seen directions are kept with an upper bound: their span scaled by
    ``unseen_fraction`` of the camera elevation. Discarding them wrongly rejects an
    elongated object lying along the line of sight, whose silhouette across the view is
    its length.

    Preference: the silhouette when it fits the jaw (``fits_m``); then a direction the
    top face vouches for; then the narrowest bounded direction. ``bounded`` marks the
    width as an upper bound rather than a measurement.

    Closing across the line of sight also places the near-side centring error along the
    finger pads, where it is harmless; closing along it can land a finger on the object.
    """
    band = np.asarray(band, dtype=np.float64)
    if len(band) < 3:
        return (0.0, 0.0, np.array([1.0, 0.0]), False)
    top = np.asarray(top, dtype=np.float64)
    centre = band[:, :2].mean(axis=0)
    xy = band[:, :2] - centre
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    proj = xy @ dirs.T
    spans = np.ptp(proj, axis=0)
    top_spans = None
    if len(top) >= 3:
        top_spans = np.ptp((top[:, :2] - centre) @ dirs.T, axis=0)
    if view_dir is None:
        # No camera to reason about. The top face still says how wide a prismatic
        # object is where a lower band shows only its near faces.
        measured = spans.copy()
        if top_spans is not None:
            measured = np.where(
                top_spans >= FLAT_TOP_FRACTION * spans, np.maximum(spans, top_spans), spans
            )
        i = int(np.argmin(measured))
        perp = np.array([-dirs[i, 1], dirs[i, 0]])
        return (float(measured[i]), float(np.ptp(xy @ perp)), dirs[i], False)

    view = np.asarray(view_dir, dtype=np.float64) / max(1e-9, np.linalg.norm(view_dir))
    across = np.abs(dirs @ view) <= np.cos(np.radians(ACROSS_VIEW_DEG))
    flat = np.zeros(len(dirs), dtype=bool)
    measured = spans.copy()
    if top_spans is not None:
        wide_enough = top_spans >= FLAT_TOP_FRACTION * spans
        # Does the top surface stay level to its far edge, looking away from the camera,
        # or fall away? Judged once, along the line of sight: that is where a dome
        # curves out of view, and a level top is level in every direction. Judging it
        # per direction lets an elongated object's length leak into the slope.
        along = (top[:, :2] - centre) @ view
        reach = max(FAR_BAND_MIN_M, FAR_BAND_FRACTION * float(np.ptp(along)))
        sel = along >= along.max() - reach
        level = False
        if sel.sum() >= 5:
            a = along[sel]
            level = np.ptp(a) < 1e-4 or float(np.polyfit(a, top[sel, 2], 1)[0]) >= FAR_SLOPE
        flat = wide_enough & level
        # Where the top face is what makes a direction measurable, the width in that
        # direction is the top face's: a band lower down sees only the near faces of a
        # box, and their outline is narrower than the box (4.7 cm for a 7 cm bottle).
        measured = np.where(flat, np.maximum(spans, top_spans), spans)
    ok = across | flat
    # Partly seen directions carry an upper bound instead of a measurement.
    bound = spans * (1.0 + unseen_fraction(elevation_rad))
    measured = np.where(ok, measured, bound)

    def pick(mask):
        return int(np.argmin(np.where(mask, measured, np.inf)))

    everything = np.ones(len(dirs), dtype=bool)
    if fits_m is None:
        i = pick(everything)
    else:
        i = pick(across) if across.any() else pick(everything)
        if measured[i] > fits_m and ok.any():
            i = pick(ok)
        if measured[i] > fits_m:
            i = pick(everything)
    bounded = not bool(ok[i])
    width = float(measured[i])
    perp = np.array([-dirs[i, 1], dirs[i, 0]])
    length = float(np.ptp(xy @ perp))
    return (width, length, dirs[i], bounded)


def plausible_container(
    centre: npt.ArrayLike, rim_z: float, pointed_at: npt.ArrayLike, max_offset_m: float = 0.30
) -> bool:
    """Reject a container model that floods across a cluttered table.

    The fitted centre must lie within ``max_offset_m`` of the point the model named.
    """
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    p = np.asarray(pointed_at, dtype=np.float64).reshape(3)
    return bool(np.linalg.norm(c[:2] - p[:2]) < max_offset_m and np.isfinite(rim_z))


def visible_from(cam: CameraFrame, point: npt.ArrayLike, tolerance_m: float = 0.03) -> bool:
    """Is this 3D point the nearest surface along its own ray? Used for occlusion checks."""
    from rung.contracts.frames import project

    if cam.depth is None or cam.K is None or cam.pose is None:
        return False
    uv, z = project(np.asarray(point, dtype=np.float64)[None], cam.K, cam.pose)
    u, v = int(round(float(uv[0, 0]))), int(round(float(uv[0, 1])))
    if not (0 <= u < cam.width and 0 <= v < cam.height) or z[0] <= 0:
        return False
    d = float(cam.depth[v, u])
    return bool(np.isfinite(d) and d >= float(z[0]) - tolerance_m)


def tcp_of(pose: Pose) -> npt.NDArray[np.float64]:
    return np.asarray(pose.pos, dtype=np.float64)


def cross_section(
    points: npt.ArrayLike, z_hi: float, slab_m: float = 0.025
) -> npt.NDArray[np.float64]:
    """Points in a horizontal slab just below ``z_hi``.

    A single camera sees an object's top face and one or two sides, so spans over the whole
    cloud inflate the footprint: a 7 cm bottle measures 10 cm across because the visible
    side wall counts as horizontal extent. Jaw fit depends on the width where the jaw
    closes, so that slab is measured. Falls back to all points if the slab has fewer than 8.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    slab = pts[(pts[:, 2] <= z_hi + 1e-6) & (pts[:, 2] >= z_hi - slab_m)]
    return slab if len(slab) >= 8 else pts


def min_width_direction(
    points: npt.ArrayLike, n_angles: int = 180
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """The narrowest horizontal width of a cloud, over every jaw orientation.

    Returns ``(width, length, closing_direction)``: the smallest span across any direction,
    the span perpendicular to it, and the unit vector the jaw should close along.

    Projecting onto the principal axis fails for boxy objects: a 7 by 7 cm bottle whose
    samples favour a diagonal measures 9.9 cm that way. The jaw can take any yaw, so the
    minimum over all directions is used, together with its direction.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return (0.0, 0.0, np.array([1.0, 0.0]))
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    proj = xy @ dirs.T
    spans = proj.max(axis=0) - proj.min(axis=0)
    i = int(np.argmin(spans))
    width = float(spans[i])
    perp = np.array([-dirs[i, 1], dirs[i, 0]])
    length = float(np.ptp(xy @ perp))
    return (width, length, dirs[i])
